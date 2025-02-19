import math
import numpy as np  # type: ignore
from scipy.interpolate import UnivariateSpline, CubicSpline, PchipInterpolator # type: ignore
from scipy.spatial import cKDTree #type: ignore
from scipy.integrate import cumulative_trapezoid #type: ignore
from functools import lru_cache


class CurveReducer:
    
    @staticmethod
    def pow_curve(x: list[float], p: float) -> list[float]:
        """
        Custom Power Function (Normalized)

        Args:
            x (list[float]): _description_
            p (float): _description_

        Returns:
            list[float]: _description_
        """
        return [(x[i] ** p) for i in range(len(x))]

    @staticmethod
    def sigmoid_curve(x: list[float], p: float, c: float) -> list[float]:
        """
        Sigmoid-like Transformation (Normalized)
        """
        return [x[i] / (1 + c * x[i] ** p) for i in range(len(x))]

    @staticmethod
    def log_curve(x: list[float], k: float) -> list[float]:
        """
        Logarithmic Compression (Normalized)
        """
        return [math.log(1 + k * x[i]) / math.log(1 + k) for i in range(len(x))]


    @staticmethod
    def spline_from_points(x_values: list[float], current_POS: list[float], smoothing_factor: float = 0.01) -> list[float]:
        """
        Smoothing Spline from Points (Normalized)
        """        
        
        #smoothing_factor = 0.01
        spline_function_smooth = UnivariateSpline(x_values, current_POS, s=smoothing_factor)
        
        return spline_function_smooth
  
        """ 
        from scipy.interpolate import UnivariateSpline

        # Create a smoothing spline with a regularization parameter
        smoothing_factor = 0.01  # Adjust this based on data
        spline_function_smooth = UnivariateSpline(x_values, current_POS, s=smoothing_factor)

        # Plot the smoothed function
        test_x = np.linspace(0, 1, 100)
        plt.plot(x_values, current_POS, 'o', label="Original Data")
        plt.plot(test_x, spline_function_smooth(test_x), '-', label="Smoothed Spline")
        plt.legend()
        plt.show() """
        
    @staticmethod
    def spline_from_points_cubic(x_values: list[float], current_POS: list[float]) -> list[float]:
        """
        Smoothing Spline from Points (Normalized)
        """        
        
        #smoothing_factor = 0.01
        spline_function_smooth = CubicSpline(x_values, current_POS)
        
        return spline_function_smooth
        
    @staticmethod
    def interpolate_spline(spline, dataX) -> list[float]:
        """
        Interpolate Spline
        """
        return spline(dataX)

class CurveEstimator3D:
    def __init__(self):
        self.points = None
        self.curve_length = None
        self.spline = None
        self.spline_derivs = None  # Precomputed derivatives
        self.t_values = None
        self.arc_spline = None
        self.sampled_points = None
        self.s_uniform = None  # Stored arc-length parameters
        self.kd_tree = None
        self._cached_queries = {}  # Cache for nearest-neighbor queries

    def calculate_curve_length(self, points: np.ndarray, 
                              integration_samples: int = 10000,
                              query_samples: int = 2000):
        """
        Enhanced version with separate sampling for integration and querying.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input must be Nx3 array.")
        if len(points) < 2:
            raise ValueError("At least two points required.")

        self.points = points

        # Chord-length parameterization with vectorized operations
        deltas = np.diff(points, axis=0)
        chord_distances = np.linalg.norm(deltas, axis=1)
        chord_lengths = np.concatenate(([0], np.cumsum(chord_distances)))
        self.t_values = chord_lengths / chord_lengths[-1]

        # Fit splines and precompute derivatives
        self.spline = [CubicSpline(self.t_values, points[:, dim]) for dim in range(3)]
        self.spline_derivs = [s.derivative() for s in self.spline]

        # High-precision arc-length parameterization
        self.parameterize_by_arc_length(integration_samples)

        # Optimized sampling for queries
        self.calculate_sampled_points(query_samples)

    def parameterize_by_arc_length(self, num_samples: int = 10000):
        """Vectorized integration with adaptive sampling."""
        t_samples = np.linspace(0, 1, num_samples)

        # Vectorized derivative calculation
        dx = np.array([d(t_samples) for d in self.spline_derivs])
        speeds = np.linalg.norm(dx, axis=0)

        # High-precision cumulative integration
        arc_lengths = cumulative_trapezoid(speeds, t_samples, initial=0)
        self.curve_length = arc_lengths[-1]
        arc_lengths_normalized = arc_lengths / self.curve_length

        # Monotonic cubic interpolation
        self.arc_spline = PchipInterpolator(arc_lengths_normalized, t_samples)

    def calculate_sampled_points(self, num_samples: int = 2000):
        """Optimized uniform sampling with precomputed parameters."""
        self.s_uniform = np.linspace(0, 1, num_samples)
        t_from_s = self.arc_spline(self.s_uniform)

        # Vectorized spline evaluation
        self.sampled_points = np.vstack([
            s(t_from_s) for s in self.spline
        ]).T

        self.kd_tree = cKDTree(self.sampled_points)
        self._cached_queries.clear()  # Clear cache when data changes

    @lru_cache(maxsize=1024)
    def interpolate_point_by_percentage(self, percentage: float) -> np.ndarray:
        """Precision-optimized interpolation with caching."""
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be in [0, 1].")

        t = self.arc_spline(percentage)
        return np.array([s(t) for s in self.spline])

    def calculate_percentage(self, target_xyz: np.ndarray) -> float:
        """Optimized query with caching and multi-threaded KD-Tree search."""
        target_key = tuple(target_xyz)
        if target_key in self._cached_queries:
            return self._cached_queries[target_key]

        _, idx = self.kd_tree.query(target_xyz, workers=-1)
        percentage = self.s_uniform[idx]

        # Cache the result
        self._cached_queries[target_key] = percentage
        return percentage
