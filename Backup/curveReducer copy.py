import math
import numpy as np  # type: ignore
from scipy.interpolate import UnivariateSpline, CubicSpline, PchipInterpolator # type: ignore

from scipy.spatial.distance import cdist # type: ignore
from scipy.integrate import quad # type: ignore
from scipy.spatial import cKDTree #type: ignore
from scipy.integrate import cumulative_trapezoid #type: ignore


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


class CurveEstimator:
    def __init__(self):
        self.points = None
        self.distances = None
        self.cumulative_lengths = None
        self.curve_length = None

    def calculate_curve_length(self, points: np.ndarray):
        """Calculates the total length of a 3D curve given by discrete points."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input points must have a shape of Nx3 (representing XYZ coordinates).")

        self.points = points
        self.distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.cumulative_lengths = np.concatenate(([0], np.cumsum(self.distances)))  # Prepend 0 for easier lookups
        self.curve_length = self.cumulative_lengths[-1]

    def interpolate_point_on_curve(self, percentage: float) -> np.ndarray:
        """Interpolates a point along a 3D curve at a specified percentage of its total length."""
        if self.points is None:
            raise RuntimeError("Curve length must be calculated first using calculate_curve_length().")
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1.")

        target_length = percentage * self.curve_length
        if target_length == 0:
            return self.points[0]
        if target_length == self.curve_length:
            return self.points[-1]

        segment_index = np.searchsorted(self.cumulative_lengths, target_length) - 1
        p1, p2 = self.points[segment_index], self.points[segment_index + 1]

        t = (target_length - self.cumulative_lengths[segment_index]) / self.distances[segment_index]
        return p1 + t * (p2 - p1)  # Linear interpolation

    def calculate_percentage(self, target_xyz: np.ndarray) -> float:
        """Finds the percentage along the curve that is closest to a given point."""
        if self.points is None:
            raise RuntimeError("Curve length must be calculated first using calculate_curve_length().")

        target_xyz = np.asarray(target_xyz)
        p1, p2 = self.points[:-1], self.points[1:]
        segment_vectors = p2 - p1
        segment_lengths_sq = np.einsum("ij,ij->i", segment_vectors, segment_vectors)  # Fast dot product
        segment_lengths_sq = np.where(segment_lengths_sq == 0, 1, segment_lengths_sq)  # Avoid div by zero

        target_vectors = target_xyz - p1
        t = np.einsum("ij,ij->i", target_vectors, segment_vectors) / segment_lengths_sq
        t = np.clip(t, 0, 1)

        closest_points = p1 + t[:, None] * segment_vectors
        distances = np.linalg.norm(closest_points - target_xyz, axis=1)

        closest_idx = np.argmin(distances)
        length_to_closest_point = self.cumulative_lengths[closest_idx] + t[closest_idx] * self.distances[closest_idx]

        return length_to_closest_point / self.curve_length

    @staticmethod
    def _closest_point_on_segment(p1: np.ndarray, p2: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Finds the closest point on a line segment [p1, p2] to a target point."""
        p1, p2, target = map(np.asarray, (p1, p2, target))
        segment = p2 - p1
        segment_length_sq = segment @ segment  # Equivalent to np.dot(segment, segment)

        if segment_length_sq == 0:
            return p1  # Return endpoint if segment is degenerate

        t = (target - p1) @ segment / segment_length_sq
        return p1 + np.clip(t, 0, 1) * segment  # Clamped interpolation


""" import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree """
from functools import lru_cache

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


# optimized by chatgpt
class CurveEstimator3D_D:
    def __init__(self):
        self.points = None
        self.curve_length = None  # True arc length of the spline
        self.spline = None
        self.t_values = None  # Chord-length parameterization (normalized)
        self.arc_spline = None  # Maps arc-length percentage to t parameter
        self.sampled_points = None  # Points uniformly sampled by arc length
        self.kd_tree = None  # KD-Tree for nearest-neighbor search

    def calculate_curve_length(self, points: np.ndarray, num_samples: int = 1000):
        """
        Fits a cubic spline using chord-length parameterization and computes the true arc length.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input must be Nx3 array.")
        if len(points) < 2:
            raise ValueError("At least two points required.")

        self.points = points

        # Chord-length parameterization (for spline fitting)
        chord_distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        chord_lengths = np.insert(np.cumsum(chord_distances), 0, 0)
        self.t_values = chord_lengths / chord_lengths[-1]  # Normalized t [0, 1]

        # Fit cubic splines
        self.spline = [CubicSpline(self.t_values, points[:, dim]) for dim in range(3)]

        # Compute true arc length and parameterization
        self.parameterize_by_arc_length(num_samples)

        # Sample points uniformly in arc length for KD-Tree
        self.calculate_sampled_points(num_samples)

    def arc_length_integrand(self, t):
        """Vectorized speed function ||dx/dt||."""
        derivatives = np.array([s.derivative()(t) for s in self.spline])
        return np.linalg.norm(derivatives, axis=0)  # Vectorized norm calculation

    def parameterize_by_arc_length(self, num_samples: int = 1000):
        """
        Computes the true arc length of the spline and builds a mapping from
        arc-length percentage to t parameter.
        """
        # High-resolution sampling for accurate integration
        t_samples = np.linspace(0, 1, num_samples)
        speeds = self.arc_length_integrand(t_samples)
        arc_lengths = cumulative_trapezoid(speeds, t_samples, initial=0)

        # Set true curve length and normalize
        self.curve_length = arc_lengths[-1]
        arc_lengths_normalized = arc_lengths / self.curve_length

        # Spline mapping: arc-length percentage (s) -> t parameter
        self.arc_spline = PchipInterpolator(arc_lengths_normalized, t_samples)

    def calculate_sampled_points(self, num_samples: int = 1000):
        """Samples points uniformly in arc length for KD-Tree."""
        if self.arc_spline is None:
            raise RuntimeError("Call parameterize_by_arc_length first.")

        # Uniform arc-length percentages
        s_uniform = np.linspace(0, 1, num_samples)

        # Convert to t parameters
        t_from_s = self.arc_spline(s_uniform)

        # Evaluate spline at these t values
        self.sampled_points = np.stack(
            [s(t_from_s) for s in self.spline], axis=-1
        )

        # Build KD-Tree
        self.kd_tree = cKDTree(self.sampled_points)

    def interpolate_point_by_percentage(self, percentage: float) -> np.ndarray:
        """Interpolates a point at the specified arc-length percentage."""
        if self.arc_spline is None:
            raise RuntimeError("Curve must be calculated first.")
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be in [0, 1].")

        # Convert percentage to t parameter
        t_interp = self.arc_spline(percentage)

        # Evaluate spline
        return np.array([s(t_interp) for s in self.spline])

    def calculate_percentage(self, target_xyz: np.ndarray) -> float:
        """Finds the closest arc-length percentage to a 3D point."""
        if self.kd_tree is None:
            raise RuntimeError("Sampled points not initialized.")

        _, closest_idx = self.kd_tree.query(target_xyz)
        s_uniform = np.linspace(0, 1, len(self.sampled_points))
        return s_uniform[closest_idx]  # More accurate mapping

# made by deepseek
class CurveEstimator3D_C:
    def __init__(self):
        self.points = None
        self.curve_length = None  # True arc length of the spline
        self.spline = None
        self.t_values = None  # Chord-length parameterization (normalized)
        self.arc_spline = None  # Maps arc-length percentage to t parameter
        self.sampled_points = None  # Points uniformly sampled by arc length
        self.kd_tree = None  # KD-Tree for nearest-neighbor search

    def calculate_curve_length(self, points: np.ndarray, num_samples: int = 1000):
        """
        Fits a cubic spline using chord-length parameterization and computes the true arc length.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input must be Nx3 array.")
        if len(points) < 2:
            raise ValueError("At least two points required.")

        self.points = points

        # Chord-length parameterization (for spline fitting)
        chord_distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        chord_lengths = np.insert(np.cumsum(chord_distances), 0, 0)
        self.t_values = chord_lengths / chord_lengths[-1]  # Normalized t [0, 1]

        # Fit cubic splines
        self.spline = [CubicSpline(self.t_values, points[:, dim]) for dim in range(3)]

        # Compute true arc length and parameterization
        self.parameterize_by_arc_length(num_samples)

        # Sample points uniformly in arc length for KD-Tree
        self.calculate_sampled_points(num_samples)

    def arc_length_integrand(self, t):
        """Speed function ||dx/dt||."""
        derivative = np.array([s.derivative()(t) for s in self.spline])
        return np.linalg.norm(derivative)

    def parameterize_by_arc_length(self, num_samples: int = 1000):
        """
        Computes the true arc length of the spline and builds a mapping from
        arc-length percentage to t parameter.
        """
        # High-resolution sampling for accurate integration
        t_samples = np.linspace(0, 1, num_samples)
        speeds = np.array([self.arc_length_integrand(t) for t in t_samples])
        arc_lengths = cumulative_trapezoid(speeds, t_samples, initial=0)
        
        # Set true curve length and normalize
        self.curve_length = arc_lengths[-1]
        arc_lengths_normalized = arc_lengths / self.curve_length

        # Spline mapping: arc-length percentage (s) -> t parameter
        self.arc_spline = PchipInterpolator(arc_lengths_normalized, t_samples)

    def calculate_sampled_points(self, num_samples: int = 1000):
        """Samples points uniformly in arc length for KD-Tree."""
        if not self.arc_spline:
            raise RuntimeError("Call parameterize_by_arc_length first.")

        # Uniform arc-length percentages
        s_uniform = np.linspace(0, 1, num_samples)
        
        # Convert to t parameters
        t_from_s = self.arc_spline(s_uniform)
        
        # Evaluate spline at these t values
        self.sampled_points = np.stack(
            [s(t_from_s) for s in self.spline], axis=-1
        )
        
        # Build KD-Tree
        self.kd_tree = cKDTree(self.sampled_points)

    def interpolate_point_by_percentage(self, percentage: float) -> np.ndarray:
        """Interpolates a point at the specified arc-length percentage."""
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be in [0, 1].")
        
        # Convert percentage to t parameter
        t_interp = self.arc_spline(percentage)
        
        # Evaluate spline
        return np.array([s(t_interp) for s in self.spline])

    def calculate_percentage(self, target_xyz: np.ndarray) -> float:
        """Finds the closest arc-length percentage to a 3D point."""
        _, closest_idx = self.kd_tree.query(target_xyz)
        return closest_idx / (len(self.sampled_points) - 1)



# First one from Deepseek
class CurveEstimator3D_OLDbutworking:
    def __init__(self):
        self.points = None
        self.curve_length = None
        self.spline = None
        self.t_values = None  # Parameter values for spline
        self.arc_lengths = None  # Cumulative arc lengths
        self.arc_spline = None  # Arc-length parameterization
        self.sampled_points = None  # Sampled points for KD-Tree
        self.kd_tree = None  # KD-Tree for nearest-neighbor search

    def calculate_curve_length(self, points: np.ndarray, num_samples: int = 500):
        """
        Fits a cubic spline to the given 3D curve points and computes arc-length parameterization.
        Handles unevenly spaced points and varying arc lengths.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input must be Nx3 array for XYZ coordinates.")
        if len(points) < 2:
            raise ValueError("At least two points are required to compute a curve.")

        self.points = points

        # Compute cumulative arc lengths between points
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.arc_lengths = np.insert(np.cumsum(distances), 0, 0)
        self.curve_length = self.arc_lengths[-1]

        # Normalize parameterization to [0, 1] based on cumulative arc lengths
        self.t_values = self.arc_lengths / self.curve_length

        # Fit cubic splines for each dimension
        self.spline = [CubicSpline(self.t_values, points[:, dim]) for dim in range(3)]

        # Compute arc-length parameterization
        self.parameterize_by_arc_length(num_samples)

        # Sample points along the curve for KD-Tree
        self.calculate_sampled_points(num_samples)

    def arc_length_integrand(self, t):
        """Compute speed function ||dx/dt|| for arc-length parameterization."""
        derivative = np.array([s.derivative()(t) for s in self.spline])
        return np.linalg.norm(derivative)

    def parameterize_by_arc_length(self, num_samples: int = 500):
        """
        Precompute an arc-length parameterized mapping using adaptive quadrature.
        """
        # Generate high-precision samples for arc-length parameterization
        sample_t = np.linspace(0, 1, num_samples)
        speeds = np.array([self.arc_length_integrand(t) for t in sample_t])
        arc_lengths = cumulative_trapezoid(speeds, sample_t, initial=0)

        # Normalize arc lengths to [0, 1]
        arc_lengths /= arc_lengths[-1]

        # Create a spline for arc-length parameterization
        self.arc_spline = PchipInterpolator(arc_lengths, sample_t, extrapolate=True)

    def calculate_sampled_points(self, num_samples: int = 500):
        """
        Computes points along the curve using arc-length parameterization.
        Builds a KD-Tree for nearest-neighbor search.
        """
        if self.spline is None:
            raise RuntimeError("Curve must be calculated first.")

        # Generate arc-length parameterized samples
        sample_t = np.linspace(0, 1, num_samples)
        self.sampled_points = np.stack([s(sample_t) for s in self.spline], axis=-1)

        # Build KD-Tree for nearest-neighbor search
        self.kd_tree = cKDTree(self.sampled_points)

    def interpolate_point_by_percentage(self, percentage: float) -> np.ndarray:
        """
        Interpolates a point along the curve based on a percentage of the total curve length.
        The percentage must be in the range [0, 1].
        """
        if self.spline is None:
            raise RuntimeError("Curve must be calculated first.")
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be in the range [0, 1].")

        # Convert percentage to normalized parameter t
        t_interp = self.arc_spline(percentage)  # Directly use percentage since arc_spline is normalized
        return np.array([s(t_interp) for s in self.spline])

    def calculate_percentage(self, target_xyz: np.ndarray) -> float:
        """
        Finds the percentage along the curve that is closest to a given 3D point.
        Uses KD-Tree for efficient nearest-neighbor search.
        """
        if self.spline is None or self.kd_tree is None:
            raise RuntimeError("Curve must be calculated first.")

        target_xyz = np.array(target_xyz)
        if target_xyz.shape != (3,):
            raise ValueError("Target point must be a 3D coordinate (shape (3,)).")

        # Find the closest point on the sampled curve
        _, closest_index = self.kd_tree.query(target_xyz, workers=-1)  # Multi-threaded search

        # Map the closest index to the corresponding percentage
        percentage = closest_index / (len(self.sampled_points) - 1)
        return percentage

# DELETE NOT IN USE
class CurveEstimator_OK:
    def __init__(self):
        self.points = None
        self.distances = None
        self.cumulative_lengths = None
        self.curve_length = None

    def calculate_curve_length(self, points: np.ndarray):
        """
        Calculates the total length of a 3D curve given by discrete points.

        Parameters:
            points (np.ndarray): An Nx3 array of points (x, y, z) along the curve.

        Raises:
            ValueError: If input points are not valid Nx3 coordinates.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Input points must have a shape of Nx3 (representing XYZ coordinates)."
            )

        self.points = points

        # Calculate pairwise distances between consecutive points
        self.distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

        # Cumulative lengths of the curve
        self.cumulative_lengths = np.cumsum(self.distances)

        # Total curve length
        self.curve_length = self.cumulative_lengths[-1]

    def interpolate_point_on_curve(self, percentage: float) -> np.ndarray:
        """
        Interpolates a point along a 3D curve at a specified percentage of its total length.

        Parameters:
            percentage (float): Percentage of the curve's length (0 to 1) where the point is desired.

        Returns:
            np.ndarray: Interpolated point (x, y, z) on the curve.
        """
        if self.points is None:
            raise RuntimeError("Curve length must be calculated first using calculate_curve_length().")
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1.")

        target_length = percentage * self.curve_length

        # Edge cases: return start or end directly
        if target_length == 0:
            return self.points[0]
        if target_length == self.curve_length:
            return self.points[-1]

        # Locate the segment using binary search
        segment_index = np.searchsorted(self.cumulative_lengths, target_length)

        # Extract segment start and end points
        p1, p2 = self.points[segment_index], self.points[segment_index + 1]

        # Compute interpolation factor
        t = (target_length - self.cumulative_lengths[segment_index - 1]) / self.distances[segment_index] if segment_index > 0 else target_length / self.distances[0]

        return p1 + t * (p2 - p1)  # Linear interpolation

    def calculate_percentage(self, target_xyz: np.ndarray) -> float:
        """
        Finds the percentage along the curve that is closest to a given point.

        Parameters:
            target_xyz (np.ndarray): Target point (x, y, z).

        Returns:
            float: Percentage (0 to 1) along the curve that is closest to the target point.
        """
        if self.points is None:
            raise RuntimeError("Curve length must be calculated first using calculate_curve_length().")

        target_xyz = np.asarray(target_xyz)

        # Compute closest points for all segments
        p1 = self.points[:-1]  # Start points of segments
        p2 = self.points[1:]   # End points of segments

        # Compute closest points on each segment
        segment_vectors = p2 - p1
        target_vectors = target_xyz - p1
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        
        # Avoid division by zero
        segment_lengths = np.where(segment_lengths == 0, 1, segment_lengths)

        # Compute t (proportion along each segment)
        t = np.sum(target_vectors * segment_vectors, axis=1) / (segment_lengths**2)
        t = np.clip(t, 0, 1)  # Ensure t is within segment bounds

        # Get closest points on segments
        closest_points = p1 + t[:, np.newaxis] * segment_vectors

        # Compute distances to target
        distances = np.linalg.norm(closest_points - target_xyz, axis=1)

        # Find closest segment
        closest_idx = np.argmin(distances)

        # Compute total curve length up to the closest point
        segment_start_length = self.cumulative_lengths[closest_idx - 1] if closest_idx > 0 else 0.0
        length_to_closest_point = segment_start_length + t[closest_idx] * self.distances[closest_idx]

        # Return percentage along the curve
        return length_to_closest_point / self.curve_length

    @staticmethod
    def _closest_point_on_segment(p1: np.ndarray, p2: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Finds the closest point on a line segment [p1, p2] to a target point.

        Parameters:
            p1, p2 (np.ndarray): The endpoints of the segment (3D points).
            target (np.ndarray): The target point (3D point).

        Returns:
            np.ndarray: Closest point on the segment to the target point.
        """
        p1, p2, target = map(np.asarray, (p1, p2, target))  # Ensure inputs are NumPy arrays
        segment = p2 - p1
        segment_length_squared = segment @ segment  # Equivalent to np.dot(segment, segment), but faster

        if segment_length_squared == 0:
            return p1  # Return one of the endpoints if the segment is a single point

        t = (target - p1) @ segment / segment_length_squared  # Compute projection scalar
        return p1 + np.clip(t, 0, 1) * segment  # Clamp t to [0,1] and compute closest point

# DELETE NOT IN USE
class CurveEstimator_OLD:
    def __init__(self):
        self.points = None
        self.distances = None
        self.cumulative_lengths = None
        self.curve_length = None

    def calculate_curve_length(self, points: np.ndarray):
        """
        Calculates the total length of a 3D curve given by discrete points.

        Parameters:
            points (np.ndarray): An Nx3 array of points (x, y, z) along the curve.

        Raises:
            ValueError: If input points are not valid Nx3 coordinates.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Input points must have a shape of Nx3 (representing XYZ coordinates)."
            )

        self.points = points

        # Calculate pairwise distances between consecutive points
        self.distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

        # Cumulative lengths of the curve
        self.cumulative_lengths = np.cumsum(self.distances)

        # Total curve length
        self.curve_length = self.cumulative_lengths[-1]

    def interpolate_point_on_curve_OLD(self, percentage: float) -> np.ndarray:
        """
        Interpolates a point along a 3D curve at a specified percentage of its total length.

        Parameters:
            percentage (float): Percentage of the curve's length (0 to 1) where the point is desired.

        Returns:
            np.ndarray: Interpolated point (x, y, z) on the curve.

        Raises:
            ValueError: If percentage is not between 0 and 1.
        """
        if self.points is None:
            raise RuntimeError(
                "Curve length must be calculated first using calculate_curve_length()."
            )
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1.")

        # Target length along the curve
        target_length = percentage * self.curve_length

        # Edge cases: start or end of the curve
        if target_length == 0:
            return self.points[0]
        if target_length == self.curve_length:
            return self.points[-1]

        # Find the segment containing the target length
        segment_index = np.searchsorted(self.cumulative_lengths, target_length)

        # Get the segment's start and end points
        p1 = self.points[segment_index]
        p2 = self.points[segment_index + 1]

        # Lengths for interpolation
        segment_start_length = (
            self.cumulative_lengths[segment_index - 1] if segment_index > 0 else 0.0
        )
        segment_length = self.distances[segment_index]

        # Interpolation factor
        t = (target_length - segment_start_length) / segment_length

        # Linear interpolation
        return p1 + t * (p2 - p1)

    def interpolate_point_on_curve(self, percentage: float) -> np.ndarray:
        """
        Interpolates a point along a 3D curve at a specified percentage of its total length.

        Parameters:
            percentage (float): Percentage of the curve's length (0 to 1) where the point is desired.

        Returns:
            np.ndarray: Interpolated point (x, y, z) on the curve.
        """
        if self.points is None:
            raise RuntimeError("Curve length must be calculated first using calculate_curve_length().")
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1.")

        target_length = percentage * self.curve_length

        # Edge cases: return start or end directly
        if target_length == 0:
            return self.points[0]
        if target_length == self.curve_length:
            return self.points[-1]

        # Locate the segment using binary search
        segment_index = np.searchsorted(self.cumulative_lengths, target_length)

        # Extract segment start and end points
        p1, p2 = self.points[segment_index], self.points[segment_index + 1]

        # Compute interpolation factor
        t = (target_length - self.cumulative_lengths[segment_index - 1]) / self.distances[segment_index] if segment_index > 0 else target_length / self.distances[0]

        return p1 + t * (p2 - p1)  # Linear interpolation

    def calculate_percentage_OLD(self, target_xyz: np.ndarray) -> float:
        """
        Finds the percentage along the curve that is closest to a given point.

        Parameters:
            target_xyz (np.ndarray): Target point (x, y, z).

        Returns:
            float: Percentage (0 to 1) along the curve that is closest to the target point.
        """
        if self.points is None:
            raise RuntimeError(
                "Curve length must be calculated first using calculate_curve_length()."
            )

        target_xyz = np.array(target_xyz)
        min_distance = float("inf")
        closest_length = 0.0

        for i in range(len(self.points) - 1):
            p1, p2 = self.points[i], self.points[i + 1]
            closest_point = self._closest_point_on_segment(p1, p2, target_xyz)

            # Calculate the length up to this segment
            segment_start_length = self.cumulative_lengths[i - 1] if i > 0 else 0.0
            segment_length = self.distances[i]

            # Get t (proportion along the segment)
            t = np.linalg.norm(closest_point - p1) / segment_length
            t = np.clip(t, 0, 1)

            # Compute total curve length up to the closest point
            length_to_closest_point = segment_start_length + t * segment_length

            # Update if this point is closer
            distance_to_target = np.linalg.norm(closest_point - target_xyz)
            if distance_to_target < min_distance:
                min_distance = distance_to_target
                closest_length = length_to_closest_point

        # Compute final percentage
        return closest_length / self.curve_length

    def calculate_percentage(self, target_xyz: np.ndarray) -> float:
        """
        Finds the percentage along the curve that is closest to a given point.

        Parameters:
            target_xyz (np.ndarray): Target point (x, y, z).

        Returns:
            float: Percentage (0 to 1) along the curve that is closest to the target point.
        """
        if self.points is None:
            raise RuntimeError("Curve length must be calculated first using calculate_curve_length().")

        target_xyz = np.asarray(target_xyz)

        # Compute closest points for all segments
        p1 = self.points[:-1]  # Start points of segments
        p2 = self.points[1:]   # End points of segments

        # Compute closest points on each segment
        segment_vectors = p2 - p1
        target_vectors = target_xyz - p1
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        
        # Avoid division by zero
        segment_lengths = np.where(segment_lengths == 0, 1, segment_lengths)

        # Compute t (proportion along each segment)
        t = np.sum(target_vectors * segment_vectors, axis=1) / (segment_lengths**2)
        t = np.clip(t, 0, 1)  # Ensure t is within segment bounds

        # Get closest points on segments
        closest_points = p1 + t[:, np.newaxis] * segment_vectors

        # Compute distances to target
        distances = np.linalg.norm(closest_points - target_xyz, axis=1)

        # Find closest segment
        closest_idx = np.argmin(distances)

        # Compute total curve length up to the closest point
        segment_start_length = self.cumulative_lengths[closest_idx - 1] if closest_idx > 0 else 0.0
        length_to_closest_point = segment_start_length + t[closest_idx] * self.distances[closest_idx]

        # Return percentage along the curve
        return length_to_closest_point / self.curve_length

    @staticmethod
    def _closest_point_on_segment_OLD(p1, p2, target):
        """
        Finds the closest point on a line segment [p1, p2] to a target point.

        Parameters:
            p1, p2 (np.ndarray): The endpoints of the segment (3D points).
            target (np.ndarray): The target point (3D point).

        Returns:
            np.ndarray: Closest point on the segment to the target point.
        """
        p1, p2, target = np.array(p1), np.array(p2), np.array(target)
        segment = p2 - p1
        segment_length_squared = np.dot(segment, segment)

        if segment_length_squared == 0:
            return p1

        t = np.dot(target - p1, segment) / segment_length_squared
        t = np.clip(t, 0, 1)
        return p1 + t * segment

    @staticmethod
    def _closest_point_on_segment(p1: np.ndarray, p2: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Finds the closest point on a line segment [p1, p2] to a target point.

        Parameters:
            p1, p2 (np.ndarray): The endpoints of the segment (3D points).
            target (np.ndarray): The target point (3D point).

        Returns:
            np.ndarray: Closest point on the segment to the target point.
        """
        p1, p2, target = map(np.asarray, (p1, p2, target))  # Ensure inputs are NumPy arrays
        segment = p2 - p1
        segment_length_squared = segment @ segment  # Equivalent to np.dot(segment, segment), but faster

        if segment_length_squared == 0:
            return p1  # Return one of the endpoints if the segment is a single point

        t = (target - p1) @ segment / segment_length_squared  # Compute projection scalar
        return p1 + np.clip(t, 0, 1) * segment  # Clamp t to [0,1] and compute closest point

#Last one from ChatGPT
class CurveEstimator3D_LastChatGPT:
    def __init__(self):
        self.points = None
        self.curve_length = None
        self.spline = None
        self.t_values = None  # Arc-length parameterized values
        self.sampled_points = None
        self.t_samples = None
        self.kd_tree = None
        self.arc_spline = None  # Store arc-length parameterization
        self.arc_lengths = None  # Store actual arc lengths

    def calculate_curve_length(self, points: np.ndarray, num_samples: int = 500):
        """
        Fits a cubic spline to the given 3D curve points and computes arc-length parameterization.
        Optimized with adaptive quadrature and higher precision.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input must be Nx3 array for XYZ coordinates.")
        if len(points) < 2:
            raise ValueError("At least two points are required to compute a curve.")

        self.points = points
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative_lengths = np.insert(np.cumsum(distances), 0, 0)
        total_length = cumulative_lengths[-1]
        
        self.t_values = cumulative_lengths / total_length  # Normalize to [0, 1]
        self.spline = [CubicSpline(self.t_values, points[:, dim]) for dim in range(3)]
        self.curve_length = total_length
        
        # Generate high-precision arc-length parameterized samples
        self.calculate_sampled_points(num_samples)
    
    def arc_length_integrand(self, t):
        """Compute speed function ||dx/dt|| for arc-length parameterization."""
        derivative = np.array([s.derivative()(t) for s in self.spline])
        return np.linalg.norm(derivative)
    
    def parameterize_by_arc_length(self, num_samples: int = 500):
        """
        Precompute an arc-length parameterized mapping using adaptive quadrature.
        """
        if self.arc_spline is None:
            sample_t = np.linspace(0, 1, num_samples)
            arc_lengths = cumulative_trapezoid([self.arc_length_integrand(t) for t in sample_t], sample_t, initial=0)
            self.arc_lengths = arc_lengths  # Store actual arc-length values
            self.arc_spline = PchipInterpolator(arc_lengths, sample_t, extrapolate=True)
    
    def calculate_sampled_points(self, num_samples: int = 500):
        """
        Computes points along the curve using arc-length parameterization without enforcing uniform spacing.
        """
        self.parameterize_by_arc_length(num_samples)
        self.t_samples = self.arc_lengths  # Use actual arc-length values instead of uniform [0,1]
        t_arc = self.arc_spline(self.t_samples)
        self.sampled_points = np.stack([s(t_arc) for s in self.spline], axis=-1)
        self.kd_tree = cKDTree(self.sampled_points)
    
    def interpolate_point_on_curve(self, arc_length: float) -> np.ndarray:
        """
        Interpolates a point along the true arc-length parameterized curve.
        """
        if self.spline is None:
            raise RuntimeError("Curve must be calculated first.")
        if not (0 <= arc_length <= self.curve_length):
            raise ValueError("Arc length must be within curve bounds.")
        
        t_interp = self.arc_spline(arc_length)
        return np.array([s(t_interp) for s in self.spline])
    
    def calculate_percentage(self, target_xyz: np.ndarray) -> float:
        """
        Finds the percentage along the curve that is closest to a given point using optimized KD-Tree search.
        """
        if self.spline is None or self.kd_tree is None:
            raise RuntimeError("Curve must be calculated first.")
        
        target_xyz = np.array(target_xyz)
        _, closest_index = self.kd_tree.query(target_xyz, workers=-1)  # Multi-threaded search
        return self.arc_lengths[closest_index] / self.curve_length

class CurveEstimator3D_FastButNotPrecise:
    def __init__(self):
        self.points = None
        self.curve_length = None
        self.spline = None
        self.t_values = None  # Arc-length parameterized values
        self.sampled_points = None
        self.t_samples = None
        self.kd_tree = None
        self.arc_spline = None  # Store arc-length parameterization

    def calculate_curve_length(self, points: np.ndarray, num_samples: int = 300):
        """
        Fits a cubic spline to the given 3D curve points and computes arc-length parameterization.
        Optimized with adaptive quadrature and reduced sampling overhead.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input must be Nx3 array for XYZ coordinates.")
        if len(points) < 2:
            raise ValueError("At least two points are required to compute a curve.")

        self.points = points
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative_lengths = np.insert(np.cumsum(distances), 0, 0)
        total_length = cumulative_lengths[-1]
        
        self.t_values = cumulative_lengths / total_length  # Normalize to [0, 1]
        self.spline = [CubicSpline(self.t_values, points[:, dim]) for dim in range(3)]
        self.curve_length = total_length
        
        # Generate high-precision arc-length parameterized samples
        self.calculate_sampled_points(num_samples)
    
    def arc_length_integrand(self, t):
        """Compute speed function ||dx/dt|| for arc-length parameterization."""
        derivative = np.array([s.derivative()(t) for s in self.spline])
        return np.linalg.norm(derivative)
    
    def parameterize_by_arc_length(self, num_samples: int = 300):
        """
        Precompute an arc-length parameterized mapping using adaptive quadrature.
        """
        if self.arc_spline is None:
            sample_t = np.linspace(0, 1, num_samples)
            arc_lengths = cumulative_trapezoid([self.arc_length_integrand(t) for t in sample_t], sample_t, initial=0)
            arc_lengths /= arc_lengths[-1]  # Normalize to [0,1]
            self.arc_spline = CubicSpline(arc_lengths, sample_t)
    
    def calculate_sampled_points(self, num_samples: int = 300):
        """
        Computes evenly spaced points along the curve using precomputed arc-length parameterization.
        """
        self.parameterize_by_arc_length(num_samples)
        self.t_samples = np.linspace(0, 1, num_samples)
        t_arc = self.arc_spline(self.t_samples)
        self.sampled_points = np.stack([s(t_arc) for s in self.spline], axis=-1)
        self.kd_tree = cKDTree(self.sampled_points)
    
    def interpolate_point_on_curve(self, percentage: float) -> np.ndarray:
        """
        Interpolates a point along the arc-length parameterized curve.
        """
        if self.spline is None:
            raise RuntimeError("Curve must be calculated first.")
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1.")
        
        t_interp = self.arc_spline(percentage)
        return np.array([s(t_interp) for s in self.spline])
    
    def calculate_percentage(self, target_xyz: np.ndarray) -> float:
        """
        Finds the percentage along the curve that is closest to a given point using optimized KD-Tree search.
        """
        if self.spline is None or self.kd_tree is None:
            raise RuntimeError("Curve must be calculated first.")
        
        target_xyz = np.array(target_xyz)
        _, closest_index = self.kd_tree.query(target_xyz, workers=-1)  # Multi-threaded search
        return self.t_samples[closest_index]

class CurveEstimator3D_A:
    def __init__(self):
        self.points = None
        self.curve_length = None
        self.spline = None
        self.t_values = None  # Arc-length parameterized values
        self.sampled_points = None
        self.t_samples = None
        self.kd_tree = None
        self.arc_spline = None  # Store arc-length parameterization

    def calculate_curve_length(self, points: np.ndarray, num_samples: int = 500):
        """
        Fits a cubic spline to the given 3D curve points and computes arc-length parameterization.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input must be Nx3 array for XYZ coordinates.")
        if len(points) < 2:
            raise ValueError("At least two points are required to compute a curve.")

        self.points = points
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative_lengths = np.insert(np.cumsum(distances), 0, 0)
        total_length = cumulative_lengths[-1]
        
        self.t_values = cumulative_lengths / total_length  # Normalize to [0, 1]
        self.spline = [CubicSpline(self.t_values, points[:, dim]) for dim in range(3)]
        self.curve_length = total_length
        
        # Generate high-precision arc-length parameterized samples
        self.calculate_sampled_points(num_samples)
    
    def arc_length_integrand(self, t):
        """Compute speed function ||dx/dt|| for arc-length parameterization."""
        derivative = np.array([s.derivative()(t) for s in self.spline])
        return np.linalg.norm(derivative)
    
    def parameterize_by_arc_length(self, num_samples: int = 500):
        """
        Precompute an arc-length parameterized mapping and store it.
        """
        if self.arc_spline is None:
            arc_lengths = np.array([quad(self.arc_length_integrand, 0, t)[0] for t in self.t_values])
            arc_lengths /= arc_lengths[-1]  # Normalize to [0,1]
            self.arc_spline = CubicSpline(arc_lengths, self.t_values)
    
    def calculate_sampled_points(self, num_samples: int = 500):
        """
        Computes evenly spaced points along the curve using precomputed arc-length parameterization.
        """
        self.parameterize_by_arc_length(num_samples)
        self.t_samples = np.linspace(0, 1, num_samples)
        t_arc = self.arc_spline(self.t_samples)
        self.sampled_points = np.array([[s(t) for s in self.spline] for t in t_arc])
        self.kd_tree = cKDTree(self.sampled_points)
    
    def interpolate_point_on_curve(self, percentage: float) -> np.ndarray:
        """
        Interpolates a point along the arc-length parameterized curve.
        """
        if self.spline is None:
            raise RuntimeError("Curve must be calculated first.")
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1.")
        
        t_interp = self.arc_spline(percentage)
        return np.array([s(t_interp) for s in self.spline])
    
    def calculate_percentage(self, target_xyz: np.ndarray) -> float:
        """
        Finds the percentage along the curve that is closest to a given point using KD-Tree.
        Optimized with precomputed samples and vectorized search.
        """
        if self.spline is None or self.kd_tree is None:
            raise RuntimeError("Curve must be calculated first.")
        
        target_xyz = np.array(target_xyz)
        _, closest_index = self.kd_tree.query(target_xyz, workers=-1)  # Multi-threaded search
        return self.t_samples[closest_index]
     
class CurveEstimator3D_OLD:

    def __init__(self):
        self.points = None
        self.curve_length = None
        self.spline = None
        self.t_values = None  # Parameterized values for the spline
        self.sample_points = None
        self.t_samples = None
        
    def calculate_sampled_points(self, num_samples: int = 1000) -> np.ndarray:
        self.t_samples = np.linspace(0, 1, num_samples)
        self.sampled_points = np.array([ [s(t) for s in self.spline] for t in self.t_samples ])
        

    def calculate_curve_length(self, points: np.ndarray):
        """
        Fits a cubic spline to the given 3D curve points and computes an improved curve length.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input must be Nx3 array for XYZ coordinates.")

        self.points = points

        # Create a parameter (t) along the curve using cumulative Euclidean distances
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative_lengths = np.insert(np.cumsum(distances), 0, 0)

        # Normalize t to be between 0 and 1
        self.t_values = cumulative_lengths / cumulative_lengths[-1]

        # Fit cubic splines separately for x, y, z
        self.spline = [
            CubicSpline(self.t_values, points[:, dim]) for dim in range(3)
        ]

        # Store total curve length
        self.curve_length = cumulative_lengths[-1]

    def interpolate_point_on_curve(self, percentage: float) -> np.ndarray:
        """
        Interpolates a point along a smooth spline approximation of the curve.
        """
        if self.spline is None:
            raise RuntimeError("Curve must be calculated first.")
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1.")

        # Evaluate the spline at the interpolated parameter value
        t_interp = percentage
        return np.array([s(t_interp) for s in self.spline])

    def calculate_percentage(self, target_xyz: np.ndarray) -> float:
        """
        Finds the percentage along the curve that is closest to a given point.

        Parameters:
            target_xyz (np.ndarray): Target point (x, y, z).
            num_samples (int): Number of points to sample along the curve.

        Returns:
            float: Percentage (0 to 1) along the curve that is closest to the target point.
        """
        if self.spline is None:
            raise RuntimeError("Curve must be calculated first.")

        target_xyz = np.array(target_xyz)

        # Compute distances to the target point
        distances = np.linalg.norm(self.sampled_points - target_xyz, axis=1)

        # Find the index of the closest sampled point
        closest_index = np.argmin(distances)
        closest_t = self.t_samples[closest_index]  # Get corresponding parameter t

        return closest_t  # Return percentage along the curve
