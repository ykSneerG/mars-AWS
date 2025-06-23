import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from tqdm import tqdm # type: ignore
from typing import List, Tuple


class LUTInverter_DS:
    def __init__(self, atob_lut, cmyk_grid_size=17, lab_grid_shape=(33, 33, 33)):
        """
        Optimized LUT inverter with hybrid approach for better speed/quality balance.
        
        Parameters:
            atob_lut: List of ((C,M,Y,K), (L,a,b)) tuples
            cmyk_grid_size: Size of the CMYK grid in AToB table
            lab_grid_shape: Resolution of BToA output grid
        """
        self.atob_lut = atob_lut
        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape
        
        # Convert to numpy arrays
        self.cmyk_points = np.array([cmyk for cmyk, _ in atob_lut])
        self.lab_points = np.array([lab for _, lab in atob_lut])
        
        # Build both interpolator and KD-tree
        self.interpolator = self._build_interpolator()
        self.kdtree = cKDTree(self.lab_points)
        
        # Precompute LAB grid
        self.lab_grid = self._create_lab_grid()
        
        # Cache for optimization results
        self.optimization_cache = {}

    def _build_interpolator(self):
        """Build 4D CMYK → LAB interpolator using RegularGridInterpolator."""
        lab_grid_4d = self.lab_points.reshape((self.cmyk_grid_size,) * 4 + (3,))
        points = [np.linspace(0, 1, self.cmyk_grid_size)] * 4
        return RegularGridInterpolator(points, lab_grid_4d, method='linear', 
                                     bounds_error=False, fill_value=None)

    def _create_lab_grid(self):
        """Create uniform LAB grid for inversion."""
        L = np.linspace(0, 100, self.lab_grid_shape[0])
        a = np.linspace(-128, 127, self.lab_grid_shape[1])
        b = np.linspace(-128, 127, self.lab_grid_shape[2])
        Lg, ag, bg = np.meshgrid(L, a, b, indexing='ij')
        return np.stack([Lg, ag, bg], axis=-1).reshape(-1, 3)

    def _invert_lab_point(self, target_lab):
        """
        Hybrid inversion with fast path for nearby points and optimization for others.
        """
        # First check cache
        cache_key = tuple(np.round(target_lab, 2))
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        # Fast path: find nearest neighbor
        _, idx = self.kdtree.query(target_lab.reshape(1, -1))
        nearest_cmyk = self.cmyk_points[idx[0]]
        nearest_lab = self.lab_points[idx[0]]
        
        # If very close, use nearest neighbor
        if np.sum((nearest_lab - target_lab) ** 2) < 1.0:  # ΔE < 1
            return nearest_cmyk
        
        # Otherwise perform optimization
        def loss(cmyk):
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e6
            lab = self.interpolator(cmyk.reshape(1, -1))[0]
            return np.sum((lab - target_lab) ** 2)

        result = minimize(
            loss,
            x0=nearest_cmyk,  # Start from nearest neighbor
            bounds=[(0, 1)] * 4,
            method='L-BFGS-B',
            options={'maxiter': 30, 'ftol': 1e-4}
        )
        
        optimized_cmyk = result.x if result.success else nearest_cmyk
        self.optimization_cache[cache_key] = optimized_cmyk
        return optimized_cmyk

    def build_btoa_lut(self, fast_approx_threshold=5.0, verbose=True):
        """
        Generate BToA LUT with optimized speed/quality tradeoff.
        
        Parameters:
            fast_approx_threshold: ΔE threshold for using fast approximation (higher=faster)
            verbose: Whether to show progress
        """
        btoa = np.zeros((len(self.lab_grid), 4))
        
        # Use tqdm for progress bar if verbose
        iterable = tqdm(enumerate(self.lab_grid), total=len(self.lab_grid)) if verbose else enumerate(self.lab_grid)
        
        for i, lab in iterable:
            # First try fast approximation
            _, idx = self.kdtree.query(lab.reshape(1, -1))
            nearest_cmyk = self.cmyk_points[idx[0]]
            nearest_lab = self.lab_points[idx[0]]
            
            # If within threshold, use fast approximation
            if np.sum((nearest_lab - lab) ** 2) < fast_approx_threshold ** 2:
                btoa[i] = nearest_cmyk
            else:
                # Otherwise use optimized inversion
                btoa[i] = self._invert_lab_point(lab)
        
        return btoa.reshape((*self.lab_grid_shape, 4))

    @staticmethod
    def smooth_lut(btoa_grid, sigma=1.0, preserve_boundaries=True):
        """
        Improved smoothing that optionally preserves gamut boundaries.
        """
        smoothed = np.zeros_like(btoa_grid)
        for i in range(4):
            channel = btoa_grid[..., i].copy()
            if preserve_boundaries:
                # Identify boundary points (0 or 1)
                boundary_mask = (channel == 0) | (channel == 1)
                # Smooth only non-boundary points
                blurred = gaussian_filter(channel, sigma=sigma)
                channel[~boundary_mask] = blurred[~boundary_mask]
            else:
                channel = gaussian_filter(channel, sigma=sigma)
            smoothed[..., i] = channel
        return smoothed



class BToA_LUT_Generator:
    def __init__(self, atob_lut: List[Tuple[Tuple[float, float, float, float], Tuple[float, float, float]]], 
                 cmyk_grid_size: int = 17, 
                 lab_grid_shape: Tuple[int, int, int] = (33, 33, 33)):
        """
        Optimized single-process BToA LUT generator for Docker/AWS environments.
        
        Parameters:
            atob_lut: List of ((C,M,Y,K), (L,a,b)) tuples
            cmyk_grid_size: Size of the input CMYK grid (default 17)
            lab_grid_shape: Resolution of output LAB grid (default 33x33x33)
        """
        self.atob_lut = atob_lut
        self.cmyk_grid_size = cmyk_grid_size
        self.lab_grid_shape = lab_grid_shape
        
        # Convert to numpy arrays
        self.cmyk_points = np.array([cmyk for cmyk, _ in atob_lut])
        self.lab_points = np.array([lab for _, lab in atob_lut])
        
        # Build interpolator and KD-tree
        self.interpolator = self._build_interpolator()
        self.tree = cKDTree(self.lab_points)
        
        # Precompute LAB grid
        self.lab_grid = self._create_lab_grid()
        
        # Cache for already computed points
        self.cache = {}

    def _build_interpolator(self) -> RegularGridInterpolator:
        """Build 4D CMYK → LAB interpolator using linear interpolation."""
        lab_grid_4d = self.lab_points.reshape((self.cmyk_grid_size,) * 4 + (3,))
        points = [np.linspace(0, 1, self.cmyk_grid_size)] * 4
        return RegularGridInterpolator(
            points, lab_grid_4d, method='linear', bounds_error=False, fill_value=None
        )

    def _create_lab_grid(self) -> np.ndarray:
        """Create uniform LAB grid for inversion."""
        L_range = (0, 100)
        a_range = (-128, 127)
        b_range = (-128, 127)

        L_lin = np.linspace(*L_range, self.lab_grid_shape[0])
        a_lin = np.linspace(*a_range, self.lab_grid_shape[1])
        b_lin = np.linspace(*b_range, self.lab_grid_shape[2])

        Lg, ag, bg = np.meshgrid(L_lin, a_lin, b_lin, indexing='ij')
        return np.stack([Lg, ag, bg], axis=-1).reshape(-1, 3)

    def _invert_point(self, target_lab: np.ndarray) -> np.ndarray:
        """
        Invert a single LAB point to CMYK using optimization with caching.
        
        Args:
            target_lab: Target LAB value (1D array of length 3)
            
        Returns:
            CMYK values (1D array of length 4)
        """
        # Check cache first
        cache_key = tuple(np.round(target_lab, 2))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 1. Find nearest neighbor as initial guess
        _, idx = self.tree.query(target_lab.reshape(1, -1))
        cmyk_init = self.cmyk_points[idx[0]]
        
        # 2. Define optimization objective
        def loss(cmyk: np.ndarray) -> float:
            if np.any((cmyk < 0) | (cmyk > 1)):
                return 1e12
            
            predicted_lab = self.interpolator(cmyk.reshape(1, -1))[0]
            delta_e = np.sqrt(np.sum((predicted_lab - target_lab) ** 2))
            
            # Add small penalty for extreme CMYK combinations
            ink_limit_penalty = 0.01 * np.sum(cmyk > 0.9)
            return delta_e + ink_limit_penalty
        
        # 3. Run bounded optimization with tighter tolerances
        bounds = [(0, 1)] * 4
        result = minimize(
            loss,
            x0=cmyk_init,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-4}
        )
        
        cmyk_result = np.clip(result.x, 0, 1) if result.success else cmyk_init
        
        # Store in cache
        self.cache[cache_key] = cmyk_result
        return cmyk_result

    def generate_btoa_lut(self, smooth_sigma: float = 0.7) -> np.ndarray:
        """
        Generate optimized BToA LUT with optional smoothing.
        
        Args:
            smooth_sigma: Sigma for Gaussian smoothing (0 for no smoothing)
            
        Returns:
            4D numpy array of shape (L, a, b, 4) containing CMYK values
        """
        print(f"Generating BToA LUT with {len(self.lab_grid)} points (single process)...")
        
        # Pre-allocate result array
        btoa_grid = np.zeros((len(self.lab_grid), 4))
        
        # Process in batches with progress bar
        batch_size = 500
        for i in tqdm(range(0, len(self.lab_grid), batch_size), 
                        desc="Inverting LAB to CMYK"):
            batch_end = min(i + batch_size, len(self.lab_grid))
            for j in range(i, batch_end):
                btoa_grid[j] = self._invert_point(self.lab_grid[j])
        
        # Reshape to grid
        btoa_grid = btoa_grid.reshape((*self.lab_grid_shape, 4))
        
        # Apply smart smoothing if requested
        if smooth_sigma > 0:
            print(f"Applying Gaussian smoothing (σ={smooth_sigma})...")
            btoa_grid = self._smart_smooth(btoa_grid, sigma=smooth_sigma)
        
        return btoa_grid

    def _smart_smooth(self, btoa_grid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian smoothing while preserving important boundaries.
        Optimized for single-process operation.
        """
        smoothed = np.zeros_like(btoa_grid)
        
        for i in range(4):  # For each CMYK channel
            channel = btoa_grid[..., i].copy()
            
            # Identify gamut boundary (where CMYK values hit 0 or 1)
            boundary_mask = (channel == 0) | (channel == 1)
            
            # Smooth only non-boundary points
            blurred = gaussian_filter(channel, sigma=sigma)
            channel[~boundary_mask] = blurred[~boundary_mask]
            
            smoothed[..., i] = channel
        
        return smoothed

    @staticmethod
    def save_for_icc(btoa_grid: np.ndarray) -> dict:
        """
        Prepare data for ICC profile creation.
        Returns a dictionary with the structure needed for MFT2 tag.
        """
        return {
            'grid_size': btoa_grid.shape[:3],
            'data': btoa_grid.astype(np.float32),  # ICC profiles typically use 32-bit floats
            'input_channels': 3,  # LAB
            'output_channels': 4   # CMYK
        }