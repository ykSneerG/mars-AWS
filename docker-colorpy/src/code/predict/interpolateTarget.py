import numpy as np # type: ignore
from scipy.interpolate import Rbf, RBFInterpolator, CubicHermiteSpline # type: ignore
from scipy.spatial.distance import cdist # type: ignore

from enum import Enum

class BaseInterpolation:
    def __init__(self):
        self.src_dcs = None  # Device color space (e.g., CMYK, RGB)
        self.src_pcs = None  # Spectral reflectance values

    def set_src_dcs(self, dcs_data):
        """Set device color space data (e.g., CMYK, RGB)."""
        self.src_dcs = np.array(dcs_data)

    def set_src_spectra(self, spectral_data):
        """Set spectral reflectance data."""
        self.src_pcs = np.array(spectral_data)

# --#-#-- INTERPOLATION scipy.interpolate.RbRBFInterpolator --#-#-- INTERPOLATION scipy.interpolate.RbRBFInterpolator --#-#--
 
class ModernRBFkernel(Enum):
    """ 
    #supported in SKIPY 1.15.2 scipy.interpolate.RBFInterpolator
    ’linear’ : -r
    ‘thin_plate_spline’ : r**2 * log(r)
    ‘cubic’ : r**3
    ‘quintic’ : -r**5
    ‘multiquadric’ : -sqrt(1 + r**2)
    ‘inverse_multiquadric’ : 1/sqrt(1 + r**2)
    ‘inverse_quadratic’ : 1/(1 + r**2)
    ‘gaussian’ : exp(-r**2)
    """
    
    MULTIQUADRATIC = 'multiquadric'
    THINPLATESPLINE = 'thin_plate_spline'
    
 
class ModernRBFInterpolator:
    
    def __init__(self):
        self.src_dcs = None
        self.src_pcs = None
        self.rbf_interpolator = None
        self.epsilon = None
        self.smoothing = 1e-8
        self.kernel = ModernRBFkernel.MULTIQUADRATIC.value

    def set_src_dcs(self, dcs_data):
        """Set input device color data with validation."""
        self.src_dcs = np.asarray(dcs_data, dtype=np.float32)
        if self.src_dcs.ndim != 2:
            raise ValueError("dcs_data must be a 2D array of shape (n_samples, n_features)")

    def set_src_spectra(self, spectral_data):
        """Set output spectral data with physics-aware validation."""
        self.src_pcs = np.asarray(spectral_data, dtype=np.float32)
        if self.src_pcs.ndim != 2:
            raise ValueError("spectral_data must be a 2D array of shape (n_samples, n_wavelengths)")
        if np.any(self.src_pcs < 0):
            raise ValueError("Spectral data contains negative values - physically impossible")
        
    def set_smoothness(self, smoothing):
        """Set regularization parameter for RBF smoothing."""
        if smoothing <= 0:
            raise ValueError("Smoothing parameter must be positive")
        self.smoothing = smoothing
        
    def set_kernel(self, kernel):
        """Set RBF kernel type."""
        try:
            self.kernel = ModernRBFkernel(kernel).value
        except ValueError:
            raise ValueError(f"Unsupported kernel type: {kernel}")

    def _compute_epsilon(self):
        """Compute adaptive epsilon based on nearest neighbor distances."""
        distances = cdist(self.src_dcs, self.src_dcs)
        np.fill_diagonal(distances, np.inf)
        return np.mean(np.min(distances, axis=1))

    def precompute_interpolator(self):
        """Create single RBF interpolator for all wavelengths."""
        #self._validate_data()
        self.epsilon = self._compute_epsilon()
        
        try:
            self.rbf_interpolator = RBFInterpolator(
                self.src_dcs,
                self.src_pcs,
                kernel=self.kernel,
                epsilon=self.epsilon,
                smoothing=self.smoothing,
                degree=4
            )
        except ValueError as e:
            raise RuntimeError(f"RBF interpolation failed: {str(e)}") from e

    def interpolate_spectral_data(self, new_color_values):
        """Batched interpolation for new color values."""
        if self.rbf_interpolator is None:
            raise ValueError("Call precompute_interpolator() first")
            
        new_color_values = np.asarray(new_color_values, dtype=np.float32)
        if new_color_values.ndim != 2:
            raise ValueError("Input must be 2D array of shape (n_points, n_features)")
        
        return self.rbf_interpolator(new_color_values)
 


# import numpy as np # type: ignore
# from scipy.spatial.distance import mahalanobis  # type: ignore
from scipy.linalg import inv, pinv  # type: ignore

class FilterMahalanobis:
    """ Outlier Detection using Mahalanobis distance. """
    
    def __init__(self, dcs_data, pcs_data, z_score: float = 3.0) -> None:
        """
        Initialize the class and apply outlier filtering.

        Args:
            dcs_data (array-like): Device-dependent color values (e.g., CMYK).
            pcs_data (array-like): Spectral or perceptual color values.
            z_score (float): The threshold multiplier for outlier detection (default: 3.0).
        """
        self.dcs_data = np.asarray(dcs_data, dtype=np.float64)
        self.pcs_data = np.asarray(pcs_data, dtype=np.float64)
        self.z_score = z_score
        
        self.distances, self.threshold, self.outliers, self.filtered_dcs, self.filtered_pcs = self.apply_filter()

    @property
    def filtered_dcs_list(self) -> list:
        return self.filtered_dcs.tolist()    

    @property
    def filtered_pcs_list(self) -> list:
        return self.filtered_pcs.tolist() 

    def apply_filter(self) -> tuple:
        """Apply Mahalanobis distance-based outlier detection."""
        
        # Combine CMYK + Spectral into one dataset
        combined_data = np.hstack((self.dcs_data, self.pcs_data))

        # Compute the mean and covariance
        mean_vec = np.mean(combined_data, axis=0)
        cov_matrix = np.cov(combined_data, rowvar=False)
        
        # Handle singular matrix (numerical stability)
        try:
            inv_cov_matrix = inv(cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov_matrix = pinv(cov_matrix)  # Use pseudo-inverse if singular

        # Compute Mahalanobis distances (vectorized)
        diff = combined_data - mean_vec
        distances = np.sqrt(np.einsum('ij,ji->i', diff @ inv_cov_matrix, diff.T))  

        # Adaptive threshold (mean + z_score * std)
        threshold = np.mean(distances) + self.z_score * np.std(distances)
        outliers = np.where(distances > threshold)[0]

        # Remove outliers
        filtered_dcs = np.delete(self.dcs_data, outliers, axis=0)
        filtered_pcs = np.delete(self.pcs_data, outliers, axis=0)
        
        return distances, threshold, outliers, filtered_dcs, filtered_pcs

# import numpy as np # type: ignore
from scipy.signal import savgol_filter # type: ignore  # for Savitzky-Golay smoothing

class SavitzkyGolaySmoothing:
    """Savitzky-Golay Smoothing for Spectral Data."""
    
    def __init__(self, pcs_data: np.ndarray, window_length: int = 11, polyorder: int = 3):
        """
        Initialize the smoothing class with spectral data.
        
        :param pcs_data: Spectral data (2D numpy array).
        :param window_length: Window size for SG filter (odd number).
        :param polyorder: Polynomial order for SG filter.
        """
        self.pcs_data = np.array(pcs_data, dtype=np.float32)
        self.window_length = window_length
        self.polyorder = polyorder

        # Ensure window_length is odd and valid
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be an odd integer.")
        
        if self.window_length <= self.polyorder:
            raise ValueError("window_length must be greater than polyorder.")

        self.smoothed_pcs = self.apply_smoothing()

    @property
    def smoothed_pcs_list(self) -> list:
        """Returns the smoothed spectral data as a list."""
        return self.smoothed_pcs.tolist()
    
    def apply_smoothing(self) -> np.ndarray:
        """Apply Savitzky-Golay smoothing using vectorized approach."""
        
        # Apply the Savitzky-Golay filter across all spectral bands (axis 0 - columns)
        smoothed_data = np.apply_along_axis(
            savgol_filter, 0, self.pcs_data, window_length=self.window_length, polyorder=self.polyorder
        )
        
        # Clip negative values to 0 (since spectral values can't be negative)
        smoothed_data = np.clip(smoothed_data, 0, None)  # Clips values below 0 to 0
        
        return smoothed_data



# --- NEW --- NEW --- NEW --- NEW --- NEW --- NEW --- NEW --- NEW --- NEW --- NEW --- NEW --- NEW --- NEW --- NEW --- NEW ---


class ModernCubicHermeticSplineInterpolator:
    
    def __init__(self):
        self.src_dcs = None
        self.src_pcs = None
        self.interpolators = []  # Store separate splines per spectral band

    def set_src_dcs(self, dcs_data):
        """Set input device color data with validation."""
        self.src_dcs = np.asarray(dcs_data, dtype=np.float32)
        if self.src_dcs.ndim != 2:
            raise ValueError("dcs_data must be a 2D array of shape (n_samples, n_features)")

    def set_src_spectra(self, spectral_data):
        """Set output spectral data with validation."""
        self.src_pcs = np.asarray(spectral_data, dtype=np.float32)
        if self.src_pcs.ndim != 2:
            raise ValueError("spectral_data must be a 2D array of shape (n_samples, n_wavelengths)")
        if np.any(self.src_pcs < 0):
            raise ValueError("Spectral data contains negative values - physically impossible")

    def estimate_derivatives(self, dcs: np.ndarray, pcs: np.ndarray) -> np.ndarray:
        """Estimate first derivatives for each spectral band."""
        num_samples = dcs.shape[0]  # Number of CMYK samples
        dydx = np.zeros_like(pcs)

        distances = np.linalg.norm(dcs[1:] - dcs[:-1], axis=1)  # Compute CMYK distances

        # Forward difference
        dydx[0, :] = (pcs[1, :] - pcs[0, :]) / distances[0]

        # Central difference
        for i in range(1, num_samples - 1):
            left_dist = np.linalg.norm(dcs[i] - dcs[i - 1])
            right_dist = np.linalg.norm(dcs[i + 1] - dcs[i])
            dydx[i, :] = (pcs[i + 1, :] - pcs[i - 1, :]) / (left_dist + right_dist)

        # Backward difference
        dydx[-1, :] = (pcs[-1, :] - pcs[-2, :]) / distances[-1]

        return dydx

    def precompute_interpolator(self):
        """Create cubic Hermite splines per spectral band, ensuring sorted input."""
        if self.src_dcs is None or self.src_pcs is None:
            raise ValueError("Set both CMYK (DCS) and spectral data before interpolation.")

        dydx = self.estimate_derivatives(self.src_dcs, self.src_pcs)  # Compute derivatives

        # Extract first CMYK dimension (if multi-dimensional, use first channel)
        x = self.src_dcs[:, 0] if self.src_dcs.shape[1] > 1 else self.src_dcs.flatten()

        # **Sort everything based on CMYK values**
        sort_idx = np.argsort(x)  # Sorting indices
        x_sorted = x[sort_idx]  # Sorted CMYK values
        pcs_sorted = self.src_pcs[sort_idx, :]  # Sort spectral data accordingly
        dydx_sorted = dydx[sort_idx, :]  # Sort derivatives accordingly

        # Create a separate spline for each spectral band
        self.interpolators = []
        for band in range(self.src_pcs.shape[1]):  # Iterate over spectral wavelengths
            try:
                spline = CubicHermiteSpline(x_sorted, pcs_sorted[:, band], dydx_sorted[:, band])
                self.interpolators.append(spline)
            except ValueError as e:
                raise RuntimeError(f"Precompute Interpolation failed for band {band}: {str(e)}") from e


    def interpolate_spectral_data(self, new_color_values):
        """Batched interpolation for new CMYK values."""
        if not self.interpolators:
            raise ValueError("Call precompute_interpolator() first.")

        new_color_values = np.asarray(new_color_values, dtype=np.float32)
        if new_color_values.ndim != 2:
            raise ValueError("Input must be a 2D array of shape (n_points, n_features)")

        x_query = new_color_values[:, 0] if new_color_values.shape[1] > 1 else new_color_values.flatten()

        interpolated_spectra = np.array([spline(x_query) for spline in self.interpolators]).T
        return interpolated_spectra


class ModernCubicHermeticSplineInterpolator_OLD:
    
    def __init__(self):
        self.src_dcs = None
        self.src_pcs = None
        self.interpolator = None

    def set_src_dcs(self, dcs_data):
        """Set input device color data with validation."""
        self.src_dcs = np.asarray(dcs_data, dtype=np.float32)
        if self.src_dcs.ndim != 2:
            raise ValueError("dcs_data must be a 2D array of shape (n_samples, n_features)")

    def set_src_spectra(self, spectral_data):
        """Set output spectral data with physics-aware validation."""
        self.src_pcs = np.asarray(spectral_data, dtype=np.float32)
        if self.src_pcs.ndim != 2:
            raise ValueError("spectral_data must be a 2D array of shape (n_samples, n_wavelengths)")
        if np.any(self.src_pcs < 0):
            raise ValueError("Spectral data contains negative values - physically impossible")
        
    def estimate_derivatives(self, dcs: np.ndarray, pcs: np.ndarray) -> np.ndarray:
        """
        Estimate first derivatives for each spectral dimension while considering CMYK (DCS).
        
        :param dcs: 2D array of shape (num_samples, num_CMYK_channels)
        :param pcs: 2D array of shape (num_samples, num_spectral_bands)
        :return: 2D array of estimated derivatives (same shape as pcs)
        """
        num_samples = dcs.shape[0]  # Number of CMYK combinations
        dydx = np.zeros_like(pcs)

        # Compute distance metric for non-uniform CMYK spacing
        distances = np.linalg.norm(dcs[1:] - dcs[:-1], axis=1)
        
        # Forward difference for the first point
        dydx[0, :] = (pcs[1, :] - pcs[0, :]) / distances[0]

        # Central difference for middle points
        for i in range(1, num_samples - 1):
            left_dist = np.linalg.norm(dcs[i] - dcs[i - 1])
            right_dist = np.linalg.norm(dcs[i + 1] - dcs[i])
            dydx[i, :] = (pcs[i + 1, :] - pcs[i - 1, :]) / (left_dist + right_dist)

        # Backward difference for the last point
        dydx[-1, :] = (pcs[-1, :] - pcs[-2, :]) / distances[-1]

        return dydx


    def precompute_interpolator(self):
        """Create piecewise cubic Hermite spline interpolator."""
        
        dydx = self.estimate_derivatives(self.src_dcs, self.src_pcs)  # Compute derivatives
        
        try:
            self.interpolator = CubicHermiteSpline(
                self.src_dcs, 
                self.src_pcs, 
                dydx)
            
        except ValueError as e:
            raise RuntimeError(f"Precompute Interpolation failed: {str(e)}") from e

    def interpolate_spectral_data(self, new_color_values):
        """Batched interpolation for new color values."""
        if self.interpolator is None:
            raise ValueError("Call precompute_interpolator() first")
            
        new_color_values = np.asarray(new_color_values, dtype=np.float32)
        if new_color_values.ndim != 2:
            raise ValueError("Input must be 2D array of shape (n_points, n_features)")
        
        return self.interpolator(new_color_values)
 


# ============================== DELETE BELOW ==============================

class RadialBasisFunction(BaseInterpolation):
    """Radial Basis Function (RBF) interpolation for spectral data."""

    def __init__(self):
        super().__init__()
        self.rbf_interpolators = None
        self.num_channels = None

    def precompute_interpolators(self):
        """Precompute RBF interpolators for each spectral wavelength."""
        if self.src_dcs is None or self.src_pcs is None:
            raise ValueError("Source data is missing. Use set_src_dcs() and set_src_spectra() first.")

        self.num_channels = self.src_dcs.shape[1]  # Auto-detect number of input channels (e.g., CMYK = 4)
        num_wavelengths = self.src_pcs.shape[1]  # Number of spectral bands

        self.rbf_interpolators = []
        for i in range(num_wavelengths):
            rbf = Rbf(*[self.src_dcs[:, j] for j in range(self.num_channels)], self.src_pcs[:, i],
                      function='gaussian', epsilon=2)
            self.rbf_interpolators.append(rbf)

    def interpolate_spectral_data_rbf(self, new_color_values):
        """Interpolates spectral data for given color space values."""
        if self.rbf_interpolators is None:
            raise ValueError("Interpolators not precomputed. Call precompute_interpolators() first.")

        new_color_values = np.atleast_2d(new_color_values)  # Ensure 2D input
        num_wavelengths = len(self.rbf_interpolators)
        num_new_samples = new_color_values.shape[0]

        interpolated_spectra = np.zeros((num_new_samples, num_wavelengths))
        for i in range(num_wavelengths):
            interpolated_spectra[:, i] = self.rbf_interpolators[i](
                *[new_color_values[:, j] for j in range(self.num_channels)]
            )

        return interpolated_spectra

    def interpolate_numpy(self, new_color_values):
        """Handles NumPy input/output for local use."""
        return self.interpolate_spectral_data_rbf(new_color_values)

    def interpolate_list(self, new_color_values_list):
        """
        Cloud-friendly method:
        - Accepts a **Python list** instead of a NumPy array.
        - Converts list → NumPy array, processes, then converts back to list.
        """
        new_color_values_np = np.array(new_color_values_list)  # Convert list to NumPy
        interpolated_np = self.interpolate_numpy(new_color_values_np)
        return interpolated_np.tolist()  # Convert back to list for serialization

class OptimizedRBFInterpolator3:
    def __init__(self):
        self.src_dcs = None  # Shape: (n_samples, 4)
        self.src_pcs = None  # Shape: (n_samples, 36)
        self.weights = None  # Shape: (n_samples, 36)
        self.epsilon = None
        self.dcs_min = None
        self.dcs_max = None

    def set_src_dcs(self, dcs_data):
        """Normalize CMYK to [0,1] and store as float32."""
        dcs_data = np.asarray(dcs_data, dtype=np.float32)
        self.dcs_min = dcs_data.min(axis=0)
        self.dcs_max = dcs_data.max(axis=0)
        self.src_dcs = (dcs_data - self.dcs_min) / (self.dcs_max - self.dcs_min + 1e-8)

    def set_src_spectra(self, spectral_data):
        """Store spectra as float32 with [0,1] validation."""
        spectral_data = np.asarray(spectral_data, dtype=np.float32)
        """ if np.any(spectral_data < 0):
            raise ValueError("Spectral data must greater than or equal to zero.") """
        self.src_pcs = spectral_data

    def precompute_interpolator(self):
        """Vectorized RBF precomputation using matrix algebra."""
        # self._validate_data()
        
        # Shared epsilon calculation
        distances = cdist(self.src_dcs, self.src_dcs)
        np.fill_diagonal(distances, np.inf)
        self.epsilon = np.mean(np.min(distances, axis=1))
        
        # Precompute RBF kernel matrix once for all wavelengths
        r = cdist(self.src_dcs, self.src_dcs, metric='euclidean')
        # Multiquadric kernel
        self.phi = np.sqrt((r/self.epsilon)**2 + 1)  
        """ if function == 'multiquadric':
            self.phi = np.sqrt((r/self.epsilon)**2 + 1)
        elif function == 'gaussian':
            self.phi = np.exp(-(r/self.epsilon)**2)
        else:
            raise ValueError(f"Unsupported function: {function}") """
        
        # Solve for weights matrix in one go using batched least squares
        self.weights = np.linalg.lstsq(self.phi, self.src_pcs, rcond=None)[0]

    def interpolate_spectral_data_rbf(self, new_color_values):
        """Vectorized interpolation using precomputed weights."""
        # Calculate distances from new points to training points
        r_new = cdist(new_color_values, self.src_dcs, metric='euclidean')
        
        # Apply RBF kernel
        if hasattr(self, 'phi'):  # Use same kernel type as precomputed
            phi_new = np.sqrt((r_new/self.epsilon)**2 + 1)
        else:
            raise ValueError("Precompute interpolators first")
        
        # Matrix multiply for all wavelengths simultaneously
        return phi_new @ self.weights
