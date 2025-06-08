import numpy as np # type: ignore
from scipy.interpolate import Rbf, RBFInterpolator, CubicHermiteSpline # type: ignore
from scipy.spatial.distance import cdist # type: ignore

from enum import Enum

class BaseInterpolation:
    def __init__(self):
        self.src_dcs = None  # Device color space (e.g., CMYK, RGB)
        self.src_pcs = None  # Spectral reflectance values

    def set_src_dcs(self, data):
        """Set device color space data (e.g., CMYK, RGB)."""
        self.src_dcs = np.asarray(data, dtype=np.float32)
        
        if self.src_dcs.ndim != 2:
            raise ValueError("dcs_data must be a 2D array of shape (n_samples, n_features)")

    def set_src_pcs(self, data):
        """Set spectral reflectance data."""
        self.src_pcs = np.asarray(data, dtype=np.float32)
        
        if self.src_pcs.ndim != 2:
            raise ValueError("pcs_data must be a 2D array of shape (n_samples, n_wavelengths)")
        
        # Set negative values to a small positive number
        if np.any(self.src_pcs < 0):
            self.src_pcs[self.src_pcs < 0] = 1e-8  


# --#-#-- INTERPOLATION scipy.interpolate.RbRBFInterpolator --#-#--
 
class ModernRBFkernel(Enum):
    """ 
    **ModernRBFkernel**
    
    supported in SKIPY 1.15.2 scipy.interpolate.RBFInterpolator
    
    ’linear’ : -r
    ‘thin_plate_spline’ : r**2 * log(r)
    ‘cubic’ : r**3
    ‘quintic’ : -r**5
    ‘multiquadric’ : -sqrt(1 + r**2)
    ‘inverse_multiquadric’ : 1/sqrt(1 + r**2)
    ‘inverse_quadratic’ : 1/(1 + r**2)
    ‘gaussian’ : exp(-r**2)
    """
    
    LINEAR = 'linear'
    THINPLATESPLINE = 'thin_plate_spline'
    CUBIC = 'cubic'
    QUINTIC = 'quintic'
    MULTIQUADRATIC = 'multiquadric'
    INVERS_MULTIQUADRATIC = 'inverse_multiquadric'
    INVERS_QUADRATIC = 'inverse_quadratic'
    GAUSSIAN = 'gaussian'
    
 
class ModernRBFInterpolator(BaseInterpolation):
    
    def __init__(self):
        super().__init__()
        """ self.src_dcs = None
        self.src_pcs = None """
        self.rbf_interpolator = None
        self.epsilon = None
        self.smoothing = 1e-8
        self.kernel = ModernRBFkernel.MULTIQUADRATIC.value
        self.degree = 4  # Default degree for polynomial fitting, can be adjusted

    """ 
    def set_src_dcs(self, dcs_data):
        #Set input device color data with validation.
        self.src_dcs = np.asarray(dcs_data, dtype=np.float32)
        if self.src_dcs.ndim != 2:
            raise ValueError("dcs_data must be a 2D array of shape (n_samples, n_features)")

    def set_src_pcs(self, spectral_data):
        #Set output spectral data with physics-aware validation.
        self.src_pcs = np.asarray(spectral_data, dtype=np.float32)
        if self.src_pcs.ndim != 2:
            raise ValueError("spectral_data must be a 2D array of shape (n_samples, n_wavelengths)")
        if np.any(self.src_pcs < 0):
            #raise ValueError("Spectral data contains negative values - physically impossible")
            self.src_pcs[self.src_pcs < 0] = 1e-8  # Set negative values to a small positive number
    """
    
    def set_smoothness(self, smoothing):
        """Set regularization parameter for RBF smoothing."""
        if smoothing <= 0:
            raise ValueError("Smoothing parameter must be positive")
        self.smoothing = smoothing
        
    def set_degree(self, degree):
        """Set polynomial degree for RBF interpolation."""
        if degree < 0:
            raise ValueError("Degree must be non-negative")
        self.degree = degree

    def set_epsilon(self, epsilon):
        """Set epsilon for RBF interpolation."""
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        self.epsilon = epsilon

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
                degree=self.degree
            )
        except ValueError as e:
            raise RuntimeError(f"RBF interpolation failed: {str(e)}") from e

    def interpolate(self, new_color_values) -> np.ndarray:
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

