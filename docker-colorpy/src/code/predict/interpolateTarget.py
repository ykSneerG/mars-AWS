import numpy as np # type: ignore
from scipy.interpolate import Rbf, RBFInterpolator # type: ignore
from scipy.spatial.distance import cdist # type: ignore

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

""" 
# ==============================
# Example Usage (Cloud & Local)
# ==============================
if __name__ == "__main__":
    # Example: CMYKOG (6D color space) with spectral data
    num_samples = 1200
    num_wavelengths = 36  # Spectral range: 380-730nm in 10nm steps
    num_channels = 6  # CMYKOG

    # Generate synthetic data
    color_samples = np.random.rand(num_samples, num_channels)
    spectral_samples = np.random.rand(num_samples, num_wavelengths)

    # Initialize and compute RBF interpolation
    rbf_model = RadialBasisFunction()
    rbf_model.set_src_dcs(color_samples)
    rbf_model.set_src_spectra(spectral_samples)
    rbf_model.precompute_interpolators()

    # ✅ **Local NumPy-based usage**
    new_colors_np = np.array([[0.2, 0.3, 0.5, 0.1, 0.6, 0.4]])
    print("Interpolated (NumPy):", rbf_model.interpolate_numpy(new_colors_np))

    # ✅ **Cloud-friendly list-based usage**
    new_colors_list = [[0.2, 0.3, 0.5, 0.1, 0.6, 0.4], [0.6, 0.1, 0.2, 0.4, 0.3, 0.7]]
    print("Interpolated (List for API):", rbf_model.interpolate_list(new_colors_list))
 """
 
 
class ModernRBFInterpolator:
    def __init__(self):
        self.src_dcs = None
        self.src_pcs = None
        self.rbf_interpolator = None
        self.epsilon = None
        self.smoothing = 1e-8

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

    def _compute_epsilon(self):
        """Compute adaptive epsilon based on nearest neighbor distances."""
        distances = cdist(self.src_dcs, self.src_dcs)
        np.fill_diagonal(distances, np.inf)
        return np.mean(np.min(distances, axis=1))

    def precompute_interpolator(self, kernel='multiquadric'):
        """Create single RBF interpolator for all wavelengths."""
        #self._validate_data()
        self.epsilon = self._compute_epsilon()
        
        try:
            self.rbf_interpolator = RBFInterpolator(
                self.src_dcs,
                self.src_pcs,
                kernel=kernel,
                epsilon=self.epsilon,
                smoothing=self.smoothing
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
 
 
 
 
# ============================== DELETE BELOW ==============================
 
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


class OptimizedRBFInterpolator2:
    def __init__(self):
        self.src_dcs = None  # Input device color data (e.g., CMYK)
        self.src_pcs = None  # Output spectral or color data
        self.rbf_interpolators = None  # RBF models per wavelength

    def set_src_dcs(self, dcs_data):
        """Set input device color data (e.g., CMYK)."""
        self.src_dcs = np.asarray(dcs_data, dtype=np.float32)
        if self.src_dcs.ndim != 2:
            raise ValueError("dcs_data must be a 2D array.")

    def set_src_spectra(self, spectral_data):
        """Set output spectral data (e.g., reflectance)."""
        self.src_pcs = np.asarray(spectral_data, dtype=np.float32)
        if self.src_pcs.ndim != 2:
            raise ValueError("spectral_data must be a 2D array.")

    def _validate_data(self):
        """Ensure input and output data are correctly set."""
        if self.src_dcs is None or self.src_pcs is None:
            raise ValueError("Missing source data. Use set_src_dcs() and set_src_spectra().")
        if self.src_dcs.shape[0] != self.src_pcs.shape[0]:
            raise ValueError("Mismatch in number of samples between dcs and spectral data.")
        if self.src_dcs.shape[0] < 2:
            raise ValueError("At least two samples are required for interpolation.")

    def precompute_interpolator(self, function='multiquadric'):
        """Precompute RBF interpolators for each wavelength with automated setup."""
        self._validate_data()
        
        # Compute pairwise distances and determine epsilon based on nearest neighbors
        distances = cdist(self.src_dcs, self.src_dcs)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distance
        nn_dists = np.min(distances, axis=1)
        epsilon_value = np.mean(nn_dists)  # Per-interpolator epsilon heuristic

        num_wavelengths = self.src_pcs.shape[1]
        self.rbf_interpolators = []
        
        # Create an RBF model for each wavelength
        for i in range(num_wavelengths):
            try:
                rbf = Rbf(*self.src_dcs.T, self.src_pcs[:, i], 
                          function=function, epsilon=epsilon_value)
                self.rbf_interpolators.append(rbf)
            except Exception as e:
                raise RuntimeError(f"Failed to create RBF for wavelength index {i}: {e}")

    def interpolate_spectral_data_rbf(self, new_color_values):
        """Interpolate spectral data for new device color values."""
        if self.rbf_interpolators is None:
            raise ValueError("Precompute interpolators first with precompute_interpolator().")
        
        new_color_values = np.asarray(new_color_values, dtype=np.float64)
        if new_color_values.ndim != 2:
            raise ValueError("new_color_values must be a 2D array.")
        if new_color_values.shape[1] != self.src_dcs.shape[1]:
            raise ValueError(f"Expected {self.src_dcs.shape[1]} input channels.")
        
        # Vectorized evaluation across all RBF interpolators
        interpolated = np.array([rbf(*new_color_values.T) for rbf in self.rbf_interpolators]).T
        return interpolated 



class OptimizedRBFInterpolator:
    def __init__(self):
        self.src_dcs = None  # Shape: (n_samples, 4)
        self.src_pcs = None  # Shape: (n_samples, 36)
        self.weights = None  # Shape: (n_samples, 36)
        self.epsilon = None
        self.kernel_function = None

    def set_src_dcs(self, dcs_data):
        """Set input CMYK data as float32 for faster computation."""
        self.src_dcs = np.asarray(dcs_data, dtype=np.float32)
        if self.src_dcs.ndim != 2:
            raise ValueError("dcs_data must be a 2D array.")

    def set_src_spectra(self, spectral_data):
        """Set spectral data as float32."""
        self.src_pcs = np.asarray(spectral_data, dtype=np.float32)
        if self.src_pcs.ndim != 2:
            raise ValueError("spectral_data must be a 2D array.")

    def _validate_data(self):
        """Check data consistency."""
        if self.src_dcs is None or self.src_pcs is None:
            raise ValueError("Missing source data.")
        if self.src_dcs.shape[0] != self.src_pcs.shape[0]:
            raise ValueError("Mismatched samples between dcs and spectra.")
        if self.src_dcs.shape[0] < 2:
            raise ValueError("At least two samples required.")

    def precompute_interpolator(
        self,
        kernel="gaussian",  # Switch to Gaussian for better stability
        regularization=1e-4,  # Increase regularization strength
        svd_cond_threshold=1e-10  # Truncate small singular values
    ):
        self._validate_data()
        self.kernel_function = kernel

        # Check for duplicates in training data (common in CMYK datasets)
        unique_data, unique_indices = np.unique(self.src_dcs, axis=0, return_index=True)
        if len(unique_data) < len(self.src_dcs):
            print(f"Removed {len(self.src_dcs) - len(unique_data)} duplicate samples")
            self.src_dcs = self.src_dcs[unique_indices]
            self.src_pcs = self.src_pcs[unique_indices]

        # Normalize input data to [0, 1] to improve conditioning
        self.dcs_min = self.src_dcs.min(axis=0)
        self.dcs_max = self.src_dcs.max(axis=0)
        self.src_dcs = (self.src_dcs - self.dcs_min) / (self.dcs_max - self.dcs_min + 1e-8)

        # Compute distances and epsilon (avoid near-zero values)
        dcs_data = self.src_dcs.astype(np.float32)
        distances = cdist(dcs_data, dcs_data, metric="euclidean")
        np.fill_diagonal(distances, np.inf)
        nn_dists = np.min(distances, axis=1)
        self.epsilon = np.maximum(np.median(nn_dists), 1e-4).astype(np.float32)

        # Build kernel matrix with regularization
        if kernel == "multiquadric":
            Phi = np.sqrt((distances / self.epsilon) ** 2 + 1)
        elif kernel == "gaussian":
            Phi = np.exp(-(distances / self.epsilon) ** 2)
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        Phi = Phi.astype(np.float64)  # Use float64 for inversion stability
        Phi += np.eye(Phi.shape[0]) * regularization  # Regularize diagonal

        # Truncated SVD solve: U @ diag(s) @ Vh = Phi
        try:
            U, s, Vh = np.linalg.svd(Phi, full_matrices=False)
            s_inv = np.zeros_like(s)
            mask = s > svd_cond_threshold
            s_inv[mask] = 1 / s[mask]
            self.weights = (Vh.T @ np.diag(s_inv) @ U.T) @ self.src_pcs.astype(np.float64)
            self.weights = self.weights.astype(np.float32)  # Save memory
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"SVD failed despite stabilization: {e}")
    
    def interpolate_spectral_data_rbf(self, new_color_values):
        """Vectorized prediction using precomputed weights."""
        if self.weights is None:
            raise ValueError("Call precompute_interpolator() first.")
        
        new_color_values = np.asarray(new_color_values, dtype=np.float32)
        if new_color_values.ndim != 2:
            raise ValueError("new_color_values must be 2D.")
        if new_color_values.shape[1] != 4:
            raise ValueError("Expected 4 input channels (CMYK).")
        
        # Compute distances between new and training points
        distances = cdist(new_color_values, self.src_dcs, metric='euclidean')
        distances = distances.astype(np.float32)
        
        # Apply kernel function
        if self.kernel_function == 'multiquadric':
            K = np.sqrt((distances / self.epsilon)**2 + 1)
        elif self.kernel_function == 'gaussian':
            K = np.exp(-(distances / self.epsilon)**2)
        K = K.astype(np.float32)
        
        # Predict: (m x n) @ (n x 36) = (m x 36)
        return K @ self.weights

