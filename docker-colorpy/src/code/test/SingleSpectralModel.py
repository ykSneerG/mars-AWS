import time

import numpy as np
from scipy.optimize import minimize
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures

class SingleCalibrationSpectralModel:
    def __init__(self, wavelengths, channels=4):
        """Initialize for a fixed printer/ink/substrate setup"""
        self.wavelengths = np.array(wavelengths)
        self.n_channels = channels  # Use the channels parameter
        self.poly_degree = 3  # Lower degree for stable single-calibration DEFAULT: 2
        self.ink_limits = np.array([1.0] * channels)  # Default CMYK limits
        self.total_ink_limit = 3.2  # Typical max ink coverage (3.2 = 320%)
        
        # Physics parameters (to be calibrated)
        self.mu_a = np.zeros((channels, len(wavelengths)))  # Absorption (channels x wavelengths)
        self.mu_s = np.zeros((channels, len(wavelengths)))  # Scattering
        
        # Correction model
        self.gp_models = []  # One GP per wavelength
        self.patches = None  # Training patches (CMYK values)
        self.measured_R = None  # Measured reflectance

    # def _km_model(self, mu_a, mu_s):
    #     """Simplified Kubelka-Munk for single setup"""
    #     mu_eff = np.sqrt(mu_a * (mu_a + 2 * mu_s))
    #     return 1 / (1 + mu_eff)
    
    def _km_model(self, mu_a, mu_s, substrate_absorption=0.02):
        """Enhanced KM model with substrate and nonlinearity"""
        mu_eff = np.sqrt((mu_a + substrate_absorption) * (mu_a + substrate_absorption + 2 * mu_s))
        return np.exp(-2.5 * mu_eff)  # Empirical factor for ink penetration
    
    # def _km_model(self, mu_a, mu_s):
    #     """True KM equation (no empirical factors)"""
    #     ratio = mu_a / (mu_s + 1e-10)  # Avoid division by zero
    #     return 1 + ratio - np.sqrt(ratio**2 + 2 * ratio)
    
    # def _km_model(self, mu_a, mu_s, n=1.8):
    #     """Yule-Nielsen for better halftone prediction"""
    #     R_inf = 1 + (mu_a / mu_s) - np.sqrt((mu_a / mu_s)**2 + 2 * (mu_a / mu_s))
    #     return R_inf ** (1 / n)

    def _physics_prediction(self, concentrations):
        """Predict reflectance using physical model only"""
        mix_mu_a = np.sum(concentrations[:, None] * self.mu_a, axis=0)
        mix_mu_s = np.sum(concentrations[:, None] * self.mu_s, axis=0)
        return self._km_model(mix_mu_a, mix_mu_s)

    def fit(self, patches, measured_R, n_iter=30):
        """Calibrate model with Neugebauer primaries + mixes"""
        self.patches = np.array(patches)
        self.measured_R = np.array(measured_R)
        
        # In fit():
        """ bounds = [
            (0.1, 10)] * (4 * len(wavelengths)) +  # μ_a bounds (strictly positive)
            (5, 50)] * (4 * len(wavelengths))       # μ_s bounds (higher than μ_a)
        ] """
        
        # Stage 1: Optimize physics parameters
        def loss(params):
            self.mu_a = params[:4*len(self.wavelengths)].reshape(4, -1)
            self.mu_s = params[4*len(self.wavelengths):].reshape(4, -1)
            R_pred = np.array([self._physics_prediction(p) for p in self.patches])
            return np.mean((R_pred - self.measured_R)**2)
        
        def sam_loss(pred, target):
            """Spectral Angle Mapper (0°=perfect match, 90°=worst)"""
            dot = np.sum(pred * target, axis=1)
            norm_pred = np.linalg.norm(pred, axis=1)
            norm_target = np.linalg.norm(target, axis=1)
            return np.mean(np.arccos(dot / (norm_pred * norm_target)))
        
        def hybrid_loss(params, alpha=0.5):
            """alpha=0.5 balances MSE and SAM"""
            self.mu_a = params[:4*len(self.wavelengths)].reshape(4, -1)
            self.mu_s = params[4*len(self.wavelengths):].reshape(4, -1)
            R_pred = np.array([self._physics_prediction(p) for p in self.patches])
            
            mse = np.mean((R_pred - self.measured_R)**2)
            sam = sam_loss(R_pred, self.measured_R)
            return alpha * mse + (1 - alpha) * sam
        
        """ initial_guess = np.concatenate([
            0.1 * np.ones(4 * len(self.wavelengths)),  # mu_a
            10 * np.ones(4 * len(self.wavelengths))     # mu_s
        ]) """
        
        initial_guess = np.concatenate([
            0.1 * np.ones(4 * len(self.wavelengths)),  # Higher initial μₐ
            20 * np.ones(4 * len(self.wavelengths))    # Higher initial μₛ
        ])
        bounds = [(0.0001, 10)] * len(initial_guess)  # Stricter bounds
        
         # μ_a bounds (strictly positive) # μ_s bounds (higher than μ_a)
        #bounds = [(0.1, 10)] * (4 * len(wavelengths)) + [(5, 50)] * (4 * len(wavelengths))      
        
        result = minimize(hybrid_loss, initial_guess, method='L-BFGS-B', 
                         bounds=bounds,
                         options={'maxiter': n_iter})
        
        
        print(f"Optimization success: {result.success}")
        print(f"Final loss: {result.fun:.6f} (RMS: {np.sqrt(result.fun):.4f})")
        
        """"
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        self.mu_a = result.x[:4*len(self.wavelengths)].reshape(4, -1)  # Explicit update
        self.mu_s = result.x[4*len(self.wavelengths):].reshape(4, -1) 
        """
        
        # Stage 2: Learn residuals with GPs
        X = PolynomialFeatures(self.poly_degree).fit_transform(self.patches)
        physics_pred = np.array([self._physics_prediction(p) for p in self.patches])
        residuals = self.measured_R - physics_pred
        
        self.gp_models = []
        for wl in range(len(self.wavelengths)):
            gp = GaussianProcessRegressor(alpha=1e-4, n_restarts_optimizer=5)
            gp.fit(X, residuals[:, wl])
            self.gp_models.append(gp)

    def predict(self, concentrations):
        """Predict reflectance for CMYK values (0-1)"""
        concentrations = np.clip(concentrations, 0, self.ink_limits)
        if np.sum(concentrations) > self.total_ink_limit:
            concentrations = self._apply_gcr(concentrations)
            
        R_phys = self._physics_prediction(concentrations)
        X = PolynomialFeatures(self.poly_degree).fit_transform(concentrations.reshape(1, -1))
        correction = np.array([gp.predict(X)[0] for gp in self.gp_models])
        return np.clip(R_phys + correction, 0, 1)
    
    def predict_all(self, concentrations_list):
        """Predict reflectance for multiple CMYK values"""
        return np.array([self.predict(c) for c in concentrations_list])

    def add_data(self, new_patches, new_spectra):
        """Incrementally update model with new measurements"""
        if self.patches is None:
            self.patches = np.array(new_patches)
            self.measured_R = np.array(new_spectra)
        else:
            self.patches = np.vstack([self.patches, new_patches])
            self.measured_R = np.vstack([self.measured_R, new_spectra])
        self.fit(self.patches, self.measured_R)  # Recalibrate with all data

    def save(self, filename):
        """Save model to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'wavelengths': self.wavelengths,
                'mu_a': self.mu_a,
                'mu_s': self.mu_s,
                'gp_models': self.gp_models,
                'patches': self.patches,
                'measured_R': self.measured_R,
                'ink_limits': self.ink_limits
            }, f)

    @classmethod
    def load(cls, filename):
        """Load model from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(data['wavelengths'])
        model.mu_a = data['mu_a']
        model.mu_s = data['mu_s']
        model.gp_models = data['gp_models']
        model.patches = data['patches']
        model.measured_R = data['measured_R']
        model.ink_limits = data.get('ink_limits', np.ones(4))
        return model

    def _apply_gcr(self, concentrations):
        """Basic gray component replacement"""
        min_cmy = np.min(concentrations[:3])
        if min_cmy > 0.1:
            k = min_cmy * 0.8  # Partial replacement
            concentrations[:3] -= k
            concentrations[3] += k
        return np.clip(concentrations, 0, self.ink_limits)

# ===========================================================================
# Example Usage
# ===========================================================================
if __name__ == "__main__":
    
    # Start timer
    start_time = time.time()
    
    
    # Initialize with measurement wavelengths (e.g., 380-730nm in 10nm steps)
    wavelengths = np.arange(380, 740, 10)  # Updated range
    model = SingleCalibrationSpectralModel(wavelengths, 4)

    # 1. Calibrate with Neugebauer primaries + some mixtures (e.g., 30 patches)
    """ patches = [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],  # C, M, Y, K
        [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0],  # CM, CY, MY
        [0.5, 0, 0, 0], [0, 0.5, 0, 0],  # Toned primaries
        [0.3, 0.3, 0.3, 0],  # CMY gray
        # ... add more mixtures as needed
    ] """
    
    patches = [
        [1,1,1,1],[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0],[0,0,1,1],[0,1,0,1],[0,1,1,0],[1,0,0,1],[1,0,1,0],[1,1,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,0]
    ]
    
    
    # Simulated measurements (replace with real spectrometer data)
    # measured_R = np.random.rand(len(patches), len(wavelengths)) * 0.98 + 0.01
    
    measured_R = np.array(
        [
            [0.0043,0.0062,0.0068,0.0075,0.0072,0.0071,0.0079,0.0083,0.0089,0.0094,0.0101,0.0104,0.0103,0.0095,0.0086,0.0078,0.0074,0.007,0.0067,0.0066,0.0066,0.0067,0.0065,0.0064,0.0062,0.006,0.0059,0.0059,0.0059,0.0058,0.0057,0.0055,0.0054,0.0056,0.0062,0.0071],
            [0.0078,0.0082,0.0093,0.0105,0.0096,0.0093,0.0094,0.0094,0.01,0.0107,0.0113,0.0118,0.0112,0.0104,0.0094,0.0082,0.0077,0.0073,0.0072,0.0073,0.0082,0.009,0.0096,0.0098,0.0098,0.0099,0.0102,0.0101,0.01,0.0099,0.01,0.0101,0.0102,0.0106,0.0109,0.0111],
            [0.0079,0.0092,0.0092,0.0103,0.0101,0.0094,0.0098,0.0104,0.0114,0.0124,0.0138,0.0153,0.0164,0.0162,0.0151,0.0139,0.0125,0.0111,0.0094,0.0084,0.0075,0.007,0.0066,0.0065,0.0063,0.0059,0.0058,0.0058,0.0056,0.0055,0.0056,0.0054,0.0053,0.0056,0.0061,0.0069],
            [0.0057,0.0078,0.0077,0.0081,0.008,0.0072,0.0073,0.007,0.0067,0.0066,0.0063,0.0058,0.0055,0.005,0.0047,0.0044,0.004,0.0044,0.0043,0.0045,0.0049,0.0051,0.0049,0.0048,0.0047,0.0045,0.0042,0.0041,0.004,0.0038,0.004,0.004,0.004,0.004,0.0045,0.0054],[0.0117,0.0128,0.0167,0.0178,0.0192,0.0221,0.0264,0.0326,0.0426,0.0566,0.0745,0.0933,0.1049,0.0959,0.0657,0.0394,0.0293,0.025,0.0183,0.0148,0.0163,0.0185,0.0181,0.0169,0.0167,0.0171,0.0183,0.0215,0.0261,0.0282,0.0267,0.0236,0.0197,0.0185,0.0229,0.0381],[0.0091,0.009,0.0086,0.0094,0.0099,0.01,0.0096,0.01,0.0104,0.0118,0.0129,0.0144,0.0153,0.0156,0.0153,0.0149,0.0144,0.0141,0.0139,0.014,0.0138,0.0138,0.0139,0.014,0.014,0.0141,0.0143,0.0146,0.0147,0.0147,0.0148,0.0152,0.0155,0.0158,0.0162,0.0168],[0.0059,0.0052,0.0056,0.0041,0.005,0.0054,0.0056,0.0057,0.0061,0.0056,0.0054,0.0051,0.0051,0.0049,0.0045,0.0042,0.004,0.004,0.0041,0.0044,0.0054,0.0067,0.0074,0.008,0.0084,0.0086,0.0088,0.009,0.0091,0.0093,0.0093,0.0096,0.01,0.0104,0.0108,0.0114],[0.0151,0.0162,0.0194,0.0224,0.0251,0.0275,0.031,0.0369,0.0468,0.0608,0.0794,0.1005,0.116,0.1119,0.0827,0.0558,0.0502,0.0564,0.0569,0.0703,0.1426,0.2894,0.4651,0.6195,0.7228,0.782,0.8146,0.8331,0.8435,0.8481,0.8497,0.8515,0.8548,0.8584,0.8591,0.8617],[0.0057,0.0066,0.0083,0.008,0.0084,0.0086,0.009,0.0094,0.009,0.0088,0.0089,0.009,0.0089,0.0086,0.0082,0.0077,0.0071,0.0062,0.0053,0.0047,0.0044,0.0042,0.004,0.0039,0.0038,0.0037,0.0035,0.0033,0.0035,0.0035,0.0033,0.0034,0.0034,0.0036,0.0043,0.005],[0.0158,0.0175,0.0182,0.0209,0.0242,0.0279,0.0342,0.045,0.0638,0.0969,0.157,0.2582,0.3798,0.4463,0.4177,0.339,0.2542,0.1739,0.1072,0.0649,0.0428,0.0322,0.0256,0.0217,0.0207,0.0211,0.0225,0.0262,0.0314,0.0336,0.032,0.0285,0.0243,0.0227,0.0275,0.0435],[0.024,0.0525,0.1089,0.1921,0.2788,0.3257,0.3443,0.3347,0.3053,0.2713,0.2277,0.1833,0.1477,0.112,0.0686,0.0377,0.0268,0.0225,0.0159,0.0134,0.017,0.0229,0.026,0.0268,0.0275,0.0288,0.0309,0.035,0.0403,0.0425,0.0408,0.0372,0.033,0.0313,0.0359,0.0521],[0.0058,0.0068,0.0063,0.0074,0.0072,0.0071,0.0071,0.0072,0.0074,0.0073,0.0074,0.0074,0.0074,0.0074,0.0076,0.0077,0.0078,0.008,0.0081,0.0082,0.0083,0.0086,0.0087,0.0089,0.0092,0.0093,0.0095,0.0098,0.0101,0.0102,0.0105,0.0107,0.011,0.0113,0.0118,0.0123],[0.0252,0.0227,0.0222,0.0239,0.0256,0.0283,0.0328,0.0422,0.0604,0.0931,0.158,0.2806,0.4691,0.6638,0.7846,0.8314,0.8458,0.8493,0.8475,0.85,0.8499,0.8519,0.8547,0.8584,0.8619,0.8653,0.8709,0.8764,0.8803,0.8817,0.8821,0.8829,0.8851,0.8878,0.8881,0.8894],[0.0694,0.1241,0.2102,0.3238,0.4465,0.4761,0.444,0.3946,0.3495,0.3066,0.2565,0.2081,0.1713,0.1356,0.089,0.0549,0.0485,0.055,0.0553,0.0701,0.1481,0.3024,0.48,0.6327,0.7349,0.794,0.8268,0.8456,0.8559,0.8604,0.862,0.8637,0.8669,0.8702,0.8709,0.8728],[0.0264,0.0684,0.1513,0.2803,0.4371,0.5539,0.6581,0.7251,0.7438,0.7401,0.7146,0.6772,0.6252,0.5529,0.4611,0.3628,0.2705,0.1843,0.1134,0.0699,0.0479,0.038,0.0319,0.0282,0.0274,0.028,0.0298,0.0339,0.0392,0.0417,0.0398,0.0363,0.0319,0.0302,0.0347,0.0506],[0.1026,0.1911,0.3315,0.5481,0.8422,0.9781,0.9859,0.9625,0.9527,0.9428,0.9272,0.9172,0.908,0.898,0.8886,0.8816,0.8785,0.8744,0.869,0.8703,0.8691,0.8705,0.8729,0.8763,0.8796,0.8829,0.8884,0.8937,0.8975,0.8988,0.8987,0.899,0.9009,0.9037,0.9038,0.9049]
        ]
    )
    
    print("First 4 training patches (CMYK):", patches[:1])
    print("First 4 training patches (CMYK):", measured_R[:1])
    
    
    # Calibrate model
    model.fit(patches, measured_R)

    # 2. Save model
    model.save("my_printer_model.pkl")

    # 3. Later... load and add more data
    """ 
    loaded_model = SingleCalibrationSpectralModel.load("my_printer_model.pkl")
    new_patches = [[0.7, 0.2, 0.1, 0], [0.2, 0.7, 0.1, 0]]  # New mixtures
    new_spectra = np.random.rand(2, len(wavelengths)) * 0.5 + 0.3
    loaded_model.add_data(new_patches, new_spectra)
    """
    
    # 4. Predict
    cmyk1 = np.array([0.8, 0.5, 0.1, 0.05])
    cmyk2 = np.array([0.2, 0.3, 0.5, 0.1])
    #cmyk = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    cmyk = np.array([[1, 1, 1, 1]])
    
    # merge 
    #cmyk = np.vstack([cmyk1.reshape(1, -1), cmyk2.reshape(1, -1), cmyk3]) 
    #cmyk = np.vstack([cmyk1.reshape(1, -1), cmyk2.reshape(1, -1), cmyk3])   
    
    print(f"Input CMYK values: {cmyk}")
    predicted_spectrum = model.predict_all(cmyk)
    #print(f"Predicted reflectance at 500nm: {predicted_spectrum[wavelengths == 500][0]:.3f}")
    print(f"Predicted reflectance: {predicted_spectrum}")
    
    
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
    
    
    # Compare predicted vs measured for a patch
    import matplotlib.pyplot as plt
    
    idx = 11  # First patch
    R_pred = model._physics_prediction(model.patches[idx])
    R_meas = model.measured_R[idx]
    print(f"Predicted reflectance: {R_pred}")
    print(f"Measured reflectance: {R_meas}")
    
    plt.plot(model.wavelengths, R_meas, 'k-', label='Measured')
    plt.plot(model.wavelengths, R_pred, 'r--', label='KM Prediction')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.show()