import numpy as np
import pickle
import json
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime


class SpectralInkModel:
    def __init__(self, wavelengths, n_channels=4, poly_degree=3):
        """Initialize with measurement wavelengths and ink channels (CMYK)"""
        self.wavelengths = np.array(wavelengths)
        self.n_channels = n_channels
        self.poly_degree = poly_degree
        self.opt_params = None  # Physics parameters (μₐ, μₛ)
        self.gp_models = []     # Per-wavelength GP correctors
        self.patches = None     # Training patches (CMYK combinations)
        self.measured_R = None  # Measured reflectance spectra
        self.ink_limits = np.ones(n_channels)  # Default ink limits (100%)
        self.total_ink_limit = 3.2  # Typical max ink coverage (320%)

    def _km_model(self, mu_a, mu_s):
        """Enhanced Kubelka-Munk with nonlinear terms"""
        mu_eff = np.sqrt(mu_a * (mu_a + 2*mu_s + 0.1*mu_a*mu_s))
        return 1 / (1 + mu_eff + 0.2*mu_eff**2)

    def _physics_prediction(self, concentrations):
        """Physics-based reflectance prediction"""
        mu_a = self.opt_params[:self.n_channels*len(self.wavelengths)].reshape(self.n_channels, -1)
        mu_s = self.opt_params[self.n_channels*len(self.wavelengths):].reshape(self.n_channels, -1)
        
        mix_mu_a = np.zeros_like(self.wavelengths, dtype=np.float64)
        mix_mu_s = np.zeros_like(self.wavelengths, dtype=np.float64)
        
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                weight = concentrations[i] * concentrations[j]
                mix_mu_a += weight * (mu_a[i] if i == j else 0.5*(mu_a[i]+mu_a[j]))
                mix_mu_s += weight * (mu_s[i] if i == j else 0.7*(mu_s[i]+mu_s[j]))
        
        return self._km_model(mix_mu_a, mix_mu_s)

    def _get_features(self, concentrations):
        """Generate polynomial features for GP correction"""
        return PolynomialFeatures(self.poly_degree).fit_transform(concentrations)

    def fit(self, patches, measured_R, n_iter=50):
        """Train the model with initial data"""
        self.patches = np.array(patches)
        self.measured_R = np.array(measured_R)
        
        # Stage 1: Physics optimization
        def loss(params):
            self.opt_params = params
            R_pred = np.array([self._physics_prediction(p) for p in self.patches])
            return np.mean((R_pred - self.measured_R)**2)
        
        # Initialize parameters
        init_params = np.concatenate([
            0.1 * np.ones(self.n_channels*len(self.wavelengths)),  # μₐ
            10 * np.ones(self.n_channels*len(self.wavelengths))    # μₛ
        ])
        
        bounds = [(1e-6, None)] * len(init_params)
        result = minimize(loss, init_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': n_iter})
        self.opt_params = result.x
        
        # Stage 2: GP correction
        physics_pred = np.array([self._physics_prediction(p) for p in self.patches])
        residuals = self.measured_R - physics_pred
        X = self._get_features(self.patches)
        
        self.gp_models = []
        for wl in range(len(self.wavelengths)):
            gp = GaussianProcessRegressor(n_restarts_optimizer=5)
            gp.fit(X, residuals[:, wl])
            self.gp_models.append(gp)

    def predict(self, concentrations):
        """Predict reflectance for given CMYK values (0-1)"""
        concentrations = np.clip(concentrations, 0, self.ink_limits)
        if np.sum(concentrations) > self.total_ink_limit:
            concentrations = self._apply_gcr(concentrations)
            
        R_phys = self._physics_prediction(concentrations)
        X = self._get_features(concentrations.reshape(1, -1))
        correction = np.array([gp.predict(X)[0] for gp in self.gp_models])
        return np.clip(R_phys + correction, 0, 1)

    def update_model(self, new_patches, new_spectra):
        """Incremental update with new measurements"""
        self.patches = np.vstack([self.patches, new_patches])
        self.measured_R = np.vstack([self.measured_R, new_spectra])
        
        # Update GP models
        physics_pred = np.array([self._physics_prediction(p) for p in self.patches])
        residuals = self.measured_R - physics_pred
        X = self._get_features(self.patches)
        
        for wl, gp in enumerate(self.gp_models):
            gp.fit(X, residuals[:, wl])  # Warm-start refit

    def save(self, path, method='joblib'):
        """Save model to disk"""
        if method == 'pickle':
            with open(path + '.pkl', 'wb') as f:
                pickle.dump(self, f)
        elif method == 'json':
            data = {
                'wavelengths': self.wavelengths.tolist(),
                'opt_params': self.opt_params.tolist(),
                'patches': self.patches.tolist(),
                'measured_R': self.measured_R.tolist(),
                'ink_limits': self.ink_limits.tolist(),
                'gp_params': [gp.get_params() for gp in self.gp_models]
            }
            with open(path + '.json', 'w') as f:
                json.dump(data, f)

    @classmethod
    def load(cls, path, method='joblib'):
        """Load model from disk"""
        if method == 'pickle':
            with open(path + '.pkl', 'rb') as f:
                return pickle.load(f)
        elif method == 'json':
            with open(path + '.json', 'r') as f:
                data = json.load(f)
            model = cls(np.array(data['wavelengths']))
            model.opt_params = np.array(data['opt_params'])
            model.patches = np.array(data['patches'])
            model.measured_R = np.array(data['measured_R'])
            model.ink_limits = np.array(data['ink_limits'])
            model.gp_models = [GaussianProcessRegressor(**params) for params in data['gp_params']]
            return model

    def _apply_gcr(self, concentrations):
        """Gray component replacement"""
        min_cmy = np.min(concentrations[:3])
        if min_cmy > 0.1:  # Only apply if significant CMY present
            k = min_cmy * 0.9  # Partial replacement
            concentrations[:3] -= k
            concentrations[3] += k
        return np.clip(concentrations, 0, self.ink_limits)

    def suggest_new_patches(self, n_candidates=1000, top_n=5):
        """Active learning: suggest most informative patches to measure"""
        candidates = np.random.dirichlet(np.ones(self.n_channels), size=n_candidates)
        uncertainties = []
        
        for c in candidates:
            X = self._get_features(c.reshape(1, -1))
            var = np.mean([gp.predict(X, return_std=True)[1] for gp in self.gp_models])
            uncertainties.append(var)
            
        return candidates[np.argsort(uncertainties)[-top_n:]]

# ================================================
# Example Usage
# ================================================

if __name__ == "__main__":
    
    # Start timer
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    
    # 1. Initialize with measurement wavelengths (e.g., 400-700nm in 10nm steps)
    wavelengths = np.arange(380, 730, 10)
    model = SpectralInkModel(wavelengths)
    
    def trainer():

        # 2. Generate and measure training patches (e.g., 24 Neugebauer + gradients)
        patches = np.vstack([
            np.eye(4),  # C, M, Y, K
            0.5 * (np.eye(4) + np.roll(np.eye(4), 1, axis=0)),  # CM, MY, YC, KC
            np.random.dirichlet(np.ones(4), size=16)  # Random mixtures
        ])
        print("Training patches (CMYK):", patches)
        
        # Simulated measurements (replace with real spectrometer data)
        measured_R = np.random.rand(len(patches), len(wavelengths)) * 0.8 + 0.2  # 0.2-1.0 reflectance
        print("Measured reflectance spectra:", measured_R)

        # 3. Train initial model
        model.fit(patches, measured_R)

        # 4. Save model
        model.save("ink_model_v1")
        
    def updater():

        # 5. Later... load and update with new data
        new_model = SpectralInkModel.load("ink_model_v1")
        new_patches = np.array([[0.7, 0.2, 0.1, 0.0], [0.1, 0.1, 0.1, 0.7]])
        new_spectra = np.random.rand(2, len(wavelengths)) * 0.5 + 0.3
        new_model.update_model(new_patches, new_spectra)

    def predictor():
        
        # 5. Later... load and update with new data
        #new_model = SpectralInkModel.load("ink_model_v1")

        # 6. Get predictions
        cmyk = np.array([0.8, 0.5, 0, 0.1])  # 80C 50M 10K
        predicted_spectrum = model.predict(cmyk)
        print(f"Predicted reflectance at 500nm: {predicted_spectrum[wavelengths == 500][0]:.3f}")
        print(f"Predicted reflectance at 500nm: {predicted_spectrum}")

    def suggestor():
        
        # 5. Later... load and update with new data
        new_model = SpectralInkModel.load("ink_model_v1")

        # 7. Active learning suggestions
        next_patches = new_model.suggest_new_patches()
        print("Suggested patches to measure next:", next_patches)


    # Run the functions
    trainer()
    
    print (f"Time taken for training: {datetime.now() - start_time}")
    
    predictor()
    
    print (f"Time taken for training: {datetime.now() - start_time}")