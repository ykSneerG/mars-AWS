import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

# Load spectral data (replace with your measurements)
wavelengths = np.linspace(400, 700, 36)  # 36 channels (400-700nm)
R_paper = np.array([...])  # Paper reflectance

# Solid inks
R_C = np.array([...])  # Cyan
R_M = np.array([...])  # Magenta
R_Y = np.array([...])  # Yellow
R_K = np.array([...])  # Black

# Primary ramps (e.g., 10%, 25%, 50%, 75%, 100%)
R_C_ramp = {10: np.array([...]), 25: np.array([...]), ..., 100: R_C}  # Cyan ramp
R_M_ramp = {10: np.array([...]), 25: np.array([...]), ..., 100: R_M}  # Magenta ramp
R_Y_ramp = {10: np.array([...]), 25: np.array([...]), ..., 100: R_Y}  # Yellow ramp
R_K_ramp = {10: np.array([...]), 25: np.array([...]), ..., 100: R_K}  # Black ramp

# Cellular Neugebauer mixtures (e.g., 50%C+50%M, 50%C+50%Y, etc.)
R_50C50M = np.array([...])  # 50%C + 50%M
R_50C50Y = np.array([...])  # 50%C + 50%Y
R_50M50Y = np.array([...])  # 50%M + 50%Y
R_50C50K = np.array([...])  # 50%C + 50%K
# ... Add other key mixtures

# Cellular grid (2x2x2x2 for CMYK)
grid_points = [
    np.array([0, 50, 100]),  # C (%)
    np.array([0, 50, 100]),  # M (%)
    np.array([0, 50, 100]),  # Y (%)
    np.array([0, 50, 100])   # K (%)
]

# Initialize parameter grid (K_C, S_C, K_M, S_M, K_Y, S_Y, K_K, S_K, n)
param_grid = np.zeros((2, 2, 2, 2, 9))  # 16 cells × 9 params

def km_reflectance(K, S):
    """Kubelka-Munk reflectance from K/S."""
    return 1 + (K/S) - np.sqrt((K/S)**2 + 2*(K/S))

def yn_predict(params, a_C, a_M, a_Y, a_K, R_C, R_M, R_Y, R_K):
    """Yule-Nielsen prediction for a CMYK mixture."""
    K_C, S_C, K_M, S_M, K_Y, S_Y, K_K, S_K, n = params
    R_C_km = km_reflectance(K_C, S_C) if a_C > 0 else 0
    R_M_km = km_reflectance(K_M, S_M) if a_M > 0 else 0
    R_Y_km = km_reflectance(K_Y, S_Y) if a_Y > 0 else 0
    R_K_km = km_reflectance(K_K, S_K) if a_K > 0 else 0
    R_paper_km = R_paper  # Paper reflectance
    return (a_C*R_C_km**(1/n) + a_M*R_M_km**(1/n) + a_Y*R_Y_km**(1/n) + a_K*R_K_km**(1/n) + (1-a_C-a_M-a_Y-a_K)*R_paper_km**(1/n))**n

def error(params, a_C, a_M, a_Y, a_K, R_measured):
    """Error between predicted and measured reflectance."""
    R_pred = yn_predict(params, a_C, a_M, a_Y, a_K, R_C, R_M, R_Y, R_K)
    return np.sum((R_measured - R_pred)**2)

# Step 1: Fit primary ramps (per-channel K/S optimization)
def fit_ramp(ramp_data, ink_name):
    """Fit K/S for a single ink using its ramp."""
    def ramp_error(params):
        K, S = params
        error = 0
        for coverage, R_measured in ramp_data.items():
            a = coverage / 100.0
            R_pred = km_reflectance(a*K, a*S)  # K-M for halftones
            error += np.sum((R_measured - R_pred)**2)
        return error
    result = minimize(ramp_error, [2.0, 1.0], bounds=[(0.1, 10), (0.1, 5)])
    return result.x

K_C, S_C = fit_ramp(R_C_ramp, 'Cyan')
K_M, S_M = fit_ramp(R_M_ramp, 'Magenta')
K_Y, S_Y = fit_ramp(R_Y_ramp, 'Yellow')
K_K, S_K = fit_ramp(R_K_ramp, 'Black')

# Step 2: Fit cellular mixtures (optimize n per cell)
for i in range(2):  # C cells
    for j in range(2):  # M cells
        for k in range(2):  # Y cells
            for l in range(2):  # K cells
                # Skip paper-only cell
                if i == 0 and j == 0 and k == 0 and l == 0:
                    continue
                
                # Assign base K/S from ramps
                params_init = [K_C, S_C, K_M, S_M, K_Y, S_Y, K_K, S_K, 2.0]
                
                # Optimize n for mixtures in this cell
                if i == 1 and j == 1 and k == 0 and l == 0:  # 50%C + 50%M
                    a_C, a_M, a_Y, a_K = 0.5, 0.5, 0.0, 0.0
                    R_measured = R_50C50M
                elif i == 1 and j == 0 and k == 1 and l == 0:  # 50%C + 50%Y
                    a_C, a_M, a_Y, a_K = 0.5, 0.0, 0.5, 0.0
                    R_measured = R_50C50Y
                # ... Add other mixtures
                
                bounds = [(0.1, 10), (0.1, 5)] * 4 + [(1.0, 3.0)]  # K/S bounds + n
                result = minimize(
                    lambda p: error(p, a_C, a_M, a_Y, a_K, R_measured),
                    params_init,
                    bounds=bounds
                )
                param_grid[i, j, k, l] = result.x

# Step 3: Prediction function
def predict_cynsn_km(c, m, y, k):
    """Predict reflectance for any CMYK mixture."""
    # Find cell and interpolate parameters
    c_idx = np.digitize(c, grid_points[0]) - 1
    m_idx = np.digitize(m, grid_points[1]) - 1
    y_idx = np.digitize(y, grid_points[2]) - 1
    k_idx = np.digitize(k, grid_points[3]) - 1
    params = param_grid[c_idx, m_idx, y_idx, k_idx]
    
    # Normalize concentrations
    a_C, a_M, a_Y, a_K = c/100, m/100, y/100, k/100
    return yn_predict(params, a_C, a_M, a_Y, a_K, R_C, R_M, R_Y, R_K)

# Example: Predict 30%C + 60%M + 10%Y + 20%K
R_pred = predict_cynsn_km(30, 60, 10, 20)