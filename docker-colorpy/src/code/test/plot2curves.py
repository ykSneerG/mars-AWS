# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize

# # Reflectance data (36 channels, 400-700nm)
# wavelengths = np.linspace(400, 700, 36)

# # Your measured reflectance (Color A)
# R_measured = np.array([
#     0.0158, 0.0175, 0.0182, 0.0209, 0.0242, 0.0279, 0.0342, 0.0450, 
#     0.0638, 0.0969, 0.1570, 0.2582, 0.3798, 0.4463, 0.4177, 0.3390, 
#     0.2542, 0.1739, 0.1072, 0.0649, 0.0428, 0.0322, 0.0256, 0.0217, 
#     0.0207, 0.0211, 0.0225, 0.0262, 0.0314, 0.0336, 0.0320, 0.0285, 
#     0.0243, 0.0227, 0.0275, 0.0435
# ])

# # Your predicted reflectance (Color B)
# R_predicted = np.array([
#     0.0257, 0.0307, 0.0332, 0.0370, 0.0402, 0.0446, 0.0515, 0.0656, 
#     0.0921, 0.1372, 0.2192, 0.3531, 0.5160, 0.6127, 0.5917, 0.5148, 
#     0.4235, 0.3224, 0.2223, 0.1494, 0.1076, 0.0875, 0.0746, 0.0666, 
#     0.0649, 0.0662, 0.0701, 0.0789, 0.0900, 0.0952, 0.0913, 0.0840, 
#     0.0747, 0.0710, 0.0806, 0.1131
# ])

# def apply_yn_correction(R, n):
#     """Apply Yule-Nielsen correction to reflectance spectrum"""
#     # Handle potential zeros to avoid numerical issues
#     R = np.clip(R, 0.001, 1.0)  # Clip to avoid division by zero
#     return R ** (1/n)

# def reconstruct_yn(R_yn, n):
#     """Reconstruct reflectance from YN-corrected values"""
#     return R_yn ** n

# def error(n):
#     """Error function to optimize n"""
#     R_corrected = reconstruct_yn(apply_yn_correction(R_predicted, n), n)
#     return np.sum((R_corrected - R_measured)**2)

# # Optimize n between 1.0-3.0 (typical paper range)
# result = minimize(error, x0=1.5, bounds=[(1.0, 3.0)])
# optimal_n = result.x[0]

# # Apply correction
# #R_yn_corrected = reconstruct_yn(apply_yn_correction(R_predicted, optimal_n), optimal_n)
# R_yn_corrected = apply_yn_correction(R_predicted, 0.7)

# # Plotting
# plt.figure(figsize=(12, 6))
# plt.plot(wavelengths, R_measured, 'b-', label='Measured', linewidth=4)
# plt.plot(wavelengths, R_predicted, 'r--', label='Original Prediction', linewidth=4)
# plt.plot(wavelengths, R_yn_corrected, 'g-', 
#          label=f'YN Corrected (n={optimal_n:.2f})', linewidth=2)

# plt.title('Reflectance Spectra with Yule-Nielsen Correction', fontsize=14)
# plt.xlabel('Wavelength (nm)', fontsize=12)
# plt.ylabel('Reflectance', fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=10, loc='upper left')
# plt.xlim(400, 700)
# plt.ylim(0, 0.7)
# plt.show()

# print(f"Optimal n: {optimal_n:.3f}")
# print(f"Mean Absolute Error:")
# print(f"Original: {np.mean(np.abs(R_predicted - R_measured)):.4f}")
# print(f"Corrected: {np.mean(np.abs(R_yn_corrected - R_measured)):.4f}")


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Your data
wavelengths = np.linspace(400, 700, 36)
# R_measured = np.array([
#     0.0158, 0.0175, 0.0182, 0.0209, 0.0242, 0.0279, 0.0342, 0.0450, 
#     0.0638, 0.0969, 0.1570, 0.2582, 0.3798, 0.4463, 0.4177, 0.3390, 
#     0.2542, 0.1739, 0.1072, 0.0649, 0.0428, 0.0322, 0.0256, 0.0217, 
#     0.0207, 0.0211, 0.0225, 0.0262, 0.0314, 0.0336, 0.0320, 0.0285, 
#     0.0243, 0.0227, 0.0275, 0.0435
# ])

# R_measured = np.array([
#     0.06974,0.143,0.2672,0.4533,0.6999,0.8358,0.8897,0.9028,0.9023,0.8957,0.877,0.8575,0.8315,0.7938,0.7424,0.6802,0.6138,0.5385,0.4606,0.3995,0.3565,0.3313,0.3122,0.2995,0.2965,0.2988,0.3062,0.319,0.3337,0.3394,0.3337,0.3233,0.3095,0.3024,0.3137,0.3505
# ])

R_measured = np.array([
    0.05575,0.1224,0.2357,0.4073,0.6258,0.7586,0.8275,0.8562,0.8622,0.855,0.8351,0.8108,0.7776,0.7288,0.6614,0.5822,0.4992,0.4099,0.3212,0.2535,0.209,0.1836,0.1649,0.1531,0.1503,0.1528,0.1597,0.1721,0.1868,0.193,0.1882,0.1783,0.1652,0.159,0.1707,0.2081
])



# Your predicted reflectance (Color B)
# R_predicted = np.array([
#     0.0257, 0.0307, 0.0332, 0.0370, 0.0402, 0.0446, 0.0515, 0.0656, 
#     0.0921, 0.1372, 0.2192, 0.3531, 0.5160, 0.6127, 0.5917, 0.5148, 
#     0.4235, 0.3224, 0.2223, 0.1494, 0.1076, 0.0875, 0.0746, 0.0666, 
#     0.0649, 0.0662, 0.0701, 0.0789, 0.0900, 0.0952, 0.0913, 0.0840, 
#     0.0747, 0.0710, 0.0806, 0.1131
# ])
# R_predicted = np.array([
#     0.0882,0.1733,0.3096,0.5155,0.7702,0.8663,0.9042,0.9185,0.9188,0.9125,0.8971,0.8823,0.864,0.8382,0.8026,0.7572,0.7022,0.6284,0.5336,0.44,0.3687,0.327,0.2968,0.2762,0.2716,0.2752,0.2857,0.3077,0.3334,0.3446,0.3361,0.3198,0.2975,0.2882,0.312,0.3804
# ])

R_predicted = np.array([
    0.0992,0.1871,0.3266,0.5408,0.8237,0.9337,0.9533,0.9497,0.9437,0.9351,0.9198,0.9086,0.8969,0.8823,0.8646,0.8439,0.8196,0.7828,0.7288,0.6684,0.6145,0.5795,0.552,0.5324,0.528,0.532,0.5429,0.5645,0.5882,0.5981,0.5908,0.5761,0.5554,0.5466,0.5694,0.6288
])

def correct_prediction(R_pred, n):
    """Apply both correction factor and YN adjustment"""
    return R_pred ** n

def error(params):
    n = params
    R_corr = correct_prediction(R_predicted, n)
    return np.mean((R_corr - R_measured) ** 2)  # MSE

# Optimize both c and n
initial_guess = [1.0]  # Start with no correction
bounds = [(0.1, 20.0)]  # Allow c and n to vary
result = minimize(error, initial_guess, bounds=bounds)
n_opt = result.x

# Apply optimal correction
R_corrected = correct_prediction(R_predicted, 3.0 * 0.75)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(wavelengths, R_measured, 'b-', label='Measured', lw=3)
plt.plot(wavelengths, R_predicted, 'r--', label='Original Prediction', lw=2)
plt.plot(wavelengths, R_corrected, 'g-',  label=f'Corrected (n={n_opt[0]:.4f})', lw=2)
plt.fill_between(wavelengths, R_measured, R_corrected, color='gray', alpha=0.2)

plt.title('Proper Reflectance Correction', fontsize=14)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Optimal parameters: n={n_opt:.4f}")
print(f"MAE improvement: {np.mean(np.abs(R_predicted-R_measured)):.4f} → {np.mean(np.abs(R_corrected-R_measured)):.4f}")