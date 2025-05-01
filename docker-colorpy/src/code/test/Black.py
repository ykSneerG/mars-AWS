import numpy as np
from scipy.interpolate import CubicSpline

# Function to calculate LAB value from reflectance (simplified)
def reflectance_to_Lab(R, wavelengths):
    # Simple calculation of L* (can be replaced by a more accurate color science calculation)
    L = 100 * (1 - np.mean(R))  # Placeholder L* calculation based on average reflectance
    return L

# Main function for extrapolation with intermediate spectra
def extrapolate_spectrum_with_intermediates(spectra, target_L, wavelengths, num_steps=300, max_concentration=2.0):
    # Assuming spectra is a list of reflectance arrays: [R_light, R_int_1, R_int_2, ..., R_mid]
    num_spectra = len(spectra)
    concentrations = np.linspace(0.0, 1.0, num_spectra)  # Concentrations from 0 to 1 for Light to Mid

    # Prepare cubic splines for each wavelength
    splines = []
    for wl in range(len(wavelengths)):
        spline = CubicSpline(concentrations, [spectra[i][wl] for i in range(num_spectra)], extrapolate=True)
        splines.append(spline)

    # Generate candidate spectra with different concentrations (up to max_concentration)
    conc_steps = np.linspace(0, max_concentration, num_steps)
    R_candidates = []
    for c in conc_steps:
        R_candidate = np.array([spl(c) for spl in splines])
        R_candidates.append(R_candidate)

    # Find the candidate spectrum whose L is closest to the target L
    best_diff = float('inf')
    best_R = None
    best_L = None

    for R_cand in R_candidates:
        L = reflectance_to_Lab(R_cand, wavelengths)
        diff = abs(L - target_L)

        if diff < best_diff:
            best_diff = diff
            best_R = R_cand
            best_L = L

    return best_R, best_L

# --------------------------
# Example usage

# Example wavelengths (36 bands from 380 to 730nm)
wavelengths = np.arange(380, 740, 10)  # 380, 390, ..., 730

# You would replace these with your actual reflectance spectra arrays (36 bands)
R_light = np.linspace(0.9, 0.7, len(wavelengths))  # Fake reflectance for Light
R_mid = np.linspace(0.5, 0.3, len(wavelengths))    # Fake reflectance for Mid

# Example intermediate spectra (replace with actual data)
R_int_1 = np.linspace(0.8, 0.6, len(wavelengths))  # Fake intermediate reflectance
R_int_2 = np.linspace(0.7, 0.5, len(wavelengths))  # Fake intermediate reflectance
# Add more intermediate spectra as needed (e.g., R_int_3, R_int_4, ...)

# Combine into a list (Light, intermediates, Mid)
spectra = [R_light, R_int_1, R_int_2, R_mid]

# Target L value for dark color
target_L = 8.0

# Run extrapolation
R_dark_corrected, achieved_L = extrapolate_spectrum_with_intermediates(spectra, target_L, wavelengths)

# Output
print("Corrected Dark Spectrum Reflectance (36 bands):")
print(R_dark_corrected)
print(f"Achieved L*: {achieved_L:.2f}")
