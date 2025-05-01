import numpy as np
# from scipy.optimize import minimize_scalar
# from skimage import color
import matplotlib.pyplot as plt

import sys
import os

# Füge das Wurzelverzeichnis zum Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.code.space.colorConverterNumpy import ColorTrafoNumpy

data_top = [
    0.14,
    0.2112,
    0.4391,
    0.6822,
    0.8248,
    0.879,
    0.8954,
    0.8981,
    0.9028,
    0.9093,
    0.9155,
    0.9212,
    0.9257,
    0.9283,
    0.9312,
    0.9326,
    0.9353,
    0.9372,
    0.9362,
    0.9399,
    0.9396,
    0.9404,
    0.9409,
    0.9414,
    0.9414,
    0.9415,
    0.9436,
    0.9456,
    0.9469,
    0.9475,
    0.9473,
    0.9469,
    0.9485,
    0.95,
    0.9495,
    0.9511,
]

data_mid = [
    0.0317,
    0.0326,
    0.0311,
    0.0278,
    0.0256,
    0.0251,
    0.025,
    0.0248,
    0.0258,
    0.028,
    0.0297,
    0.0312,
    0.0372,
    0.0548,
    0.0925,
    0.136,
    0.1513,
    0.1389,
    0.1172,
    0.0989,
    0.0868,
    0.0795,
    0.0735,
    0.0693,
    0.0675,
    0.0667,
    0.0669,
    0.069,
    0.0721,
    0.0727,
    0.0704,
    0.0668,
    0.0625,
    0.0601,
    0.0623,
    0.0705
]

data_low = [
    0.0089,
    0.0088,
    0.009,
    0.0089,
    0.0088,
    0.0089,
    0.0091,
    0.0092,
    0.0097,
    0.0101,
    0.0101,
    0.0104,
    0.012,
    0.0152,
    0.0208,
    0.026,
    0.0257,
    0.0217,
    0.0178,
    0.0152,
    0.0137,
    0.0128,
    0.0122,
    0.0119,
    0.0116,
    0.0115,
    0.0114,
    0.0113,
    0.0111,
    0.0108,
    0.0106,
    0.0104,
    0.0106,
    0.0112,
    0.0123,
    0.0139
]


# Kubelka-Munk K/S calculation
def reflectance_to_KS(R):
    R = np.clip(R, 1e-5, 1.0)
    return (1 - R)**2 / (2 * R)

# Reverse: K/S back to Reflectance
def KS_to_reflectance(KS):
    KS = np.clip(KS, 0, None)
    return (1 + KS - np.sqrt(KS**2 + 2 * KS))

# Konvertiere Spektrum zu LAB L-Wert (angenommenes Illuminant D65 und 2 Grad Beobachter)
def spectrum_to_L(spectrum):
    trafo = ColorTrafoNumpy()
    lab = trafo.Cs_SNM2LAB(spectrum)
    L = lab[0]
    return L

# Hauptfunktion mit binärer Suche
def estimate_full_spectrum_target_L(R_top, R_mid, target_L, tol=0.1, max_iter=50):
    KS_top = reflectance_to_KS(R_top)
    KS_mid = reflectance_to_KS(R_mid)

    low = 0.01
    high = 0.99
    iteration = 0

    best_c = None
    best_diff = np.inf
    best_R_full = None

    while iteration < max_iter:
        mid_c = (low + high) / 2
        KS_full = (KS_mid - (1 - mid_c) * KS_top) / mid_c
        R_full = KS_to_reflectance(KS_full)
        L_value = spectrum_to_L(R_full)

        diff = abs(L_value - target_L)

        print(f"Iteration {iteration}: c={mid_c:.6f}, L={L_value:.4f}, diff={diff:.6f}")

        if diff < best_diff:
            best_diff = diff
            best_c = mid_c
            best_R_full = R_full

        if L_value > target_L:
            high = mid_c
        else:
            low = mid_c

        if best_diff < tol:
            print(f"Breaking at iteration {iteration} with diff={best_diff:.6f}")
            break

        iteration += 1


    if best_c is None:
        raise RuntimeError("Binary search failed!")

    return best_R_full, best_c, iteration

# Beispiel: Wellenlängen und Spektren definieren
# wavelengths = np.arange(380, 740, 10)

# Dummy-Daten (ersetzen durch echte Spektren)
R_top = np.asarray(data_top)
R_mid = np.asarray(data_mid)

# Vorgabe eines Ziel-L-Wertes
target_L = spectrum_to_L(np.asarray(data_low))

# Berechnung
R_full_estimated, estimated_c, iteration = estimate_full_spectrum_target_L(R_top, R_mid, target_L, tol=0.0001, max_iter=500)
estimated_L = spectrum_to_L(R_full_estimated)

# Ausgabe
print(f"Geschätzter Konzentrationsanteil der dunklen Farbe (für Ziel-L={target_L}): {estimated_c:.4f}")
print(f"estimated_L: {estimated_L:.4f}")
print("Geschätztes Vollton-Reflexionsspektrum:")
print(R_full_estimated)
print("\t".join([f"{x:.6f}" for x in R_full_estimated]))
print(f"Anzahl der Iterationen: {iteration}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(R_top, label='Top (Hell)', linestyle='--')
plt.plot(R_mid, label='Mid (Mischung)', linestyle='-.')
plt.plot(R_full_estimated, label=f'Geschätzter Vollton (Act-L={estimated_L})', linewidth=2)
plt.plot(data_low, label=f'Low (Ziel-L={target_L})', linestyle=':')
plt.xlabel('Wellenlänge (nm)')
plt.ylabel('Reflexionsgrad')
plt.title('Spektrale Reflexionen')
plt.legend()
plt.grid(True)
plt.show()
