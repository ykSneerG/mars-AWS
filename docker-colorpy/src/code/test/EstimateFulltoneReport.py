import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# Füge das Wurzelverzeichnis zum Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.code.space.colorConverterNumpy import ColorTrafoNumpy

data_top = [
    0.14,0.2112,0.4391,0.6822,0.8248,0.879,0.8954,0.8981,0.9028,0.9093,0.9155,0.9212,0.9257,0.9283,0.9312,0.9326,0.9353,0.9372,0.9362,0.9399,0.9396,0.9404,0.9409,0.9414,0.9414,0.9415,0.9436,0.9456,0.9469,0.9475,0.9473,0.9469,0.9485,0.95,0.9495,0.9511
]

data_mid = [
    0.1196,0.1657,0.2696,0.335,0.3626,0.3872,0.4053,0.4017,0.3888,0.3637,0.3136,0.2465,0.1853,0.1403,0.1095,0.0889,0.0756,0.0668,0.0609,0.0572,0.0546,0.053,0.052,0.0517,0.052,0.053,0.0549,0.0579,0.0626,0.0695,0.0797,0.0945,0.1153,0.1422,0.1741,0.2085
]

data_low = [
    0.0534,0.0557,0.0599,0.0644,0.0709,0.0801,0.0899,0.0924,0.0888,0.0783,0.061,0.0451,0.0364,0.0323,0.0302,0.0289,0.0282,0.0277,0.0273,0.0272,0.0271,0.027,0.027,0.0271,0.0272,0.0273,0.0276,0.0279,0.0284,0.0291,0.03,0.0316,0.0342,0.0382,0.0439,0.0526
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

def calculate_rmse(reconstructed, original):
    return np.sqrt(np.mean((reconstructed - original) ** 2))

def mix_spectra(c, R_top, R_full):
    KS_top = reflectance_to_KS(R_top)
    KS_full = reflectance_to_KS(R_full)
    KS_mix = c * KS_full + (1 - c) * KS_top
    R_mix = KS_to_reflectance(KS_mix)
    return R_mix

def find_best_c_to_match_mid(R_top, R_full, R_mid, tol=1e-6, max_iter=100):
    KS_top = reflectance_to_KS(R_top)
    KS_full = reflectance_to_KS(R_full)

    low = 0.0
    high = 1.0

    best_c = None
    best_rmse = float('inf')
    best_R_mix = None

    for i in range(max_iter):
        c = (low + high) / 2
        KS_mix = c * KS_full + (1 - c) * KS_top
        R_mix = KS_to_reflectance(KS_mix)
        rmse = np.sqrt(np.mean((R_mix - R_mid) ** 2))

        print(f"Iter {i}: c={c:.6f}, RMSE={rmse:.6f}")

        # Update best
        if rmse < best_rmse:
            best_rmse = rmse
            best_c = c
            best_R_mix = R_mix

        # Try checking RMSE for a slightly lower and higher c
        delta = 1e-4
        c1 = max(0, c - delta)
        c2 = min(1, c + delta)

        rmse1 = np.sqrt(np.mean((KS_to_reflectance(c1 * KS_full + (1 - c1) * KS_top) - R_mid)**2))
        rmse2 = np.sqrt(np.mean((KS_to_reflectance(c2 * KS_full + (1 - c2) * KS_top) - R_mid)**2))

        # Move in direction of lower RMSE
        if rmse1 < rmse2:
            high = c  # better towards left
        else:
            low = c   # better towards right

        # Stop early if RMSE change is small
        if abs(rmse2 - rmse1) < tol:
            print(f"Breaking at iteration {i} with ΔRMSE={abs(rmse2 - rmse1):.10f}")
            break

    return best_c, best_R_mix, best_rmse


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

# Estimate the Mid Color from the new corrected R_full_estimated and R_top
best_c, R_mix_reconstructed, rmse = find_best_c_to_match_mid(R_top, R_full_estimated, R_mid)
print(f"Best concentration to reconstruct Mid: {best_c:.10f}, RMSE: {rmse:.10f}")

remixed_R_full = mix_spectra(estimated_c, R_top, R_full_estimated)

# Ausgabe
print(f"Geschätzter Konzentrationsanteil der dunklen Farbe (für Ziel-L={target_L}): {estimated_c:.6f}")
print(f"estimated_L: {estimated_L:.4f}")
print("Geschätztes Vollton-Reflexionsspektrum:")
print(R_full_estimated)
print("\t".join([f"{x:.6f}" for x in R_full_estimated]))
print(f"Anzahl der Iterationen: {iteration}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(R_top, label='Top (Hell)', linestyle='--')
plt.plot(R_mid, label='Mid (Mischung)', linestyle='-.', linewidth=2)
plt.plot(R_full_estimated, label=f'Geschätzter Vollton (Act-L={estimated_L})', linewidth=2)
plt.plot(data_low, label=f'Low (Ziel-L={target_L})', linestyle=':')
plt.plot(remixed_R_full, label='Remixed Full', linestyle=':')
plt.xlabel('Wellenlänge (nm)')
plt.ylabel('Reflexionsgrad')
plt.title('Spektrale Reflexionen')
plt.legend()
plt.grid(True)
plt.show()


class MasstoneEstimator:
    def __init__(self, R_top, R_mid, R_low, tol=0.0001, max_iter=100):
        self.R_top = np.asarray(R_top)
        self.R_mid = np.asarray(R_mid)
        self.R_low = np.asarray(R_low)
        self.target_L = self.spectrum_to_L(self.R_low)
        self.tolerance = tol
        self.max_iter = max_iter
        self.minimum = 1e-5
        
        self.iteration = 0
        self.log = {
            'steps': []
        }
        

    def estimate(self):
        R_full_estimated, estimated_c, iteration =  self.estimate_full_spectrum_target_L(self.R_top, self.R_mid, self.target_L)
        
        self.R_est = R_full_estimated
        self.C_est = estimated_c
        self.iteration = iteration
        
    
    def estimate_full_spectrum_target_L(self, R_top: np.ndarray, R_mid: np.ndarray, target_L: float) -> tuple:
        
        KS_top = self.reflectance_to_KS(R_top)
        KS_mid = self.reflectance_to_KS(R_mid)

        low = self.minimum
        high = 1-self.minimum
        iteration = 0

        best_c = None
        best_diff = np.inf
        best_R_full = None

        while iteration < self.max_iter:
            mid_c = (low + high) / 2
            KS_full = (KS_mid - (1 - mid_c) * KS_top) / mid_c
            R_full = self.KS_to_reflectance(KS_full)
            L_value = self.spectrum_to_L(R_full)

            diff = abs(L_value - target_L)

            # Log the iteration
            self.log['steps'].append({
                'iteration': iteration,
                'c': mid_c,
                'L': L_value,
                'diff': diff
            })

            if diff < best_diff:
                best_diff = diff
                best_c = mid_c
                best_R_full = R_full

            if L_value > target_L:
                high = mid_c
            else:
                low = mid_c

            if best_diff < self.tolerance:
                # print(f"Breaking at iteration {iteration} with diff={best_diff:.6f}")
                self.log['steps'].append({
                    'iteration': iteration,
                    'c': mid_c,
                    'L': L_value,
                    'diff': diff
                })
                break

            iteration += 1


        if best_c is None:
            raise RuntimeError("Binary search failed!")

        return best_R_full, best_c, iteration


    @staticmethod
    def reflectance_to_KS(R: np.ndarray) -> np.ndarray:
        """
        Kubelka-Munk K/S calculation
        Args:
            R (np.ndarray): Reflectance values.
        Returns:
            np.ndarray: K/S values.
        """
        R = np.clip(R, 1e-5, 1.0)
        return (1 - R)**2 / (2 * R)

    @staticmethod
    def KS_to_reflectance(KS: np.ndarray) -> np.ndarray:
        """
        Reverse: K/S back to Reflectance
        Args:
            KS (np.ndarray): K/S values.
        Returns:
            np.ndarray: Reflectance values.
        """
        KS = np.clip(KS, 0, None)
        return (1 + KS - np.sqrt(KS**2 + 2 * KS))

    @staticmethod
    def spectrum_to_L(R: np.ndarray) -> float:
        """
        Convert reflectance spectrum to L value.
        Args:
            R (np.ndarray): Reflectance values.
        Returns:
            float: L value.
        """
        trafo = ColorTrafoNumpy()
        lab = trafo.Cs_SNM2LAB(R)
        L = lab[0]
        return L
    
    @staticmethod
    def mix_KS_to_reflectance(concentration: float, ks_full: np.ndarray, ks_top: np.ndarray) -> np.ndarray:
        KS_mix = concentration * ks_full + (1 - concentration) * ks_top
        R_mix = MasstoneEstimator.KS_to_reflectance(KS_mix)
        return R_mix
        
        