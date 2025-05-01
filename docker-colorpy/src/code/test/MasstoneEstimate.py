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


class ConcentrationEstimator:
    def __init__(self, R_top, R_mid, R_low, tol=0.0001, max_iter=100):
        self.R_top = np.asarray(R_top)
        self.R_mid = np.asarray(R_mid)
        self.R_low = np.asarray(R_low)
        
        self.tolerance = tol
        self.max_iter = max_iter
       
     
    def find_best_c_to_match_mid(self, R_top, R_full, R_mid, tol=1e-6, max_iter=100):
        KS_top = MasstoneEstimator.reflectance_to_KS(R_top)
        KS_full = MasstoneEstimator.reflectance_to_KS(R_full)

        low = 0.0
        high = 1.0

        best_c = None
        best_rmse = float('inf')
        best_R_mix = None

        for i in range(max_iter):
            c = (low + high) / 2
            KS_mix = c * KS_full + (1 - c) * KS_top
            R_mix = MasstoneEstimator.KS_to_reflectance(KS_mix)
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

            rmse1 = np.sqrt(np.mean((MasstoneEstimator.KS_to_reflectance(c1 * KS_full + (1 - c1) * KS_top) - R_mid)**2))
            rmse2 = np.sqrt(np.mean((MasstoneEstimator.KS_to_reflectance(c2 * KS_full + (1 - c2) * KS_top) - R_mid)**2))

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
        

    def estimate(self, target_L=0):

        R_full_estimated, estimated_c, iteration = self.estimate_full_spectrum_target_L(self.R_top, self.R_mid, self.target_L + target_L)
        
        self.R_est = R_full_estimated
        self.C_est = estimated_c
        self.iteration = iteration
        
    
    def estimate_full_spectrum_target_L(self, R_top: np.ndarray, R_mid: np.ndarray, target_L: float) -> tuple:
        
        KS_top = self.reflectance_to_KS(R_top)
        KS_mid = self.reflectance_to_KS(R_mid)

        # low = self.minimum
        # high = 1-self.minimum
        low = 0
        high = 1
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

    def evaluate(self):
        return MasstoneEstimator.mix_spectra(self.C_est, self.R_top, self.R_est)


    @staticmethod
    def mix_spectra(c_full: float, R_top: np.ndarray, R_full: np.ndarray) -> np.ndarray:
        KS_top = MasstoneEstimator.reflectance_to_KS(R_top)
        KS_full = MasstoneEstimator.reflectance_to_KS(R_full)
        KS_mix = c_full * KS_full + (1 - c_full) * KS_top
        R_mix = MasstoneEstimator.KS_to_reflectance(KS_mix)
        return R_mix

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
             
        
if __name__ == "__main__":
    
    mt = MasstoneEstimator(data_top, data_mid, data_low, tol=0.0001, max_iter=100)
    mt.estimate(0)
    
    R_top = mt.R_top
    R_mid = mt.R_mid
    R_low = mt.R_low
    target_L = mt.target_L
    r_est = mt.R_est
    iteration = mt.iteration
    estimated_L = mt.spectrum_to_L(mt.R_est)
    estimated_c = mt.C_est


    # Estimate the Mid Color from the new corrected r_est and R_top
    """     
    best_c, R_mix_reconstructed, rmse = find_best_c_to_match_mid(R_top, r_est, R_mid)
    print(f"Best concentration to reconstruct Mid: {best_c:.10f}, RMSE: {rmse:.10f}")
    """
    remixed_mid = mt.evaluate()

    # Ausgabe
    print(f"Geschätzter Konzentrationsanteil der dunklen Farbe (für Ziel-L={target_L}): {estimated_c:.6f}")
    print(f"estimated_L: {estimated_L:.4f}")
    print("Geschätztes Vollton-Reflexionsspektrum:")
    print("\t".join([f"{x:.6f}" for x in r_est]))
    print(f"Anzahl der Iterationen: {iteration}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(R_top, label='Top', linestyle='-', linewidth=2, color='blue')
    
    plt.plot(data_low, label=f'Low (L={target_L:.4f})', linestyle='-', linewidth=2, color='lime')
    plt.plot(r_est, label=f'Low (L={estimated_L:.4f}) (Remixed)', linestyle='-.', linewidth=1, color='black')
    
    plt.plot(R_mid, label='Mid', linestyle='-', linewidth=3, color='orange')
    plt.plot(remixed_mid, label='Mid (Remixed)', linestyle='-.', linewidth=1.5, color='black')
    
    plt.xlabel('Wellenlänge (nm)')
    plt.ylabel('Reflexionsgrad')
    plt.title('Spektrale Reflexionen')
    plt.legend()
    plt.grid(True)
    plt.show()
