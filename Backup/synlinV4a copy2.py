from typing import Union
import numpy as np  # type: ignore
from src.code.space.colorConverter import ( Cs_Spectral2Multi, ColorTrafo )
from src.code.predict.linearization.baselinearization import BaseLinearization
from src.code.curveReducer import CurveEstimator3D


class SynLinSolidV4a(BaseLinearization):
    """
    Predict a linearization based an spectral data in a range of 380-730nm with 10nm steps.

    Using KEILE, the concentration is for the first ink, and the last ink is always 100%.
    The second color can be optimized by a correction factor.

    .-----.
    'ABBBB'
    'AABBB'
    'AAABB'
    'AAAAB'
    'AAAAA'
    .-----.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.precision = 100
        self.space = "XYZ"

    def set_precision(self, value: int):
        self.precision = value

    def set_space(self, value):
        if value not in ["XYZ", "LAB"]:
            raise ValueError("Space must be 'XYZ', 'LAB'")
        self.space = value

    def start_Curve3D(self):

        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -

        ksMedia = OptcolorNumpy.ksFromSnm(np.array(self.media))
        ksSolid = OptcolorNumpy.ksFromSnm(np.array(self.solid))

        # - + - + - + - + ESTIMATE THE CURVE IN 3D SPACE - + - + - + - + -

        cSolid = np.linspace(0, 1, self.precision)
        cMedia = 1 - cSolid

        trafo: ColorTrafo = ColorTrafo()

        nps_snm = OptcolorNumpy.ksMix(ksMedia, ksSolid, cMedia, cSolid)
        nps = trafo.CS_SNM2XYZ(nps_snm)
        if self.space == "LAB":
            nps = trafo.Cs_XYZ2LAB(nps)

        ce = CurveEstimator3D()
        ce.calculate_curve_length(nps, 10000, 2000)

        LENC = len(self.gradient)
        num_bands = len(self.media)  # Anzahl der Spektralbänder

        estimat_SNM = np.zeros((LENC, num_bands))
        current_POS = [0.0] * LENC
        est_cSolid = [0.0] * LENC
        loop = [0] * LENC

        for j in range(LENC):
            low, high = 0.0, 1.0  # Binary search range for each point

            if self.gradient[j] == low or self.gradient[j] == high:
                est_cSolid[j] = self.gradient[j]
                estimat_SNM[j] = self.media if self.gradient[j] == low else self.solid
                continue

            while loop[j] < 21:

                mid = (low + high) * 0.5  # Midpoint for binary search
                estimat_SNM[j] = OptcolorNumpy.ksMix(ksMedia, ksSolid, 1 - mid, mid)

                tmp = trafo.CS_SNM2XYZ(estimat_SNM[j])
                if self.space == "LAB":
                    tmp = trafo.Cs_XYZ2LAB(tmp)

                current_POS[j] = ce.calculate_percentage(tmp)

                dfactor = current_POS[j] - self.gradient[j]  # Error difference

                if abs(dfactor) < self.tolerance:  # Convergence condition
                    break  # Stop early if within tolerance

                if dfactor > 0:
                    high = mid  # Shift upper bound down
                else:
                    low = mid  # Shift lower bound up

                loop[j] += 1

            est_cSolid[j] = mid  # Assign final optimized value

        #color = Cs_Spectral2Multi(estimat_SNM, self.destination_types)
        color = trafo.Cs_Spectral2Multi(estimat_SNM, self.destination_types)

        response = {
            "color": color
        }

        if self.debug:
            response.update(
                {
                    "ksMedia": ksMedia.tolist(),
                    "ksSolid": ksSolid.tolist(),
                    "loops": loop,
                    "loopsSum": sum(loop),
                    "current_POS": [round(elem, 6) for elem in current_POS],
                    "curve_length": ce.curve_length,
                    "nps": [nps[i].tolist() for i in range(len(nps))],
                    "space": self.space,
                    "ramp": [round(elem, 2) for elem in self.gradient],
                    "cSolid": [round(elem, 4) for elem in est_cSolid],
                    "tolerance": self.tolerance,
                    "precision": self.precision,
                    "version": "MARS.4a.082",
                }
            ) 

        return response

    def start_Curve3D_CHATGPT_notworking(self):
        """
        Optimized function to estimate the color mixing curve in 3D space.
        """

        # Convert to NumPy arrays once (Avoid redundant conversion)
        ksMedia = OptcolorNumpy.ksFromSnm(np.asarray(self.media))
        ksSolid = OptcolorNumpy.ksFromSnm(np.asarray(self.solid))

        # Generate concentration values
        cSolid = np.linspace(0, 1, self.precision)
        cMedia = 1 - cSolid

        # Perform initial ksMix calculation in bulk
        trafo = ColorTrafo()
        nps_snm = OptcolorNumpy.ksMix(ksMedia, ksSolid, cMedia, cSolid)

        # Convert to the target color space in bulk
        nps = trafo.CS_SNM2XYZ(nps_snm)
        if self.space == "LAB":
            nps = trafo.Cs_XYZ2LAB(nps)

        # Initialize curve estimator
        ce = CurveEstimator3D()
        ce.calculate_curve_length(nps, 10000, 2000)

        # Setup result arrays
        LENC = len(self.gradient)
        num_bands = len(self.media)  # Number of spectral bands

        estimat_SNM = np.zeros((LENC, num_bands))  # Optimized allocation
        est_cSolid = np.zeros(LENC)
        loop = np.zeros(LENC, dtype=int)  # Track loop iterations
        current_POS = np.zeros(LENC)  # Store calculated percentages

        # Precompute logical conditions to skip redundant calculations
        is_low = self.gradient == 0.0
        is_high = self.gradient == 1.0

        # Assign precomputed values directly
        estimat_SNM[is_low] = self.media
        estimat_SNM[is_high] = self.solid
        est_cSolid[is_low] = 0.0
        est_cSolid[is_high] = 1.0

        # Sicherstellen, dass `mask` ein 1D-Array ist
        mask = np.logical_not(is_low | is_high)  

        # `indices` immer als 1D-Array sicherstellen
        indices = np.where(mask)[0]

        # Falls `indices` leer ist, explizit als leeres Array setzen
        if indices.size == 0:
            indices = np.array([], dtype=int)

        for j in indices:
            low, high = 0.0, 1.0  # Binary search range
            target = self.gradient[j]

            while loop[j] < 21:
                mid = (low + high) * 0.5  # Midpoint for binary search
                estimat_SNM[j] = OptcolorNumpy.ksMix(ksMedia, ksSolid, 1 - mid, mid)

                # Convert to color space
                tmp = trafo.CS_SNM2XYZ(estimat_SNM[j])
                if self.space == "LAB":
                    tmp = trafo.Cs_XYZ2LAB(tmp)

                current_POS[j] = ce.calculate_percentage(tmp)
                dfactor = current_POS[j] - target  # Error difference

                # Early stopping if within tolerance
                if abs(dfactor) < self.tolerance:
                    break

                # Update binary search range
                if dfactor > 0:
                    high = mid
                else:
                    low = mid

                loop[j] += 1

            est_cSolid[j] = mid  # Assign final optimized value

        # Final spectral color transformation
        # color = Cs_Spectral2Multi(estimat_SNM, self.destination_types)
        color = trafo.Cs_Spectral2Multi(estimat_SNM, self.destination_types)
        
        response = {
            "color": color,
        }

        if self.debug:
            response.update(
                {
                    "ksMedia": ksMedia.tolist(),
                    "ksSolid": ksSolid.tolist(),
                    "loops": loop,
                    "loopsSum": sum(loop),
                    "current_POS": [round(elem, 6) for elem in current_POS],
                    "curve_length": ce.curve_length,
                    "nps": [nps[i].tolist() for i in range(len(nps))],
                    "space": self.space,
                    "ramp": [round(elem, 2) for elem in self.gradient],
                    "cSolid": [round(elem, 4) for elem in est_cSolid],
                    "tolerance": self.tolerance,
                    "precision": self.precision,
                    "version": "MARS.4a.082",
                }
            )

        return response

    def start_Curve3D_DeepSeek(self):
        # Convert gradient to numpy array to ensure consistent array operations
        gradient = np.asarray(self.gradient)
        
        # Precompute ksMedia and ksSolid using NumPy
        ksMedia = OptcolorNumpy.ksFromSnm(np.array(self.media))
        ksSolid = OptcolorNumpy.ksFromSnm(np.array(self.solid))

        # Generate cSolid and cMedia for curve estimation
        cSolid = np.linspace(0, 1, self.precision)
        cMedia = 1 - cSolid

        trafo = ColorTrafo()
        nps_snm = OptcolorNumpy.ksMix(ksMedia, ksSolid, cMedia, cSolid)
        nps = trafo.CS_SNM2XYZ(nps_snm)
        if self.space == "LAB":
            nps = trafo.Cs_XYZ2LAB(nps)

        # Precompute nps as a NumPy array for distance calculations
        nps_array = np.array(nps)
        ce = CurveEstimator3D()
        ce.calculate_curve_length(nps, 10000, 2000)

        LENC = len(gradient)
        num_bands = len(self.media)

        # Initialize arrays for results
        estimat_SNM = np.zeros((LENC, num_bands))
        est_cSolid = np.zeros(LENC)
        loop = np.zeros(LENC, dtype=int)

        # Handle gradient values at boundaries (0.0 or 1.0)
        mask_0 = (gradient == 0.0)
        mask_1 = (gradient == 1.0)
        mask_other = ~(mask_0 | mask_1)

        est_cSolid[mask_0] = 0.0
        est_cSolid[mask_1] = 1.0
        estimat_SNM[mask_0] = self.media
        estimat_SNM[mask_1] = self.solid

        # Ensure active is always at least 1D to prevent scalar issues
        active = np.atleast_1d(mask_other.copy())
        low = np.where(active, 0.0, np.nan)
        high = np.where(active, 1.0, np.nan)
        mid = np.zeros(LENC)

        # Binary search iterations (max 21)
        for _ in range(21):
            if not np.any(active):
                break

            current_indices = np.where(active)[0]
            mid_active = (low[current_indices] + high[current_indices]) * 0.5
            mid[current_indices] = mid_active

            # Vectorized ksMix for active elements
            estimat_SNM_active = OptcolorNumpy.ksMix(ksMedia, ksSolid, 1 - mid_active, mid_active)
            estimat_SNM[current_indices] = estimat_SNM_active

            # Batch transform to XYZ/LAB
            tmp = trafo.CS_SNM2XYZ(estimat_SNM_active)
            if self.space == "LAB":
                tmp = trafo.Cs_XYZ2LAB(tmp)

            # Vectorized distance calculation
            expanded_tmp = tmp[:, np.newaxis, :]
            distances = np.sum((expanded_tmp - nps_array) ** 2, axis=2)
            closest_indices = np.argmin(distances, axis=1)
            current_POS_active = (closest_indices / 2000) * 100

            dfactor = current_POS_active - gradient[current_indices]
            tolerance_mask = np.abs(dfactor) < self.tolerance

            # Update converged elements
            converged_indices = current_indices[tolerance_mask]
            est_cSolid[converged_indices] = mid[converged_indices]
            active[converged_indices] = False

            # Adjust bounds for non-converged elements
            non_converged = current_indices[~tolerance_mask]
            dfactor_non_converged = dfactor[~tolerance_mask]
            high[non_converged] = np.where(dfactor_non_converged > 0, mid[non_converged], high[non_converged])
            low[non_converged] = np.where(dfactor_non_converged <= 0, mid[non_converged], low[non_converged])
            loop[non_converged] += 1

        # Final estimate for remaining elements
        remaining = np.where(active)[0]
        est_cSolid[remaining] = mid[remaining]
        estimat_SNM[remaining] = OptcolorNumpy.ksMix(ksMedia, ksSolid, 1 - mid[remaining], mid[remaining])

        # Compute final colors
        color = Cs_Spectral2Multi(estimat_SNM, self.destination_types)

        response = {"color": color}

        if self.debug:
            # Debug calculations remain unchanged
            tmp_debug = trafo.CS_SNM2XYZ(estimat_SNM)
            if self.space == "LAB":
                tmp_debug = trafo.Cs_XYZ2LAB(tmp_debug)
            distances_debug = np.sum((tmp_debug[:, np.newaxis, :] - nps_array) ** 2, axis=2)
            closest_debug = np.argmin(distances_debug, axis=1)
            current_POS_debug = (closest_debug / 2000) * 100

            response.update({
                "ksMedia": ksMedia.tolist(),
                "ksSolid": ksSolid.tolist(),
                "loops": loop.tolist(),
                "loopsSum": int(np.sum(loop)),
                "current_POS": [round(pos, 6) for pos in current_POS_debug],
                "curve_length": ce.curve_length,
                "nps": [arr.tolist() for arr in nps],
                "space": self.space,
                "ramp": [round(g, 2) for g in gradient],
                "cSolid": [round(c, 4) for c in est_cSolid],
                "tolerance": self.tolerance,
                "precision": self.precision,
                "version": "FIXED.1.0",
            })

        return response


class OptcolorNumpy_OLD:

    @staticmethod
    def ksFromSnm(snm: np.ndarray) -> np.ndarray:
        return (1 - snm) ** 2 / (2 * snm)

    @staticmethod
    def ksToSnm(ks: np.ndarray) -> np.ndarray:
        return 1 + ks - np.sqrt(ks**2 + 2 * ks)

    @staticmethod
    def ksFulltoneInk(ksMedia: np.ndarray, ksSolid: np.ndarray) -> np.ndarray:
        return ksSolid - ksMedia

    @staticmethod
    def ksMix(
        ksMedia: np.ndarray, ksSolid: np.ndarray, cMedia: Union[float, np.ndarray], cSolid: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Optimized version of ksMix using vectorized operations.
        """
        # Wenn cMedia und cSolid float sind, in np.ndarray umwandeln
        if isinstance(cMedia, float):
            cMedia = np.array([cMedia])
        if isinstance(cSolid, float):
            cSolid = np.array([cSolid])

        ksMix = (ksMedia[:, np.newaxis] * cMedia) + (ksSolid[:, np.newaxis] * cSolid)
        spectrals = OptcolorNumpy.ksToSnm(ksMix)
        return spectrals.T


class OptcolorNumpy:
    @staticmethod
    def ksFromSnm(snm: np.ndarray) -> np.ndarray:
        """Computes ks from snm using vectorized NumPy operations."""
        return np.square(1 - snm) / (2 * snm)

    @staticmethod
    def ksToSnm(ks: np.ndarray) -> np.ndarray:
        """Computes snm from ks using vectorized NumPy operations."""
        return 1 + ks - np.sqrt(ks**2 + 2 * ks)

    @staticmethod
    def ksFulltoneInk(ksMedia: np.ndarray, ksSolid: np.ndarray) -> np.ndarray:
        """Computes full-tone ink ks."""
        return ksSolid - ksMedia

    @staticmethod
    def ksMix(
        ksMedia: np.ndarray, ksSolid: np.ndarray, 
        cMedia: Union[float, np.ndarray], cSolid: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Optimized ksMix using broadcasting.
        - ksMedia and ksSolid have shape (36,)
        - cMedia and cSolid are scalars or arrays of shape (M,)
        - Broadcasting ensures correct multiplication without loops.
        - Result: Spectral data of shape (M, 36)
        """
        # Ensure scalar factors are converted to 1D NumPy arrays
        cMedia = np.array([cMedia]) if isinstance(cMedia, float) else np.asarray(cMedia)
        cSolid = np.array([cSolid]) if isinstance(cSolid, float) else np.asarray(cSolid)

        # Reshape ksMedia and ksSolid to (36, 1) for broadcasting
        ksMix = (ksMedia[:, np.newaxis] * cMedia) + (ksSolid[:, np.newaxis] * cSolid)

        # Convert to snm and return with correct transposition
        return OptcolorNumpy.ksToSnm(ksMix).T  # Shape: (M, 36)