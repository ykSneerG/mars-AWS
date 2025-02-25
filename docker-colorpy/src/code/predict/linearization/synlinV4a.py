from typing import Union
import numpy as np  # type: ignore
from src.code.space.colorConverter import ( ColorTrafo )
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

        color = trafo.Cs_Spectral2Multi(estimat_SNM, self.destination_types)
        #color = trafo.Cs_Spectral2Multi_Parallel(estimat_SNM, self.destination_types)
        
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


    def start_Curve3D_Parallel(self):
        import concurrent.futures
        from functools import partial

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
        num_bands = len(self.media)

        # Initialize arrays
        estimat_SNM = np.zeros((LENC, num_bands))
        current_POS = np.zeros(LENC)
        est_cSolid = np.zeros(LENC)
        loop = np.zeros(LENC, dtype=int)

        gradient_array = np.array(self.gradient)
        low_mask = gradient_array == 0.0
        high_mask = gradient_array == 1.0
        
        # Handle edge cases
        est_cSolid[low_mask] = 0.0
        est_cSolid[high_mask] = 1.0
        estimat_SNM[low_mask] = self.media
        estimat_SNM[high_mask] = self.solid

        process_mask = ~(low_mask | high_mask)
        points_to_process = np.where(process_mask)[0]

        def process_point(j, ksMedia, ksSolid, gradient_array, trafo, ce):
            low, high = 0.0, 1.0
            current_pos = 0.0
            loop_count = 0
            estimat = np.zeros(num_bands)
            
            for _ in range(21):
                mid = (low + high) * 0.5
                estimat = OptcolorNumpy.ksMix(ksMedia, ksSolid, 1 - mid, mid)
                tmp = trafo.CS_SNM2XYZ(estimat)
                if self.space == "LAB":
                    tmp = trafo.Cs_XYZ2LAB(tmp)
                
                # Convert numpy array to list of floats
                tmp_list = tmp.ravel().tolist()
                current_pos = ce.calculate_percentage(tmp_list)
                dfactor = current_pos - gradient_array[j]
                
                if abs(dfactor) < self.tolerance:
                    break
                    
                if dfactor > 0:
                    high = mid
                else:
                    low = mid
                    
                loop_count += 1
                
            return j, mid, estimat, current_pos, loop_count

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            process_func = partial(process_point, 
                                 ksMedia=ksMedia,
                                 ksSolid=ksSolid, 
                                 gradient_array=gradient_array,
                                 trafo=trafo,
                                 ce=ce)
            
            futures = [executor.submit(process_func, j) for j in points_to_process]
            
            for future in concurrent.futures.as_completed(futures):
                j, mid, estimat, pos, loop_count = future.result()
                estimat_SNM[j] = estimat
                current_POS[j] = pos
                est_cSolid[j] = mid
                loop[j] = loop_count

        # Use parallel version of spectral conversion
        color = trafo.Cs_Spectral2Multi(estimat_SNM, self.destination_types)
        
        response = {
            "color": color
        }

        if self.debug:
            response.update({
                "ksMedia": ksMedia.tolist(),
                "ksSolid": ksSolid.tolist(), 
                "loops": loop.tolist(),
                "loopsSum": np.sum(loop),
                "current_POS": np.round(current_POS, 6).tolist(),
                "curve_length": ce.curve_length,
                "nps": [nps[i].tolist() for i in range(len(nps))],
                "space": self.space,
                "ramp": np.round(gradient_array, 2).tolist(),
                "cSolid": np.round(est_cSolid, 4).tolist(),
                "tolerance": self.tolerance,
                "precision": self.precision,
                "version": "MARS.4a.082",
            })

        return response






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
