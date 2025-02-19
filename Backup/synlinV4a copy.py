from typing import Union
import numpy as np  # type: ignore
from src.code.space.colorSpace import CsXYZ
from src.code.space.colorConverter import (
    Cs_Spectral2Multi,
    CS_Spectral2XYZ_Numpy,
    Cs_XYZ2LAB,
    Cs_XYZ2LAB_Numpy,
    ColorTrafo
)
from src.code.predict.linearization.baselinearization import BaseLinearization
from src.code.curveReducer import CurveEstimator3D

from src.code.space.colorConstants.illuminant import Illuminant


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

    def start_Curve3D_OLD(self):

        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -

        ksMedia = OptcolorNumpy.ksFromSnm(np.array(self.media))
        ksSolid = OptcolorNumpy.ksFromSnm(np.array(self.solid))

        # - + - + - + - + ESTIMATE THE CURVE IN 3D SPACE - + - + - + - + -

        cSolid = np.linspace(0, 1, self.precision)
        cMedia = 1 - cSolid

        space_func = (
            OptcolorNumpy.ksMixToLAB
            if self.space == "LAB"
            else OptcolorNumpy.ksMixToXYZ
        )
        nps = space_func(ksMedia, ksSolid, cMedia, cSolid)

        ce = CurveEstimator3D()
        ce.calculate_curve_length(nps, 10000, 2000)

        LENC = len(self.gradient)

        estimat_SNM = [0.0] * LENC
        current_POS = [0.0] * LENC
        est_cSolid = [0.0] * LENC
        loop = [0] * LENC

        refWhite: np.ndarray = Illuminant.D50_DEG2.to_numpy()

        for j in range(LENC):
            low, high = 0.0, 1.0  # Binary search range for each point

            if self.gradient[j] == low or self.gradient[j] == high:
                est_cSolid[j] = self.gradient[j]
                estimat_SNM[j] = self.media if self.gradient[j] == low else self.solid
                continue

            while loop[j] < 21:  # A reasonable max for binary search

                mid = (low + high) * 0.5  # Midpoint for binary search
                estimat_SNM[j] = OptcolorNumpy.ksMixToSNM(
                    ksMedia, ksSolid, 1 - mid, mid
                )

                tmp = CS_Spectral2XYZ_Numpy(estimat_SNM[j])
                if self.space == "LAB":
                    # tmp = Cs_XYZ2LAB(CsXYZ(tmp[0], tmp[1], tmp[2])).to_numpy()
                    tmp = Cs_XYZ2LAB_Numpy(tmp, refWhite)

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

        color = Cs_Spectral2Multi(estimat_SNM, self.destination_types)

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

        color = Cs_Spectral2Multi(estimat_SNM, self.destination_types)

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


class OptcolorNumpy:

    @staticmethod
    def ksFromSnm(snm: np.ndarray) -> np.ndarray:
        return (1 - snm) ** 2 / (2 * snm)

    @staticmethod
    def ksToSnm(ks: np.ndarray) -> np.ndarray:
        return 1 + ks - np.sqrt(ks**2 + 2 * ks)

    @staticmethod
    def ksFulltoneInk(ksMedia: np.ndarray, ksSolid: np.ndarray) -> np.ndarray:
        return ksSolid - ksMedia

    """ @staticmethod
    def ksMix(
        ksMedia: np.ndarray, ksSolid: np.ndarray, cMedia: float, cSolid: float
    ) -> np.ndarray:
        return (ksMedia * cMedia) + (ksSolid * cSolid) """

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

    @staticmethod
    def ksMixToXYZ(
        ksMedia: np.ndarray, ksSolid: np.ndarray, cMedia: np.ndarray, cSolid: np.ndarray
    ) -> np.ndarray:
        """
        Optimized version of ksMixToXYZ using vectorized operations.
        """
        
        spectrals = OptcolorNumpy.ksMix(ksMedia, ksSolid, cMedia, cSolid)
        return np.array([CS_Spectral2XYZ_Numpy(spectral) for spectral in spectrals])
        
        """ 
        ksMix = OptcolorNumpy.ksMix(
            ksMedia[:, np.newaxis], ksSolid[:, np.newaxis], cMedia, cSolid
        )
        spectrals = OptcolorNumpy.ksToSnm(ksMix)
        return np.array([CS_Spectral2XYZ_Numpy(spectral) for spectral in spectrals.T]) """

    @staticmethod
    def ksMixToLAB(
        ksMedia: np.ndarray, ksSolid: np.ndarray, cMedia: np.ndarray, cSolid: np.ndarray
    ) -> np.ndarray:
        """
        Optimized version of ksMixToXYZ using vectorized operations.
        """
        
        spectrals = OptcolorNumpy.ksMix(ksMedia, ksSolid, cMedia, cSolid)
        xyz = np.array([CS_Spectral2XYZ_Numpy(spectral) for spectral in spectrals])
        return np.array([Cs_XYZ2LAB(CsXYZ(x[0], x[1], x[2])).to_numpy() for x in xyz])
        
        
        """ ksMix = OptcolorNumpy.ksMix(
            ksMedia[:, np.newaxis], ksSolid[:, np.newaxis], cMedia, cSolid
        )
        spectrals = OptcolorNumpy.ksToSnm(ksMix)
        xyz = np.array([CS_Spectral2XYZ_Numpy(spectral) for spectral in spectrals.T])
        lab = np.array([Cs_XYZ2LAB(CsXYZ(x[0], x[1], x[2])).to_numpy() for x in xyz])
        return lab """
