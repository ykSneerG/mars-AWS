import json
import math
from src.code.space.colorSpace import CsXYZ
from src.code.space.colorConverter import (
    Cs_Spectral2Multi,
    CS_Spectral2XYZ,
    CS_Spectral2XYZ_Numpy,
    Cs_XYZ2LAB,
)
from src.code.predict.linearization.baselinearization import BaseLinearization
import numpy as np  # type: ignore
from src.code.curveReducer import CurveReducer, CurveEstimator3D, CurveEstimator


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
        self.precision = 2

    def set_precision(self, value: int):
        self.precision = value

    def start(self):

        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -

        rRumMedia = sum(self.media)
        rRumSolid = sum(self.solid)

        invert = False
        if rRumMedia < rRumSolid:
            invert = True

        rSumValue = rRumSolid / rRumMedia if invert else rRumMedia / rRumSolid
        rSumValue = math.pow(rSumValue, 1 / 2.25)

        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -

        ksMedia = Optcolor.ksFromSnm(self.solid if invert else self.media)
        ksSolid = Optcolor.ksFromSnm(self.media if invert else self.solid)

        # cFactor = 4
        # estGiga = self.calculate_gradient_by_steps(self.precision)
        estGiga = self.calculate_gradient_by_steps(self.precision)
        cSolid = CurveReducer.pow_curve(estGiga, rSumValue)
        # cSolid = estGiga.copy()
        # cSolid = [0,1]

        LENC: int = len(cSolid)

        spectrals = [
            Optcolor.ksMixWithConcentrationsToSpectral(
                ksMedia, ksSolid, (1 - cSolid[i]), cSolid[i]
            )
            for i in range(LENC)
        ]

        # - + - + - + - + ESTIMATE THE CURVE IN 3D SPACE - + - + - + - + -

        nps = [CS_Spectral2XYZ(spectrals[i]).to_numpy() for i in range(LENC)]

        ce = CurveEstimator()
        ce.calculate_curve_length(np.array(nps))

        LENC_USER = len(self.gradient)

        estimat_SNM = [0.0] * LENC_USER
        current_POS = [0.0] * LENC_USER

        est_cSolid = CurveReducer.pow_curve(self.gradient, rSumValue)

        loop: list[int] = [0] * LENC_USER

        for j in range(LENC_USER):
            low, high = 0.0, 1.0  # Binary search range for each point
            loop[j] = 0  # Reset iteration counter

            if j == 0:
                current_POS[j] = low
                estimat_SNM[j] = Optcolor.ksToSnm(ksMedia)
                continue
            if j == LENC_USER - 1:
                current_POS[j] = high
                estimat_SNM[j] = Optcolor.ksToSnm(ksSolid)
                continue

            while loop[j] < 50:  # A reasonable max for binary search

                mid = (low + high) / 2  # Midpoint for binary search
                estimat_SNM[j] = Optcolor.ksMixWithConcentrationsToSpectral(
                    ksMedia, ksSolid, 1 - mid, mid
                )
                current_POS[j] = ce.calculate_percentage(
                    CS_Spectral2XYZ(estimat_SNM[j]).to_numpy()
                )

                dfactor = current_POS[j] - self.gradient[j]  # Error difference

                if abs(dfactor) < self.tolerance:  # Convergence condition
                    break  # Stop early if within tolerance

                if dfactor > 0:
                    high = mid  # Shift upper bound down
                else:
                    low = mid  # Shift lower bound up

                loop[j] += 1

            est_cSolid[j] = mid  # Assign final optimized value

        if invert:
            estimat_SNM = estimat_SNM[::-1]
            est_cSolid = est_cSolid[::-1]

        formatted_SNM = [round(elem, 4) for elem in est_cSolid]
        formatted_POS = [round(elem, 10) for elem in current_POS]
        formatted_RMP = [round(elem, 2) for elem in self.gradient]

        response = {
            "color": Cs_Spectral2Multi(estimat_SNM),
            "ramp": formatted_RMP,
            "cSolid": formatted_SNM,
            "tolerance": self.tolerance,
            "precision": self.precision,
            "version": "MARS.4a.078",
        }
        if self.debug:
            response.update(
                {
                    "current_POS": formatted_POS,
                    "curve_length": ce.curve_length,
                    "loops": loop,
                    "loopsSum": sum(loop),
                    "rSumValue": rSumValue,
                }
            )

        return response

    def start_SCTV(self):

        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -

        rRumMedia = sum(self.media)
        rRumSolid = sum(self.solid)

        invert = False
        if rRumMedia < rRumSolid:
            invert = True

        rSumValue = rRumSolid / rRumMedia if invert else rRumMedia / rRumSolid
        rSumValue = math.pow(rSumValue, 1 / 2.25)

        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -

        ksMedia = Optcolor.ksFromSnm(self.solid if invert else self.media)
        ksSolid = Optcolor.ksFromSnm(self.media if invert else self.solid)

        cSolid = CurveReducer.pow_curve(self.gradient, rSumValue)

        LENC: int = len(cSolid)

        spectrals = [
            Optcolor.ksMixWithConcentrationsToSpectral(
                ksMedia, ksSolid, (1 - cSolid[i]), cSolid[i]
            )
            for i in range(LENC)
        ]

        # - + - + - + - + ESTIMATE THE CURVE IN 3D SPACE - + - + - + - + -

        est_XYZ = [CS_Spectral2XYZ(spectrals[i]) for i in range(LENC)]

        estimat_SNM = [0.0] * LENC
        current_POS = [0.0] * LENC

        loop: list[int] = [0] * LENC

        for j in range(LENC):
            low, high = 0.0, 1.0  # Binary search range for each point
            loop[j] = 0  # Reset iteration counter

            if j == 0:
                current_POS[j] = low
                estimat_SNM[j] = Optcolor.ksToSnm(ksMedia)
                continue
            if j == LENC - 1:
                current_POS[j] = high
                estimat_SNM[j] = Optcolor.ksToSnm(ksSolid)
                continue

            while loop[j] < 50:  # A reasonable max for binary search

                mid = (low + high) / 2  # Midpoint for binary search
                estimat_SNM[j] = Optcolor.ksMixWithConcentrationsToSpectral(
                    ksMedia, ksSolid, 1 - mid, mid
                )
                tmp_XYZ = CS_Spectral2XYZ(estimat_SNM[j])
                current_POS[j] = Optcolor.calculate_sctv(
                    est_XYZ[0], tmp_XYZ, est_XYZ[-1]
                )

                dfactor = current_POS[j] - self.gradient[j]  # Error difference

                if abs(dfactor) < self.tolerance:  # Convergence condition
                    break  # Stop early if within tolerance

                if dfactor > 0:
                    high = mid  # Shift upper bound down
                else:
                    low = mid  # Shift lower bound up

                loop[j] += 1

            cSolid[j] = mid  # Assign final optimized value

        formatted_CSO = [round(elem, 4) for elem in cSolid]
        formatted_POS = [round(elem, 10) for elem in current_POS]
        formatted_RMP = [round(elem, 2) for elem in self.gradient]

        return {
            "color": Cs_Spectral2Multi(estimat_SNM),
            "ramp": formatted_RMP,
            "cSolid": formatted_CSO,
            "current_POS": formatted_POS,
            "loops": loop,
            "loopsSum": sum(loop),
            "tolerance": self.tolerance,
            "precision": self.precision,
            "version": "MARS.4a.078",
            "rSumValue": rSumValue,
        }

    def start_Curve(self):

        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -

        rRumMedia = sum(self.media)
        rRumSolid = sum(self.solid)

        invert = False
        if rRumMedia < rRumSolid:
            invert = True

        rSumValue = rRumSolid / rRumMedia if invert else rRumMedia / rRumSolid
        rSumValue = math.pow(rSumValue, 1 / 2.25)

        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -

        ksMedia = Optcolor.ksFromSnm(self.solid if invert else self.media)
        ksSolid = Optcolor.ksFromSnm(self.media if invert else self.solid)

        # cFactor = 4
        # estGiga = self.calculate_gradient_by_steps(self.precision)
        estGiga = self.calculate_gradient_by_steps(self.precision)
        cSolid = CurveReducer.pow_curve(estGiga, rSumValue)
        # cSolid = estGiga.copy()
        # cSolid = [0,1]

        LENC: int = len(cSolid)

        spectrals = [
            Optcolor.ksMixWithConcentrationsToSpectral(
                ksMedia, ksSolid, (1 - cSolid[i]), cSolid[i]
            )
            for i in range(LENC)
        ]

        # - + - + - + - + ESTIMATE THE CURVE IN 3D SPACE - + - + - + - + -

        nps = [CS_Spectral2XYZ(spectrals[i]).to_numpy() for i in range(LENC)]

        ce = CurveEstimator()
        ce.calculate_curve_length(np.array(nps))

        # LENC_USER = len(self.gradient)

        estimat_SNM = [0.0] * LENC  # LENC_USER
        current_POS = [0.0] * LENC  # LENC_USER

        est_cSolid = cSolid.copy()

        loop: list[int] = [0] * LENC  # LENC_USER

        for j in range(LENC):
            low, high = 0.0, 1.0  # Binary search range for each point
            loop[j] = 0  # Reset iteration counter

            if j == 0:
                current_POS[j] = low
                estimat_SNM[j] = Optcolor.ksToSnm(ksMedia)
                continue
            if j == LENC - 1:
                current_POS[j] = high
                estimat_SNM[j] = Optcolor.ksToSnm(ksSolid)
                continue

            while loop[j] < 50:  # A reasonable max for binary search

                mid = (low + high) / 2  # Midpoint for binary search
                estimat_SNM[j] = Optcolor.ksMixWithConcentrationsToSpectral(
                    ksMedia, ksSolid, 1 - mid, mid
                )
                current_POS[j] = ce.calculate_percentage(
                    CS_Spectral2XYZ(estimat_SNM[j]).to_numpy()
                )

                dfactor = current_POS[j] - self.gradient[j]  # Error difference

                if abs(dfactor) < self.tolerance:  # Convergence condition
                    break  # Stop early if within tolerance

                if dfactor > 0:
                    high = mid  # Shift upper bound down
                else:
                    low = mid  # Shift lower bound up

                loop[j] += 1

            est_cSolid[j] = mid  # Assign final optimized value

        # IDEA: Using only 20 colors to predict the curve would increase the speed.
        # I should be saved in a SKIPY file and loaded when needed.
        # It should have the option to smooth the curve.
        # As the first part of the curve is mostly flat, myabe here it room for improvement
        # by reducing the datapoint in that area. Does not need to be equaliy spaced.

        spline = CurveReducer.spline_from_points(estGiga, est_cSolid)

        # x_values = np.linspace(0, 1, self.precision)
        interpolated = spline(self.gradient)
        formatted_iSNM = [round(elem, 10) for elem in interpolated]

        result_100 = [
            Optcolor.ksMixWithConcentrationsToSpectral(
                ksMedia, ksSolid, 1 - interpolated[i], interpolated[i]
            )
            for i in range(len(interpolated))
        ]

        formatted_SNM = [round(elem, 4) for elem in est_cSolid]
        formatted_POS = [round(elem, 10) for elem in current_POS]
        formatted_RMP = [round(elem, 2) for elem in self.gradient]

        return {
            "color": Cs_Spectral2Multi(result_100),
            "ramp": formatted_RMP,
            "cSolid": formatted_iSNM,
            "curve_length": ce.curve_length,
            "current_POS": formatted_POS,
            "loops": loop,
            "loopsSum": sum(loop),
            "tolerance": self.tolerance,
            "precision": self.precision,
            "version": "MARS.4a.078.1",
            "rSumValue": rSumValue,
        }

    def start_Curve3D(self):

        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -
        
        ksMedia = OptcolorNumpy.ksFromSnm(np.array(self.media))
        ksSolid = OptcolorNumpy.ksFromSnm(np.array(self.solid))

        # - + - + - + - + ESTIMATE THE CURVE IN 3D SPACE - + - + - + - + -

        cSolid = np.linspace(0, 1, self.precision)
        cMedia = 1 - cSolid

        nps = OptcolorNumpy.ksMixWithConcentrationsToXYZ_V2(ksMedia, ksSolid, cMedia, cSolid)
        
        ce = CurveEstimator3D()
        ce.calculate_curve_length(nps, 10000, 2000)

        LENC = len(self.gradient)

        estimat_SNM = [0.0] * LENC
        current_POS = [0.0] * LENC
        est_cSolid  = [0.0] * LENC
        loop        = [0]   * LENC

        for j in range(LENC):
            low, high = 0.0, 1.0  # Binary search range for each point

            if self.gradient[j] == low or self.gradient[j] == high:
                est_cSolid[j] = self.gradient[j]
                estimat_SNM[j] = self.media if self.gradient[j] == low else self.solid
                continue

            while loop[j] < 21:  # A reasonable max for binary search

                mid = (low + high) * 0.5  # Midpoint for binary search
                estimat_SNM[j] = OptcolorNumpy.ksMixWithConcentrationsToSpectral(
                    ksMedia, ksSolid, 1 - mid, mid
                )
                current_POS[j] = ce.calculate_percentage(
                    CS_Spectral2XYZ_Numpy(estimat_SNM[j])
                )
        
                dfactor = current_POS[j] - self.gradient[j]  # Error difference
            
                if abs(dfactor) < self.tolerance:  # Convergence condition
                    break  # Stop early if within tolerance
                
                if dfactor > 0:
                    high = mid  # Shift upper bound down
                else:
                    low = mid  # Shift lower bound up

                loop[j] += 1

            est_cSolid[j] = mid  # Assign final optimized value

        response = {
            "color": Cs_Spectral2Multi(estimat_SNM),
            "ramp": [round(elem, 2) for elem in self.gradient],
            "cSolid": [round(elem, 4) for elem in est_cSolid],
            "tolerance": self.tolerance,
            "precision": self.precision,
            "version": "MARS.4a.081.3D",
        }
        
        if self.debug:
            
            xyz_color = ce.interpolate_point_by_percentage(0.5).tolist()
            lab_color = Cs_XYZ2LAB(CsXYZ(xyz_color[0], xyz_color[1], xyz_color[2])).to_list()
            
            response.update(
                {
                    "ksMedia": ksMedia.tolist(),
                    "ksSolid": ksSolid.tolist(),
                    "ksMediaSum": np.mean(np.array(self.media)),
                    "ksSolidSum": np.mean(np.array(self.solid)),
                    "loops": loop,
                    "loopsSum": sum(loop),
                    "current_POS": [round(elem, 6) for elem in current_POS],
                    "curve_length": ce.curve_length,
                    "nps": [nps[i].tolist() for i in range(len(nps))],
                    "nps50_xyz": xyz_color,
                    "nps50_lab": lab_color,
                }
            )
        
        return response


class Optcolor:

    @staticmethod
    def ksFromSnm(snm: list[float]) -> list[float]:
        return [(1 - x) ** 2 / (2 * x) for x in snm]

    @staticmethod
    def ksFulltoneInk(ksMedia: list[float], ksSolid: list[float]) -> list[float]:
        return [ksSolid[i] - ksMedia[i] for i in range(len(ksMedia))]

    @staticmethod
    def ksMix(
        ksMedia: list[float],
        ksSolid: list[float],
        concentrations: float,
        correct: float,
    ) -> list[float]:
        return [
            ksMedia[i] * (1 - concentrations)
            + (ksSolid[i] * (concentrations) * correct)
            for i in range(len(ksMedia))
        ]

    @staticmethod
    def ksMixWithConcentrations(
        ksMedia: list[float], ksSolid: list[float], cMedia: float, cSolid: float
    ) -> list[float]:
        return [
            (ksMedia[i] * cMedia) + (ksSolid[i] * cSolid) for i in range(len(ksMedia))
        ]

    @staticmethod
    def ksMixWithConcentrationsToSpectral(
        ksMedia: list[float], ksSolid: list[float], cMedia: float, cSolid: float
    ) -> list[float]:

        if cMedia > 1:
            cMedia = 1

        if cSolid > 1:
            cSolid = 1

        if cMedia < 0:
            cMedia = 0

        if cSolid < 0:
            cSolid = 0

        ksMix = Optcolor.ksMixWithConcentrations(ksMedia, ksSolid, cMedia, cSolid)
        spectrals = Optcolor.ksToSnm(ksMix)
        return spectrals

    @staticmethod
    def ksToSnm(ks: list[float]) -> list[float]:
        return [1 + x - math.sqrt(x**2 + 2 * x) for x in ks]

    @staticmethod
    def calculate_sctv(xyz_p: CsXYZ, xyz_t: CsXYZ, xyz_s: CsXYZ):
        """
        Calculate the spectral contrast visibility of a color.

        Parameters:
            xyz_p (CsXYZ): Substrate Color.
            xyz_t (CsXYZ): Mix Color.
            xyz_s (CsXYZ): Solid Color.

        Returns:
            float: The spot color tonal value.
        """

        numerator = (
            (xyz_t.X - xyz_p.X) ** 2
            + (xyz_t.Y - xyz_p.Y) ** 2
            + (xyz_t.Z - xyz_p.Z) ** 2
        )
        denominator = (
            (xyz_s.X - xyz_p.X) ** 2
            + (xyz_s.Y - xyz_p.Y) ** 2
            + (xyz_s.Z - xyz_p.Z) ** 2
        )
        return math.sqrt(numerator / denominator)

class OptcolorNumpy:

    @staticmethod
    def ksFromSnm(snm: np.ndarray) -> np.ndarray:
        return (1 - snm) ** 2 / (2 * snm)

    @staticmethod
    def ksFulltoneInk(ksMedia: np.ndarray, ksSolid: np.ndarray) -> np.ndarray:
        return ksSolid - ksMedia

    @staticmethod
    def ksMix(
        ksMedia: np.ndarray,
        ksSolid: np.ndarray,
        concentrations: float,
        correct: float,
    ) -> np.ndarray:
        return ksMedia * (1 - concentrations) + (ksSolid * concentrations * correct)

    @staticmethod
    def ksMixWithConcentrations(
        ksMedia: np.ndarray, ksSolid: np.ndarray, cMedia: float, cSolid: float
    ) -> np.ndarray:
        return (ksMedia * cMedia) + (ksSolid * cSolid)

    @staticmethod
    def ksMixWithConcentrationsToSpectral(
        ksMedia: np.ndarray, ksSolid: np.ndarray, cMedia: float, cSolid: float
    ) -> np.ndarray:

        cMedia = np.clip(cMedia, 0, 1)
        cSolid = np.clip(cSolid, 0, 1)

        ksMix = OptcolorNumpy.ksMixWithConcentrations(ksMedia, ksSolid, cMedia, cSolid)
        spectrals = OptcolorNumpy.ksToSnm(ksMix)
        return spectrals

    @staticmethod
    def ksMixWithConcentrationsToXYZ(
        ksMedia: np.ndarray, ksSolid: np.ndarray, cMedia: float, cSolid: float
    ) -> np.ndarray:

        cMedia = np.clip(cMedia, 0, 1)
        cSolid = np.clip(cSolid, 0, 1)

        ksMix = OptcolorNumpy.ksMixWithConcentrations(ksMedia, ksSolid, cMedia, cSolid)
        spectrals = OptcolorNumpy.ksToSnm(ksMix)
        xyz = CS_Spectral2XYZ_Numpy(spectrals)
        return xyz
    
    @staticmethod
    def ksMixWithConcentrationsToXYZ_V2(
        ksMedia: np.ndarray, ksSolid: np.ndarray, cMedia: np.ndarray, cSolid: np.ndarray
    ) -> np.ndarray:
        """
        Optimized version of ksMixWithConcentrationsToXYZ using vectorized operations.

        Parameters:
            ksMedia (np.ndarray): Media K/S values.
            ksSolid (np.ndarray): Solid K/S values.
            cMedia (np.ndarray): Media concentrations.
            cSolid (np.ndarray): Solid concentrations.

        Returns:
            np.ndarray: XYZ values.
        """
        
        """ 
        cMedia = np.clip(cMedia, 0, 1)
        cSolid = np.clip(cSolid, 0, 1) 
        """

        ksMix = OptcolorNumpy.ksMixWithConcentrations(ksMedia[:, np.newaxis], ksSolid[:, np.newaxis], cMedia, cSolid)
        spectrals = OptcolorNumpy.ksToSnm(ksMix)
        xyz = np.array([CS_Spectral2XYZ_Numpy(spectral) for spectral in spectrals.T])
        return xyz

    @staticmethod
    def ksToSnm(ks: np.ndarray) -> np.ndarray:
        return 1 + ks - np.sqrt(ks**2 + 2 * ks)

    @staticmethod
    def calculate_sctv(xyz_p, xyz_t, xyz_s):
        """
        Calculate the spectral contrast visibility of a color.

        Parameters:
            xyz_p (CsXYZ): Substrate Color.
            xyz_t (CsXYZ): Mix Color.
            xyz_s (CsXYZ): Solid Color.

        Returns:
            float: The spot color tonal value.
        """
        numerator = (
            (xyz_t.X - xyz_p.X) ** 2
            + (xyz_t.Y - xyz_p.Y) ** 2
            + (xyz_t.Z - xyz_p.Z) ** 2
        )
        denominator = (
            (xyz_s.X - xyz_p.X) ** 2
            + (xyz_s.Y - xyz_p.Y) ** 2
            + (xyz_s.Z - xyz_p.Z) ** 2
        )
        return math.sqrt(numerator / denominator)
