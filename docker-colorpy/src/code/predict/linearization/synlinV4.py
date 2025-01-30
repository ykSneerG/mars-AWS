import math
from src.code.space.colorSpace import CsXYZ
from src.code.space.colorConverter import Cs_Spectral2Multi, CS_Spectral2XYZ
from src.code.predict.linearization.baselinearization import BaseLinearization
import numpy as np  # type: ignore


class SynLinSolidV4(BaseLinearization):
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

    """ def setCorrection(self, correction: float):
        self.correction = correction """

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
        estGiga = self.calculate_gradient_by_steps(2)
        cSolid = CurveReducer.reduce_curve2(estGiga, rSumValue)
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

        est_cSolid = CurveReducer.reduce_curve2(self.gradient, rSumValue)

        loop = [0] * LENC_USER
        stop = [False] * LENC_USER

        for i in range(1000): # self.maxLoops
            for j in range(LENC_USER):
                if stop[j]:
                    continue

                estimat_SNM[j] = Optcolor.ksMixWithConcentrationsToSpectral(
                    ksMedia, ksSolid, est_cSolid[-1] - est_cSolid[j], est_cSolid[j]
                )
                current_POS[j] = ce.calculate_percentage(
                    CS_Spectral2XYZ(estimat_SNM[j]).to_numpy()
                )

                loop[j] += 1

                dfactor = abs(current_POS[j] - self.gradient[j])
                if dfactor < self.tolerance: # or est_cSolid[j] > 1:
                    stop[j] = True
                else:
                    
                    """ if current_POS[j] > 0.75 and current_POS[j] <= 0.95:
                        dfactor = 0.01
                    
                    if current_POS[j] > 0.95 and current_POS[j] <= 0.98:
                        dfactor = 0.001
                        
                    if current_POS[j] > 0.98 and current_POS[j]:
                        dfactor = 0.00001 """
                        
                    if current_POS[j] > 0.97:
                        dfactor = 0.000001
                    
                    est_cSolid[j] *= (
                        (1 - dfactor)
                        if current_POS[j] > self.gradient[j]
                        else (1 + dfactor)
                    )
                    
        # IDEA: Using only 20 colors to predict the curve would increase the speed.
        # I should be saved in a SKIPY file and loaded when needed.
        # It should have the option to smooth the curve.
        # As the first part of the curve is mostly flat, myabe here it room for improvement 
        # by reducing the datapoint in that area. Does not need to be equaliy spaced.


        formatted_SNM = [round(elem, 4) for elem in est_cSolid]
        formatted_POS = [round(elem, 4) for elem in current_POS]
        formatted_RMP = [round(elem, 2) for elem in self.gradient]

        return {
            "color": Cs_Spectral2Multi(estimat_SNM),
            #"ramp": self.gradient,
            "ramp": formatted_RMP,
            #"cSolid": est_cSolid,
            "cSolid": formatted_SNM,
            "curve_length": ce.curve_length,
            #"current_POS": current_POS,
            "current_POS": formatted_POS,
            "loops": loop,
            "loopsSum": sum(loop),
            "tolerance": self.tolerance,
            "version": "MARS.4.070",
            "rSumValue": rSumValue
        }


class CurveReducer:
    """
    # Original array
    x = np.linspace(0, 1, 100)

    # Custom Power Function (Normalized)
    p = 2
    c1 = 0.5
    y1 = (x**p * (1 + c1 * x)) / (1 + c1)

    # Sigmoid-like Transformation (Normalized)
    c2 = 2
    y2 = x / (1 + c2 * x**p)

    # Logarithmic Compression (Normalized)
    k = 10
    y3 = np.log(1 + k * x) / np.log(1 + k)
    """

    @staticmethod
    def reduce_curve(x: list[float], p: float, c: float) -> list[float]:
        return [(x[i] ** p * (1 + c * x[i])) / (1 + c) for i in range(len(x))]

    @staticmethod
    def reduce_curve2(x: list[float], p: float) -> list[float]:
        return [(x[i] ** p) for i in range(len(x))]

    @staticmethod
    def sigmoid_curve(x: list[float], p: float, c: float) -> list[float]:
        return [x[i] / (1 + c * x[i] ** p) for i in range(len(x))]

    @staticmethod
    def log_curve(x: list[float], k: float) -> list[float]:
        return [math.log(1 + k * x[i]) / math.log(1 + k) for i in range(len(x))]


class CurveEstimator:
    def __init__(self):
        self.points = None
        self.distances = None
        self.cumulative_lengths = None
        self.curve_length = None

    def calculate_curve_length(self, points: np.ndarray):
        """
        Calculates the total length of a 3D curve given by discrete points.

        Parameters:
            points (np.ndarray): An Nx3 array of points (x, y, z) along the curve.

        Raises:
            ValueError: If input points are not valid Nx3 coordinates.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Input points must have a shape of Nx3 (representing XYZ coordinates)."
            )

        self.points = points

        # Calculate pairwise distances between consecutive points
        self.distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

        # Cumulative lengths of the curve
        self.cumulative_lengths = np.cumsum(self.distances)

        # Total curve length
        self.curve_length = self.cumulative_lengths[-1]

    def interpolate_point_on_curve(self, percentage: float) -> np.ndarray:
        """
        Interpolates a point along a 3D curve at a specified percentage of its total length.

        Parameters:
            percentage (float): Percentage of the curve's length (0 to 1) where the point is desired.

        Returns:
            np.ndarray: Interpolated point (x, y, z) on the curve.

        Raises:
            ValueError: If percentage is not between 0 and 1.
        """
        if self.points is None:
            raise RuntimeError(
                "Curve length must be calculated first using calculate_curve_length()."
            )
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1.")

        # Target length along the curve
        target_length = percentage * self.curve_length

        # Edge cases: start or end of the curve
        if target_length == 0:
            return self.points[0]
        if target_length == self.curve_length:
            return self.points[-1]

        # Find the segment containing the target length
        segment_index = np.searchsorted(self.cumulative_lengths, target_length)

        # Get the segment's start and end points
        p1 = self.points[segment_index]
        p2 = self.points[segment_index + 1]

        # Lengths for interpolation
        segment_start_length = (
            self.cumulative_lengths[segment_index - 1] if segment_index > 0 else 0.0
        )
        segment_length = self.distances[segment_index]

        # Interpolation factor
        t = (target_length - segment_start_length) / segment_length

        # Linear interpolation
        return p1 + t * (p2 - p1)

    def calculate_percentage(self, target_xyz: np.ndarray) -> float:
        """
        Finds the percentage along the curve that is closest to a given point.

        Parameters:
            target_xyz (np.ndarray): Target point (x, y, z).

        Returns:
            float: Percentage (0 to 1) along the curve that is closest to the target point.
        """
        if self.points is None:
            raise RuntimeError(
                "Curve length must be calculated first using calculate_curve_length()."
            )

        target_xyz = np.array(target_xyz)

        min_distance = float("inf")
        closest_length = 0.0

        for i in range(len(self.points) - 1):
            # Get the closest point on the segment
            p1, p2 = self.points[i], self.points[i + 1]
            closest_point = self._closest_point_on_segment(p1, p2, target_xyz)

            # Calculate length to the closest point
            segment_start_length = self.cumulative_lengths[i - 1] if i > 0 else 0.0
            segment_length = self.distances[i]
            t = (
                np.linalg.norm(closest_point - p1) / segment_length
            )  # Proportion within the segment
            length_to_closest_point = segment_start_length + t * segment_length

            # Update minimum distance
            distance_to_target = np.linalg.norm(closest_point - target_xyz)
            if distance_to_target < min_distance:
                min_distance = distance_to_target
                closest_length = length_to_closest_point

        # Calculate percentage
        return closest_length / self.curve_length

    @staticmethod
    def _closest_point_on_segment(p1, p2, target):
        """
        Finds the closest point on a line segment [p1, p2] to a target point.

        Parameters:
            p1, p2 (np.ndarray): The endpoints of the segment (3D points).
            target (np.ndarray): The target point (3D point).

        Returns:
            np.ndarray: Closest point on the segment to the target point.
        """
        p1, p2, target = np.array(p1), np.array(p2), np.array(target)
        segment = p2 - p1
        segment_length_squared = np.dot(segment, segment)

        if segment_length_squared == 0:
            return p1

        t = np.dot(target - p1, segment) / segment_length_squared
        t = np.clip(t, 0, 1)
        return p1 + t * segment


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
        """ result = []
        for x in ks:
            value = x ** 2 + 2 * x
            if value < 0:
                #raise ValueError(f"Negative value encountered in ksToSnm: {value}")
                result.append(1)
            else:
                result.append(1 + x - math.sqrt(value))
        return result """

    @staticmethod
    def calculate_sctv(xyz_p: CsXYZ, xyz_t: CsXYZ, xyz_s: CsXYZ):
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
