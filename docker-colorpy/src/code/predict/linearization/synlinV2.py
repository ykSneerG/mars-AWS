import math
from src.code.space.colorSpace import CsXYZ
from src.code.space.colorConstants.illuminant import OBSERVER
from src.code.space.colorConverter import (
    CS_Spectral2XYZ,
    Cs_XYZ2RGB,
    Cs_XYZ2LAB,
    Cs_Lab2LCH,
    Cs_XYZ2Denisty,
)

# from space.colorSpace import CsXYZ, CsRGB, CsLAB, CsLCH
# from linear.sctv import SCTV
# from predict.kubelkaMunkSubstrate import KubelkaMunkSubstrate


class SynLinSolidV2:
    """
    Predict a linearization based an spectral data in a range of 380-730nm with 10nm steps.
    """

    def __init__(self, **kwargs):
        self.media = None
        """
        Spectral data for the media.
        """
        self.solid = None
        """
        Spectral data for the solid ink colors.
        """
        self.gradient: list = []
        """
        DCS values for the gradient.
        From 0 to 100% in x% steps.
        """

        self.gradientCorrected = []

        self.linearSolid = []
        self.linearSctv = []
        self.linearIters = []

        self.places = 4
        self.tolerance = 0.5
        self.maxLoops = 100

    def get_prediction(self):
        xyzs, hexs, labs, lchs, dens = self.convert_to_color_spaces(self.linearSolid)
        return {
            "pcs": self.linearSolid,
            "dcs": self.gradient,
            "sctv": self.linearSctv,
            "dcsCorrected": self.gradientCorrected,
            "its": self.linearIters,
            "xyz": xyzs,
            "hex": hexs,
            "lab": labs,
            "lch": lchs,
            "density": dens,
        }

    def set_media(self, media):
        self.media = media

    def set_solid(self, solid):
        self.solid = solid

    def set_places(self, places):
        self.places = places

    def set_max_loops(self, max_loops):
        self.maxLoops = max_loops

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value

    def set_gradient(self, gradient: list):
        self.gradient = gradient
        self.gradientCorrected = [1.0] * len(self.gradient)
        self.linearSctv = [None] * len(self.gradient)
        self.linearIters = [0] * len(self.gradient)
        self.linearSolid = [None] * len(self.gradient)

    def set_gradient_by_steps(self, subdivision) -> None:
        if subdivision < 1:
            raise ValueError("Subdivision must be greater than 1.")

        ramp = [0.0] * subdivision
        stepzize = 1 / (subdivision - 1)

        for i in range(subdivision):
            ramp[i] = round(i * stepzize, self.places)

        self.set_gradient(ramp)

    def convert_to_color_spaces(self, linearSolid: list) -> tuple:
        xyzs = []
        hexs = []
        labs = []
        lchs = []
        dens = []
        for item in linearSolid:
            xyz = CS_Spectral2XYZ(item, OBSERVER.DEG2)
            lab = Cs_XYZ2LAB(xyz)
            lch = Cs_Lab2LCH(lab)
            rgb = Cs_XYZ2RGB(xyz)
            den = Cs_XYZ2Denisty(xyz)
            hexs.append(rgb.to_hex())
            xyzs.append(xyz.to_json(2))
            labs.append(lab.to_json(2))
            lchs.append(lch.to_json(2))
            dens.append(round(den, 2))

        return xyzs, hexs, labs, lchs, dens

    def in_tolerance(self, actualValue, targetValue) -> bool:
        return abs(actualValue - targetValue) <= self.tolerance

    def start(self):

        LENC: int = len(self. gradient)

        spectrals = [0] * LENC
        sctvs = [0.0] * LENC
        iter = [0] * LENC
        iterstop = [False] * LENC

        #  add media
        ks = [Optcolor.ksFromSnm(self.media)]

        # add solid
        tmp = Optcolor.ksFromSnm(self.solid)
        ks.append(Optcolor.ksFulltoneInk(ks[0], tmp))

        # generate initial steps for the concentration
        concentrations = self.gradient
        conccorections = [1.0] * LENC

        mixXYZ = [CsXYZ] * LENC

        for r in range(self.maxLoops):

            for i in range(LENC):
                ksMix = Optcolor.ksMix(
                    ks[0], ks[1], concentrations[i] * conccorections[i]
                )
                spectrals[i] = Optcolor.ksToSnm(ksMix)
                mixXYZ[i] = CS_Spectral2XYZ(spectrals[i], OBSERVER.DEG2)

            for i in range(LENC):
                if iterstop[i]:
                    continue

                sctvs[i] = Optcolor.calculate_sctv(mixXYZ[0], mixXYZ[i], mixXYZ[-1])
                conccorections[i] *= Optcolor.calculate_sctv_correction(sctvs[i], concentrations[i])
                iter[i] = r
                iterstop[i] = self.in_tolerance(sctvs[i], concentrations[i])

            if all(iterstop):
                break

        roundedSpectrals = [[round(s, self.places) for s in spec] for spec in spectrals]

        return {
            "spectrals": roundedSpectrals,
            "sctvs": sctvs,
            "iter": iter,
            "operations": sum(iter),
            "loops": max(iter),
            "conccorections": conccorections,
        }


class Optcolor:
    @staticmethod
    def ksFromSnm(snm):
        ks = [(1 - x) ** 2 / (2 * x) for x in snm]
        return ks

    @staticmethod
    def ksFulltoneInk(ksMedia, ksSolid):
        ks = [ksSolid[i] - ksMedia[i] for i in range(len(ksMedia))]
        return ks

    @staticmethod
    def ksMix(ksMedia, ksSolid, concentrations):
        ks = [ksMedia[i] + (ksSolid[i] * concentrations) for i in range(len(ksMedia))]
        return ks

    @staticmethod
    def ksToSnm(ks):
        snm = [1 + x - math.sqrt(x**2 + 2 * x) for x in ks]
        return snm

    @staticmethod
    def generate_steps_array(steps):
        array = [i / (steps - 1) for i in range(steps)]
        return array

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

    @staticmethod
    def calculate_sctv_correction(sctv_value, target_value):
        if sctv_value == 0:
            return 0
        return target_value / sctv_value
