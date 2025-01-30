from abc import abstractmethod
import math
from typing import Union

from src.code.colorMath import CmMath
import numpy as np # type: ignore


MINVAL_CS_RGB = 0
MAXVAL_CS_RGB = 255
MINVAL_CS_XYZ = 0
MAXVAL_CS_XYZ = 100


class CsBase:

    @abstractmethod
    def __init__() -> None:
        raise NotImplementedError("CsBase is an abstract class and cannot be instantiated directly")

    # def to_json(self) -> dict[str, any]:
    #     return {k: v for k, v in vars(self).items() if not k.startswith('__')}

    def to_json(self, decimal_places: int = None) -> dict[str, Union[str, int, float, None]]:
        def round_numeric(value):
            if decimal_places is not None and isinstance(value, float):
                return round(value, decimal_places)
            return value

        return {k: round_numeric(v) for k, v in vars(self).items() if not k.startswith('__')}

    def to_list(self) -> list:
        return [v for k, v in vars(self).items() if not k.startswith('__')]


class CsLAB(CsBase):
    """
    Represents a color in the CieLAB color space.

    :param L (float): Lightness component, range 0 - 100.
    :param A (float): Green-red component, range -128 - 127.
    :param B (float): Blue-yellow component, range -128 - 127.
    """

    def __init__(self, lab_L: float, lab_A: float, lab_B: float) -> None:
        if (
            not isinstance(lab_L, (float, int)) or
            not isinstance(lab_A, (float, int)) or
            not isinstance(lab_B, (float, int))
        ):
            raise TypeError("The parameter must be float.")

        self.L = lab_L
        self.A = lab_A
        self.B = lab_B

    def to_chroma(self) -> float:
        """
        Calculates the chroma of the color.
        """
        return math.sqrt(self.A ** 2 + self.B ** 2)


class CsLCH(CsBase):

    def __init__(self, lch_L: float, lch_C: float, lch_H: float) -> None:
        if (
            not isinstance(lch_L, (float, int)) or
            not isinstance(lch_C, (float, int)) or
            not isinstance(lch_H, (float, int))
        ):
            raise TypeError("The parameter must be float.")

        if lch_C < 0:
            raise ValueError("The chroma is negative.")

        self.L = lch_L
        self.C = lch_C
        self.H = CsLCH.reduce_angle(lch_H)

    def to_chroma(self) -> float:
        """
        Calculates the chroma of the color.
        """
        return self.C

    @staticmethod
    def reduce_angle(angle: float) -> float:
        return angle % 360


class CsXYZ(CsBase):

    def __init__(self, xyz_X: float, xyz_Y: float, xyz_Z: float) -> None:
        if (
            not isinstance(xyz_X, (float, int)) or
            not isinstance(xyz_Y, (float, int)) or
            not isinstance(xyz_Z, (float, int))
        ):
            raise TypeError("The parameter must be float.")

        self.X = xyz_X
        self.Y = xyz_Y
        self.Z = xyz_Z

    def normalize(self):
        """
        Scales the CsXYZ into the range 0-1.
        """
        return CsXYZ(
            self.X / MAXVAL_CS_XYZ,
            self.Y / MAXVAL_CS_XYZ,
            self.Z / MAXVAL_CS_XYZ
        )

    def denormalize(self):
        """
        Scales the CsXYZ into the range 0-100.
        """
        return CsXYZ(
            self.X * MAXVAL_CS_XYZ,
            self.Y * MAXVAL_CS_XYZ,
            self.Z * MAXVAL_CS_XYZ
        )

    def to_density(self) -> float:
        """
        Returns the density value of a CsXYZ color.
        """
        return -math.log10(self.Y / 100)
    
    def to_numpy(self):
        return np.array([self.X, self.Y, self.Z])


class CsRGB(CsBase):

    def __init__(self, rgb_R: float, rgb_G: float, rgb_B: float) -> None:
        if (
            not isinstance(rgb_R, (float, int)) or
            not isinstance(rgb_G, (float, int)) or
            not isinstance(rgb_B, (float, int))
        ):
            raise TypeError("The parameter must be float.")

        self.R = rgb_R
        self.G = rgb_G
        self.B = rgb_B

    def normalize(self) -> 'CsRGB':
        """
        Scales the CsXYZ into the range 0-1.
        """
        return CsRGB(
            self.R / MAXVAL_CS_RGB,
            self.G / MAXVAL_CS_RGB,
            self.B / MAXVAL_CS_RGB
        )

    def denormalize(self) -> 'CsRGB':
        """
        Scales the CsXYZ into the range 0-255.
        """
        return CsRGB(
            self.R * MAXVAL_CS_RGB,
            self.G * MAXVAL_CS_RGB,
            self.B * MAXVAL_CS_RGB
        )

    def clamp(self) -> 'CsRGB':
        return CsRGB(
            CmMath.clamp(self.R, 0, MAXVAL_CS_RGB),
            CmMath.clamp(self.G, 0, MAXVAL_CS_RGB),
            CmMath.clamp(self.B, 0, MAXVAL_CS_RGB)
        )

    def as_int(self) -> dict[str, int]:
        return {
            "R": CmMath.clampToInt(self.R, 0, MAXVAL_CS_RGB),
            "G": CmMath.clampToInt(self.G, 0, MAXVAL_CS_RGB),
            "B": CmMath.clampToInt(self.B, 0, MAXVAL_CS_RGB)
        }

    def to_hex(self, prefix: bool = True) -> str:
        intRGB = self.as_int()
        tmphex = '#{:02x}{:02x}{:02x}'.format(intRGB['R'], intRGB['G'], intRGB['B'])
        hexstr = tmphex.upper()
        return hexstr if prefix else hexstr[1:]


class CsCMYK(CsBase):

    def __init__(self,
                 cmyk_C: Union[float, int],
                 cmyk_M: Union[float, int],
                 cmyk_Y: Union[float, int],
                 cmyk_K: Union[float, int]):
        if (
            not isinstance(cmyk_C, (float, int)) or
            not isinstance(cmyk_M, (float, int)) or
            not isinstance(cmyk_Y, (float, int)) or
            not isinstance(cmyk_K, (float, int))
        ):
            raise TypeError("The parameter must be float or integer.")

        self.C = cmyk_C
        self.M = cmyk_M
        self.Y = cmyk_Y
        self.K = cmyk_K


class CsSpectral(CsBase):

    def __init__(self, nm380, nm390, nm400, nm410, nm420, nm430, nm440, nm450, nm460, nm470, nm480, nm490, nm500, nm510, nm520, nm530, nm540, nm550, nm560,
                 nm570, nm580, nm590, nm600, nm610, nm620, nm630, nm640, nm650, nm660, nm670, nm680, nm690, nm700, nm710, nm720, nm730):

        self.nm380 = nm380
        self.nm390 = nm390

        self.nm400 = nm400
        self.nm410 = nm410
        self.nm420 = nm420
        self.nm430 = nm430
        self.nm440 = nm440
        self.nm450 = nm450
        self.nm460 = nm460
        self.nm470 = nm470
        self.nm480 = nm480
        self.nm490 = nm490

        self.nm500 = nm500
        self.nm510 = nm510
        self.nm520 = nm520
        self.nm530 = nm530
        self.nm540 = nm540
        self.nm550 = nm550
        self.nm560 = nm560
        self.nm570 = nm570
        self.nm580 = nm580
        self.nm590 = nm590

        self.nm600 = nm600
        self.nm610 = nm610
        self.nm620 = nm620
        self.nm630 = nm630
        self.nm640 = nm640
        self.nm650 = nm650
        self.nm660 = nm660
        self.nm670 = nm670
        self.nm680 = nm680
        self.nm690 = nm690

        self.nm700 = nm700
        self.nm710 = nm710
        self.nm720 = nm720
        self.nm730 = nm730


class CsPC01:
    def __init__(self, c1):
        self.C1 = c1


class CsPC02(CsPC01):
    def __init__(self, c1, c2):
        CsPC01.__init__(self, c1)
        self.C2 = c2


class CsPC03(CsPC02):
    def __init__(self, c1, c2, c3):
        CsPC02.__init__(self, c1, c2)
        self.C3 = c3


class CsPC04(CsPC03):
    def __init__(self, c1, c2, c3, c4):
        CsPC03.__init__(self, c1, c2, c3)
        self.C4 = c4


class CsPC05(CsPC04):
    def __init__(self, c1, c2, c3, c4, c5):
        CsPC03.__init__(self, c1, c2, c3, c4)
        self.C5 = c5


class CsPC06(CsPC05):
    def __init__(self, c1, c2, c3, c4, c5, c6):
        CsPC05.__init__(self, c1, c2, c3, c4, c5)
        self.C6 = c6


class CsPC07(CsPC06):
    def __init__(self, c1, c2, c3, c4, c5, c6, c7):
        CsPC06.__init__(self, c1, c2, c3, c4, c5, c6)
        self.C7 = c7


class CsPC08(CsPC07):
    def __init__(self, c1, c2, c3, c4, c5, c6, c7, c8):
        CsPC07.__init__(self, c1, c2, c3, c4, c5, c6, c7)
        self.C8 = c8


class CsPC09(CsPC08):
    def __init__(self, c1, c2, c3, c4, c5, c6, c7, c8, c9):
        CsPC08.__init__(self, c1, c2, c3, c4, c5, c6, c7, c8)
        self.C9 = c9


class CsPC10(CsPC09):
    def __init__(self, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10):
        CsPC09.__init__(self, c1, c2, c3, c4, c5, c6, c7, c8, c9)
        self.C10 = c10


class CsPC11(CsPC10):
    def __init__(self, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11):
        CsPC10.__init__(self, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10)
        self.C11 = c11


class CsPC12(CsPC11):
    def __init__(self, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12):
        CsPC11.__init__(self, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
        self.C12 = c12


class CsPC13(CsPC12):
    def __init__(self, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13):
        CsPC12.__init__(self, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12)
        self.C13 = c13
