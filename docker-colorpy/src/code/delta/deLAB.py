from typing import Union
from src.code.space.colorSpace import CsLAB, CsLCH
from src.code.space.colorConverter import Cs_Lab2LCH


def delta_lightness(r: Union[CsLAB, CsLCH], s: Union[CsLAB, CsLCH]) -> float:
    '''
    Calculates the difference in lightness (DeltaL) between two colors in the LAB or LCH color space.
    :param r: The first color (either a CsLAB or CsLCH object).
    :param s: The second color (either a CsLAB or CsLCH object).
    :return: The difference in lightness (DeltaL) between the two colors.
    '''

    return r.L - s.L


def delta_chroma(r: Union[CsLAB, CsLCH], s: Union[CsLAB, CsLCH]) -> float:
    '''
    Calculates the difference in chroma between two colors in the CieLAB or LCH color space.

    :param r: A color object in the CieLAB or LCH color space.
    :param s: A color object in the CieLAB or LCH color space.
    :return: The difference in chroma between the two colors.
    '''

    def get_chroma(obj: Union[CsLAB, CsLCH]) -> float:
        """
        Returns the chroma of a color object in the CieLAB or LCH color space.

        :param obj: A color object in the CieLAB or LCH color space.
        :return: The chroma of the color object.
        """

        return obj.C if isinstance(obj, CsLCH) else obj.to_chroma()

    return abs(get_chroma(r) - get_chroma(s))


def delta_hue(r: Union[CsLAB, CsLCH], s: Union[CsLAB, CsLCH]) -> float:
    '''
    Delta Hue
    '''

    def get_hue(obj: Union[CsLAB, CsLCH]) -> float:
        """
        Returns the hue component of a CsLAB or CsLCH color.

        :param obj: The CsLAB or CsLCH color.
        :return: The hue component of the color as a float.
        """

        return obj.H if isinstance(obj, CsLCH) else Cs_Lab2LCH(obj).H

    return abs(get_hue(r) - get_hue(s))


def delta_lch(r: Union[CsLAB, CsLCH], s: Union[CsLAB, CsLCH]) -> dict[str, float]:
    '''
    Delta CsLCH
    '''

    return {
        "dL": delta_lightness(r, s),
        "dC": delta_chroma(r, s),
        "dH": delta_hue(r, s)
    }


def delta_lab(r: CsLAB, s: CsLAB) -> dict[str, float]:
    '''
    Delta CsLAB
    '''

    return {
        "dL": r.L - s.L,
        "dA": r.A - s.A,
        "dB": r.B - s.B
    }
