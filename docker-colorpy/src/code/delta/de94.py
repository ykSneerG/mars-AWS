import math
from enum import Enum
from src.code.space.colorSpace import CsLAB


class WEIGHTINGFACTOR(Enum):
    '''
    Weighting factor for different applications, e.g. garphic arts or textile.
    '''
    GraphicArt = (1.000, 0.045, 0.015)
    Textile = (2.000, 0.048, 0.014)


def deltaE94(r: CsLAB, s: CsLAB, weightingFactor: WEIGHTINGFACTOR):
    '''
    Returns the DeltaE94 between a given reference LAB and another LAB color.
    ΔE (1994) is defined in the L*C*h* color space with differences in
    lightness, chroma and hue calculated from L*a*b* coordinates.
    '''
    KL, K1, K2 = weightingFactor.value
    KC = 1.0
    KH = 1.0

    rChroma = r.to_chroma()
    sChroma = s.to_chroma()

    SL = 1
    SC = 1 + K1 * rChroma
    SH = 1 + K2 * sChroma

    deltaL = r.L - s.L
    deltaA = r.A - s.A
    deltaB = r.B - s.B
    deltaC = rChroma - sChroma
    deltaH = math.sqrt(max(deltaA ** 2 + deltaB ** 2 - deltaC ** 2, 0))

    return math.sqrt(
        (deltaL / (KL * SL)) ** 2 +
        (deltaC / (KC * SC)) ** 2 +
        (deltaH / (KH * SH)) ** 2)
