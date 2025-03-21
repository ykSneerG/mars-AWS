import math
from src.code.space.colorSpace import CsLAB


def deltaECMC(r: CsLAB, s: CsLAB, CMC_L: float = 2, CMC_C: float = 1) -> float:
    """
    Calculates the color difference between two colors in the CMC color space.

    Parameters:
    :param r (CsLAB): The reference color.
    :param s (CsLAB): The sample color.
    :param CMC_L (float): The lightness factor. Default is 2.
    :param CMC_C (float): The chroma factor. Default is 1.

    Returns:
    :return: The color difference between the two colors.
    """
    rChroma = r.to_chroma()
    sChroma = s.to_chroma()

    deltaL = r.L - s.L
    deltaA = r.A - s.A
    deltaB = r.B - s.B
    deltaC = rChroma - sChroma
    deltaH = math.sqrt(max(deltaA**2 + deltaB**2 - deltaC**2, 0))

    SL = 0.511 if r.L < 16 else (0.040975 * r.L) / (1 + 0.01765 * r.L)

    SC = (0.0638 * rChroma) / (1 + 0.0131 * rChroma) + 0.638

    F = math.sqrt(pow(rChroma, 4) / (pow(rChroma, 4) + 1900))

    if r.A == 0 or r.B == 0:
        H = 0  # Either r.A or r.B is zero
    else:
        H = math.atan(r.B / r.A)

    H1 = H if H >= 0 else H + 360

    T = 0
    if 164 <= H1 and H1 <= 345:
        T = 0.56 + abs(0.2 * math.cos(H1 + 168))
    else:
        T = 0.36 + abs(0.4 * math.cos(H1 + 35))

    SH = SC * (F * T + 1 - F)

    return math.sqrt(
        (deltaL / (CMC_L * SL)) ** 2 +
        (deltaC / (CMC_C * SC)) ** 2 +
        (deltaH / SH) ** 2
    )
