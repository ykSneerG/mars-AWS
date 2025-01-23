import math

from src.code.colorMath import CmMath
from src.code.space.colorSpace import CsLAB, CsLCH, CsXYZ, CsRGB
from src.code.space.colorConstants.illuminant import OBSERVER, Illuminant
from src.code.space.colorConstants.matrixRGB import MatrixRGB2XYZ, MatrixXYZ2RGB
from src.code.space.colorConstants.weightningFunctions import Cmf2deg, Cmf10deg


CIE_E = 0.008856
"""
CIE constants / Actual CIE standard
"""
CIE_K = 903.3
"""
CIE constants / Actual CIE standard
"""


def Cs_Lab2LCH(lab: CsLAB) -> CsLCH:
    """
    Convert a CsLAB color to a CsLCH color.

    :param lab: The CsLAB color to convert.
    :return: The resulting CsLCH color.
    """

    '''
    hue: float = math.degrees(math.atan2(lab.B, lab.A))
    hue: float = (hue + 360) % 360
    '''

    hue: float = math.degrees(math.atan2(lab.B, lab.A)) % 360

    return CsLCH(lab.L, lab.to_chroma(), hue)


def Cs_LCH2Lab(lch: CsLCH) -> CsLAB:
    """
    Convert a CsLCH color to a CsLAB color.

    :param lch: The CsLCH color to convert.
    :return: The converted CsLAB color.
    """
    labA: float = lch.C * math.cos(math.radians(lch.H))
    labB: float = lch.C * math.sin(math.radians(lch.H))

    return CsLAB(lch.L, labA, labB)


def Cs_XYZ2LAB(xyz: CsXYZ, refWhite: CsXYZ = Illuminant.D50_DEG2) -> CsLAB:
    """
    Convert a CsXYZ color to a CsLAB color.

    :param xyz: The CsXYZ color to convert.
    :param refWhite: The reference white for the conversion as a normalized CsXYZ color.
    :return: The converted CsLAB color.
    """

    normalized_xyz: CsXYZ = xyz.normalize()

    rXYZ: list[float] = [
        normalized_xyz.X / refWhite.X,
        normalized_xyz.Y / refWhite.Y,
        normalized_xyz.Z / refWhite.Z
    ]

    for i in range(len(rXYZ)):
        if (rXYZ[i] > CIE_E):
            rXYZ[i] = pow(rXYZ[i], 1 / 3)
        else:
            rXYZ[i] = (CIE_K * rXYZ[i] + 16) / 116.0

    labL: float = 116.0 * rXYZ[1] - 16.0
    labA: float = 500.0 * (rXYZ[0] - rXYZ[1])
    labB: float = 200.0 * (rXYZ[1] - rXYZ[2])

    return CsLAB(labL, labA, labB)


def Cs_Lab2XYZ(lab: CsLAB, refWhite: CsXYZ) -> CsXYZ:
    """
    Convert a CsLAB color to a CsXYZ color.

    :param lab: The CsLAB color to convert.
    :param refWhite: The reference white for the conversion as a normalized CsXYZ color.
    :return: The converted CsXYZ color.
    """

    y: float = (lab.L + 16) / 116
    x: float = y + lab.A / 500
    z: float = y - lab.B / 200

    y = pow(y, 3) if (lab.L > CIE_E * CIE_K) else lab.L / CIE_K
    x = pow(x, 3) if (pow(x, 3) > CIE_E) else (116 * x - 16) / CIE_K
    z = pow(z, 3) if (pow(z, 3) > CIE_E) else (116 * z - 16) / CIE_K

    return CsXYZ(
        refWhite.X * x,
        refWhite.Y * y,
        refWhite.Z * z
    ).denormalize()


def Cs_XYZ2Denisty(xyz: CsXYZ) -> float:
    """
    Convert a CsXYZ color to a density value.

    :param xyz: The CsXYZ color to convert.
    :return: The calculated denisty value.
    """

    return -math.log10(xyz.Y / 100)


def Cs_XYZ2RGB(xyz: CsXYZ) -> CsRGB:
    """
    Convert a CsXYZ color to a CsRGB color.

    :param xyz: The CsXYZ color to convert.
    :return: The converted CsRGB color.
    """

    # linear RGB
    varRGB: list[float] = CmMath.matrix3x3_1x3(MatrixRGB2XYZ.SRGB_D50, xyz.normalize().to_list())

    # companded RGB
    normalized_rgb: CsRGB = __sRGBcompanding(CsRGB(varRGB[0], varRGB[1], varRGB[2]))

    return normalized_rgb.denormalize().clamp()


def Cs_RGB2XYZ(rgb: CsRGB) -> CsXYZ:
    """
    Convert a CsRGB color to a XYZ color.

    :param rgb: The CsRGB color to convert.
    :return: The converted CsXYZ color.
    """

    # Convert sRGB to linear RGB
    linear_rgb: CsRGB = __SRGBcompandingInvers(rgb.normalize())

    # Apply the inverse of the SRGB_D50 matrix to get XYZ
    normalized_xyz: list[float] = CmMath.matrix3x3_1x3(MatrixXYZ2RGB.SRGB_D50, linear_rgb.to_list())

    return CsXYZ(normalized_xyz[0], normalized_xyz[1], normalized_xyz[2]).denormalize()


def CS_Spectral2XYZ(curveNM: list[float], observer: OBSERVER = OBSERVER.DEG2) -> CsXYZ:
    """
    Spectral Data (380-730nm in 10nm steps) to XYZ Color (default: D50, 2 deg.)

    :param curveNM: array with 36 wavelengths (380-730nm in 10nm steps)
    :param observer: "Deg2" or "Deg10" specifies the color matching functions.
    """

    if observer == OBSERVER.DEG2:
        cmf = Cmf2deg
    elif observer == OBSERVER.DEG10:
        cmf = Cmf10deg
    else:
        raise ValueError("observer must be '2deg' or '10deg'")

    xyz = CsXYZ(0.0, 0.0, 0.0)

    for i, wavelength in enumerate(curveNM):
        xyz.X += cmf.WX[i] * wavelength
        xyz.Y += cmf.WY[i] * wavelength
        xyz.Z += cmf.WZ[i] * wavelength

    return xyz


# -x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.

def CS_Spectral2LAB(curveNM: list[float], observer: OBSERVER = OBSERVER.DEG2) -> CsLAB:
    cs_xyz = CS_Spectral2XYZ(curveNM, observer)
    cs_lab = Cs_XYZ2LAB(cs_xyz)
    return cs_lab


def CS_Spectral2LCH(curveNM: list[float], observer: OBSERVER = OBSERVER.DEG2) -> CsLCH:
    cs_xyz = CS_Spectral2XYZ(curveNM, observer)
    cs_lab = Cs_XYZ2LAB(cs_xyz)
    cs_lch = Cs_Lab2LCH(cs_lab)
    return cs_lch


def CsXYZ2LCH(xyz: CsXYZ, refWhite: CsXYZ = Illuminant.D50_DEG2) -> CsLCH:
    cs_xyz = Cs_XYZ2LAB(xyz, refWhite)
    cs_lch = Cs_Lab2LCH(cs_xyz)
    return cs_lch


# -x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.


def __sRGBcompanding(linearRGB: CsRGB) -> list[float]:
    """
    sRGB companding to delinearize a given CsRGB value.

    :param rgb: CsRGB color
    :return: new CsRGB
    """
    tmp: list[float] = [
            1.055 * math.pow(linrgb, 1.0 / 2.4) - 0.055
            if linrgb > 0.0031308
            else linrgb * 12.92
            for linrgb in linearRGB.to_list()
            ]

    return CsRGB(tmp[0], tmp[1], tmp[2])


def __SRGBcompandingInvers(rgb: CsRGB) -> CsRGB:
    """
    Inverse sRGB companding to linearize a given CsRGB value.

    :param rgb: CsRGB color
    :return: new CsRGB
    """

    sRGB: list[float] = [rgb.R, rgb.G, rgb.B]
    for i in range(3):
        if (sRGB[i] > 0.04045):
            sRGB[i] = math.pow((sRGB[i] + 0.055) / 1.055, 2.4)
        else:
            sRGB[i] /= 12.92
    return CsRGB(sRGB[0], sRGB[1], sRGB[2])

# -x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.-x-.
