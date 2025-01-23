from enum import Enum
from src.code.space.colorSpace import CsXYZ
from src.code.colorMath import CmMath


class OBSERVER(str, Enum):
    DEG2 = 'Deg2'
    DEG10 = 'Deg10'


class Illuminant:
    """
    ### Observer 2° & Standard Illuminants, e.g. D50, F11, ...
    ASTM E308-01 except B which comes from Wyszecki & Stiles, p. 769):
    """

    A_DEG2 = CsXYZ(1.09850, 1.00000, 0.35585)
    """
    Observer= 2°, Illuminant= A (CsXYZ)
    """

    B_DEG2 = CsXYZ(0.99072, 1.00000, 0.85223)
    """
    Observer= 2°, Illuminant= B (CsXYZ)
    """

    C_DEG2 = CsXYZ(0.98074, 1.00000, 1.18232)
    """
    Observer= 2°, Illuminant= C (CsXYZ)
    """

    D50_DEG2 = CsXYZ(0.96422, 1.00000, 0.82521)
    """
    Observer= 2°, Illuminant= D50, ISO 13655 (CsXYZ)
    """

    D55_DEG2 = CsXYZ(0.95682, 1.00000, 0.92149)
    """
    Observer= 2°, Illuminant= D55 (CsXYZ)
    """

    D65_DEG2 = CsXYZ(0.95047, 1.00000, 1.08883)
    """
    Observer= 2°, Illuminant= D65, ISO 13655 (CsXYZ)
    """

    D75_DEG2 = CsXYZ(0.94972, 1.00000, 1.22638)
    """
    Observer= 2°, Illuminant= D75 (CsXYZ)
    """

    E_DEG2 = CsXYZ(1.00000, 1.00000, 1.00000)
    """
    Observer= 2°, Illuminant= E, equal energy (CsXYZ)
    """

    F2_DEG2 = CsXYZ(0.99186, 1.00000, 0.67393)
    """
    Observer= 2°, Illuminant= F2 (CsXYZ)
    """

    F7_DEG2 = CsXYZ(0.95041, 1.00000, 1.08747)
    """
    Observer= 2°, Illuminant= F7 (CsXYZ)
    """

    F11_DEG2 = CsXYZ(1.00962, 1.00000, 0.64350)
    """
    Observer= 2°, Illuminant= F11 (CsXYZ)
    """

    D50_DEG10 = CsXYZ(0.9672, 1.000, 0.8143)
    """
    Observer= 10°, Illuminant= D50 (CsXYZ)
    """

    D55_DEG10 = CsXYZ(0.958, 1.000, 0.9093)
    """
    Observer= 10°, Illuminant= D55 (CsXYZ)
    """

    D65_DEG10 = CsXYZ(0.9481, 1.000, 1.073)
    """
    Observer= 10°, Illuminant= D65 (CsXYZ)
    """

    D75_DEG10 = CsXYZ(0.94416, 1.000, 1.2064)
    """
    Observer= 10°, Illuminant= D75 (CsXYZ)
    """

    @staticmethod
    def get_List():
        return {
            "A_DEG2": Illuminant.A_DEG2.to_json(),
            "B_DEG2": Illuminant.B_DEG2.to_json(),
            "C_DEG2": Illuminant.C_DEG2.to_json(),
            "D50_DEG2": Illuminant.D50_DEG2.to_json(),
            "D55_DEG2": Illuminant.D55_DEG2.to_json(),
            "D65_DEG2": Illuminant.D65_DEG2.to_json(),
            "D75_DEG2": Illuminant.D75_DEG2.to_json(),
            "E_DEG2": Illuminant.E_DEG2.to_json(),
            "F2_DEG2": Illuminant.F2_DEG2.to_json(),
            "F7_DEG2": Illuminant.F7_DEG2.to_json(),
            "F11_DEG2": Illuminant.F11_DEG2.to_json(),
            "D50_DEG10": Illuminant.D50_DEG10.to_json(),
            "D55_DEG10": Illuminant.D55_DEG10.to_json(),
            "D65_DEG10": Illuminant.D65_DEG10.to_json(),
            "D75_DEG10": Illuminant.D75_DEG10.to_json()
        }


class AdaptionBaseMatrix:

    XYZscaling = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]

    XYZscalingInvers = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]

    Bradford = [
        [ 0.8951,  0.2664, -0.1614],
        [-0.7502,  1.7135,  0.0367],
        [ 0.0389, -0.0685,  1.0296]]

    BradfordInvers = [
        [ 0.98699311, -0.14705423, 0.15996270],
        [ 0.43230528,  0.51836027, 0.04929123],
        [-0.00852866,  0.04004282, 0.96848673]]

    VonKries = [
        [ 0.4002400, 0.7076000, -0.0808100],
        [-0.2263000, 1.1653200,  0.0457000],
        [ 0.0000000, 0.0000000,  0.9182200]]

    VonKriesInvers = [
        [1.8599364, -1.1293816,  0.2198974],
        [0.3611914,  0.6388125, -0.0000064],
        [0.0000000,  0.0000000,  1.0890636]]

    @staticmethod
    def get_matrix(
        baseMatrix: list[list[float]],
        baseMatrixInvers: list[list[float]],
        srcXYZ: CsXYZ,
        dstXYZ: CsXYZ
    ) -> list[list[float]]:

        src = CmMath.matrix3x3_1x3(baseMatrix, srcXYZ.to_list())
        dst = CmMath.matrix3x3_1x3(baseMatrix, dstXYZ.to_list())

        dia = [
            [(dst[0] / src[0]), 0, 0],
            [0, (dst[1] / src[1]), 0],
            [0, 0, (dst[2] / src[2])]]

        return CmMath.matrix3x3_3x3(
            CmMath.matrix3x3_3x3(baseMatrixInvers, dia),
            baseMatrix
        )
