from src.code.space.colorSpace import CsXYZ
from src.code.space.colorConstants.illuminant import Illuminant


class SCTV:
    '''
    Spot Color Tonal Value (ISO 20654)
    '''

    def __init__(self, xyzp, xyzs):
        '''
        Initializes the SCTV object with the given substrate and spot ink solid XYZ values.

        Parameters:
        - xyzp: CsXYZ for the substrate
        - xyzs: CsXYZ for the spot ink solid

        Returns: None
        '''
        self.illuminant = Illuminant.D50_DEG2
        self.set_substrate(xyzp)
        self.set_solidink(xyzs)

    def set_substrate(self, value: CsXYZ) -> None:
        '''
        Sets the XYZp value and recalculates the SCTV value.
        * @param {*} value CsXYZ for the substrate
        '''
        self.XYZp = value
        self.vp = self.adapt_illuminant(self.XYZp)

    def set_solidink(self, value: CsXYZ) -> None:
        '''
        Sets the XYZs value and recalculates the SCTV value.
        * @param {*} value CsXYZ for the spot ink solid
        '''
        self.XYZs = value
        self.vs = self.adapt_illuminant(self.XYZs)

    def get_sctv(self, XYZt: CsXYZ) -> float:
        '''
        Calculates the SCTV value from the given XYZ values.
        - XYZt CsXYZ for the spot ink tone

        Returns: SCTV value for the given XYZ values
        '''

        Vs: CsXYZ = self.vs
        Vp: CsXYZ = self.vp
        Vt: CsXYZ = self.adapt_illuminant(XYZt)

        numerator: float = (Vt.X - Vp.X) ** 2 + (Vt.Y - Vp.Y) ** 2 + (Vt.Z - Vp.Z) ** 2
        denominator: float = (Vs.X - Vp.X) ** 2 + (Vs.Y - Vp.Y) ** 2 + (Vs.Z - Vp.Z) ** 2

        return 100 * ((numerator / denominator) ** 0.5)

    def adapt_illuminant(self, xyz: CsXYZ) -> CsXYZ:
        if not isinstance(xyz, CsXYZ):
            raise TypeError('Invalid input parameters. Expected CsXYZ objects.')

        def adaptValue(sptValue, nValue):
            if not isinstance(sptValue, (int, float)) or not isinstance(nValue, (int, float)):
                raise TypeError('Invalid input parameters. Expected numbers.')

            sptU = sptValue / nValue
            sptU = sptU ** (1 / 3) if sptU > (6 / 29) ** 3 else (941 / 108) * sptU + (4 / 29)
            return sptU * 116 - 16

        adapted_X = adaptValue(xyz.X, self.illuminant.X)
        adapted_Y = adaptValue(xyz.Y, self.illuminant.Y)
        adapted_Z = adaptValue(xyz.Z, self.illuminant.Z)

        return CsXYZ(adapted_X, adapted_Y, adapted_Z)

    @staticmethod
    def correction(sctvValue, targetValue):
        if sctvValue == 0:
            return 0

        return targetValue / sctvValue
