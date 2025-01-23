from src.code.predict.kubelkaMunk import KubelkaMunk
from src.code.space.colorConstants.illuminant import OBSERVER
from src.code.space.colorConverter import CS_Spectral2XYZ, Cs_XYZ2RGB, Cs_XYZ2LAB, Cs_Lab2LCH, Cs_XYZ2Denisty
# from space.colorSpace import CsXYZ, CsRGB, CsLAB, CsLCH
from src.code.linear.sctv import SCTV


class SynLinSolid:
    '''
    Predict a linearization based an spectral data in a range of 380-730nm with 10nm steps.
    '''

    def __init__(self, **kwargs):
        self.media = None
        '''
        Spectral data for the media.
        '''
        self.solid = None
        '''
        Spectral data for the solid ink colors.
        '''
        self.gradient: list = []
        '''
        DCS values for the gradient.
        From 0 to 100% in x% steps.
        '''

        self.gradientCorrected = []

        self.linearSolid = []
        self.linearSctv = []
        self.linearIters = []

        self.places = 4
        self.tolerance = 0.5

    def get_prediction(self):

        xyzs, hexs, labs, lchs, dens = self.convert_to_color_spaces(self.linearSolid)

        result = {
            'pcs': self.linearSolid,
            'dcs': self.gradient,
            'sctv': self.linearSctv,
            'dcsCorrected': self.gradientCorrected,
            'its': self.linearIters,
            'xyz': xyzs,
            'hex': hexs,
            'lab': labs,
            'lch': lchs,
            'density': dens
        }

        return result

    def set_media(self, media):
        self.media = media

    def set_solid(self, solid):
        self.solid = solid

    def set_places(self, places):
        self.places = places

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value

    def set_gradient(self, gradient: list):
        self.gradient = gradient
        self.gradientCorrected = self.gradient.copy()
        self.linearSctv = [None] * len(self.gradient)
        self.linearIters = [0] * len(self.gradient)
        self.linearSolid = [None] * len(self.gradient)

    def set_gradient_by_steps(self, subdivision) -> None:

        if (subdivision < 1):
            raise ValueError('Subdivision must be greater than 1.')

        gradient = [
                round(100 * i / (subdivision - 1), self.places)
                for i in range(subdivision)
            ]
        self.set_gradient(gradient)

    def calculate(self) -> None:

        if (len(self.gradient) == 0):
            raise ValueError('Gradient is empty.')

        if (self.solid is None):
            raise ValueError('Solid is empty.')

        if (self.media is None):
            raise ValueError('Media is empty.')

        # self.linearSolid = []
        # self.linearSctv = []

        xyzMedia = CS_Spectral2XYZ(self.media, OBSERVER.DEG2)
        xyzSolid = CS_Spectral2XYZ(self.solid, OBSERVER.DEG2)
        sctv = SCTV(xyzMedia, xyzSolid)

        for i, item in enumerate(self.gradientCorrected):
            if self.linearSctv[i] is not None and self.in_tolerance_sctv(self.linearSctv[i], self.gradient[i]):
                continue

            item = round(item, 6)
            km = KubelkaMunk()
            km.add_paint(self.media, float(100 - item))
            km.add_paint(self.solid, float(item))
            curve = km.mix()
            self.linearSolid[i] = [round(value, self.places) for value in curve]

            # Calculate SCTV
            xyz = CS_Spectral2XYZ(curve, OBSERVER.DEG2)
            resSctv = sctv.get_sctv(xyz)
            self.linearSctv[i] = round(resSctv, self.places)

            # Calculate the corrected gradient
            factor = SCTV.correction(resSctv, item)
            self.gradientCorrected[i] = self.gradient[i] * factor
            self.linearIters[i] += 1

    def calculate_loops(self, loops: int = 1) -> None:
        if (loops < 1):
            raise ValueError('Loops must be greater than 1.')

        for i in range(loops):
            self.calculate()

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

    def in_tolerance_sctv(self, actualValue, targetValue) -> bool:
        return abs(actualValue - targetValue) <= self.tolerance
