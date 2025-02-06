import math
from src.code.space.colorSpace import CsXYZ
from src.code.space.colorConverter import Cs_Spectral2Multi, CS_Spectral2XYZ
from src.code.predict.linearization.baselinearization import BaseLinearization


class SynLinSolidV3(BaseLinearization):
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
        
    def setCorrection(self, correction: float):
        self.correction = correction

    def start(self):
        
        cFactor = self.correction if hasattr(self, 'correction') else 4.5
        
        LENC: int = len(self.gradient)

        spectrals = [0.0] * LENC
        
        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -
        
        rRumMedia = sum(self.media)
        rRumSolid = sum(self.solid)
        
        invert = False
        if rRumMedia < rRumSolid :
            invert = True
            
        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -
            
        ksMedia = Optcolor.ksFromSnm(self.solid if invert else self.media)
        ksSolid = Optcolor.ksFromSnm(self.media if invert else self.solid)
        
        # generate initial concentration and corrections
        cSolid = [(c ** cFactor) for c in self.gradient]
        #cMedia = [(1 - c) for c in self.gradient] 
        cMedia = [(1 - c) for c in cSolid] 
        
        for i in range(LENC):
            ksMix = Optcolor.ksMixWithConcentrations( ksMedia, ksSolid, cMedia[i], cSolid[i] )
            spectrals[i] = Optcolor.ksToSnm(ksMix)
            
        colors = spectrals if not invert else spectrals[::-1]
        
        
        """ mixXYZ: list[CsXYZ] = [CS_Spectral2XYZ(spectrals[i]) for i in range(LENC)]
        sctvs: list[CsXYZ] = [Optcolor.calculate_sctv(mixXYZ[0], mixXYZ[i], mixXYZ[-1]) for i in range(LENC)] """

        return {
            "color": Cs_Spectral2Multi(colors),
            "ramp": self.gradient,
            "cMedia": cMedia,
            "cSolid": cSolid,
            #"sctvs": sctvs,
            "version": "MARS.3.001"
            
        }


class Optcolor:
    
    @staticmethod
    def ksFromSnm(snm: list[float]) -> list[float]:
        return [(1 - x) ** 2 / (2 * x) for x in snm]

    @staticmethod
    def ksFulltoneInk(ksMedia: list[float], ksSolid: list[float]) -> list[float]:
        return [ksSolid[i] - ksMedia[i] for i in range(len(ksMedia))]
        
    @staticmethod
    def ksMix(ksMedia: list[float], ksSolid: list[float], concentrations: float, correct: float) -> list[float]:
        return [ksMedia[i] * (1 - concentrations) + (ksSolid[i] * (concentrations) * correct) for i in range(len(ksMedia))]
    
    @staticmethod
    def ksMixWithConcentrations(ksMedia: list[float], ksSolid: list[float], cMedia: float, cSolid: float) -> list[float]:
        return [(ksMedia[i] * cMedia) + (ksSolid[i] * cSolid) for i in range(len(ksMedia))]

    @staticmethod
    def ksToSnm(ks: list[float]) -> list[float]:
        return [1 + x - math.sqrt(x ** 2 + 2 * x) for x in ks]
    
    @staticmethod
    def calculate_sctv(xyz_p: CsXYZ, xyz_t: CsXYZ, xyz_s: CsXYZ):
        numerator = (
            (xyz_t.X - xyz_p.X) ** 2 + 
            (xyz_t.Y - xyz_p.Y) ** 2 + 
            (xyz_t.Z - xyz_p.Z) ** 2
        )
        denominator = (
            (xyz_s.X - xyz_p.X) ** 2 + 
            (xyz_s.Y - xyz_p.Y) ** 2 + 
            (xyz_s.Z - xyz_p.Z) ** 2
        )
        return math.sqrt(numerator / denominator)
