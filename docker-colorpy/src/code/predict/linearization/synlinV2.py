import math
from src.code.space.colorSpace import CsSpectral, CsXYZ
from src.code.space.colorConstants.illuminant import OBSERVER
from src.code.space.colorConverter import (CS_Spectral2XYZ, Cs_Spectral2Multi)

from src.code.predict.linearization.baselinearization import BaseLinearization

class SynLinSolidV2(BaseLinearization):
    """
    Predict a linearization based an spectral data in a range of 380-730nm with 10nm steps.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def start(self):
        
        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -
        
        rRumMedia = sum(self.media)
        rRumSolid = sum(self.solid)
        
        invert = False
        if rRumMedia < rRumSolid :
            invert = True
            
        # - + - + - + - + CHECK THE LIGHT and DARK TENDENCIES - + - + - + - + -
            

        LENC: int = len(self.gradient)

        spectrals = [0.0] * LENC
        sctvs = [0.0] * LENC
        iter = [0] * LENC
        stop = [False] * LENC

        # ks media & solid
        
        if invert:
            ksMedia = Optcolor.ksFromSnm(self.solid)
            ksSolid = Optcolor.ksFulltoneInk(ksMedia, Optcolor.ksFromSnm(self.media))
        else:
            ksMedia = Optcolor.ksFromSnm(self.media)
            ksSolid = Optcolor.ksFulltoneInk(ksMedia, Optcolor.ksFromSnm(self.solid))
        
        """ ksMedia = Optcolor.ksFromSnm(self.media)
        ksSolid = Optcolor.ksFulltoneInk(ksMedia, Optcolor.ksFromSnm(self.solid)) """
        #ksSolid = Optcolor.ksFromSnm(self.solid)
        
        # generate initial concentration and corrections
        concentrations = self.gradient
        conccorections = [1.0] * LENC

        mixXYZ: list[CsXYZ] = [0.0] * LENC

        for r in range(self.maxLoops):

            for i in range(LENC):
                ksMix = Optcolor.ksMix( ksMedia , ksSolid , concentrations[i] * conccorections[i] )
                spectrals[i] = Optcolor.ksToSnm(ksMix)
                mixXYZ[i] = CS_Spectral2XYZ(spectrals[i], OBSERVER.DEG2)

            for i in range(LENC):
                if stop[i]:
                    continue

                sctvs[i] = Optcolor.calculate_sctv(mixXYZ[0], mixXYZ[i], mixXYZ[-1])
                conccorections[i] *= Optcolor.calculate_sctv_correction( sctvs[i], concentrations[i] )
                
                iter[i] = r
                stop[i] = self.in_tolerance(sctvs[i], concentrations[i])

            if all(stop):
                break

        colors = spectrals
        if invert:
            colors = spectrals[::-1]

        return {
            "color": Cs_Spectral2Multi(colors),
            "ramp": self.gradient,
            #"sctvs": sctvs,
            "iter": iter,
            "operations": sum(iter),
            "loops": max(iter),
            #"conccorections": conccorections,
            "invert": invert
        }


class Optcolor:
    
    @staticmethod
    def ksFromSnm(snm: list[float]) -> list[float]:
        return [(1 - x) ** 2 / (2 * x) for x in snm]

    @staticmethod
    def ksFulltoneInk(ksMedia: list[float], ksSolid: list[float]) -> list[float]:
        return [ksSolid[i] - ksMedia[i] for i in range(len(ksMedia))]
        
    @staticmethod
    def ksMix(ksMedia: list[float], ksSolid: list[float], concentrations: float) -> list[float]:
        return [ksMedia[i] + (ksSolid[i] * concentrations) for i in range(len(ksMedia))]

    @staticmethod
    def ksToSnm(ks: list[float]) -> list[float]:
        """ return [1 + x - math.sqrt(x ** 2 + 2 * x) for x in ks] """
        # This is a workaround to avoid negative values, but it is maybe not the correct way to do
        result = []
        for x in ks:
            if x < 0:
                result.append(1)
                continue
            result.append(1 + x - math.sqrt(x ** 2 + 2 * x))
        return result

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

    @staticmethod
    def calculate_sctv_correction(sctv_value, target_value):
        if sctv_value == 0:
            return 0
        return target_value / sctv_value
