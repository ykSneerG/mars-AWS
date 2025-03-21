from src.code.delta.colorDifference import ColorDiffernce
from src.code.space.colorSpace import CsLAB
from src.code.space.colorConverterNumpy import ColorTrafoNumpy as ctn
from src.code.space.colorConstants.illuminant import OBSERVER, Illuminant

from src.handlers import BaseLambdaHandler


class Trafo_Delta_Handler(BaseLambdaHandler):

    def handle(self):

        labRef = self.event["refLAB"]
        labSam = self.event["samLAB"]

        csLabRef = CsLAB(labRef["L"], labRef["A"], labRef["B"])
        csLabSam = CsLAB(labSam["L"], labSam["A"], labSam["B"])

        delta = ColorDiffernce(csLabRef, csLabSam)

        jd = {
            "de00": delta.de00(),
            "de76": delta.de76(),
            "deCMC": delta.deCMC(),
            "de94graphic": delta.de94_graphic(),
            "de94textile": delta.de94_textile(),
        }

        return self.get_common_response(jd)


class Trafo_ConvertSpectral_Handler(BaseLambdaHandler):

    def handle(self):
        
        """ 
        const body = {
            "SNM": snm,
            "observer": observer,
            "illuminant": illuminant,
            "inclSourceValues": true,
            "inclLAB": true,
            "inclLCH": true,
            "inclXYZ": true,
            "inclRGB": false,
            "inclDensity": false,
            "inclHEX": true
        }
        """

        spectral = self.event["SNM"]
        
        usr_observer = OBSERVER.DEG10 if self.event.get("observer") == "Deg10" else OBSERVER.DEG2
                
        usr_illuminant = Illuminant.find_illuminant(self.event.get("illuminant"), usr_observer)

        
        colortrafo = ctn(usr_observer, usr_illuminant)
        colortrafo._set_illuminant(usr_illuminant)
        
        dst_values = {
            "XYZ": True,
            "HEX": True,
            "LAB": True,
            "LCH": True
        }
        result = colortrafo.Cs_SNM2MULTI(spectral, dst_values)
        
        return self.get_common_response(result)

