from src.code.space.colorSpace import CsLAB
from src.code.delta.de00 import deltaE00
from src.code.delta.de76 import deltaE76
from src.code.delta.de94 import deltaE94, WEIGHTINGFACTOR
from src.code.delta.deCMC import deltaECMC


class ColorDiffernce:
    def __init__(self, labRef: CsLAB, labSam: CsLAB):
        if not isinstance(labRef, CsLAB) or not isinstance(labSam, CsLAB):
            raise TypeError("The parameter must be a CsLAB object.")

        self.LabRef = labRef
        self.LabSam = labSam

    def de00(self) -> float:
        return deltaE00(self.LabRef, self.LabSam)

    def de76(self) -> float:
        return deltaE76(self.LabRef, self.LabSam)

    def deCMC(self) -> float:
        return deltaECMC(self.LabRef, self.LabSam)

    def de94_graphic(self) -> float:
        return deltaE94(self.LabRef, self.LabSam, weightingFactor=WEIGHTINGFACTOR.GraphicArt)

    def de94_textile(self) -> float:
        return deltaE94(self.LabRef, self.LabSam, weightingFactor=WEIGHTINGFACTOR.Textile)
    
    def de94(self, weightingFactor: WEIGHTINGFACTOR) -> float:
        return deltaE94(self.LabRef, self.LabSam, weightingFactor=WEIGHTINGFACTOR.GraphicArt)
