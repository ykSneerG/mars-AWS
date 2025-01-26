from src.code.space.colorConverter import Cs_Spectral2Multi
from src.code.predict.linearization.baselinearization import BaseLinearization


class LinearInterpolation(BaseLinearization):
    """
    Predict a linearization based an spectral data in a range of 380-730nm with 10nm steps.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start(self):

        spectrals = InterpolateColor.mix_reflectance(
            self.media, self.solid, self.gradient
        )

        return {"color": Cs_Spectral2Multi(spectrals), "ramp": self.gradient}


class InterpolateColor:

    @staticmethod
    def mix_reflectance(
        reflectance_A: list[float],
        reflectance_B: list[float],
        concentration: list[float],
    ) -> list[list[float]]:
        return [
            InterpolateColor.mix_two_spectras(reflectance_A, reflectance_B, 1 - c, c)
            for c in concentration
        ]

    @staticmethod
    def mix_two_spectras(
        ra: list[float], rb: list[float], cA: float, cB: float
    ) -> list[float]:
        return [ra[i] * cA + rb[i] * cB for i in range(len(ra))]
