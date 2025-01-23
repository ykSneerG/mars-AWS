class DotGainMurrayDavies:
    """
    This class calculates the tonal value based on the density of the media and solid.
    """
    def __init__(self, densityMedia: float, densitySolid: float) -> None:
        self.densityMedia = densityMedia
        self.densitySolid = densitySolid
        self.maxAreaValue = 100.0
        self.yuleNielsen = 1.0

    def get_relative_tonal_value(self, densityHalftone: float) -> float:

        # relative optical density of the halftone
        rodHalft: float = densityHalftone - self.densityMedia

        # relative optical density of the solid ink
        rodSolid: float = self.densitySolid - self.densityMedia

        return (1 - pow(10, -rodHalft)) / (1 - pow(10, -rodSolid)) * self.maxAreaValue

    def get_tonal_value(self, densityHalftone: float) -> float:

        return (1 - pow(10, -densityHalftone)) / (1 - pow(10, -self.densitySolid)) * self.maxAreaValue

    def get_dot_area_YN(self, densityHalftone: float) -> float:
        '''
        %Dot Area = 100(1 – 10–Dtp/n)/(1 – 10–Dsp/n)
        '''

        return (1 - pow(10, -densityHalftone / self.yuleNielsen)) / (1 - pow(10, -self.densitySolid / self.yuleNielsen)) * self.maxAreaValue

    def get_percentage_dot_area_YN(self, densityHalftone: float) -> float:
        '''
        My version of the formula: %Dot Area = 100(1 – 10–Dtp/n)/(1 – 10–Dsp/n)
        with a relative optical density of the halftone and solid ink
        '''

        # relative optical density of the halftone
        rodHalft: float = densityHalftone - self.densityMedia

        # relative optical density of the solid ink
        rodSolid: float = self.densitySolid - self.densityMedia

        return (1 - pow(10, -rodHalft / self.yuleNielsen)) / (1 - pow(10, -rodSolid / self.yuleNielsen)) * self.maxAreaValue
