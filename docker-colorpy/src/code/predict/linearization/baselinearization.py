class BaseLinearization:
    """
    Predict a linearization based an spectral data in a range of 380-730nm with 10nm steps.
    """

    def __init__(self, **kwargs):
        self.media = None
        """
        Spectral data for the media.
        """
        self.solid = None
        """
        Spectral data for the solid ink colors.
        """
        self.gradient: list = []
        """
        DCS values for the gradient.
        From 0 to 100% in x% steps.
        """

        """ self.places = 4 """

        self.tolerance: float = 0.5
        self.maxLoops = 100

    def set_media(self, media):
        self.media = media

    def set_solid(self, solid):
        self.solid = solid

    """ def set_places(self, places: int) -> None:
        self.places = places """

    def set_max_loops(self, max_loops: int) -> None:
        self.maxLoops = max_loops

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float):
        self._tolerance: float = value

    def set_gradient(self, gradient: list):
        self.gradient = gradient

    def set_gradient_by_steps(self, subdivision: int) -> None | dict:
        if subdivision <= 1:
            # raise ValueError("Subdivision must be greater than 1.")
            return "Subdivision must be greater than 1."

        size = 1 / (subdivision - 1)
        ramp = [i * size for i in range(subdivision)]

        self.set_gradient(ramp)

    def in_tolerance(self, actualValue, targetValue) -> bool:
        return abs(actualValue - targetValue) <= self.tolerance

    def start(self):
        pass
