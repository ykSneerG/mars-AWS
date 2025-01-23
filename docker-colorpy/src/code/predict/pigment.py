class Pigment:
    """
    A Pigment contains a colorant, e.g. RGB, XYZ, reflectance curve and a quantity
    """

    def __init__(self, colorant: list[float], quantity: float = 0.0) -> None:

        if (
            not isinstance(colorant, list) or
            not all(isinstance(c, float) for c in colorant) or
            len(colorant) < 1
        ):
            raise TypeError("colorant must be a list of floats")

        if (
            not isinstance(quantity, float) or
            quantity < 0
        ):
            raise TypeError("quantity must be a non-negative float")

        self.colorant: list[float] = colorant
        """
        Color of the pigment.
        """

        self.quantity: float = quantity
        """
        Quantity of the pigment.
        """
