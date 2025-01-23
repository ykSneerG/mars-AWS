import math
import sys
from src.code.predict.pigment import Pigment


class KubelkaMunk:
    """
    Kubelka-Munk algorithm
    """

    def __init__(self) -> None:

        self.substrate: Pigment

        self.pigments: list[Pigment] = []

        self.k1: float = 0.0
        """
        Saunderson_s correction variable k1
        """

        self.k2: float = 0.0
        """
        Saunderson_s correction variable k2
        """

    def set_k1(self, custom_k1: float) -> None:
        if not isinstance(custom_k1, float):
            raise TypeError("The parameter must be float.")
        self.k1 = custom_k1

    def set_k2(self, custom_k2: float) -> None:
        if not isinstance(custom_k2, float):
            raise TypeError("The parameter must be float.")
        self.k2 = custom_k2

    def add_media(self, reflectance: list[float]) -> None:
        try:
            self.substrate = Pigment(reflectance)
        except Exception as e:
            raise ValueError("An error occurred in the subclass") from e

    def add_paint(self, reflectance: list[float], quantity: float) -> None:
        try:
            self.pigments.append(Pigment(reflectance, quantity))
        except Exception as e:
            raise ValueError("An error occurred in the subclass") from e

    def mix(self) -> list[float]:

        # rs = KubelkaMunk.saunderson($this->substrate.pig, self.k1, self.k2)

        lfc = self.len_first_colorant()

        total_qty = self.sum_pigments_quantity()

        ks: list[float] = [0.0] * lfc

        for pigment in self.pigments:

            if pigment.quantity <= 0:
                continue

            conc: float = KubelkaMunk.concentration(pigment.quantity, total_qty)

            for i in range(lfc):
                k: float = KubelkaMunk.absorption(pigment.colorant[i])
                s: float = KubelkaMunk.scattering(pigment.colorant[i])

                ks[i] += (k / s) * conc

        return KubelkaMunk.reflectance_from_ks(ks)
        # return KubelkaMunk.saundersonInvers(r, self.k1, self.k2)

    def mix_saunderson(self) -> list[float]:

        lfc = self.len_first_colorant()
        k: list[float] = [0.0] * lfc
        s: list[float] = [0.0] * lfc
        ks: list[float] = [0.0] * lfc

        saunderson: bool = self.use_saunderson_correction()

        if saunderson:
            k = KubelkaMunk.saunderson(self.substrate.colorant, self.k1, self.k2)
            s = k

        total_qty = self.sum_pigments_quantity()

        for pigment in self.pigments:

            if pigment.quantity <= 0:
                continue

            conc: float = KubelkaMunk.concentration(pigment.quantity, total_qty)

            for i in range(lfc):
                k[i] += KubelkaMunk.absorption(pigment.colorant[i])
                s[i] += KubelkaMunk.scattering(pigment.colorant[i])

                ks[i] += (k[i] / s[i]) * conc

        res: list[float] = KubelkaMunk.reflectance_from_ks(ks)

        if saunderson:
            return KubelkaMunk.saunderson_invers(res, self.k1, self.k2)

        return res

    def use_saunderson_correction(self):
        if (
            len(self.substrate.colorant) == self.len_first_colorant() and
            self.substrate.quantity >= 0.0
        ):
            return True
        return False

    def sum_pigments_quantity(self):
        return sum(pigment.quantity for pigment in self.pigments)

    def len_first_colorant(self) -> int:
        return len(self.pigments[0].colorant)

    @staticmethod
    def concentration(qty: float, qty_total: float) -> float:
        return qty / qty_total

    @staticmethod
    def absorption(r: float) -> float:
        """
        Returns the absorption of a pigment.
        """
        return math.pow(1.0 - r, 2)

    @staticmethod
    def scattering(r: float) -> float:
        return 2.0 * r

    @staticmethod
    def reflectance(ks: float) -> float:
        return 1.0 + ks - math.sqrt(ks * ks + (2.0 * ks))

    @staticmethod
    def reflectance_from_ks(ks: list[float]) -> list[float]:
        return [KubelkaMunk.reflectance(ks[i]) for i in range(len(ks))]

    @staticmethod
    def saunderson(rx: list[float], k1: float, k2: float) -> list[float]:
        """
        Saundersons Correction of the spectral reflectance maesurements,
        an extension of the Kubelka-Munk algorithm

        Args:
            rx (list[float]): Refectance data array with n-items between 0-1.
            k1 (float): Correction factor k1, e.g. 0.03
                        (needs to be calculated, the value that gives the
                        lowest delta between predicted and measured - TBD)
            k2 (float): Correction factor k2, e.g. 0.60
                        (needs to be calculated, the value that gives the
                        lowest delta between predicted and measured - TBD)

        Returns:
            list[float]: reflectance curve
        """
        return [max(sys.float_info.min, (r - k1) / (1 - k1 - k2 + k2 * r)) for r in rx]

    @staticmethod
    def saunderson_invers(rx: list[float], k1: float, k2: float) -> list[float]:
        """
        Inverse of Saunderson's correction

        Args:
            rx (list[float]): input reflectance curve
            k1 (float): Correction factor k1
            k2 (float): Correction factor k2

        Returns:
            list[float]: reflectance curve
        """
        return [(k1 + ((1 - k1) * (1 - k2) * r) / (1 - k2 * r)) for r in rx]
