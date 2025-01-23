import math


class KubelkaMunkSubstrate:
    """
    Kubelka-Munk algorithm in consideration of the substrate.
    Spectral reflectance is used as input and output.
    """

    def __init__(self) -> None:
        self.substrate: KSobject
        self.pigments: list[KSobject] = []

    def set_substrate(self, reflectance: list[float]) -> None:
        '''
        Set the substrate reflectance.
        '''
        try:
            ks_from_r = KubelkaMunkHelper.ks_from_reflectance(reflectance)
            self.substrate = KSobject(ks_from_r)
        except Exception as e:
            raise ValueError("An error occurred in the subclass") from e

    def add_paint(self, reflectance: list[float]) -> None:
        '''
        Add a pigment or ink reflectance.
        '''
        try:
            ks_from_r = KubelkaMunkHelper.ks_from_reflectance(reflectance)
            ks_netto = KubelkaMunkHelper.ks_from_fulltoneink(ks_from_r, self.substrate.ks)
            self.pigments.append(KSobject(ks_netto))
        except Exception as e:
            raise ValueError("An error occurred in the subclass") from e

    def mix(self, recipe: list[float]) -> list[float]:

        if (len(recipe) != len(self.pigments)):
            raise ValueError("The recipe must have the same length as the number of pigments")

        total_qty: float = sum(recipe)

        if (total_qty == 0):
            return KubelkaMunkHelper.reflectance_from_ks(self.substrate.ks)

        # lfc is a number of first colorant, add a type hint to avoid confusion
        lfc: int = self.len_first_colorant()
        ks: list[float] = [0.0] * lfc

        for i in range(len(recipe)):

            if recipe[i] <= 0:
                continue

            conc: float = KubelkaMunkHelper.concentration(recipe[i], total_qty)

            for j in range(lfc):
                ks[j] += self.pigments[i].ks[j] * conc

        ks = KubelkaMunkHelper.add_substrate(ks, self.substrate.ks)
        return KubelkaMunkHelper.reflectance_from_ks(ks)

    def len_first_colorant(self) -> int:
        return len(self.pigments[0].ks)


class KubelkaMunkHelper:

    @staticmethod
    def ks_from_reflectance(r: list[float]) -> list[float]:
        return [KubelkaMunkHelper.absorption(r[i]) / KubelkaMunkHelper.scattering(r[i]) for i in range(len(r))]

    @staticmethod
    def ks_from_fulltoneink(ksFulltone: list[float], ksSubstrate: list[float]) -> list[float]:
        return [ksSubstrate[i] - ksFulltone[i] for i in range(len(ksFulltone))]

    @staticmethod
    def concentration(qty: float, qty_total: float) -> float:
        '''
        normailze the quantity to the range [0, 1]
        '''
        return qty / qty_total

    @staticmethod
    def absorption(r: float) -> float:
        '''
        Returns the absorption of a pigment.
        '''
        return math.pow(1.0 - r, 2)

    @staticmethod
    def scattering(r: float) -> float:
        '''
        Returns the scattering of a pigment.
        '''
        return 2.0 * r

    @staticmethod
    def add_substrate(ksMedia: list[float], ksSolid: list[float]) -> list[float]:
        return [ksMedia[i] + ksSolid[i] for i in range(len(ksMedia))]

    @staticmethod
    def reflectance(ks: float) -> float:
        '''
        Returns the reflectance of a KS element.
        '''
        try:
            return 1.0 + ks - math.sqrt(ks * ks + (2.0 * ks))
        except Exception as e:
            raise Exception("An error occurred in the subclass - def reflectance(ks: float) -> float:", ks) from e

    @staticmethod
    def reflectance_from_ks(ks: list[float]) -> list[float]:
        '''
        Returns the reflectance of a KS array.
        '''
        # return [KubelkaMunkHelper.reflectance(ks[i]) for i in range(len(ks))]
        snm = []
        for i in range(len(ks)):
            if ks[i] < 0:
                tmp = 0.0
            else:
                tmp: float = 1 + ks[i] - math.sqrt(math.pow(ks[i], 2) + 2 * ks[i])
            snm.append(tmp)
        return snm


class KSobject:
    """
    A KSobject contains ...
    """

    def __init__(self, ks: list[float]) -> None:

        if (
            not isinstance(ks, list) or
            not all(isinstance(c, float) for c in ks) or
            len(ks) < 1
        ):
            raise TypeError("ks must be a list of floats")

        self.ks: list[float] = ks
        """
        Color of the pigment.
        """
