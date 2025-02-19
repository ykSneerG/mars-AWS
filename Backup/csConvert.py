from typing import overload
from src.code.space.colorConverter import Cs_XYZ2LAB
from src.code.space.colorConverter import CsLAB, CsXYZ
from src.code.space.colorConverter import Illuminant


@overload
def convert_XYZ2LAB(xyz: CsXYZ, refWhite: CsXYZ = Illuminant.D50_DEG2) -> CsLAB:
    return Cs_XYZ2LAB(xyz, refWhite)


@overload
def convert_XYZ2LAB(*xyz: CsXYZ, refWhite: CsXYZ = Illuminant.D50_DEG2) -> list[CsLAB]:
    return [convert_XYZ2LAB(color, refWhite) for color in xyz]
