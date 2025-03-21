import math
from src.code.space.colorSpace import CsLAB


def deltaE76(r: CsLAB, s: CsLAB) -> float:
    return math.sqrt(
          (r.L - s.L) ** 2 +
          (r.A - s.A) ** 2 +
          (r.B - s.B) ** 2
    )
