import math
import statistics
from src.code.space.colorSpace import CsLAB
from src.code.colorMath import CmAngle


# declare constants
KH = 1
KL = 1
KC = 1


def deltaE00(r: CsLAB, s: CsLAB) -> float:
    """
    r Reference color
    s Sample color
    Implemented according to:
    Sharma, Gaurav; Wencheng Wu, Edul N. Dalal (2005).
    "The CIEDE2000 color-difference formula: Implementation notes,
    supplementary test data, and mathematical observations"
    (http://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf)
    """

    labL = [r.L, s.L]
    labA = [r.A, s.A]
    labB = [r.B, s.B]

    # 1. Calculate C_prime, h_prime
    a_prime = prime_a(labA, labB)
    C_prime = prime_c(a_prime, labB)
    h_prime = prime_h(a_prime, labB)

    # 2. Calculate dL_prime, dC_prime, dH_prime
    dL_prime = labL[1] - labL[0]
    dC_prime = C_prime[1] - C_prime[0]
    dH_prime = 2 * math.sqrt(C_prime[0] * C_prime[1]) \
        * CmAngle.SinDeg(prime_dh(C_prime, h_prime) * 0.5)

    # 3. Calculate CIEDE2000 Color-Difference dE00
    L_prime_mean = statistics.mean(labL)
    C_prime_mean = statistics.mean(C_prime)
    h_prime_mean = prime_mean_h(h_prime, C_prime)

    T = 1 - 0.17 * CmAngle.CosDeg(h_prime_mean - 30) \
        + 0.24 * CmAngle.CosDeg(2 * h_prime_mean) \
        + 0.32 * CmAngle.CosDeg(3 * h_prime_mean + 6) \
        - 0.20 * CmAngle.CosDeg(4 * h_prime_mean - 63)

    dTheta = 30 * math.exp(-pow(((h_prime_mean - 275) / 25), 2))
    R_C = 2 * math.sqrt(pow(C_prime_mean, 7) / (pow(C_prime_mean, 7) + pow(25, 7)))
    S_L = 1 + (0.015 * pow(L_prime_mean - 50, 2)) / math.sqrt(20 + pow(L_prime_mean - 50, 2))
    S_C = 1 + 0.045 * C_prime_mean
    S_H = 1 + 0.015 * C_prime_mean * T
    R_T = -CmAngle.SinDeg(2 * dTheta) * R_C

    return math.sqrt(
            (dL_prime / (KL * S_L)) ** 2
            + (dC_prime / (KC * S_C)) ** 2
            + (dH_prime / (KH * S_H)) ** 2
            + R_T * (dC_prime / (KC * S_C)) * (dH_prime / (KH * S_H))
        )


def prime_a(a: list[float], b: list[float]) -> list[float]:
    C_ab = [math.hypot(a[i], b[i]) for i in range(len(a))]
    C_ab_mean = statistics.mean(C_ab)
    G = 1 + (0.5 * (1 - math.sqrt(C_ab_mean ** 7 / (C_ab_mean ** 7 + 25 ** 7))))

    return [G * a[i] for i in range(len(a))]


def prime_c(a_prime: list[float], b: list[float]) -> list[float]:
    """
    Calculates the hypotenuse of a right triangle given the lengths of its two legs.

    Args:
        a_prime (List[float]): List of floats representing the lengths of the legs of the right triangle.
        b (List[float]): List of floats representing the lengths of the legs of the right triangle.

    Returns:
        List[float]: List of floats representing the length of the hypotenuse of the right triangle.
    """
    return [math.hypot(a_prime[i], b[i]) for i in range(len(a_prime))]


def prime_h(a_prime: list[float], b: list[float]) -> list[float]:
    res = [0, 0]

    for i in range(len(res)):
        if b[i] != 0 and a_prime[i] != 0:
            hue = math.degrees(math.atan2(b[i], a_prime[i]))
            res[i] = hue if hue >= 0 else hue + 360

    return res


def prime_dh(C_prime: list[float], h_prime: list[float]) -> float:
    if C_prime[0] * C_prime[1] == 0.0:
        return 0

    delta_h = (h_prime[1] - h_prime[0]) % 360
    return delta_h if delta_h <= 180 else delta_h - 360


def prime_mean_h(h_prime: list[float], C_prime: list[float]) -> float:
    if C_prime[0] * C_prime[1] == 0.0:
        return sum(h_prime)

    if abs(h_prime[0] - h_prime[1]) > 180:
        return statistics.mean([h_prime[0], h_prime[1] + 360 if sum(h_prime) < 360 else h_prime[1] - 360])

    return statistics.mean(h_prime)
