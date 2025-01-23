import math
from decimal import Decimal, ROUND_HALF_UP


class CmAngle:

    TWOPI = 2 * math.pi
    '''
    Constant - Double PI (6,2831853...)
    '''

    @staticmethod
    def RadianToDegree(rad: float) -> float:
        '''
        Convert rad to degree.
        '''
        return 360 * (rad / CmAngle.TWOPI)

    @staticmethod
    def DegreeToRadian(deg: float) -> float:
        '''
        Convert degree to rad.
        '''
        return CmAngle.TWOPI * (deg / 360.0)

    @staticmethod
    def SinDeg(degree: float) -> float:
        '''
        Compute sine of a given angle in degrees

        :param degree: angle in degrees
        '''
        return math.sin(CmAngle.DegreeToRadian(degree))

    @staticmethod
    def CosDeg(degree: float) -> float:
        '''
        Compute cosine of a given angle in degrees

        :param degree: angle in degrees
        '''
        return math.cos(CmAngle.DegreeToRadian(degree))


class CmMath:
    """
    Colormath
    """
    @staticmethod
    def clamp(current: float, down: float, up: float) -> float:
        """
        Returns the current value clamped between the down and up values.
        """
        return max(down, min(up, current))

    @staticmethod
    def clampToInt(current: float, down: int, up: int) -> int:
        """
        Returns the current value rounded to the nearest integer and clamped between the down and up values.
        """
        rounded = Decimal(current).to_integral_value(rounding=ROUND_HALF_UP)
        return int(max(down, min(up, rounded)))

    @staticmethod
    def matrix3x3_1x3(mx3x3: list[list[float]], vec: list[float]) -> list[float]:
        """
        Multiplies a 3x3 matrix with a 1x3 vector and returns a 1x3 vector.
        """
        return [
            sum([mx3x3[i][j] * vec[j] for j in range(3)])
            for i in range(3)
        ]

    @staticmethod
    def matrix3x3_3x3(matrix1: list[list[float]], matrix2: list[list[float]]) -> list[list[float]]:
        """
        Multiplies two 3x3 matrices and returns a 3x3 matrix.
        """
        return [
            [sum([matrix1[i][k] * matrix2[k][j] for k in range(3)]) for j in range(3)]
            for i in range(3)
        ]

    @staticmethod
    def round_list(lst: list, num_decimals: int) -> list:
        """
        Rounds each element of a list to the specified number of decimals.
        """
        if isinstance(lst[0], list):
            # If lst contains lists, recursively round each element
            return [CmMath.round_list(sub_lst, num_decimals)
                    for sub_lst in lst]
        else:
            # Otherwise, round each element using round() function
            return [round(elem, num_decimals)
                    for elem in lst]

    @staticmethod
    def toNum(value: str):
        '''
        Check if the value is a number, and convert to float or int.
        Otherwise it will return the given string.
        '''
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value

    @staticmethod
    def inRange(arr: list[float], min_vals: list[float], max_vals: list[float]) -> bool:
        """
        #### Check if all array elements are within the given range.

        Parameters:
        :param arr (list): The array to check.
        :param min_vals (list): The minimum values of the ranges.
        :param max_vals (list): The maximum values of the ranges.

        Returns:
        bool: True if all elements are within range, False otherwise.
        """
        for i in range(len(arr)):
            if not (min_vals[i] <= arr[i] <= max_vals[i]):
                return False
        return True

    @staticmethod
    def check_same_value(lst):
        return all(elem == lst[0] for elem in lst)
