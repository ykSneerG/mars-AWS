class MatrixRGB2XYZ:
    '''
    RGB to XYZ Matrices
    '''

    SRGB_D50 = [
        [ 3.1338561, -1.6168667, -0.4906146],
        [-0.9787684,  1.9161415,  0.0334540],
        [ 0.0719453, -0.2289914,  1.4052427]]

    SRGB_D65 = [
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]]


class MatrixXYZ2RGB:
    '''
    XYZ to RGB Matrices
    '''

    SRGB_D50 = [
        [0.4360747, 0.3850649, 0.1430804],
        [0.2225045, 0.7168786, 0.0606169],
        [0.0139322, 0.0971045, 0.7141733]]

    SRGB_D65 = [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]]
