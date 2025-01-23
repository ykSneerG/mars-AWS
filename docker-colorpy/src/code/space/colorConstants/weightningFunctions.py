class Cmf2deg:
    """
    ### Weightning function: 380-730nm in 10nm steps for Illuminant D50 and Observer 2°
    ISO 13655
    """
    WX = [0.004, 0.012,
          0.060, 0.234, 0.775, 1.610, 2.453, 2.777, 2.500, 1.717, 0.861, 0.283,
          0.040, 0.088, 0.593, 1.590, 2.799, 4.207, 5.657, 7.132, 8.540, 9.255,
          9.835, 9.469, 8.009, 5.926, 4.171, 2.609, 1.541, 0.855, 0.434, 0.194,
          0.097, 0.050, 0.022, 0.012]
    WY = [0.000, 0.000,
          0.002, 0.006, 0.023, 0.066, 0.162, 0.313, 0.514, 0.798, 1.239, 1.839,
          2.948, 4.632, 6.587, 8.308, 9.197, 9.650, 9.471, 8.902, 8.112, 6.829,
          5.838, 4.753, 3.573, 2.443, 1.629, 0.984, 0.570, 0.313, 0.158, 0.070,
          0.035, 0.018, 0.008, 0.004]
    WZ = [0.019, 0.057,
          0.285, 1.113, 3.723, 7.862, 12.309, 14.647, 14.346, 11.299, 7.309, 4.128,
          2.466, 1.447, 0.736, 0.401, 0.196, 0.085, 0.037, 0.020, 0.015, 0.010,
          0.007, 0.004, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
          0.000, 0.000, 0.000, 0.000]


class Cmf10deg:
    """
    ### Weightning function: 380-730nm in 10nm steps for Illuminant D50 and Observer 10°
    R.W.G.Hunt / M.R.Pointer (Measuring Color 4th edition, p.404)
    """
    WX = [0.001, 0.002,
          0.059, 0.385, 1.087, 1.598, 2.556, 2.888, 2.437, 1.574, 0.630, 0.096,
          0.006, 0.284, 0.965, 2.101, 3.317, 4.745, 6.194, 7.547, 8.847, 9.218,
          9.712, 9.035, 7.465, 5.426, 3.713, 2.208, 1.289, 0.714, 0.338, 0.114,
          0.075, 0.035, 0.014, 0.008]
    WY = [0.000, 0.000,
          0.006, 0.040, 0.112, 0.190, 0.398, 0.675, 1.000, 1.469, 2.130, 2.715,
          3.842, 5.138, 6.500, 7.872, 8.532, 8.931, 8.780, 8.214, 7.557, 6.375,
          5.663, 4.597, 3.447, 2.366, 1.541, 0.882, 0.509, 0.279, 0.131, 0.056,
          0.029, 0.014, 0.005, 0.003]
    WZ = [0.002, 0.009,
          0.263, 1.751, 5.154, 7.864, 13.066, 15.511, 14.023, 10.623, 6.312, 3.227,
          1.796, 0.919, 0.501, 0.263, 0.114, 0.031, -0.003, 0.001, 0.000, 0.000,
          0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
          0.000, 0.000, 0.000, 0.000]
