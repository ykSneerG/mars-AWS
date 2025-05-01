import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

import matplotlib.pyplot as plt

import sys
import os

# Füge das Wurzelverzeichnis zum Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.code.space.colorConverterNumpy import ColorTrafoNumpy

""" from "opt/venv/docker-colorpy/src/code/space/colorConverterNumpy" import ColorTrafoNumpy """
# from opt/venv/bin:$PATH"

""" 
data = [
    [
        0.1291,
        0.1993,
        0.431,
        0.6716,
        0.8103,
        0.8661,
        0.8895,
        0.8996,
        0.9139,
        0.9281,
        0.9373,
        0.9447,
        0.951,
        0.956,
        0.9615,
        0.9641,
        0.9668,
        0.9686,
        0.9676,
        0.9713,
        0.9711,
        0.9719,
        0.9723,
        0.9727,
        0.9723,
        0.9719,
        0.9732,
        0.9748,
        0.9754,
        0.9756,
        0.9749,
        0.9739,
        0.9751,
        0.9757,
        0.9741,
        0.9751,
    ],
    [
        0.1287,
        0.1987,
        0.4285,
        0.6641,
        0.7951,
        0.8454,
        0.8659,
        0.8733,
        0.8825,
        0.8888,
        0.8887,
        0.8845,
        0.8775,
        0.8682,
        0.8574,
        0.8439,
        0.829,
        0.8126,
        0.7948,
        0.7776,
        0.7598,
        0.7427,
        0.7264,
        0.7108,
        0.6967,
        0.6841,
        0.6734,
        0.6647,
        0.657,
        0.6504,
        0.6448,
        0.6403,
        0.6382,
        0.6352,
        0.6336,
        0.6371,
    ],
    [
        0.1274,
        0.1962,
        0.4194,
        0.6379,
        0.7484,
        0.7882,
        0.8045,
        0.8081,
        0.8105,
        0.8082,
        0.7984,
        0.7833,
        0.7648,
        0.7439,
        0.7211,
        0.6955,
        0.6682,
        0.6394,
        0.6098,
        0.5812,
        0.5533,
        0.5272,
        0.503,
        0.4808,
        0.4611,
        0.444,
        0.4298,
        0.4184,
        0.4087,
        0.4003,
        0.3934,
        0.3879,
        0.3854,
        0.3817,
        0.3798,
        0.384,
    ],
    [
        0.124,
        0.1902,
        0.398,
        0.5843,
        0.6679,
        0.699,
        0.7132,
        0.7141,
        0.7115,
        0.7028,
        0.6852,
        0.661,
        0.6332,
        0.6031,
        0.5715,
        0.5374,
        0.5022,
        0.4667,
        0.4315,
        0.3989,
        0.3684,
        0.3409,
        0.3166,
        0.295,
        0.2766,
        0.261,
        0.2484,
        0.2386,
        0.2303,
        0.2233,
        0.2177,
        0.2132,
        0.2111,
        0.2082,
        0.2067,
        0.21,
    ],
    [
        0.1167,
        0.1774,
        0.3565,
        0.4971,
        0.5556,
        0.5815,
        0.5951,
        0.5941,
        0.5881,
        0.5746,
        0.5509,
        0.5201,
        0.486,
        0.4507,
        0.4149,
        0.3779,
        0.3416,
        0.3066,
        0.2737,
        0.2446,
        0.2188,
        0.1967,
        0.1779,
        0.1619,
        0.1487,
        0.1379,
        0.1294,
        0.1229,
        0.1175,
        0.1131,
        0.1095,
        0.1067,
        0.1054,
        0.1036,
        0.1027,
        0.1047,
    ],
    [
        0.1023,
        0.1529,
        0.2875,
        0.3784,
        0.4174,
        0.4397,
        0.453,
        0.4509,
        0.4428,
        0.4267,
        0.4003,
        0.3674,
        0.3326,
        0.2982,
        0.2651,
        0.2328,
        0.2028,
        0.1755,
        0.1513,
        0.131,
        0.1139,
        0.0999,
        0.0885,
        0.0791,
        0.0716,
        0.0656,
        0.0609,
        0.0574,
        0.0546,
        0.0523,
        0.0504,
        0.0489,
        0.0483,
        0.0473,
        0.0469,
        0.0479,
    ],
    [
        0.0781,
        0.1137,
        0.1964,
        0.2459,
        0.2706,
        0.2886,
        0.3003,
        0.2979,
        0.2898,
        0.2745,
        0.2507,
        0.2224,
        0.1941,
        0.1678,
        0.1439,
        0.122,
        0.1027,
        0.0862,
        0.0723,
        0.0612,
        0.0522,
        0.045,
        0.0393,
        0.0348,
        0.0312,
        0.0284,
        0.0262,
        0.0246,
        0.0233,
        0.0222,
        0.0214,
        0.0208,
        0.0205,
        0.02,
        0.0198,
        0.0203,
    ],
    [
        0.0469,
        0.0663,
        0.1054,
        0.1277,
        0.1412,
        0.1529,
        0.161,
        0.1592,
        0.1533,
        0.1426,
        0.1265,
        0.1085,
        0.0916,
        0.0767,
        0.0639,
        0.0527,
        0.0433,
        0.0356,
        0.0294,
        0.0245,
        0.0207,
        0.0177,
        0.0153,
        0.0135,
        0.012,
        0.0109,
        0.0101,
        0.0094,
        0.0089,
        0.0085,
        0.0082,
        0.0079,
        0.0078,
        0.0076,
        0.0075,
        0.0077,
    ],
]

dst_data= [
        0.0469,0.0663,0.1054,0.1277,0.1412,0.1529,0.161,0.1592,0.1533,0.1426,0.1265,0.1085,0.0916,0.0767,0.0639,0.0527,0.0433,0.0356,0.0294,0.0245,0.0207,0.0177,0.0153,0.0135,0.012,0.0109,0.0101,0.0094,0.0089,0.0085,0.0082,0.0079,0.0078,0.0076,0.0075,0.0077
    ]
"""

data = [
    [
        0.14,
        0.2112,
        0.4391,
        0.6822,
        0.8248,
        0.879,
        0.8954,
        0.8981,
        0.9028,
        0.9093,
        0.9155,
        0.9212,
        0.9257,
        0.9283,
        0.9312,
        0.9326,
        0.9353,
        0.9372,
        0.9362,
        0.9399,
        0.9396,
        0.9404,
        0.9409,
        0.9414,
        0.9414,
        0.9415,
        0.9436,
        0.9456,
        0.9469,
        0.9475,
        0.9473,
        0.9469,
        0.9485,
        0.95,
        0.9495,
        0.9511,
    ],
    [
        0.1389,
        0.2084,
        0.4228,
        0.6217,
        0.7015,
        0.7205,
        0.7248,
        0.7246,
        0.73,
        0.7407,
        0.7482,
        0.7546,
        0.7735,
        0.809,
        0.8495,
        0.874,
        0.8812,
        0.8773,
        0.8668,
        0.8573,
        0.8481,
        0.842,
        0.8363,
        0.8319,
        0.8298,
        0.8289,
        0.8297,
        0.8328,
        0.8365,
        0.8374,
        0.8348,
        0.8305,
        0.8254,
        0.8226,
        0.8254,
        0.8358,
    ],
    [
        0.1361,
        0.2018,
        0.3891,
        0.526,
        0.5616,
        0.5681,
        0.5694,
        0.5684,
        0.575,
        0.5891,
        0.5987,
        0.6068,
        0.6342,
        0.6895,
        0.7564,
        0.7999,
        0.8117,
        0.8033,
        0.7846,
        0.766,
        0.7504,
        0.7397,
        0.73,
        0.7225,
        0.7191,
        0.7175,
        0.7182,
        0.7226,
        0.7282,
        0.7295,
        0.7254,
        0.7183,
        0.7098,
        0.7048,
        0.7095,
        0.7259,
    ],
    [
        0.1314,
        0.1907,
        0.3407,
        0.4222,
        0.4336,
        0.4348,
        0.4349,
        0.4336,
        0.4407,
        0.4562,
        0.4666,
        0.4756,
        0.5074,
        0.5746,
        0.6608,
        0.7199,
        0.7359,
        0.7237,
        0.6981,
        0.6724,
        0.6516,
        0.6375,
        0.6248,
        0.6152,
        0.6108,
        0.6087,
        0.6094,
        0.6148,
        0.622,
        0.6235,
        0.6183,
        0.6093,
        0.5983,
        0.5919,
        0.5978,
        0.6185,
    ],
    [
        0.1239,
        0.1742,
        0.2824,
        0.3227,
        0.3218,
        0.3207,
        0.3203,
        0.319,
        0.3257,
        0.3408,
        0.351,
        0.3599,
        0.3923,
        0.4642,
        0.5627,
        0.6341,
        0.6538,
        0.6385,
        0.6071,
        0.5759,
        0.5513,
        0.5348,
        0.5202,
        0.5092,
        0.5041,
        0.5018,
        0.5025,
        0.5085,
        0.5167,
        0.5184,
        0.5124,
        0.5023,
        0.4898,
        0.4826,
        0.4893,
        0.5126,
    ],
    [
        0.1131,
        0.1522,
        0.2208,
        0.2353,
        0.2292,
        0.2273,
        0.2267,
        0.2256,
        0.2314,
        0.2446,
        0.2537,
        0.2616,
        0.2914,
        0.3611,
        0.4639,
        0.5435,
        0.5661,
        0.5483,
        0.5127,
        0.478,
        0.4514,
        0.4337,
        0.4183,
        0.4067,
        0.4015,
        0.3992,
        0.3998,
        0.406,
        0.4144,
        0.4162,
        0.41,
        0.3995,
        0.3866,
        0.3793,
        0.3861,
        0.4101,
    ],
    [
        0.0984,
        0.1253,
        0.1622,
        0.163,
        0.1558,
        0.1538,
        0.1533,
        0.1524,
        0.157,
        0.1675,
        0.1749,
        0.1813,
        0.2062,
        0.2677,
        0.3663,
        0.4487,
        0.473,
        0.4537,
        0.4161,
        0.3803,
        0.3536,
        0.3362,
        0.3212,
        0.3102,
        0.3052,
        0.303,
        0.3035,
        0.3094,
        0.3175,
        0.3192,
        0.3132,
        0.3032,
        0.2911,
        0.2842,
        0.2906,
        0.3133,
    ],
    [
        0.0806,
        0.0964,
        0.1121,
        0.1075,
        0.1013,
        0.0997,
        0.0993,
        0.0986,
        0.102,
        0.1097,
        0.1151,
        0.1199,
        0.1389,
        0.1883,
        0.2748,
        0.3537,
        0.3779,
        0.3586,
        0.3218,
        0.2877,
        0.2631,
        0.2474,
        0.2341,
        0.2244,
        0.2201,
        0.2181,
        0.2186,
        0.2237,
        0.2308,
        0.2323,
        0.227,
        0.2183,
        0.2079,
        0.2021,
        0.2075,
        0.2271,
    ],
    [
        0.0622,
        0.07,
        0.0746,
        0.0692,
        0.0645,
        0.0634,
        0.0631,
        0.0626,
        0.0649,
        0.0702,
        0.074,
        0.0774,
        0.0909,
        0.1277,
        0.1977,
        0.2673,
        0.2897,
        0.2717,
        0.2384,
        0.2087,
        0.1878,
        0.1747,
        0.1639,
        0.156,
        0.1526,
        0.151,
        0.1514,
        0.1555,
        0.1611,
        0.1624,
        0.1581,
        0.1512,
        0.1429,
        0.1384,
        0.1426,
        0.1582,
    ],
    [
        0.0454,
        0.0486,
        0.0486,
        0.044,
        0.0408,
        0.04,
        0.0398,
        0.0395,
        0.041,
        0.0445,
        0.0471,
        0.0493,
        0.0585,
        0.0845,
        0.1373,
        0.1943,
        0.2135,
        0.198,
        0.1701,
        0.146,
        0.1295,
        0.1195,
        0.1112,
        0.1053,
        0.1027,
        0.1016,
        0.1019,
        0.1049,
        0.1091,
        0.11,
        0.1069,
        0.1017,
        0.0956,
        0.0922,
        0.0953,
        0.1069,
    ],
    [
        0.0317,
        0.0326,
        0.0311,
        0.0278,
        0.0256,
        0.0251,
        0.025,
        0.0248,
        0.0258,
        0.028,
        0.0297,
        0.0312,
        0.0372,
        0.0548,
        0.0925,
        0.136,
        0.1513,
        0.1389,
        0.1172,
        0.0989,
        0.0868,
        0.0795,
        0.0735,
        0.0693,
        0.0675,
        0.0667,
        0.0669,
        0.069,
        0.0721,
        0.0727,
        0.0704,
        0.0668,
        0.0625,
        0.0601,
        0.0623,
        0.0705,
    ],
]

dst_data = [
    0.0089,
    0.0088,
    0.009,
    0.0089,
    0.0088,
    0.0089,
    0.0091,
    0.0092,
    0.0097,
    0.0101,
    0.0101,
    0.0104,
    0.012,
    0.0152,
    0.0208,
    0.026,
    0.0257,
    0.0217,
    0.0178,
    0.0152,
    0.0137,
    0.0128,
    0.0122,
    0.0119,
    0.0116,
    0.0115,
    0.0114,
    0.0113,
    0.0111,
    0.0108,
    0.0106,
    0.0104,
    0.0106,
    0.0112,
    0.0123,
    0.0139,
]


def reflectance_to_Lab(snm):

    trafo = ColorTrafoNumpy()
    lab = trafo.Cs_SNM2LAB(snm)
    return lab

def reflectance_to_L(snm):

    trafo = ColorTrafoNumpy()
    lab = trafo.Cs_SNM2LAB(snm)
    L = lab[0]
    return L


def extrapolate_spectrum_with_intermediates(
    spectra, target_L, num_steps=1000, max_concentration=1.3
):
    """
    Main function for extrapolation with intermediate spectra
    """

    # Assuming spectra is a list of reflectance arrays: [R_light, R_int_1, R_int_2, ..., R_mid]
    num_spectra = len(spectra)
    concentrations = np.linspace(
        0.0, 1.0, num_spectra
    )  # Concentrations from 0 to 1 for Light to Mid

    # Prepare cubic splines for each wavelength
    splines = []
    for wl in range(len(spectra[0])):
        spline = CubicSpline(
            concentrations,
            [spectra[i][wl] for i in range(num_spectra)],
            extrapolate=True,
        )
        splines.append(spline)

    # Generate candidate spectra with different concentrations (up to max_concentration)
    conc_steps = np.linspace(0, max_concentration, num_steps)
    R_candidates = []
    for c in conc_steps:
        R_candidate = np.array([spl(c) for spl in splines])
        R_candidates.append(R_candidate)

    # Find the candidate spectrum whose L is closest to the target L
    best_diff = float("inf")
    best_R = None
    best_L = None
    loops = 0
    for R_cand in R_candidates:
        loops += 1
        L = reflectance_to_L(R_cand)
        diff = abs(L - target_L)
        # print(f"Loop: {loops}, L: {L:.2f}, Target L: {target_L:.2f}, Diff: {diff:.2f}")

        if diff < best_diff:
            best_diff = diff

            R_cand_clipped = np.clip(R_cand, 0.001, 1)

            best_R = R_cand_clipped
            best_L = L

    return best_R, best_L, loops


def extrapolate_spectrum_with_intermediates_Pchip(spectra, target_L, num_steps=300, max_concentration=1.5):
    num_spectra = len(spectra)
    concentrations = np.linspace(0.0, 1.0, num_spectra)  # Concentrations from 0 to 1

    # Use PchipInterpolator for each wavelength
    interpolators = []
    for wl in range(len(spectra[0])):  # Updated to use spectra[0] for wavelength count
        interpolator = PchipInterpolator(concentrations, [spectra[i][wl] for i in range(num_spectra)], extrapolate=True)
        interpolators.append(interpolator)

    # Generate candidate spectra
    conc_steps = np.linspace(0, max_concentration, num_steps)
    R_candidates = []
    for c in conc_steps:
        R_candidate = np.array([interp(c) for interp in interpolators])
        R_candidates.append(R_candidate)

    # Find the candidate spectrum whose L is closest to the target L
    best_diff = float("inf")
    best_R = None
    best_L = None
    loops = 0
    for R_cand in R_candidates:
        loops += 1
        L = reflectance_to_L(R_cand)
        diff = abs(L - target_L)
        # print(f"Loop: {loops}, L: {L:.2f}, Target L: {target_L:.2f}, Diff: {diff:.2f}")

        if diff < best_diff:
            best_diff = diff

            R_cand_clipped = np.clip(R_cand, 0.001, 1)

            best_R = R_cand_clipped
            best_L = L

    return best_R, best_L, loops

# --------------------------
# Example usage

# Example wavelengths (36 bands from 380 to 730nm)
wavelengths = np.arange(380, 740, 10)  # 380, 390, ..., 730

# Combine into a list (Light, intermediates, Mid)
spectra = np.asarray(data)

# Target L value for dark color
target_Lab = reflectance_to_Lab(np.array(dst_data))
target_L = target_Lab[0]
print("Target L*: ", target_L)
print("Target Lab: ", target_Lab)

""" R_dark_corrected, achieved_L, loops = extrapolate_spectrum_with_intermediates(
    spectra, target_L, 300, 1.3
) """

R_dark_corrected, achieved_L, loops = extrapolate_spectrum_with_intermediates_Pchip(
    spectra, target_L, 600, 6
)

trafo = ColorTrafoNumpy()
corrected_lab = trafo.Cs_SNM2LAB(R_dark_corrected)
print("Corrected Lab values: ", corrected_lab)

# Output
print("Corrected Dark Spectrum Reflectance (36 bands):")
print(R_dark_corrected)
print(f"Achieved L*: {achieved_L:.2f}")
print(f"Number of loops: {loops}")

# Plotting
plt.figure(figsize=(10, 6))

for i in range(len(spectra)):
    plt.plot(wavelengths, spectra[i], label=f"R_{i}", color="black", alpha=0.5)

""" plt.plot(wavelengths, spectra[0], label="Light Spectrum", color="blue")
plt.plot(wavelengths, spectra[-1], label="Mid Spectrum", color="orange") """
plt.plot(wavelengths, R_dark_corrected, label="Corrected Dark Spectrum", color="red")

plt.plot(
    wavelengths,
    dst_data,
    label="Target Dark Spectrum",
    color="green",
    linestyle="dashed",
)
# plt.axhline(y=target_L, color="green", linestyle="--", label="Target L*")
plt.title("Reflectance Spectra")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.legend()
plt.grid()
plt.show()
