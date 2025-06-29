import numpy as np

class YNSNPredictor:
    
    def __init__(self, R_primaries, n=2.0):
        self.R_primaries = np.asarray(R_primaries)
        self.n = n

    @staticmethod
    def get_demichel_weights_batch(cmyk: np.ndarray) -> np.ndarray:
        C, M, Y, K = cmyk[:, 0], cmyk[:, 1], cmyk[:, 2], cmyk[:, 3]
        one_minus = lambda x: 1.0 - x

        weights = np.stack([
            one_minus(C) * one_minus(M) * one_minus(Y) * one_minus(K),  # P
            C * one_minus(M) * one_minus(Y) * one_minus(K),             # C
            one_minus(C) * M * one_minus(Y) * one_minus(K),             # M
            one_minus(C) * one_minus(M) * Y * one_minus(K),             # Y
            one_minus(C) * one_minus(M) * one_minus(Y) * K,             # K
            C * M * one_minus(Y) * one_minus(K),                        # CM
            C * one_minus(M) * Y * one_minus(K),                        # CY
            C * one_minus(M) * one_minus(Y) * K,                        # CK
            one_minus(C) * M * Y * one_minus(K),                        # MY
            one_minus(C) * M * one_minus(Y) * K,                        # MK
            one_minus(C) * one_minus(M) * Y * K,                        # YK
            C * M * Y * one_minus(K),                                   # CMY
            C * M * one_minus(Y) * K,                                   # CMK
            C * one_minus(M) * Y * K,                                   # CYK
            one_minus(C) * M * Y * K,                                   # MYK
            C * M * Y * K                                               # CMYK
        ], axis=1)
        return weights  # Shape: (N, 16)
    

    # !!! THIS IS NOT WORKING YET !!!
    @staticmethod
    def get_demichel_weights_batch_NP(channels: np.ndarray) -> np.ndarray:
        """
        Generalized Demichel weights using NumPy only.
        Channel order: first channel = most significant bit (like C in CMYK).
        """
        B, N = channels.shape
        one_minus = 1.0 - channels

        num_combinations = 2 ** N

        # Correct order: first channel is MSB, last is LSB (C=bit3, K=bit0)
        bit_order = np.arange(N - 1, -1, -1)
        combinations = ((np.arange(num_combinations)[:, None] >> bit_order) & 1).astype(np.uint8)

        weights = np.ones((B, num_combinations), dtype=np.float32)

        for i in range(num_combinations):
            mask = combinations[i]  # shape: (N,)
            selected = np.where(mask == 1, channels, one_minus)  # shape: (B, N)
            weights[:, i] = np.prod(selected, axis=1)

        return weights



    def predict_spectrum_batch(self, cmyk: np.ndarray) -> np.ndarray:        
        weights = self.get_demichel_weights_batch(cmyk)         # (N, 16)
        # print("weights.shape:", weights.shape)            # sollte (N, 16) sein
        R_primaries_n = np.power(self.R_primaries, 1/self.n)         # (16, L)
        R_pred_n = weights @ R_primaries_n                 # (N, L)
        R_pred = np.power(R_pred_n, self.n)                     # (N, L)
        return R_pred
    
    def fit_n(self, cmyk_samples, target_spectra, n_min=0.01, n_max=20.0, tol=1e-2, max_iter=25):
        '''
        Dies ist eine Variante der ternären Suche (für unimodale Funktionen noch besser als binäre Suche).
        '''
        def compute_error(n):
            self.n = n
            pred = self.predict_spectrum_batch(cmyk_samples)
            return np.mean((pred - target_spectra) ** 2)


        iter = 0
        for _ in range(max_iter):
            n1 = n_min + (n_max - n_min) / 3
            n2 = n_max - (n_max - n_min) / 3

            e1 = compute_error(n1)
            e2 = compute_error(n2)

            if e1 < e2:
                n_max = n2
            else:
                n_min = n1

            iter += 1

            if abs(n_max - n_min) < tol:
                break

        best_n = (n_min + n_max) / 2
        self.n = best_n
        return (best_n, iter)
    
    
'''

predictor = YNSNPredictor(R_primaries=your_16x31_matrix)
best_n = predictor.fit_n(cmyk_train, target_spectra) # ---> Hier werden die CMYK-Werte und die Ziel-Spektren übergeben
print("Optimales n:", best_n)

# Danach wird die Vorhersage mit dem besten n durchgeführt:
predicted_spectra = predictor.predict_spectrum_batch(cmyk_test)  # cmyk_test ist ein Array von CMYK-Werten, hier wird nun das optimierte n verwendet!!!



'''

# --- FROM HERE ONLY LOCAL TESTING --- FROM HERE ONLY LOCAL TESTING --- FROM HERE ONLY LOCAL TESTING ---

'''
# Corner / Neugebaur Primaries for CMYK in 380–730nm
corners = [
    [0.0212,0.0225,0.0247,0.0247,0.0256,0.0272,0.0278,0.0288,0.0289,0.0298,0.0314,0.0325,0.033,0.0325,0.0312,0.0299,0.0285,0.0274,0.0261,0.0256,0.0261,0.0277,0.0284,0.0282,0.0282,0.0283,0.0284,0.0287,0.0289,0.0291,0.0291,0.029,0.029,0.0292,0.0296,0.0308],
    [0.0296,0.031,0.0364,0.0412,0.0443,0.046,0.0482,0.0499,0.0506,0.0527,0.0566,0.0596,0.0606,0.0587,0.055,0.0509,0.0474,0.0443,0.041,0.0393,0.0409,0.0457,0.0492,0.0505,0.051,0.0513,0.0518,0.0525,0.0536,0.0544,0.0543,0.0537,0.0534,0.0532,0.0539,0.0554],
    [0.0246,0.0289,0.0325,0.0353,0.038,0.0414,0.0432,0.0443,0.0445,0.0449,0.0451,0.0448,0.044,0.0419,0.0388,0.0358,0.0331,0.0306,0.0284,0.0274,0.0286,0.0317,0.0335,0.0339,0.034,0.034,0.0342,0.0346,0.0351,0.0354,0.0355,0.0353,0.0349,0.0348,0.0354,0.0369],
    [0.0441,0.0656,0.111,0.1796,0.2239,0.2686,0.3224,0.3528,0.3492,0.3321,0.3066,0.2778,0.2491,0.2143,0.17,0.128,0.0959,0.0699,0.0504,0.0402,0.0396,0.0452,0.0482,0.0481,0.0476,0.0469,0.0469,0.0484,0.0515,0.0535,0.0527,0.0503,0.0473,0.0451,0.0461,0.0505],
    [0.0245,0.0244,0.0267,0.029,0.0298,0.0327,0.0343,0.0358,0.0382,0.0424,0.0493,0.0555,0.0586,0.0584,0.0557,0.0519,0.0472,0.0427,0.0385,0.036,0.0344,0.0337,0.0329,0.0325,0.0324,0.0324,0.0326,0.033,0.0337,0.0342,0.0342,0.0339,0.0335,0.0333,0.0341,0.0358],
    [0.0219,0.0287,0.0353,0.0435,0.0459,0.0506,0.0568,0.0651,0.078,0.1108,0.182,0.2795,0.3472,0.3573,0.3252,0.2747,0.2245,0.1779,0.1409,0.1181,0.105,0.0975,0.0913,0.0869,0.0848,0.0839,0.0841,0.0864,0.0906,0.0935,0.0928,0.0898,0.0856,0.0823,0.0827,0.0881],
    [0.0248,0.0276,0.033,0.0397,0.0426,0.0462,0.0503,0.053,0.0549,0.0567,0.0587,0.0598,0.0592,0.0567,0.0516,0.0457,0.0394,0.0334,0.0287,0.0259,0.0243,0.0235,0.0227,0.0222,0.0221,0.022,0.0221,0.022,0.0225,0.0226,0.0226,0.0224,0.0223,0.0225,0.0235,0.0256],
    [0.0483,0.081,0.1459,0.2522,0.3248,0.408,0.5221,0.6115,0.6379,0.6382,0.6151,0.574,0.5193,0.4466,0.3572,0.2676,0.1925,0.1315,0.0883,0.0648,0.0533,0.0474,0.0429,0.0397,0.0382,0.0374,0.0373,0.0388,0.0415,0.0434,0.0428,0.0408,0.038,0.0362,0.0372,0.0417],
    [0.0228,0.0232,0.0234,0.0236,0.0254,0.0261,0.0262,0.0263,0.0267,0.0283,0.0308,0.0326,0.0334,0.0332,0.0321,0.0312,0.0306,0.0304,0.0299,0.0302,0.0327,0.0375,0.0408,0.0422,0.0429,0.0432,0.0437,0.0442,0.0445,0.0447,0.0451,0.0453,0.0454,0.0453,0.0459,0.0468],
    [0.0491,0.0495,0.0542,0.0608,0.0628,0.0655,0.0667,0.0671,0.0677,0.0738,0.086,0.0966,0.0985,0.0906,0.0763,0.0666,0.0649,0.066,0.0634,0.0649,0.1008,0.2127,0.3658,0.4844,0.5507,0.5842,0.6022,0.613,0.6196,0.622,0.6226,0.6225,0.6247,0.6269,0.6281,0.631],
    [0.0275,0.0298,0.0321,0.0354,0.0366,0.0378,0.0391,0.0391,0.0391,0.039,0.0388,0.0384,0.0377,0.0364,0.034,0.0322,0.031,0.0301,0.0291,0.0291,0.0327,0.0403,0.046,0.0483,0.0492,0.0497,0.0501,0.0508,0.0515,0.0517,0.0519,0.052,0.0516,0.0518,0.0523,0.0535],
    [0.1161,0.1309,0.1511,0.1903,0.2113,0.2233,0.2228,0.209,0.1846,0.1573,0.13,0.1074,0.0923,0.0783,0.0627,0.053,0.0514,0.0522,0.0499,0.0513,0.0856,0.2009,0.3688,0.5042,0.5817,0.6209,0.6419,0.6541,0.6616,0.665,0.665,0.664,0.6651,0.6669,0.6679,0.672],
    [0.0224,0.0202,0.0204,0.0206,0.0195,0.0203,0.0208,0.0214,0.0228,0.026,0.0319,0.0386,0.0425,0.0439,0.0437,0.0427,0.0417,0.0406,0.0393,0.0387,0.0388,0.0397,0.0402,0.0404,0.0406,0.0408,0.0413,0.0418,0.0424,0.0427,0.0429,0.0428,0.043,0.0433,0.0437,0.0447],
    [0.0413,0.0376,0.0349,0.0342,0.0358,0.0354,0.0361,0.04,0.0499,0.0828,0.1812,0.3655,0.5664,0.6991,0.7545,0.7669,0.7692,0.7681,0.7618,0.7608,0.7695,0.7926,0.8111,0.8222,0.8312,0.8397,0.8494,0.8582,0.8652,0.8677,0.8671,0.8658,0.8679,0.8711,0.8721,0.8759],
    [0.024,0.027,0.0294,0.0328,0.0336,0.0358,0.0373,0.0382,0.0391,0.0406,0.0429,0.0448,0.0451,0.0441,0.0415,0.0388,0.0366,0.0343,0.032,0.031,0.0322,0.0353,0.0372,0.0375,0.0377,0.0378,0.038,0.0384,0.0392,0.0395,0.0397,0.0395,0.0392,0.039,0.0396,0.0411],
    [0.3396,0.3996,0.4943,0.6843,0.7867,0.8418,0.8575,0.8647,0.8703,0.8727,0.8723,0.8701,0.8651,0.8568,0.8472,0.8351,0.8257,0.8181,0.8074,0.8016,0.7984,0.8015,0.8045,0.8083,0.8139,0.8212,0.8302,0.839,0.8468,0.8494,0.8487,0.8468,0.8476,0.8492,0.8506,0.8562]
]
R_primaries = np.array(corners)[::-1, :]  # Nur Zeilen (16) umkehren, Form bleibt (16, 36)

# Test mit echten CMYK-Werten:
cmyk_input = np.array([
    [0.1, 0.0, 0.0, 0.0],
    [0.2, 0.0, 0.0, 0.0],
    [0.3, 0.0, 0.0, 0.0],
    [0.4, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0],
    [0.6, 0.0, 0.0, 0.0],
    [0.7, 0.0, 0.0, 0.0],
    [0.8, 0.0, 0.0, 0.0],
    [0.9, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0]
]) 

predictor = YNSNPredictor(R_primaries, n=2.0)
predicted_spectra = predictor.predict_spectrum_batch(cmyk_input)  # Shape (3, 36)
print(predicted_spectra)

# Ausgabe (optional plotten)
import matplotlib.pyplot as plt
wavelengths = np.linspace(380, 730, 36)
plt.plot(wavelengths, predicted_spectra.T, label='Predicted Spectrum')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Yule-Nielsen Spektrumsvorhersage')
plt.legend()
plt.grid(True)
plt.show()

'''