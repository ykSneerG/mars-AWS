class SampleSpectral:

    c1 = [0.468875, 0.520225, 0.623975, 0.746375, 0.8459, 0.88845, 0.90095, 0.90345, 0.90495, 0.903825, 0.8992, 0.89275, 0.884, 0.876125, 0.871, 0.8663, 0.8627, 0.859325, 0.85245, 0.850175, 0.8457, 0.84325, 0.8426, 0.84405, 0.847875, 0.852425, 0.858975, 0.86695, 0.87525, 0.882675, 0.88925, 0.8943, 0.9002, 0.9044, 0.9061, 0.9091]
    c2 = [0.01546, 0.03345, 0.08979, 0.1687, 0.2368, 0.3236, 0.4573, 0.5647, 0.6008, 0.601, 0.5674, 0.5074, 0.4293, 0.3293, 0.2155, 0.1144, 0.04581, 0.01301, 0.003531, 0.002278, 0.002196, 0.002257, 0.002279, 0.002432, 0.002555, 0.002558, 0.002456, 0.002414, 0.002465, 0.002456, 0.002503, 0.002572, 0.002805, 0.003239, 0.003574, 0.004216]
    c3 = [0.08673, 0.1224, 0.1604, 0.1837, 0.179, 0.1456, 0.1008, 0.06157, 0.0366, 0.02015, 0.01093, 0.008395, 0.007481, 0.006906, 0.006981, 0.007294, 0.007143, 0.007248, 0.008299, 0.01172, 0.04748, 0.1947, 0.4287, 0.6131, 0.704, 0.7476, 0.7743, 0.7947, 0.8117, 0.8264, 0.8396, 0.8516, 0.8634, 0.8725, 0.8776, 0.8828]
    c4 = [0.02042, 0.01935, 0.01826, 0.01946, 0.02074, 0.02084, 0.02089, 0.02053, 0.0208, 0.02226, 0.02427, 0.0357, 0.1132, 0.3331, 0.591, 0.7431, 0.8052, 0.8269, 0.8307, 0.834, 0.833, 0.8327, 0.8335, 0.8359, 0.8402, 0.8449, 0.8516, 0.8594, 0.8675, 0.8748, 0.8811, 0.8862, 0.8922, 0.8963, 0.8979, 0.9007]
    c5 = [0.004273, 0.004891, 0.00504, 0.004812, 0.004611, 0.004958, 0.004862, 0.004645, 0.004411, 0.004304, 0.004136, 0.004057, 0.003918, 0.003714, 0.0037, 0.003564, 0.003552, 0.003331, 0.003399, 0.00359, 0.003686, 0.003643, 0.003611, 0.003569, 0.003618, 0.003503, 0.003519, 0.003455, 0.003576, 0.003769, 0.003725, 0.003517, 0.003533, 0.003805, 0.00409, 0.004772]
    c6 = [0.02612,0.01807,0.01272,0.01048,0.008178,0.007709,0.007334,0.007556,0.00752,0.007086,0.00646,0.00607,0.0063,0.006233,0.00603,0.006488,0.008809,0.03126,0.1547,0.3899,0.5922,0.7021,0.7568,0.7847,0.7984,0.8058,0.8124,0.8186,0.823,0.8261,0.8292,0.8316,0.8357,0.8371,0.8369,0.8372]
    c7 = [0.00693,0.007275,0.008984,0.01228,0.01965,0.03107,0.04908,0.07929,0.1346,0.2425,0.3942,0.5308,0.6312,0.6738,0.6632,0.6243,0.5676,0.4887,0.3913,0.2926,0.194,0.1043,0.0388,0.009636,0.002669,0.00151,0.001288,0.001219,0.001411,0.00178,0.002346,0.003792,0.006523,0.008653,0.008735,0.00877]
    c8 = [0.1318,0.203,0.2639,0.3236,0.4224,0.4915,0.5059,0.4835,0.4341,0.3646,0.2704,0.1627,0.08235,0.03925,0.01772,0.01113,0.01096,0.01171,0.01154,0.01331,0.02227,0.04705,0.07916,0.08605,0.0682,0.06774,0.1097,0.2053,0.3413,0.4774,0.5907,0.6739,0.733,0.7702,0.7932,0.8104]


    def __init__(self):
        self.spectral = [self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8]
        
    def get_list(self):
        return self.spectral
    
    def get_idx(self, idx):
        return self.spectral[idx]