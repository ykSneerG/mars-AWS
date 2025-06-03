class AreaCoverage:

    def __init__(self, dcs, tac):
        self.dcs: list = dcs
        self.tac: float = len(dcs) * 100

    def get_coverage(self, dcs):

        # C=0.4,M=0.6,Y=0,K=0

        """
        const dcsMap = {
            "0,0,0,0": 0,    // White
            "100,0,0,0": 1,  // C
            "0,100,0,0": 2,  // M
            "0,0,100,0": 3,  // Y
            "0,0,0,100": 4,  // K
            "100,100,0,0": 5,  // CM
            "100,0,100,0": 6,  // CY
            "100,0,0,100": 7,  // CK
            "0,100,100,0": 8,  // MY
            "0,100,0,100": 9,  // MK
            "0,0,100,100": 10, // YK
            "100,100,100,0": 11, // CMY
            "100,100,0,100": 12, // CMK
            "100,0,100,100": 13, // CYK
            "0,100,100,100": 14, // MYK
            "100,100,100,100": 15 // CMYK
        };
        """

        C = dcs[0]
        M = dcs[1]
        Y = dcs[2]
        K = dcs[3]

        wP = (1 - C) * (1 - M) * (1 - Y) * (1 - K)
        wC = C * (1 - M) * (1 - Y) * (1 - K)
        wM = (1 - C) * M * (1 - Y) * (1 - K)
        wY = (1 - C) * (1 - M) * Y * (1 - K)
        wK = (1 - C) * (1 - M) * (1 - Y) * K

        wCM = C * M * (1 - Y) * (1 - K)
        wCY = C * (1 - M) * Y * (1 - K)
        wCK = C * (1 - M) * (1 - Y) * K
        wMY = (1 - C) * M * Y * (1 - K)
        wMK = (1 - C) * M * (1 - Y) * K
        wYK = (1 - C) * (1 - M) * Y * K

        wCMY = C * M * Y * (1 - K)
        wCMK = C * M * (1 - Y) * K
        wCYK = C * (1 - M) * Y * K
        wMYK = (1 - C) * M * Y * K

        wCMYK = C * M * Y * K

        return [
            wP,
            wC,
            wM,
            wY,
            wK,
            wCM,
            wCY,
            wCK,
            wMY,
            wMK,
            wYK,
            wCMY,
            wCMK,
            wCYK,
            wMYK,
            wCMYK,
        ]


    def get_demichel_weights(cmyk):
        weights = []
        for i in range(16):
            w = 1.0
            for j in range(4):  # C=0, M=1, Y=2, K=3
                bit = (i >> (3 - j)) & 1
                channel_val = cmyk[j]
                w *= channel_val if bit == 1 else (1 - channel_val)
            weights.append(w)
        return weights


res = AreaCoverage.get_demichel_weights([0.4, 0.6, 0.0, 0.0])
print(res)