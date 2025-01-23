class Ynsn_4C:
    def __init__(self, max_percentage, n):
        self.MaxPercentage = max_percentage
        self.n = self.setNfactor(n)
        self.nps = [0] * 16
        self.dcs = [None] * 4

    def setColor1(self, quantity):
        self.dcs[0] = self.normalize(quantity)

    def setColor2(self, quantity):
        self.dcs[1] = self.normalize(quantity)

    def setColor3(self, quantity):
        self.dcs[2] = self.normalize(quantity)

    def setColor4(self, quantity):
        self.dcs[3] = self.normalize(quantity)

    def setColor(self, qty1, qty2, qty3, qty4):
        self.setColor1(qty1)
        self.setColor2(qty2)
        self.setColor3(qty3)
        self.setColor4(qty4)

    def setNfactor(self, n_factor):
        if n_factor == 0:
            raise ValueError('n cannot be zero')
        self.n = n_factor

    def setNP_0000(self, r):
        self.nps[0] = r

    def setNP_1000(self, r):
        self.nps[1] = r

    def setNP_0100(self, r):
        self.nps[2] = r

    def setNP_0010(self, r):
        self.nps[3] = r

    def setNP_0001(self, r):
        self.nps[4] = r

    def setNP_1100(self, r):
        self.nps[5] = r

    def setNP_1010(self, r):
        self.nps[6] = r

    def setNP_1001(self, r):
        self.nps[7] = r

    def setNP_0110(self, r):
        self.nps[8] = r

    def setNP_0101(self, r):
        self.nps[9] = r

    def setNP_0011(self, r):
        self.nps[10] = r

    def setNP_1110(self, r):
        self.nps[11] = r

    def setNP_1101(self, r):
        self.nps[12] = r

    def setNP_1011(self, r):
        self.nps[13] = r

    def setNP_0111(self, r):
        self.nps[14] = r

    def setNP_1111(self, r):
        self.nps[15] = r

    def mix(self):
        ara = self.area()

        n_invers = 1 / self.n
        for j in range(len(self.nps)):
            self.nps[j] = [item ** n_invers for item in self.nps[j]]

        nps_w = []
        for i in range(16):
            tmp = []
            for item in self.nps[i]:
                result = (item * ara[i]) ** self.n
                tmp.append(result)
            nps_w.append(tmp)

        r_out = [0] * 36
        for i in range(16):
            for j in range(36):
                r_out[j] += nps_w[i][j]

        return r_out

    def normalize(self, value):
        return value / self.MaxPercentage

    def area(self):
        c, m, y, k = self.dcs

        aw = (1 - c) * (1 - m) * (1 - y) * (1 - k)
        ac = c * (1 - m) * (1 - y) * (1 - k)
        am = (1 - c) * m * (1 - y) * (1 - k)
        ay = (1 - c) * (1 - m) * y * (1 - k)
        ak = (1 - c) * (1 - m) * (1 - y) * k

        acm = c * m * (1 - y) * (1 - k)
        acy = c * (1 - m) * y * (1 - k)
        ack = c * (1 - m) * (1 - y) * k
        amy = (1 - c) * m * y * (1 - k)
        amk = (1 - c) * m * (1 - y) * k
        ayk = (1 - c) * (1 - m) * y * k

        acmy = c * m * y * (1 - k)
        acmk = c * m * (1 - y) * k
        acyk = c * (1 - m) * y * k
        amyk = (1 - c) * m * y * k

        acmyk = c * m * y * k

        return [aw, ac, am, ay, ak, acm, acy, ack, amy, amk, ayk, acmy, acmk, acyk, amyk, acmyk]


class CellularYnsn_4C:
    def __init__(self, max_percentage, n, subdivision=4):
        self.MaxPercentage = max_percentage
        self.n = n
        self.subdivision = subdivision
        self.nps = [0] * 16
        self.dcs = [0] * 4
        self.snm = []
        self.info = "Leerlauf"

    def setColor1(self, quantity):
        self.dcs[0] = self.normalize(quantity)

    def setColor2(self, quantity):
        self.dcs[1] = self.normalize(quantity)

    def setColor3(self, quantity):
        self.dcs[2] = self.normalize(quantity)

    def setColor4(self, quantity):
        self.dcs[3] = self.normalize(quantity)

    def setColor(self, qty1, qty2, qty3, qty4):
        self.setColor1(qty1)
        self.setColor2(qty2)
        self.setColor3(qty3)
        self.setColor4(qty4)

    def setNfactor(self, n_factor):
        self.n = n_factor

    def setSpectralData(self, snm):
        self.snm = snm

    def mix(self):
        subcube_index = HyperCube.findSubcubeIndex(self.dcs, self.subdivision)
        subcube_corners = HyperCube.getSubcube_corners_4D(subcube_index)

        subcube_size = self.MaxPercentage / self.subdivision
        subcube_corners_scaled = HyperCube.multiply_array(subcube_corners, subcube_size)

        nps = []
        for corner_scaled in subcube_corners_scaled:
            id_ = HyperCube.find_matching_array_index(corner_scaled, self.snm)
            nps.append(self.snm[id_][4:])

        min_limits = [min(c) for c in zip(*subcube_corners_scaled)]
        max_limits = [max(c) for c in zip(*subcube_corners_scaled)]
        scaled_dcs = self.scale_dcs(min_limits, max_limits)

        ynsn = Ynsn_4C(self.MaxPercentage, self.n)

        if ynsn.dcs != scaled_dcs:
            ynsn.nps = nps
            ynsn.dcs = scaled_dcs
            return ynsn.mix()

    def normalize(self, value):
        return value / self.MaxPercentage

    def scale_dcs(self, min_limits, max_limits):
        return [
            self.scale_dcs_value(dcs, min_, max_, self.MaxPercentage)
            if dcs > 0
            else 0
            for dcs, min_, max_ in zip(self.dcs, min_limits, max_limits)
        ]

    @staticmethod
    def scale_dcs_value(value, min_, max_, max_percentage):
        return (value * max_percentage - min_) / (max_ - min_)


class HyperCube:
    '''
    The 'HyperCube' class provides static methods for manipulating and calculating subcubes in a hypercube. It is used as a helper class for
    the CellularYnsn4C class, which uses the 'HyperCube' methods to calculate and mix colors for a 4-dimensional cellular automaton.
    '''
    @staticmethod
    def findSubcubeIndex(point, subdivisions):
        '''
        subcubeIndex = []
        for i in range(len(point)):
            index = int(point[i] * subdivisions) - 1
            if index < 0:
                index = 0
            subcubeIndex.append(index)
        return subcubeIndex
        '''
        return [max(0, int(p * subdivisions) - 1) for p in point]

    @staticmethod
    def getSubcubeCorners4D(subcubeIndex):
        cornersUnsorted = []
        for i in range(16):
            corner = subcubeIndex.copy()
            for j in range(4):
                if i & (1 << j):
                    corner[j] += 1
            cornersUnsorted.append(corner)

        cornersSorted = [
            cornersUnsorted[0],
            cornersUnsorted[1],
            cornersUnsorted[2],
            cornersUnsorted[3],
            cornersUnsorted[4],
            cornersUnsorted[5],
            cornersUnsorted[6],
            cornersUnsorted[7],
            cornersUnsorted[8],
            cornersUnsorted[9],
            cornersUnsorted[10],
            cornersUnsorted[11],
            cornersUnsorted[12],
            cornersUnsorted[13],
            cornersUnsorted[14],
            cornersUnsorted[15]
        ]

        return cornersSorted

    @staticmethod
    def multiplyArray(arr, factor):
        return [[value * factor for value in row] for row in arr]
        '''
        for row in arr:
            for i, value in enumerate(row):
                row[i] *= factor
        return arr
        '''

    @staticmethod
    def findMatchingArrayIndex(arr, searchArray):
        for i, search_item in enumerate(searchArray):
            if arr == search_item:
                return i
        return None
        '''
        for i in range(len(searchArray)):
            match = True
            for j in range(len(arr)):
                if arr[j] != searchArray[i][j]:
                    match = False
                    break
            if match:
                return i
        return None
        '''
