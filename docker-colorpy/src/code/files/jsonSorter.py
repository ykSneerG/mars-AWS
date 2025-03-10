# from functools import cmp_to_key
import random
from enum import Enum
from src.code.files.cgatsConstants import CgatsConstants
from src.code.space.colorConverter import CS_Spectral2LAB, CS_Spectral2LCH, Cs_XYZ2LCH
from src.code.space.colorSpace import CsLAB, CsXYZ


class CompareDcs():
    @staticmethod
    def compare1(x, y):

        x_weight = 0
        y_weight = 0

        # cn = len(x['dcs']) - x['dcs'].count(0)
        for i in range(len(x['dcs'])):
            '''
            x_weight += (float(x['dcs'][i]) + ((i+1) * 100)) if float(x['dcs'][i]) > 0 else 0
            y_weight += (float(y['dcs'][i]) + ((i+1) * 100)) if float(y['dcs'][i]) > 0 else 0
            '''
            '''
            factor = pow(10, (i + 1))  # * (cn + 1)
            x_weight += float(x['dcs'][i]) + factor if float(x['dcs'][i]) > 0 else 0
            y_weight += float(y['dcs'][i]) + factor if float(y['dcs'][i]) > 0 else 0
            '''
            factor = i * 100
            # factor = pow(10, i)  # * pow(20, cn)
            x_weight += float(x['dcs'][i]) * 0.01 + factor if float(x['dcs'][i]) > 0 else 0
            y_weight += float(y['dcs'][i]) * 0.01 + factor if float(y['dcs'][i]) > 0 else 0

        return x_weight - y_weight

    @staticmethod
    def compare_OK(x, y):

        res = 0

        for i in range(len(x['dcs'])):
            res = (x['dcs'][i] - y['dcs'][i]) * (i + 1)

        return res

    @staticmethod
    def compare_OK2(x, y):

        res = 0
        dcs_length = len(x['dcs'])
        nul_length = x['dcs'].count(0) - y['dcs'].count(0)
        for i in range(dcs_length):
            res = pow((x['dcs'][i] - y['dcs'][i]), (dcs_length - i) + nul_length)

        return res

    @staticmethod
    def compare(x, y):

        dcs_length = len(x['dcs'])

        x_nulls = x['dcs'].count(0) * pow(100, dcs_length)
        y_nulls = y['dcs'].count(0) * pow(100, dcs_length)

        x_weight = 0
        y_weight = 0
        for i in range(dcs_length):
            '''
            x_weight += pow(x['dcs'][i] * 0.01, i + 1) if x['dcs'][i] > 0 else 0
            y_weight += pow(y['dcs'][i] * 0.01, i + 1) if x['dcs'][i] > 0 else 0
            '''
            x_weight += x['dcs'][i] * 0.01 + pow(10, dcs_length-1 - i) if x['dcs'][i] > 0 else 0
            y_weight += y['dcs'][i] * 0.01 + pow(10, dcs_length-1 - i) if x['dcs'][i] > 0 else 0

        return (x_weight - x_nulls) - (y_weight - y_nulls)


class JsonDataSorter:
    '''
    Sort the json data object
    '''

    class OrderType(Enum):
        VISUAL = 'visual'
        RANDOM = 'random'
        HUE = 'hue'
        LIGHTNESS = 'lightness'
        CHROMA = 'chroma'

    def __init__(self, data, pcs_type=None):
        self.data = data
        self.pcs_type = pcs_type

    '''
    def sort_visual(self):

        res = []

        for j in range(len(self.data[0]['dcs'])):
            tmp = []
            for i in range(len(self.data)):
                cn = self.data[i]['dcs'].count(0)

                if len(self.data[0]['dcs']) - cn == j:
                    tmp.append(self.data[i])

            tmp_sordted = sorted(tmp, key=lambda x: x['dcs'])
            res += tmp_sordted

        return res

        # res = sorted(self.data, key=lambda x: [float(i) for i in x['dcs']][::-1])

        # return sorted(self.data, key=lambda x: x['dcs'])

        # res = sorted(self.data, key=lambda x: [float(i) for i in x['dcs']])
        # res = sorted(self.data, key=cmp_to_key(CompareDcs.compare))
    '''

    def sort(self, order_type=OrderType.VISUAL):

        # handle empty data
        if (len(self.data) == 0 or
                self.data is None):
            return []

        # handle data with only one row
        if (len(self.data) == 1 or
                len(self.data[0]['dcs']) == 1):
            return self.data

        sort_methods = {
            self.OrderType.VISUAL: self.sort_visual,
            self.OrderType.RANDOM: self.sort_random,
            self.OrderType.LIGHTNESS: self.sort_lightness,
            self.OrderType.CHROMA: self.sort_chroma,
            self.OrderType.HUE: self.sort_hue
        }

        return sort_methods.get(order_type, self.sort_visual)()

    def sort_visual(self):
        res = []

        dcs_length = len(self.data[0]['dcs'])

        for i in range(dcs_length):
            tmp = [data for data in self.data if data['dcs'].count(0) == dcs_length - i]
            tmp_sorted = sorted(tmp, key=lambda x: x['dcs'])
            res += tmp_sorted

        return res

    def sort_random(self):
        '''
        random.shuffle(self.data)
        return self.data
        '''
        return random.sample(self.data, len(self.data))

    def sort_lightness(self):
        if self.pcs_type == CgatsConstants.LAB_TPLS[0]:
            return sorted(self.data, key=lambda x: x['pcs'][0], reverse=True)

        elif self.pcs_type == CgatsConstants.SNM_TPLS[0]:
            return sorted(self.data, key=lambda x: CS_Spectral2LAB(x['pcs']).L, reverse=True)

        elif self.pcs_type == CgatsConstants.LCH_TPLS[0]:
            return sorted(self.data, key=lambda x: x['pcs'][0], reverse=True)

        elif self.pcs_type == CgatsConstants.XYZ_TPLS[0]:
            return sorted(self.data, key=lambda x: Cs_XYZ2LCH(CsXYZ(x['pcs'][0], x['pcs'][1], x['pcs'][2])).L, reverse=True)

    def sort_chroma(self):
        if self.pcs_type == CgatsConstants.LAB_TPLS[0]:
            return sorted(self.data, key=lambda x: CsLAB(x['pcs'][0], x['pcs'][1], x['pcs'][2]).to_chroma())

        elif self.pcs_type == CgatsConstants.SNM_TPLS[0]:
            return sorted(self.data, key=lambda x: CS_Spectral2LCH(x['pcs']).C)

        elif self.pcs_type == CgatsConstants.LCH_TPLS[0]:
            return sorted(self.data, key=lambda x: x['pcs'][1])

        elif self.pcs_type == CgatsConstants.XYZ_TPLS[0]:
            return sorted(self.data, key=lambda x: Cs_XYZ2LCH(CsXYZ(x['pcs'][0], x['pcs'][1], x['pcs'][2])).C)

    def sort_hue(self):
        if self.pcs_type == CgatsConstants.LAB_TPLS[0]:
            return sorted(self.data, key=lambda x: CsLAB(x['pcs'][0], x['pcs'][1], x['pcs'][2]).to_chroma())

        elif self.pcs_type == CgatsConstants.SNM_TPLS[0]:
            return sorted(self.data, key=lambda x: CS_Spectral2LCH(x['pcs']).C)

        elif self.pcs_type == CgatsConstants.LCH_TPLS[0]:
            return sorted(self.data, key=lambda x: x['pcs'][2])

        elif self.pcs_type == CgatsConstants.XYZ_TPLS[0]:
            return sorted(self.data, key=lambda x: Cs_XYZ2LCH(CsXYZ(x['pcs'][0], x['pcs'][1], x['pcs'][2])).H)

    @staticmethod
    def sortorder(channel=4, step=5):
        res = [[0] * channel]  # Start with [0, 0, 0, ...] list
        last_combination = [step] * channel  # Create the last combination

        while res[-1] != last_combination:
            current_combination = res[-1][:]
            current_combination[-1] += step

            for i in range(channel-1, 0, -1):
                if current_combination[i] > step:
                    current_combination[i] = 0
                    current_combination[i-1] += step

            res.append(current_combination)

        res.append(last_combination)  # Add the last combination

        return res
