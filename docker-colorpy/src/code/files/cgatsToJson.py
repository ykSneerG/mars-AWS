import json

from src.code.files.cgats import Cgats
from src.code.space.colorConverter import Cs_Spectral2Multi


def neugebauer_sort(colors):
    # Function to calculate the weight of the color (how many color channels are active)
    def color_weight(dcs):
        return sum(1 for val in dcs if val > 0)

    # Sort by color weight first, then by primary color values in the order: C > M > Y > K
    def sort_key(color):
        dcs = color["dcs"]
        weight = color_weight(dcs)
        return (weight, [-val for val in dcs])  # Negative to sort descending

    # Sort the colors
    colors.sort(key=sort_key)

    return colors


def linearization_sort(colors):
    # Sort by color weight first, then by primary color values in the order: C > M > Y > K
    # starting with the first column ascending, than the second column asc, etc.
    # Fiist [1,0,0,0], [2,0,0,0], [3,0,0,0],
    # Second [0,1,0,0], [0,2,0,0], [0,3,0,0],
    # Third [0,0,1,0], [0,0,2,0], [0,0,3,0],
    # ...[0,0,0,1], [0,0,0,2], [0,0,0,3]

    def sort_key(color):
        dcs = color["dcs"]
        # Find the first nonzero index (which determines the primary component)
        first_nonzero_index = next(
            (i for i, val in enumerate(dcs) if val > 0), len(dcs)
        )
        # Sorting key: (first nonzero index, actual dcs values in ascending order)
        return (first_nonzero_index, *dcs)

    # Sort the colors
    colors.sort(key=sort_key)

    return colors


class CgatsToJson:

    def __init__(self, params):

        self.cgatstext: str = params.get("txt", "")
        self.doublets_average = params.get("doublets_average", False)
        self.doublets_remove = params.get("doublets_remove", False)
        self.reduction_type = params.get("reduction_type", None)

        self.dst_dcs = params.get("dst_dcs", self.DCS_CMYK_48)

        self.dst_space = {
            "XYZ": params.get("inclXYZ", False),
            "LAB": params.get("inclLAB", False),
            "LCH": params.get("inclLCH", False),
            "HEX": params.get("inclHEX", False),
        }

        lines = self.cgatstext.split("\n")
        self.cgats = Cgats(lines)

        self.entries = []
        self.result = self._convert()

    @property
    def get_result(self):
        return self.result

    def _convert(self):

        try:
            cgats_table = self.cgats.row_all()

            pcs_values = [entry.get("pcs", []) for entry in cgats_table]
            dcs_values = [entry.get("dcs", []) for entry in cgats_table]

            result = [
                {"dcs": dcs or [], "pcs": pcs}
                for dcs, pcs in zip(dcs_values, pcs_values)
            ]
            self.entries = result

            # check if dcs_values are all array have length greater 0
            if all(
                isinstance(item.get("dcs", []), list) and len(item["dcs"]) > 0
                for item in result
            ):

                if self.doublets_average:
                    from collections import defaultdict

                    dcs_map = defaultdict(list)

                    for item in result:
                        dcs_key = item.get("dcs")
                        if isinstance(dcs_key, list):
                            dcs_key = tuple(
                                dcs_key
                            )  # Convert lists to tuples for hashing
                        dcs_map[dcs_key].append(item)

                    averaged_result = [
                        (
                            {
                                "pcs": [
                                    sum(values) / len(values)
                                    for values in zip(*[x["pcs"] for x in items])
                                ],
                                "dcs": dcs,
                            }
                            if dcs is not None and len(items) > 1
                            else items[0]
                        )
                        for dcs, items in dcs_map.items()
                    ]

                    result[:] = averaged_result

                if self.doublets_remove:
                    seen_dcs = set()
                    removed_result = []

                    for item in result:
                        try:
                            dcs_key = item["dcs"]
                            if isinstance(dcs_key, list):  # Ensure hashability
                                dcs_key = tuple(dcs_key)
                        except KeyError:
                            removed_result.append(item)
                            continue

                        if dcs_key not in seen_dcs:
                            seen_dcs.add(dcs_key)
                            removed_result.append(item)

                    # Modify result in-place using slice assignment (faster in AWS Lambda)
                    result[:] = removed_result

                if self.reduction_type == "corner":
                    result = self.extract_corner(result)

                if self.reduction_type == "full":
                    result = self.extract_full(result)

                if self.reduction_type == "substrate":
                    result = self.extract_substrate(result)

                if self.reduction_type == "primaries":
                    result = self.extract_primes(result)

                if self.reduction_type == "closest":
                    result = self.extract_closestDCS(result, self.dst_dcs)

            # if self.dst_space is all False, return only the actual list of colors
            '''if not any(self.dst_space.values()):
                return result

            spectrals = [item["pcs"] for item in result]
            result_2 = Cs_Spectral2Multi(spectrals, self.dst_space)

            for i in range(len(result)):
                result_2[i]["dcs"] = result[i]["dcs"

            return result_2'''
            return self.convertSpace(result, self.dst_space)

        except Exception as e:
            return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    @staticmethod
    def convertSpace(result, dst_space):
        # if self.dst_space is all False, return only the actual list of colors
        if not any(dst_space.values()):
            return result

        spectrals = [item["pcs"] for item in result]
        result_2 = Cs_Spectral2Multi(spectrals, dst_space)

        for i in range(len(result)):
            result_2[i]["dcs"] = result[i]["dcs"]

        return result_2


    '''def _convertOnlyPCS(self):

        try:
            pcs_values = self.cgats.get_values_pcs()

            if not any(self.dst_space.values()):
                return pcs_values

            return Cs_Spectral2Multi(pcs_values, self.dst_space)

        except Exception as e:
            return {"statusCode": 500, "body": json.dumps({"error": str(e)})}'''

    @staticmethod
    def is_extractable(data) -> bool:
        """
        Checks if the data is extractable

        :param data: list of dictionaries with dcs and pcs values
        """

        if not isinstance(data, list):
            return False

        return all(
            isinstance(item.get("dcs", []), list) and len(item["dcs"]) > 0
            for item in data
        )

    @staticmethod
    def extract_primes(data) -> list:
        """
        Extracts only the primaries from the data

        :param data: list of dictionaries with dcs and pcs values
        """

        """ valid = CgatsToJson.is_extractable(data)
        if not valid:
            return data """

        mylist = [
            element
            for element in data
            if sum(1 for dcs in element["dcs"] if dcs > 0) == 1
        ]
        return linearization_sort(mylist)

    @staticmethod
    def extract_substrate(data) -> list:
        """
        Extracts only the substrate from the data

        :param data: list of dictionaries with dcs and pcs values
        """

        """ valid = CgatsToJson.is_extractable(data)
        if not valid:
            return data """

        mylist = [
            element
            for element in data
            if all(dcs == 0 for dcs in element["dcs"]) and sum(element["dcs"]) == 0
        ]
        return mylist

    @staticmethod
    def extract_full(data) -> list:
        """
        Extracts only the full colors from the data

        :param data: list of dictionaries with dcs and pcs values
        """

        """ valid = CgatsToJson.is_extractable(data)
        if not valid:
            return data """

        mylist = [
            element
            for element in data
            if all(dcs == 100 or dcs == 0 for dcs in element["dcs"])
            and sum(element["dcs"]) == 100
        ]
        return neugebauer_sort(mylist)

    @staticmethod
    def extract_corner(data) -> list:
        """
        Extracts only the corner colors from the data

        :param data: list of dictionaries with dcs and pcs values
        """

        """ valid = CgatsToJson.is_extractable(data)
        if not valid:
            return data """

        mylist = [
            element
            for element in data
            if all(dcs == 100 or dcs == 0 for dcs in element["dcs"])
        ]
        return neugebauer_sort(mylist)

    @staticmethod
    def extract_closestDCS(data: list, dcs: list) -> list:
        """
        Extracts the closest color to the given dcs from the data

        :param data: list of dictionaries with dcs and pcs values
        :param dcs: list of dcs values to find the closest color
        """

        """ valid = CgatsToJson.is_extractable(data)
        if not valid:
            return data """

        def distance(dcs1, dcs2):
            return sum((a - b) ** 2 for a, b in zip(dcs1, dcs2)) ** 0.5

        """closest_color = min(data, key=lambda x: distance(x['dcs'], dcs))
        return closest_color"""
        closest_colors = []
        for target_dcs in dcs:
            closest_color = min(data, key=lambda x: distance(x["dcs"], target_dcs))
            closest_colors.append(closest_color)

        return closest_colors


    DCS_CMYK_48 = [
        [0, 0, 0, 0],
        [33, 0, 0, 0],
        [66, 0, 0, 0],
        [100, 0, 0, 0],
        [0, 33, 0, 0],
        [0, 66, 0, 0],
        [0, 100, 0, 0],
        [0, 0, 33, 0],
        [0, 0, 66, 0],
        [0, 0, 100, 0],
        [0, 0, 0, 33],
        [0, 0, 0, 66],
        [0, 0, 0, 100],
        [33, 33, 0, 0],
        [66, 66, 0, 0],
        [100, 100, 0, 0],
        [0, 33, 33, 0],
        [0, 66, 66, 0],
        [0, 100, 100, 0],
        [33, 0, 33, 0],
        [66, 0, 66, 0],
        [100, 0, 100, 0],
        [100, 33, 0, 0],
        [100, 66, 0, 0],
        [0, 100, 33, 0],
        [0, 100, 66, 0],
        [100, 0, 33, 0],
        [100, 0, 66, 0],
        [33, 100, 0, 0],
        [66, 100, 0, 0],
        [0, 33, 100, 0],
        [0, 66, 100, 0],
        [33, 0, 100, 0],
        [66, 0, 100, 0],
        [50, 0, 0, 33],
        [100, 0, 0, 33],
        [0, 50, 0, 33],
        [0, 100, 0, 33],
        [0, 0, 50, 33],
        [0, 0, 100, 33],
        [50, 50, 0, 33],
        [100, 100, 0, 33],
        [0, 50, 50, 33],
        [0, 100, 100, 33],
        [50, 0, 50, 33],
        [100, 0, 100, 33],
        [100, 50, 0, 33],
        [0, 100, 50, 33],
        [100, 0, 50, 33],
        [50, 100, 0, 33],
        [0, 50, 100, 33],
        [50, 0, 100, 33],
        
        [50, 0, 0, 66],
        [100, 0, 0, 66],
        [0, 50, 0, 66],
        [0, 100, 0, 66],
        [0, 0, 50, 66],
        [0, 0, 100, 66],
        [50, 50, 0, 66],
        [100, 100, 0, 66],
        [0, 50, 50, 66],
        [0, 100, 100, 66],
        [50, 0, 50, 66],
        [100, 0, 100, 66],
        [100, 50, 0, 66],
        [0, 100, 50, 66],
        [100, 0, 50, 66],
        [50, 100, 0, 66],
        [0, 50, 100, 66],
        [50, 0, 100, 66],
        
        [100, 0, 0, 100],
        [0, 100, 0, 100],
        [0, 0, 100, 100],
        
        [100, 100, 100, 0],
        [100, 100, 0, 100],
        [100, 0, 100, 100],
        [0, 100, 100, 100],
        
        [100, 100, 100, 100],
        
        
        #random mix of colors with 4 
        [33, 33, 33, 33],
        [66, 66, 66, 66],
        [50, 50, 50, 50],
        [25, 25, 25, 25],
        [75, 75, 75, 75]
    ]
    
    DCS_CMYK_88 = [
        [0, 0, 0, 0],
        [33, 0, 0, 0],
        [66, 0, 0, 0],
        [100, 0, 0, 0],
        [0, 33, 0, 0],
        [0, 66, 0, 0],
        [0, 100, 0, 0],
        [0, 0, 33, 0],
        [0, 0, 66, 0],
        [0, 0, 100, 0],
        [0, 0, 0, 33],
        [0, 0, 0, 66],
        [0, 0, 0, 100],
        [33, 33, 0, 0],
        [66, 66, 0, 0],
        [100, 100, 0, 0],
        [0, 33, 33, 0],
        [0, 66, 66, 0],
        [0, 100, 100, 0],
        [33, 0, 33, 0],
        [66, 0, 66, 0],
        [100, 0, 100, 0],
        [100, 33, 0, 0],
        [100, 66, 0, 0],
        [0, 100, 33, 0],
        [0, 100, 66, 0],
        [100, 0, 33, 0],
        [100, 0, 66, 0],
        [33, 100, 0, 0],
        [66, 100, 0, 0],
        [0, 33, 100, 0],
        [0, 66, 100, 0],
        [33, 0, 100, 0],
        [66, 0, 100, 0],
        [50, 0, 0, 33],
        [100, 0, 0, 33],
        [0, 50, 0, 33],
        [0, 100, 0, 33],
        [0, 0, 50, 33],
        [0, 0, 100, 33],
        [50, 50, 0, 33],
        [100, 100, 0, 33],
        [0, 50, 50, 33],
        [0, 100, 100, 33],
        [50, 0, 50, 33],
        [100, 0, 100, 33],
        [100, 50, 0, 33],
        [0, 100, 50, 33],
        [100, 0, 50, 33],
        [50, 100, 0, 33],
        [0, 50, 100, 33],
        [50, 0, 100, 33],
        
        [50, 0, 0, 66],
        [100, 0, 0, 66],
        [0, 50, 0, 66],
        [0, 100, 0, 66],
        [0, 0, 50, 66],
        [0, 0, 100, 66],
        [50, 50, 0, 66],
        [100, 100, 0, 66],
        [0, 50, 50, 66],
        [0, 100, 100, 66],
        [50, 0, 50, 66],
        [100, 0, 100, 66],
        [100, 50, 0, 66],
        [0, 100, 50, 66],
        [100, 0, 50, 66],
        [50, 100, 0, 66],
        [0, 50, 100, 66],
        [50, 0, 100, 66],
        
        [100, 0, 0, 100],
        [0, 100, 0, 100],
        [0, 0, 100, 100],
        [100, 100, 100, 0],
        [100, 100, 0, 100],
        [100, 0, 100, 100],
        [0, 100, 100, 100],
        [100, 100, 100, 100]
    ]
    
    DCS_CMYK_66 = [
        [0, 0, 0, 0],
        [33, 0, 0, 0],
        [66, 0, 0, 0],
        [100, 0, 0, 0],
        [0, 33, 0, 0],
        [0, 66, 0, 0],
        [0, 100, 0, 0],
        [0, 0, 33, 0],
        [0, 0, 66, 0],
        [0, 0, 100, 0],
        [0, 0, 0, 33],
        [0, 0, 0, 66],
        [0, 0, 0, 100],
        [33, 33, 0, 0],
        [66, 66, 0, 0],
        [100, 100, 0, 0],
        [0, 33, 33, 0],
        [0, 66, 66, 0],
        [0, 100, 100, 0],
        [33, 0, 33, 0],
        [66, 0, 66, 0],
        [100, 0, 100, 0],
        [100, 33, 0, 0],
        [100, 66, 0, 0],
        [0, 100, 33, 0],
        [0, 100, 66, 0],
        [100, 0, 33, 0],
        [100, 0, 66, 0],
        [33, 100, 0, 0],
        [66, 100, 0, 0],
        [0, 33, 100, 0],
        [0, 66, 100, 0],
        [33, 0, 100, 0],
        [66, 0, 100, 0],
        [50, 0, 0, 33],
        [100, 0, 0, 33],
        [0, 50, 0, 33],
        [0, 100, 0, 33],
        [0, 0, 50, 33],
        [0, 0, 100, 33],
        [50, 50, 0, 33],
        [100, 100, 0, 33],
        [0, 50, 50, 33],
        [0, 100, 100, 33],
        [50, 0, 50, 33],
        [100, 0, 100, 33],
        [100, 50, 0, 33],
        [0, 100, 50, 33],
        [100, 0, 50, 33],
        [50, 100, 0, 33],
        [0, 50, 100, 33],
        [50, 0, 100, 33],
        [100, 0, 0, 66],
        [0, 100, 0, 66],
        [0, 0, 100, 66],
        [100, 100, 0, 66],
        [0, 100, 100, 66],
        [100, 0, 100, 66], 
        [100, 0, 0, 100],
        [0, 100, 0, 100],
        [0, 0, 100, 100],
        [100, 100, 100, 0],
        [100, 100, 0, 100],
        [100, 0, 100, 100],
        [0, 100, 100, 100],
        [100, 100, 100, 100]
    ]

    DCS_CMYK_48a = [
        [0, 0, 0, 0],
        [33, 0, 0, 0],
        [66, 0, 0, 0],
        [100, 0, 0, 0],
        [0, 33, 0, 0],
        [0, 66, 0, 0],
        [0, 100, 0, 0],
        [0, 0, 33, 0],
        [0, 0, 66, 0],
        [0, 0, 100, 0],
        [0, 0, 0, 33],
        [0, 0, 0, 66],
        [0, 0, 0, 100],
        [33, 33, 0, 0],
        [66, 66, 0, 0],
        [100, 100, 0, 0],
        [0, 33, 33, 0],
        [0, 66, 66, 0],
        [0, 100, 100, 0],
        [33, 0, 33, 0],
        [66, 0, 66, 0],
        [100, 0, 100, 0],
        [100, 33, 0, 0],
        [100, 66, 0, 0],
        [0, 100, 33, 0],
        [0, 100, 66, 0],
        [100, 0, 33, 0],
        [100, 0, 66, 0],
        [33, 100, 0, 0],
        [66, 100, 0, 0],
        [0, 33, 100, 0],
        [0, 66, 100, 0],
        [33, 0, 100, 0],
        [66, 0, 100, 0],
        [50, 0, 0, 33],
        [100, 0, 0, 33],
        [0, 50, 0, 33],
        [0, 100, 0, 33],
        [0, 0, 50, 33],
        [0, 0, 100, 33],
        [50, 50, 0, 33],
        [100, 100, 0, 33],
        [0, 50, 50, 33],
        [0, 100, 100, 33],
        [50, 0, 50, 33],
        [100, 0, 100, 33],
        [100, 50, 0, 33],
        [0, 100, 50, 33],
        [100, 0, 50, 33],
        [50, 100, 0, 33],
        [0, 50, 100, 33],
        [50, 0, 100, 33],
    ]
