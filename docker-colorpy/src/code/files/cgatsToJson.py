import json

from src.code.files.cgats import Cgats
from src.code.space.colorConverter import Cs_Spectral2Multi


def neugebauer_sort(colors):
    # Function to calculate the weight of the color (how many color channels are active)
    def color_weight(dcs):
        return sum(1 for val in dcs if val > 0)

    # Sort by color weight first, then by primary color values in the order: C > M > Y > K
    def sort_key(color):
        dcs = color['dcs']
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
        dcs = color['dcs']
        # Find the first nonzero index (which determines the primary component)
        first_nonzero_index = next((i for i, val in enumerate(dcs) if val > 0), len(dcs))
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
        
        self.dst_space = {
            "XYZ": params.get("inclXYZ", False),
            "LAB": params.get("inclLAB", False),
            "LCH": params.get("inclLCH", False),
            "HEX": params.get("inclHEX", False)
        }
        
        self.result = self._convert()

    @property
    def get_result(self):
        return self.result
    
    def _convert(self):

        try:
            
            # Process text data line by line
            lines = self.cgatstext.split("\n")
            myCgats = Cgats(lines)

            # cgats_head    = myCgats.get_header()
            # cgats_typeDCS = myCgats.get_type_dcs()
            cgats_typePCS = myCgats.get_type_pcs()
            cgats_table = myCgats.row_all()

            pcs_values = [entry.get("pcs", []) for entry in cgats_table]
            dcs_values = [entry.get("dcs", []) for entry in cgats_table]

            result = []
            
            for i in range(len(pcs_values)):
                res = {
                    "dcs": dcs_values[i],
                    "pcs": pcs_values[i]
                }
                result.append(res)

        
            if self.doublets_average:
                from collections import defaultdict

                dcs_map = defaultdict(list)

                for item in result:
                    dcs_key = item.get("dcs")
                    if isinstance(dcs_key, list):  
                        dcs_key = tuple(dcs_key)  # Convert lists to tuples for hashing
                    dcs_map[dcs_key].append(item)

                averaged_result = [
                    {"pcs": [sum(values) / len(values) for values in zip(*[x["pcs"] for x in items])], "dcs": dcs}
                    if dcs is not None and len(items) > 1 else items[0]
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
                mylist = [
                    element for element in result 
                    if all(dcs == 100 or dcs == 0 for dcs in element['dcs'])
                ]
                result = neugebauer_sort(mylist)
                
            if self.reduction_type == "full":
                mylist = [
                    element for element in result 
                    if all(dcs == 100 or dcs == 0 for dcs in element['dcs']) and sum(element['dcs']) == 100
                ]
                result = neugebauer_sort(mylist)

            if self.reduction_type == "substrate":
                mylist = [
                    element for element in result 
                    if all(dcs == 0 for dcs in element['dcs']) and sum(element['dcs']) == 0
                ]
                result = neugebauer_sort(mylist)
                
            if self.reduction_type == "primaries":
                mylist = [
                    element for element in result 
                    if sum(1 for dcs in element['dcs'] if dcs > 0) == 1
                ]
                result = linearization_sort(mylist)
            
            # if self.dst_space is all False, return only the actual list of colors
            if not any(self.dst_space.values()):
                return result
            
            spectrals = [item["pcs"] for item in result]
            result_2 = Cs_Spectral2Multi(spectrals, self.dst_space)
            
            for i in range(len(result)):
                result_2[i]["dcs"] = result[i]["dcs"]
            
            return result_2

        except Exception as e:
            return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
