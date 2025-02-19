import json
import boto3  # type: ignore

""" import base64 """
from src.code.files.cgats import Cgats
from src.code.space.colorConverter import CS_Spectral2XYZ, Cs_XYZ2LAB, Cs_Lab2XYZ, Cs_XYZ2RGB, Cs_XYZ2LCH
from src.code.space.colorSpace import CsLCH, CsSpectral, CsXYZ, CsLAB
from src.code.space.colorConstants.illuminant import OBSERVER, Illuminant

from src.handlers import BaseLambdaHandler


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


class Files_CgatsToJson_Handler(BaseLambdaHandler):
    def handle(self):

        jd = {}

        try:
            json_data = self.event

            # Extract values from the JSON part
            txt_value = json_data.get("txt", "")
            fid_value = json_data.get("fid", "")
            doublets_average = json_data.get("doublets_average", False)
            doublets_remove = json_data.get("doublets_remove", False)
            observer_value = json_data.get("observer", OBSERVER.DEG2)
            
            reduction_type = json_data.get("reduction_type", None)
            
            inclDCS_value = json_data.get("inclDCS", False)
            inclLAB_value = json_data.get("inclLAB", False)
            inclLCH_value = json_data.get("inclLCH", False)
            inclXYZ_value = json_data.get("inclXYZ", False)
            inclRGB_value = json_data.get("inclRGB", False)
            inclDensity_value = json_data.get("inclDensity", False)
            inclHEX_value = json_data.get("inclHEX", False)
            inclSourceValues_value = json_data.get("inclSourceValues", False)

            # drop3din
            """ clientS3 = boto3.client("s3")
            s3_upload = clientS3.put_object(
                Bucket="drop3din",
                Key=fid_value,
                Body=txt_value,
                ContentType="text/plain",
            ) """

            # Process text data line by line
            lines = txt_value.split("\n")

            myCgats = Cgats(lines)

            # cgats_head    = myCgats.get_header()
            # cgats_typeDCS = myCgats.get_type_dcs()
            cgats_typePCS = myCgats.get_type_pcs()
            cgats_table = myCgats.row_all()

            pcs_values = [entry.get("pcs", []) for entry in cgats_table]
            dcs_values = [entry.get("dcs", []) for entry in cgats_table]

            result = None

            if cgats_typePCS == "SNM":
                result = self.snmToXyz(pcs_values, dcs_values, observer_value)
            elif cgats_typePCS == "XYZ":
                result = self.xyzColor(pcs_values, dcs_values)
            elif cgats_typePCS == "LAB":
                result = self.labColor(pcs_values, dcs_values)

                      
            if doublets_average:
                from collections import defaultdict

                dcs_map = defaultdict(list)

                for item in result:
                    dcs_key = item.get("dcs")
                    if isinstance(dcs_key, list):  
                        dcs_key = tuple(dcs_key)  # Convert lists to tuples for hashing
                    dcs_map[dcs_key].append(item)

                averaged_result = [
                    {"snm": [sum(values) / len(values) for values in zip(*[x["snm"] for x in items])], "dcs": dcs}
                    if dcs is not None and len(items) > 1 else items[0]
                    for dcs, items in dcs_map.items()
                ]

                result[:] = self.snmToXyz(
                    [x["snm"] for x in averaged_result], 
                    [x.get("dcs") for x in averaged_result], 
                    observer_value
                )
            
            if doublets_remove:
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


            if reduction_type == "corner":
                mylist = [
                    element for element in result 
                    if all(dcs == 100 or dcs == 0 for dcs in element['dcs'])
                ]
                result = neugebauer_sort(mylist)
                
            if reduction_type == "full":
                mylist = [
                    element for element in result 
                    if all(dcs == 100 or dcs == 0 for dcs in element['dcs']) and sum(element['dcs']) == 100
                ]
                result = neugebauer_sort(mylist)

            if reduction_type == "substrate":
                mylist = [
                    element for element in result 
                    if all(dcs == 0 for dcs in element['dcs']) and sum(element['dcs']) == 0
                ]
                result = neugebauer_sort(mylist)
                

            jd.update({"result": result})
            jd.update({"elapsed": self.get_elapsed_time()})

            return self.get_common_response(jd)

        except Exception as e:
            return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    def snmToXyz(self, snmArr, dcsArr, observer=OBSERVER.DEG2):
        result = []
        i = 0
        for snm_entry in snmArr:
            res = {}
            xyz: CsXYZ = CS_Spectral2XYZ(snm_entry, observer)
            lab: CsLAB = Cs_XYZ2LAB(xyz, Illuminant.D50_DEG2)
            hex: str = Cs_XYZ2RGB(xyz).to_hex()
            lch: CsLCH = Cs_XYZ2LCH(xyz, Illuminant.D50_DEG2)

            #res["XYZ"] = self.minifyCs(xyz, 3).to_json()
            #res["LAB"] = self.minifyCs(lab, 3).to_json()
            res["snm"] = snm_entry
            res["lch"] = self.minifyCs(lch, 3).to_json()
            res["dcs"] = dcsArr[i]
            res["hex"] = hex
            i += 1

            result.append(res)
        return result

    def xyzColor(self, xyzArr, dcsArr):
        result = []
        i = 0
        for xyz_entry in xyzArr:
            res = {}
            xyz: CsXYZ = CsXYZ(xyz_entry[0], xyz_entry[1], xyz_entry[2])
            lab: CsLAB = Cs_XYZ2LAB(xyz, Illuminant.D50_DEG2)

            res["XYZ"] = self.minifyCs(xyz, 3).to_json()
            res["LAB"] = self.minifyCs(lab, 3).to_json()
            res["DCS"] = dcsArr[i]
            i += 1

            result.append(res)
        return result

    def labColor(self, labArr, dcsArr):
        result = []
        i = 0
        for lab_entry in labArr:
            res = {}
            lab: CsLAB = CsLAB(lab_entry[0], lab_entry[1], lab_entry[2])
            xyz: CsXYZ = Cs_Lab2XYZ(lab, Illuminant.D50_DEG2)

            res["XYZ"] = self.minifyCs(xyz, 3).to_json()
            res["LAB"] = self.minifyCs(lab, 3).to_json()
            res["DCS"] = dcsArr[i]
            i += 1

            result.append(res)
        return result

    def minifyCs(self, cs, precision):
        for key in cs.__dict__:
            setattr(cs, key, round(getattr(cs, key), precision))
        return cs
