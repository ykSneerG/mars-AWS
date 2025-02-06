import json
import boto3  # type: ignore

""" import base64 """
from src.code.files.cgats import Cgats
from src.code.space.colorConverter import CS_Spectral2XYZ, Cs_XYZ2LAB, Cs_Lab2XYZ, Cs_XYZ2RGB, CsXYZ2LCH
from src.code.space.colorSpace import CsLCH, CsSpectral, CsXYZ, CsLAB
from src.code.space.colorConstants.illuminant import OBSERVER, Illuminant


from src.handlers import BaseLambdaHandler


class Files_CgatsToJson_Handler(BaseLambdaHandler):
    def handle(self):

        jd = {}

        try:
            json_data = self.event

            # Extract values from the JSON part
            txt_value = json_data.get("txt", "")
            fid_value = json_data.get("fid", "")
            observer_value = json_data.get("observer", OBSERVER.DEG2)
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
            lch: CsLCH = CsXYZ2LCH(xyz, Illuminant.D50_DEG2)

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
