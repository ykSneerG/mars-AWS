from src.code.space.colorSample import SampleSpectral
from src.code.files.botox import Botox
from src.code.files.cgatsToJson import CgatsToJson
from src.handlers import BaseLambdaHandler
from src.code.space.colorConverterNumpy import ColorTrafoNumpy
from src.code.delta.colorDifference import ColorDiffernce
from src.code.space.colorSpace import CsLAB, CsRGB
import json
import numpy as np  # type: ignore

class IntersectionDelta_Handler(BaseLambdaHandler):
    
    def __init__(self, event, context):
        super().__init__(event, context)
        self.sam_id = None
        self.ref_id = None
        self.bucket = "mars-predicted-data"
        
    
    def _bucket_to_json(self, upload_id):
        try:
            txt_value = Botox(self.bucket).load_S3(f"data/{upload_id}.txt")
        except Exception as e:
            return self.get_error_response(str(e))

        try:
            entries = CgatsToJson({
                "txt": txt_value,
                "doublets_average": True,
                "doublets_remove": True
            }).entries
        
            return entries
        except Exception as e:
            return self.get_error_response(str(e))


    def distribution_from_data(self, data):
        # Prozentuale Schwellen definieren
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

        # Perzentilwerte berechnen
        percentile_values = np.percentile(data, percentiles)

        result = {}
        # Ausgabe
        for p, val in zip(percentiles, percentile_values):
            result[p] = val
        return result

    def handle(self):
        #from src.code.space.intersection import IntersectionDelta
        
        self.sam_id = self.event.get("sampleId", 0)
        if self.sam_id == 0:
            return self.get_error_response("No sampleId provided")
        
        self.ref_id = self.event.get("referenceId", 0)
        if self.ref_id == 0:
            return self.get_error_response("No referenceId provided")
        
        
        
        jd = {}
        
        color_sam = self._bucket_to_json(self.sam_id)
        if isinstance(color_sam, dict) and color_sam.get("statusCode") == 400:
            return color_sam
        
        color_ref = self._bucket_to_json(self.ref_id)
        if isinstance(color_ref, dict) and color_ref.get("statusCode") == 400:
            return color_ref


        # find all objects of the two arrays
        """ 
        Object
        dcs: [0, 0, 0, 0] (4)
        pcs: [0.3468, 0.4072, 0.5049, 0.7031, 0.9137, 0.9884, 0.9974, 0.9716, 0.9453, 0.9267, …] (36)
        """
        intersection = []
        for sam_obj in color_sam:
            for ref_obj in color_ref:
                if sam_obj["dcs"] == ref_obj["dcs"]:
                    intersection.append({
                        "dcs": sam_obj["dcs"],
                        "pcs_sam": sam_obj["pcs"],
                        "pcs_ref": ref_obj["pcs"]
                    })


        pcs_intersect_ref = [obj["pcs_ref"] for obj in intersection]
        pcs_intersect_sam = [obj["pcs_sam"] for obj in intersection]
        dcs_intersect = [obj["dcs"] for obj in intersection]
        
        trafo = ColorTrafoNumpy()
        
        col_intersect_ref = trafo.Cs_SNM2MULTI_NP(pcs_intersect_ref, {"SNM": True, "LCH": True, "HEX": True, "LAB": True})
        for entry, dcs in zip(col_intersect_ref, dcs_intersect):
            entry.update({"dcs": dcs})
        
        col_intersect_sam = trafo.Cs_SNM2MULTI_NP(pcs_intersect_sam, {"SNM": True, "LCH": True, "HEX": True, "LAB": True})
        for entry, dcs in zip(col_intersect_sam, dcs_intersect):
            entry.update({"dcs": dcs})

        col_intersect_delta = []
        for( col_ref, col_sam) in zip(col_intersect_ref, col_intersect_sam):
            lab_ref = CsLAB(col_ref["lab"][0], col_ref["lab"][1], col_ref["lab"][2])
            lab_sam = CsLAB(col_sam["lab"][0], col_sam["lab"][1], col_sam["lab"][2])
            de_trafo = ColorDiffernce(lab_ref, lab_sam)
            #col_ref["dE00"] = de_trafo.de00()

            de00 = de_trafo.de00()
            max_de00 = 12
            act_de00 = de00 if de00 < max_de00 else max_de00
            r = act_de00 / max_de00 * 255
            rgb = (255, 255 - int(r), 255 - int(r))
            hex = CsRGB(rgb[0], rgb[1], rgb[2]).to_hex()


            col_intersect_delta.append({
                "dE00": de00,
                "dE00HEX": hex
            })
            
        de00_values = [col["dE00"] for col in col_intersect_delta]
        de00_average = sum(de00_values) / len(col_intersect_delta) if col_intersect_delta else 0
        de00_95quantile = np.percentile(de00_values, 95) if col_intersect_delta else 0
        de00_50quantile = np.percentile(de00_values, 50) if col_intersect_delta else 0
        de00_max = max(de00_values) if col_intersect_delta else 0
        de00_min = min(de00_values) if col_intersect_delta else 0
        de00_median = np.median(de00_values) if col_intersect_delta else 0
        de00_iqr = np.percentile(de00_values, 75) - np.percentile(de00_values, 25) if col_intersect_delta else 0
        de00_sigma = np.std(de00_values, ddof=1) if col_intersect_delta else 0
        de00_distribution = self.distribution_from_data(de00_values)

        result = []
        for (col_ref, col_sam, col_intersect_delta) in zip(col_intersect_ref, col_intersect_sam, col_intersect_delta):
            result.append({
                "sample": col_sam,
                "reference": col_ref,
                "dE00": col_intersect_delta["dE00"],
                "dE00HEX": col_intersect_delta["dE00HEX"],
            })

        jd = {
            "id": self.event.get("id", None),
            "sampleId": self.sam_id,
            "referenceId": self.ref_id,
            #"sample": color_sam,
            #"reference": color_ref,
            "intersection": result,
            "delta": {
                "de00Average": de00_average,
                "de00Median": de00_median,
                "de00Quantile50": de00_50quantile,
                "de00Quantile95": de00_95quantile,
                "de00Sigma": de00_sigma,
                "de00IQR": de00_iqr,
                "de00Max": de00_max,
                "de00Min": de00_min,
                "de00Distribution": de00_distribution
            },
            "message": "Intersection delta calculated successfully.",
            "elapsed": self.get_elapsed_time()
        }
        
        
        return self.get_common_response(jd)