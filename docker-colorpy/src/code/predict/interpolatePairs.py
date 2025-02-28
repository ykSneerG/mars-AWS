from src.handlers import BaseLambdaHandler, SlsHelper

class InterpolatePairs(BaseLambdaHandler):

    def handle(self):

        rail = self.event.get("rail", [])
        if len(rail) == 0:
            return self.get_error_response("No pairs provided.")

        debug = self.event.get("debug", False)
        space = self.event.get("space", "XYZ")
        preci = self.event.get("precision", 100)
        toler = self.event.get("tolerance", 0.00025)
        steps = self.event.get("steps", 5)
        id = self.event.get("id", -1)
        
        dst_types = self.event.get("destination_types", {"SNM": True, "LCH": True, "HEX": True})

        SLS_PARAMS = (debug, space, preci, toler, steps)

        jd = {"id": id}

        try:
            jd.update({"color": SlsHelper.process_colors_batch(SLS_PARAMS, rail, dst_types)})
        except Exception as e:
            return self.get_error_response(str(e))

        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)
