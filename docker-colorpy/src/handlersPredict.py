from concurrent.futures import ThreadPoolExecutor
from typing import Any
from src.handlers import BaseLambdaHandler
from src.code.predict.linearization.linearInterpolation import LinearInterpolation
from src.code.predict.linearization.synlinV4a import SynLinSolidV4a

import numpy as np  # type: ignore


class Predict_LinearInterpolation_Handler(BaseLambdaHandler):
    def handle(self):
        jd = {}

        media = self.event["media"]
        solid = self.event["solid"]
        steps = int(self.event.get("steps", 5))
        iters = int(self.event.get("iterations", 1))
        toler = float(self.event.get("tolerance", 0.004))

        sls = LinearInterpolation()
        """ sls.set_places = int(event.get('round', 3))  """
        sls.set_media(media)
        sls.set_solid(solid)

        err = sls.set_gradient_by_steps(steps)
        if err:
            return self.get_error_response(err)

        # sls.set_gradient([0.0, 50.0, 100.0])
        sls.tolerance = toler
        sls.set_max_loops = iters
        res = sls.start()

        jd.update(res)

        return self.get_common_response(jd)


class Predict_SynlinV4_Handler(BaseLambdaHandler):

    def handle(self):

        jd = {}

        media = self.event["media"]
        solid = self.event["solid"]
        steps = int(self.event.get("steps", 5))
        iters = int(self.event.get("iterations", 1))
        toler = float(self.event.get("tolerance", 0.004))
        preci = int(self.event.get("precision", 3))
        """ cfact = float(self.event.get('correction', 4.5)) """
        debug = self.event.get("debug", False)
        space = self.event.get("space", "XYZ")

        sls = SynLinSolidV4a()
        sls.set_destination_types({ "LCH": True, "HEX": True, "SNM": True })
        sls.set_debug(debug)
        sls.set_space(space)
        """ sls.set_places = int(event.get('round', 3))  """
        sls.set_precision(preci)

        sls.set_media(media)
        sls.set_solid(solid)

        err = sls.set_gradient_by_steps(steps)
        if err:
            return self.get_error_response(err)

        # sls.set_gradient([0.0, 50.0, 100.0])
        sls.tolerance = toler
        sls.set_max_loops(iters)
        """ sls.setCorrection(cfact) """
        # res: dict[str, Any] = sls.start()
        res: dict[str, Any] = sls.start_Curve3D()

        jd.update(res)

        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)


class Predict_SynAreaV4_Handler(BaseLambdaHandler):

    def handle(self):

        jd = {}

        c1 = self.event["c1"]
        c2 = self.event["c2"]
        c3 = self.event["c3"]
        """ c4 = self.event['c4'] """

        steps = int(self.event.get("steps", 5))
        iters = int(self.event.get("iterations", 1))
        toler = float(self.event.get("tolerance", 0.004))
        preci = int(self.event.get("precision", 3))
        """ cfact = float(self.event.get('correction', 4.5)) """
        debug = self.event.get("debug", False)
        space = self.event.get("space", "XYZ")

        def initClass():
            sls = SynLinSolidV4a()
            sls.set_destination_types({ "SNM": True })
            sls.set_debug(debug)
            sls.set_space(space)
            sls.set_precision(preci)
            err = sls.set_gradient_by_steps(steps)
            if err:
                return self.get_error_response(err)
            sls.tolerance = toler
            return sls

        sls = initClass()
        sls.set_gradient_by_steps(3)
        sls.set_media(c2)
        sls.set_solid(c3)
        res: dict[str, Any] = sls.start_Curve3D()
        c4 = res["color"][1]["snm"]

        media = [c1, c2]
        solid = [c3, c4]

        rail = []
        for i in range(len(media)):
            sls = initClass()
            sls.set_media(media[i])
            sls.set_solid(solid[i])
            res: dict[str, Any] = sls.start_Curve3D()
            rail.append(res)

        def process_colors(i):
            sls = initClass()
            sls.set_destination_types({ "SNM": True, "LCH": True, "HEX": True })
            sls.set_media(rail[0]["color"][i]["snm"])
            sls.set_solid(rail[1]["color"][i]["snm"])
            res = sls.start_Curve3D()
            return res["color"]

        with ThreadPoolExecutor() as executor:
            colors = (
                np.array(
                    list(executor.map(process_colors, range(len(rail[0]["color"]))))
                )
                .ravel()
                .tolist()
            )

            jd.update({"color": colors})

        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)


class Predict_SynVolumeV4_Handler(BaseLambdaHandler):

    def handle(self):

        jd = {}

        c1 = self.event["c1"]
        c2 = self.event["c2"]
        c3 = self.event["c3"]
        """ c4 = self.event['c4'] """
        c5 = self.event["c5"]

        steps = int(self.event.get("steps", 5))
        iters = int(self.event.get("iterations", 1))
        toler = float(self.event.get("tolerance", 0.004))
        preci = int(self.event.get("precision", 3))
        """ cfact = float(self.event.get('correction', 4.5)) """
        debug = self.event.get("debug", False)
        space = self.event.get("space", "XYZ")

        def initClass():
            sls = SynLinSolidV4a()
            sls.set_debug(debug)
            sls.set_space(space)
            sls.set_precision(preci)

            sls.set_destination_types({ "XYZ": False, "LAB": False, "LCH": False, "HEX": False, "SNM": True })

            err = sls.set_gradient_by_steps(steps)
            if err:
                return self.get_error_response(err)
            sls.tolerance = toler
            return sls

        # --- 1. PREDICT c4 by mixing c2 and c3 ---
        # after this the four corners of the area are defined, where c1 is treated as the substrate color

        sls = initClass()
        sls.set_gradient_by_steps(3)
        sls.set_media(c2)
        sls.set_solid(c3)
        res: dict[str, Any] = sls.start_Curve3D()
        c4 = res["color"][1]["snm"]

        media = [c1, c2]
        solid = [c3, c4]

        # --- 2. PREDICT TOWER ---
        # predict the corner colors of the top area, where c5 is located
        # tower 15, 25, 35, 45

        # Tower Corners c5, c6, c7, c8

        sls = initClass()
        sls.set_gradient_by_steps(3)
        sls.set_media(c2)
        sls.set_solid(c5)
        res: dict[str, Any] = sls.start_Curve3D()
        c6 = res["color"][1]["snm"]

        sls = initClass()
        sls.set_gradient_by_steps(3)
        sls.set_media(c4)
        sls.set_solid(c5)
        res: dict[str, Any] = sls.start_Curve3D()
        c7 = res["color"][1]["snm"]

        sls = initClass()
        sls.set_gradient_by_steps(3)
        sls.set_media(c3)
        sls.set_solid(c5)
        res: dict[str, Any] = sls.start_Curve3D()
        c8 = res["color"][1]["snm"]

        towercorner = [[c1, c5], [c3, c8], [c2, c6], [c4, c7]]

        towergradient = []
        for i in range(len(towercorner)):
            sls = initClass()
            sls.set_media(towercorner[i][0])
            sls.set_solid(towercorner[i][1])
            res: dict[str, Any] = sls.start_Curve3D()
            towergradient.append(res["color"])

        # jd.update({ "tower": towergradient })

        """ tower_L = [[c1, c5], [c3, c8]]
        toler_R = [[c2, c6], [c4, c7]] """

        # --- 2. PREDICT RAIL ---
        # the left rail is defined by c1 and c3, the right rail by c2 and c4

        rail = [[], []]
        for i in range(len(towergradient[0])):
            sls = initClass()
            sls.set_media(towergradient[0][i]["snm"])
            sls.set_solid(towergradient[1][i]["snm"])
            res: dict[str, Any] = sls.start_Curve3D()
            for item in res["color"]:
                rail[0].append(item["snm"])

            sls = initClass()
            sls.set_media(towergradient[2][i]["snm"])
            sls.set_solid(towergradient[3][i]["snm"])
            res: dict[str, Any] = sls.start_Curve3D()
            for item in res["color"]:
                rail[1].append(item["snm"])

        # jd.update({ "rail": rail })

        # --- 3. PREDICT COLORS ---
        # the colors are predicted by mixing the colors of the left and right rail

        def process_colors(i):
            sls = initClass()
            sls.set_destination_types({ "SNM": True, "LCH": True, "HEX": True })
            sls.set_media(rail[0][i])
            sls.set_solid(rail[1][i])
            res = sls.start_Curve3D()
            return res["color"]

        with ThreadPoolExecutor() as executor:
            colors = (
                np.array(list(executor.map(process_colors, range(len(rail[0])))))
                .ravel()
                .tolist()
            )

            jd.update({"color": colors})

        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)
