""" from concurrent.futures import ThreadPoolExecutor """
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


""" 
FROM HERE ONWARDS, THE CODE USES THE SynLinSolidV4a CLASS to predict n-dimensional colors.
"""

class GradientMixGenerator:
    
    @staticmethod
    def generate_dcs(gradient, dimension):
        """
        Generate a list of mixtures for a given gradient and dimension, without using itertools.

        :param gradient: A list or array of gradient points (values from 0 to 1).
        :param dimension: The dimension of the space (1D, 2D, 3D, etc.).
        :return: A 2D array of mixtures where each row represents a mixture.
        """
        # If gradient is a single value, repeat it to form a gradient
        if isinstance(gradient, (int, float)):
            gradient = [gradient]
        
        # Helper function to generate all Cartesian products recursively
        def cartesian_product(gradients, depth, current):
            # Base case: if we've reached the desired dimension, add the current combination
            if depth == dimension:
                result.append(current)
                return
            
            # Recur for all gradient values at this depth level
            for value in gradients:
                cartesian_product(gradients, depth + 1, current + [value])
        
        # To store the result
        result = []
        
        # Generate all Cartesian products starting from an empty combination
        cartesian_product(gradient, 0, [])
        
        # Convert the result into a numpy array for consistency
        return result


class SlsHelper:
    
    @staticmethod
    def evalEvent(event):
        steps = int(event.get("steps", 5))
        toler = float(event.get("tolerance", 0.00025))
        preci = int(event.get("precision", 100))
        debug = event.get("debug", False)
        space = event.get("space", "XYZ")
        return (debug, space, preci, toler, steps)
    
    @staticmethod
    def initClass(debug, space, preci = 100, toler = 0.00025):
        sls = SynLinSolidV4a()
        sls.set_debug(debug)
        sls.set_space(space)
        sls.set_precision(preci)
        sls.set_destination_types(
            {"XYZ": False, "LAB": False, "LCH": False, "HEX": False, "SNM": True}
        )
        sls.tolerance = toler
        return sls

    @staticmethod
    def mix_concentration(sls: SynLinSolidV4a, color1, color2, concentration):
        sls.set_gradient([concentration])
        sls.set_media(color1)
        sls.set_solid(color2)
        res: dict[str, Any] = sls.start_Curve3D()
        return res["color"][0]["snm"]

    @staticmethod
    def mix_1to1(sls, color1, color2):
        return SlsHelper.mix_concentration(sls, color1, color2, 0.5)

    @staticmethod
    def mix_2to1(sls, color1, color2):
        return SlsHelper.mix_concentration(sls, color1, color2, 0.67)

    @staticmethod
    def mix_3to1(sls, color1, color2):
        return SlsHelper.mix_concentration(sls, color1, color2, 0.75)

    @staticmethod
    def mix_all(sls, color1, color2, steps):
        sls.set_gradient_by_steps(steps)
        sls.set_media(color1)
        sls.set_solid(color2)
        res: dict[str, Any] = sls.start_Curve3D()
        return [item["snm"] for item in res["color"]]
    
    """ @staticmethod
    def process_colors(params, i, rail):
        
        (debug, space, preci, toler, steps) = params
        
        sls = SlsHelper.initClass(debug, space, preci, toler)
        sls.set_destination_types({"SNM": True, "LCH": True, "HEX": True})
        sls.set_gradient_by_steps(steps)
        sls.set_media(rail[0][i])
        sls.set_solid(rail[1][i])
        res = sls.start_Curve3D()
        return res["color"]

    @staticmethod
    def process_all_colors(params, rail):
        rail_length = len(rail[0])

        with ThreadPoolExecutor() as executor:
            colors = np.array(
                list(executor.map(lambda i: SlsHelper.process_colors(params, i, rail), range(rail_length)))
            ).ravel().tolist()

        return colors """
    
    @staticmethod
    def process_colors_batch(params, rail):
        (debug, space, preci, toler, steps) = params
        sls = SlsHelper.initClass(debug, space, preci, toler)
        sls.set_destination_types({"SNM": True, "LCH": True, "HEX": True})
        sls.set_gradient_by_steps(steps)
        
        colors = [
            value
            for media, solid in zip(rail[0], rail[1])
            if (sls.set_media(media) or True) and (sls.set_solid(solid) or True)
            for value in sls.start_Curve3D()["color"]
        ]
        
        return colors
    

class Predict_SynlinV4_Handler(BaseLambdaHandler):
    
    def handle(self):
        
        (debug, space, preci, toler, STEPS) = SlsHelper.evalEvent(self.event)
        
        SLS_PARAMS = (debug, space, preci, toler, STEPS)
        
        jd = {}
        
        """
        1.	No ink (substrate color)
        2.	C (Cyan)
        """

        c1 = self.event["c1"]                   # (W)
        c2 = self.event["c2"]                   # (C)
        
        # --- 2. PREDICT RAIL ---
        
        rail = [[c1], [c2]]
        
        # --- 3. PREDICT COLORS ---
                
        jd.update({ "color": SlsHelper.process_colors_batch(SLS_PARAMS, rail) })
        
        dcs_gradient = np.linspace(0, 1, STEPS)
        dcs = GradientMixGenerator.generate_dcs(dcs_gradient, 1)
        for i in range(len(jd["color"])):
            jd["color"][i].update({ "dcs": dcs[i] })
        #jd.update({ "dcs": dcs })
        
        jd.update({ "elapsed": self.get_elapsed_time() })
        return self.get_common_response(jd)


class Predict_SynAreaV4_Handler(BaseLambdaHandler):

    def handle(self):
        
        (debug, space, preci, toler, STEPS) = SlsHelper.evalEvent(self.event)
        
        SLS_PARAMS = (debug, space, preci, toler, STEPS)
        
        sls = SlsHelper.initClass(debug, space, preci, toler)

        jd = {}
        
        c1 = self.event["c1"]                   # (W)
        c2 = self.event["c2"]                   # (C)
        c3 = self.event["c3"]                   # (M)
        c4 = SlsHelper.mix_1to1(sls, c2, c3)    # (C) + (M)

        # --- 1. PREDICT TOWER ---
        
        #edges = [[c1, c2], [c3, c4]]
        edges = [[c1, c3], [c2, c4]]

        # --- 2. PREDICT RAIL ---

        rail = [
            SlsHelper.mix_all(sls, edges[0][0], edges[1][0], STEPS),
            SlsHelper.mix_all(sls, edges[0][1], edges[1][1], STEPS)   
        ]

        # --- 3. PREDICT COLORS ---
        
        jd.update({ "color": SlsHelper.process_colors_batch(SLS_PARAMS, rail) })
        
        dcs_gradient = np.linspace(0, 1, STEPS)
        dcs = GradientMixGenerator.generate_dcs(dcs_gradient, 2)
        for i in range(len(jd["color"])):
            jd["color"][i].update({ "dcs": dcs[i] })
        #jd.update({ "dcs": dcs })
            
        jd.update({ "elapsed": self.get_elapsed_time() })
        return self.get_common_response(jd)


class Predict_SynVolumeV4_Handler(BaseLambdaHandler):

    def handle(self):
        
        (debug, space, preci, toler, STEPS) = SlsHelper.evalEvent(self.event)
        
        SLS_PARAMS = (debug, space, preci, toler, STEPS)
        
        sls = SlsHelper.initClass(debug, space, preci, toler)

        jd = {}
                
        c1 = self.event["c1"]                   # (W)
        c2 = self.event["c2"]                   # (C)
        c3 = self.event["c3"]                   # (M)
        c4 = self.event["c4"]                   # (Y)
        c5 = SlsHelper.mix_1to1(sls, c2, c3)    # (C) + (M)
        c6 = SlsHelper.mix_1to1(sls, c2, c4)    # (C) + (Y)
        c7 = SlsHelper.mix_1to1(sls, c3, c4)    # (M) + (Y)
        c8 = SlsHelper.mix_2to1(sls, c5, c4)    # (C + M) + (Y)

        # --- 1. PREDICT TOWER ---
        # edges = [[c1, c3], [c2, c4]]
        # edges = [[c1, c2], [c3, c4], [c5, c6], [c7, c8]]
        
        #edges = [[c1, c3], [c2, c4], [c5, c6], [c7, c8]]
        # edges = [[c1, c5], [c2, c6], [c3, c7], [c4, c8]]
        
        #edges = [[c1, c5], [c3, c7], [c2, c6], [c4, c8]]
        # #edges = [[c1, c4], [c2, c6], [c5, c8], [c3, c7]]
        
                #[[W, C],  [M, CM],  [MY, CMY],[Y, CY]]
        edges = [[c1, c2], [c3, c5], [c7, c8], [c4, c6]]

        # --- 1. PREDICT TOWER ---

        tower = [SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges]

        # --- 2. PREDICT RAIL ---

        TG_LENGTH = len(tower[0])
        rail = [
            [item for i in range(TG_LENGTH) for item in SlsHelper.mix_all(sls, tower[0][i], tower[1][i], STEPS)],
            [item for i in range(TG_LENGTH) for item in SlsHelper.mix_all(sls, tower[3][i], tower[2][i], STEPS)]
        ]
        
        # --- 3. PREDICT COLORS ---
            
        jd.update({ "color": SlsHelper.process_colors_batch(SLS_PARAMS, rail) })
        
        dcs_gradient = np.linspace(0, 1, STEPS)
        dcs = GradientMixGenerator.generate_dcs(dcs_gradient, 3)
        for i in range(len(jd["color"])):
            jd["color"][i].update({ "dcs": dcs[i] })
        #jd.update({ "dcs": dcs })

        jd.update({ "elapsed": self.get_elapsed_time() })
        return self.get_common_response(jd)


class Predict_SynHyperFourV4_Handler(BaseLambdaHandler):
    
    def handle(self):
        
        (debug, space, preci, toler, STEPS) = SlsHelper.evalEvent(self.event)
        
        SLS_PARAMS = (debug, space, preci, toler, STEPS)
        
        sls = SlsHelper.initClass(debug, space, preci, toler)

        jd = {}
        
        c1 = self.event["c1"]                   # (W)
        c2 = self.event["c2"]                   # (C)
        c3 = self.event["c3"]                   # (M)
        c4 = self.event["c4"]                   # (Y)
        c5 = self.event["c5"]                   # (K)
        c6 = SlsHelper.mix_1to1(sls, c2, c3)    # (C) + (M)
        c7 = SlsHelper.mix_1to1(sls, c2, c4)    # (C) + (Y)
        c8 = SlsHelper.mix_1to1(sls, c2, c5)    # (C) + (K)
        c9 = SlsHelper.mix_1to1(sls, c3, c4)    # (M) + (Y)
        c10 = SlsHelper.mix_1to1(sls, c3, c5)   # (M) + (K)
        c11 = SlsHelper.mix_1to1(sls, c4, c5)   # (Y) + (K)
        c12 = SlsHelper.mix_2to1(sls, c6, c4)   # (C + M) + (Y)
        c13 = SlsHelper.mix_2to1(sls, c6, c5)   # (C + M) + (K)
        c14 = SlsHelper.mix_2to1(sls, c7, c5)   # (C + Y) + (K)
        c15 = SlsHelper.mix_2to1(sls, c9, c5)   # (M + Y) + (K)
        c16 = SlsHelper.mix_3to1(sls, c12, c5)  # (C + M + Y) + (K)
        
        # --- 1. PREDICT TOWER ---
        
        """ edges_T = [[c4, c11], [c7, c14], [c12, c16], [c9, c15]]
        edges_D = [[c1, c5], [c2, c8], [c6, c13], [c3, c10]] """
        
        """          # [[C, CM], [CY, CMY],  [YK, MYK],  [K, MK]]
        edges_T = [[c2, c6], [c7, c12], [c14, c16], [c8, c13]]
                # [[W, M],  [Y, MY],  [YK, MYK],  [K, MK]]
        edges_D = [[c1, c3], [c4, c9], [c11, c15], [c5, c10]] """
        
                 # [[M, CM], [MY, CMY], [MYK, CMYK], [MK, MCK]]
        edges_T = [[c3, c6], [c9, c12], [c15, c16], [c10, c13]]
                # [[W, C],  [Y, CY],  [YK, CYK],  [K, CK]]
        edges_D = [[c1, c2], [c4, c7], [c11, c14], [c5, c8]]
        
        interpolated_edges_T = [SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges_T]
        interpolated_edges_D = [SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges_D]
        
        ENTRY_LENGTH = len(interpolated_edges_T[0])
        EDGES_LENGTH = len(interpolated_edges_T)
        
        tower = [[] for _ in range(EDGES_LENGTH)]
        for i in range(ENTRY_LENGTH):
            for j in range(EDGES_LENGTH):
                tower[j].extend(SlsHelper.mix_all(sls, interpolated_edges_D[j][i], interpolated_edges_T[j][i], STEPS))
        
        # --- 2. PREDICT RAIL ---

        TG_LENGTH = len(tower[0])
        rail = [
            [item for i in range(TG_LENGTH) for item in SlsHelper.mix_all(sls, tower[0][i], tower[1][i], STEPS)],
            [item for i in range(TG_LENGTH) for item in SlsHelper.mix_all(sls, tower[3][i], tower[2][i], STEPS)]
        ]

        # --- 3. PREDICT COLORS ---
            
        jd.update({ "color": SlsHelper.process_colors_batch(SLS_PARAMS, rail) })
        
        dcs_gradient = np.linspace(0, 1, STEPS)
        dcs = GradientMixGenerator.generate_dcs(dcs_gradient, 4)
        for i in range(len(jd["color"])):
            jd["color"][i].update({ "dcs": dcs[i] })
        #jd.update({ "dcs": dcs })
        
        jd.update({ "elapsed": self.get_elapsed_time() })
        return self.get_common_response(jd)
