from typing import Any
from src.handlers import BaseLambdaHandler
from src.code.predict.linearization.linearInterpolation import LinearInterpolation
from src.code.predict.linearization.synlinV4a import SynLinSolidV4a
from src.code.space.colorConverter import Cs_Spectral2Multi, ColorTrafo

import numpy as np  # type: ignore
import itertools

import boto3  # type: ignore
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
import orjson # type: ignore


class Predict_LinearInterpolation_Handler(BaseLambdaHandler):
    def handle(self):
        jd = {}

        media = self.event["media"]
        solid = self.event["solid"]
        steps = int(self.event.get("steps", 5))
        # iters = int(self.event.get("iterations", 1))
        # toler = float(self.event.get("tolerance", 0.004))

        sls = LinearInterpolation()
        """ sls.set_places = int(event.get('round', 3))  """
        sls.set_media(media)
        sls.set_solid(solid)

        err = sls.set_gradient_by_steps(steps)
        if err:
            return self.get_error_response(err)

        # sls.set_gradient([0.0, 50.0, 100.0])
        # sls.tolerance = toler
        # sls.set_max_loops = iters
        res = sls.start()

        jd.update(res)

        return self.get_common_response(jd)


""" 
FROM HERE ONWARDS, THE CODE USES THE SynLinSolidV4a CLASS to predict n-dimensional colors.
"""


class GradientMixGenerator:

    @staticmethod
    def generate_dcs(gradient, dimension, scale=1):
        """
        Generate a list of mixtures for a given gradient and dimension using NumPy.

        :param gradient: A list or array of gradient points (values from 0 to 1).
        :param dimension: The dimension of the space (1D, 2D, 3D, etc.).
        :return: A 2D NumPy array of mixtures where each row represents a mixture.
        """
        
        # Ensure gradient is a NumPy array
        gradient = np.asarray(gradient * scale)

        # Generate meshgrid for all dimensions
        grids = np.meshgrid(*[gradient] * dimension, indexing="ij")

        # Reshape into a 2D array where each row is a combination
        result = np.stack(grids, axis=-1).reshape(-1, dimension)

        return result

    @staticmethod
    def generate_dcs_tolist(gradient, dimension, scale=1):
        return GradientMixGenerator.generate_dcs(gradient, dimension, scale).tolist()


class SlsHelper:

    @staticmethod
    def apply_dynamic_factor(values, alpha=0.75, value_range=(0, 100)):
        min_val, max_val = value_range
        scaled = (np.array(values) - min_val) / (max_val - min_val)
        adjusted = scaled**alpha
        result = adjusted * (max_val - min_val) + min_val
        return result

    @staticmethod
    def evalEvent(event):
        steps = int(event.get("steps", 5))
        toler = float(event.get("tolerance", 0.00025))
        preci = int(event.get("precision", 100))
        debug = event.get("debug", False)
        space = event.get("space", "XYZ")
        darkf = event.get("darken", 0.0)
        return (debug, space, preci, toler, steps, darkf)

    @staticmethod
    def initClass(debug, space, preci=100, toler=0.00025):
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
    def mix_concentration(
        sls: SynLinSolidV4a, color1, color2, concentration, darken=0.0
    ):
        sls.set_gradient([concentration])
        sls.set_media(color1)
        sls.set_solid(color2)
        res: dict[str, Any] = sls.start_Curve3D()
        snm = res["color"][0]["snm"]

        if darken > 0.0:
            """
            factor = 1.0 - darken
            snm = [factor * item for item in snm]
            """

            result = ColorTrafo.Cs_Spectral2Multi([snm], {"LCH": True})
            lchl_normalized = result[0]["lch"]["L"] / 100
            factor = (1 - lchl_normalized) ** (darken)
            snm = [factor * item for item in snm]

        return snm

    @staticmethod
    def mix_1to1(sls, color1, color2, darken=0.0):
        return SlsHelper.mix_concentration(sls, color1, color2, 0.5, darken)

    @staticmethod
    def mix_2to1(sls, color1, color2, darken=0.0):
        return SlsHelper.mix_concentration(sls, color1, color2, 0.67, darken)

    @staticmethod
    def mix_3to1(sls, color1, color2, darken=0.0):
        return SlsHelper.mix_concentration(sls, color1, color2, 0.75, darken)

    @staticmethod
    def mix_all(sls, color1, color2, steps):
        sls.set_gradient_by_steps(steps)
        sls.set_media(color1)
        sls.set_solid(color2)
        res: dict[str, Any] = sls.start_Curve3D()
        return [item["snm"] for item in res["color"]]


    @staticmethod
    def process_colors_batch(
        params, rail, destination_types={"SNM": True, "LCH": True, "HEX": True}
    ):
        (debug, space, preci, toler, steps) = params
        sls = SlsHelper.initClass(debug, space, preci, toler)
        sls.set_destination_types(destination_types)
        sls.set_gradient_by_steps(steps)

        return [
            value
            for media, solid in zip(rail[0], rail[1])
            if (sls.set_media(media) or True) and (sls.set_solid(solid) or True)
            for value in sls.start_Curve3D()["color"]
        ]


class Predict_SynlinV4_Handler(BaseLambdaHandler):

    def handle(self):

        (debug, space, preci, toler, STEPS, darkf) = SlsHelper.evalEvent(self.event)

        SLS_PARAMS = (debug, space, preci, toler, STEPS)

        jd = {}

        """
        1.	No ink (substrate color)
        2.	C (Cyan)
        """

        c1 = self.event["c1"]  # (W)
        c2 = self.event["c2"]  # (C)

        # --- 2. PREDICT RAIL ---

        rail = [[c1], [c2]]

        # --- 3. PREDICT COLORS ---

        jd.update({"color": SlsHelper.process_colors_batch(SLS_PARAMS, rail)})

        dcs_gradient = np.linspace(0, 1, STEPS)
        dcs = GradientMixGenerator.generate_dcs_tolist(dcs_gradient, 1)
        for color, dcs_value in zip(jd["color"], dcs):
            color["dcs"] = dcs_value

        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)


class Predict_SynAreaV4_Handler(BaseLambdaHandler):

    def handle(self):

        (debug, space, preci, toler, STEPS, darkf) = SlsHelper.evalEvent(self.event)

        SLS_PARAMS = (debug, space, preci, toler, STEPS)

        sls = SlsHelper.initClass(debug, space, preci, toler)

        jd = {}

        c1 = self.event.get("c1")  # (W)
        c2 = self.event.get("c2")  # (C)
        c3 = self.event.get("c3")  # (M)
        c4 = self.event.get("c4", SlsHelper.mix_1to1(sls, c2, c3, darkf))  # (C) + (M)
        # c4 = SlsHelper.mix_1to1(sls, c2, c3)    # (C) + (M)

        # --- 1. PREDICT TOWER ---

        # edges = [[c1, c2], [c3, c4]]
        edges = [[c1, c3], [c2, c4]]

        # --- 2. PREDICT RAIL ---

        rail = [
            SlsHelper.mix_all(sls, edges[0][0], edges[1][0], STEPS),
            SlsHelper.mix_all(sls, edges[0][1], edges[1][1], STEPS),
        ]

        # --- 3. PREDICT COLORS ---

        jd.update({"color": SlsHelper.process_colors_batch(SLS_PARAMS, rail)})

        dcs_gradient = np.linspace(0, 1, STEPS)
        dcs = GradientMixGenerator.generate_dcs_tolist(dcs_gradient, 2)
        for color, dcs_value in zip(jd["color"], dcs):
            color["dcs"] = dcs_value

        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)


class Predict_SynVolumeV4_Handler(BaseLambdaHandler):

    def handle(self):

        (debug, space, preci, toler, STEPS, darkf) = SlsHelper.evalEvent(self.event)

        SLS_PARAMS = (debug, space, preci, toler, STEPS)

        sls = SlsHelper.initClass(debug, space, preci, toler)

        jd = {}

        c1 = self.event.get("c1")  # (W)
        c2 = self.event.get("c2")  # (C)
        c3 = self.event.get("c3")  # (M)
        c4 = self.event.get("c4")  # (Y)
        c5 = self.event.get("c5", SlsHelper.mix_1to1(sls, c2, c3, darkf))  # (C) + (M)
        c6 = self.event.get("c6", SlsHelper.mix_1to1(sls, c2, c4, darkf))  # (C) + (Y)
        c7 = self.event.get("c7", SlsHelper.mix_1to1(sls, c3, c4, darkf))  # (M) + (Y)
        c8 = self.event.get(
            "c8", SlsHelper.mix_2to1(sls, c5, c4, darkf)
        )  # (C + M) + (Y)

        # --- 1. PREDICT TOWER ---

        # [[W, C],  [M, CM],  [MY, CMY],[Y, CY]]
        edges = [[c1, c2], [c3, c5], [c7, c8], [c4, c6]]

        # --- 1. PREDICT TOWER ---

        tower = [SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges]

        # --- 2. PREDICT RAIL ---

        TG_LENGTH = len(tower[0])
        rail = [
            [
                item
                for i in range(TG_LENGTH)
                for item in SlsHelper.mix_all(sls, tower[0][i], tower[1][i], STEPS)
            ],
            [
                item
                for i in range(TG_LENGTH)
                for item in SlsHelper.mix_all(sls, tower[3][i], tower[2][i], STEPS)
            ],
        ]

        # --- 3. PREDICT COLORS ---

        jd.update({"color": SlsHelper.process_colors_batch(SLS_PARAMS, rail)})

        dcs_gradient = np.linspace(0, 1, STEPS)
        dcs = GradientMixGenerator.generate_dcs_tolist(dcs_gradient, 3)
        for color, dcs_value in zip(jd["color"], dcs):
            color["dcs"] = dcs_value

        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)


class Predict_SynHyperFourV4_Handler(BaseLambdaHandler):

    def handle(self):

        (debug, space, preci, toler, STEPS, darkf) = SlsHelper.evalEvent(self.event)

        SLS_PARAMS = (debug, space, preci, toler, STEPS)

        sls = SlsHelper.initClass(debug, space, preci, toler)

        jd = {}

        c1 = self.event.get("c1")  # (W)
        c2 = self.event.get("c2")  # (C)
        c3 = self.event.get("c3")  # (M)
        c4 = self.event.get("c4")  # (Y)
        c5 = self.event.get("c5")  # (K)
        c6 = self.event.get("c6", SlsHelper.mix_1to1(sls, c2, c3, darkf))  # (C) + (M)
        c7 = self.event.get("c7", SlsHelper.mix_1to1(sls, c2, c4, darkf))  # (C) + (Y)
        c8 = self.event.get("c8", SlsHelper.mix_1to1(sls, c2, c5, darkf))  # (C) + (K)
        c9 = self.event.get("c9", SlsHelper.mix_1to1(sls, c3, c4, darkf))  # (M) + (Y)
        c10 = self.event.get("c10", SlsHelper.mix_1to1(sls, c3, c5, darkf))  # (M) + (K)
        c11 = self.event.get("c11", SlsHelper.mix_1to1(sls, c4, c5, darkf))  # (Y) + (K)
        c12 = self.event.get(
            "c12", SlsHelper.mix_2to1(sls, c6, c4, darkf)
        )  # (C + M) + (Y)
        c13 = self.event.get(
            "c13", SlsHelper.mix_2to1(sls, c6, c5, darkf)
        )  # (C + M) + (K)
        c14 = self.event.get(
            "c14", SlsHelper.mix_2to1(sls, c7, c5, darkf)
        )  # (C + Y) + (K)
        c15 = self.event.get(
            "c15", SlsHelper.mix_2to1(sls, c9, c5, darkf)
        )  # (M + Y) + (K)
        c16 = self.event.get(
            "c16", SlsHelper.mix_3to1(sls, c12, c5, darkf)
        )  # (C + M + Y) + (K)

        jd.update({"elapsed-1-init": self.get_elapsed_time()})

        # --- 1. PREDICT TOWER ---

        # [[M, CM],  [MY, CMY], [MYK, CMYK],[MK, MCK]]
        edges_T = [[c3, c6], [c9, c12], [c15, c16], [c10, c13]]
        # [[W, C],   [Y, CY],  [YK, CYK],  [K, CK]]
        edges_D = [[c1, c2], [c4, c7], [c11, c14], [c5, c8]]

        interpolated_edges_T = [
            SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges_T
        ]
        interpolated_edges_D = [
            SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges_D
        ]

        jd.update({"elapsed-2a-interpolated_edges": self.get_elapsed_time()})
        jd.update(
            {"elapsed-2a-interpolated_edges-length": len(interpolated_edges_T[0])}
        )

        ENTRY_LENGTH = len(interpolated_edges_T[0])
        EDGES_LENGTH = len(interpolated_edges_T)
        tower = [[] for _ in range(EDGES_LENGTH)]

        for i in range(ENTRY_LENGTH):
            for j in range(EDGES_LENGTH):
                tower[j].extend(
                    SlsHelper.mix_all(
                        sls,
                        interpolated_edges_D[j][i],
                        interpolated_edges_T[j][i],
                        STEPS,
                    )
                )
        """ 
        for j in range(EDGES_LENGTH):
            tower[j].extend(
                itertools.chain.from_iterable(
                    SlsHelper.mix_all(sls, interpolated_edges_D[j][i], interpolated_edges_T[j][i], STEPS)
                    for i in range(ENTRY_LENGTH)
                )
            )
        """

        jd.update({"elapsed-2b-tower": self.get_elapsed_time()})
        jd.update({"elapsed-2b-tower-length": len(tower[0])})

        # --- 2. PREDICT RAIL ---

        TG_LENGTH = len(tower[0])
        rail = [
            [
                item
                for i in range(TG_LENGTH)
                for item in SlsHelper.mix_all(sls, tower[0][i], tower[1][i], STEPS)
            ],
            [
                item
                for i in range(TG_LENGTH)
                for item in SlsHelper.mix_all(sls, tower[3][i], tower[2][i], STEPS)
            ],
        ]

        jd.update({"elapsed-3-rail": self.get_elapsed_time()})
        jd.update({"elapsed-3-rail-length": len(rail[0])})

        # --- 3. PREDICT COLORS ---

        jd.update({"color": SlsHelper.process_colors_batch(SLS_PARAMS, rail)})

        dcs_gradient = np.linspace(0, 1, STEPS)
        dcs = GradientMixGenerator.generate_dcs_tolist(dcs_gradient, 4)
        for color, dcs_value in zip(jd["color"], dcs):
            color["dcs"] = dcs_value

        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)


class Predict_SynHyperFourV4_Parallel_Handler(BaseLambdaHandler):

    @staticmethod
    def chunk_rail_numpy(rail, chunk_size):
        """
        Splits both sub-arrays of rail into chunks using NumPy.
        returns: A list of NumPy arrays where each array is a chunk.

        # Example Usage
        chunksize = 50
        rail_chunked_json = chunk_rail_numpy(rail, chunksize)
        """

        return np.array_split(
            np.array(rail), range(chunk_size, len(rail[0]), chunk_size), axis=1
        )

    @staticmethod
    def invoke_aws(client, payload):
        """Invokes an AWS Lambda function and handles possible errors."""
        try:
            # lambda_client = boto3.client('lambda')

            response = client.invoke(
                FunctionName="mars-colorpy-predict-interpolate-pairs",
                InvocationType="RequestResponse",  # Wait for response
                Payload=orjson.dumps(payload),
            )

            # Read the payload response
            response_payload = orjson.loads(response["Payload"].read())

            # Check if Lambda returned an error
            if "errorMessage" in response_payload:
                raise RuntimeError(f"Lambda error: {response_payload['errorMessage']}")

            return response_payload

        except Exception as e:
            print(f"Error invoking Lambda: {e}")
            return {"error": str(e)}


    def excute_lambda_rails(self, debug, space, preci, toler, STEPS, rail_chunked_list):
        payloads = [
            {
                "id": i,
                "rail": rail_chunked_list[i].tolist(),
                "debug": debug,
                "space": space,
                "preci": preci,
                "toler": toler,
                "steps": STEPS,
            }
            for i in range(len(rail_chunked_list))
        ]

        # Using ThreadPoolExecutor to submit multiple requests in parallel
        unflattened_results = [None] * len(payloads)
        flattened_elapsed = []

        lambda_client = boto3.client("lambda")
        with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
            futures = {
                executor.submit(self.invoke_aws, lambda_client, payload): payload["id"]
                for payload in payloads
            }

            for future in as_completed(futures):
                result = future.result()
                unflattened_results[result["body"]["id"]] = result["body"]["color"]
                flattened_elapsed.append(result["body"]["elapsed"])

        flattened_results = list(chain.from_iterable(unflattened_results))
        return flattened_elapsed,flattened_results


    def handle(self):

        (debug, space, preci, toler, STEPS, darkf) = SlsHelper.evalEvent(self.event)

        # SLS_PARAMS = (debug, space, preci, toler, STEPS)

        sls = SlsHelper.initClass(debug, space, preci, toler)

        jd = {}

        c1 = self.event.get("c1")  # (W)
        c2 = self.event.get("c2")  # (C)
        c3 = self.event.get("c3")  # (M)
        c4 = self.event.get("c4")  # (Y)
        c5 = self.event.get("c5")  # (K)
        c6 = self.event.get("c6", SlsHelper.mix_1to1(sls, c2, c3, darkf))  # (C) + (M)
        c7 = self.event.get("c7", SlsHelper.mix_1to1(sls, c2, c4, darkf))  # (C) + (Y)
        c8 = self.event.get("c8", SlsHelper.mix_1to1(sls, c2, c5, darkf))  # (C) + (K)
        c9 = self.event.get("c9", SlsHelper.mix_1to1(sls, c3, c4, darkf))  # (M) + (Y)
        c10 = self.event.get("c10", SlsHelper.mix_1to1(sls, c3, c5, darkf))  # (M) + (K)
        c11 = self.event.get("c11", SlsHelper.mix_1to1(sls, c4, c5, darkf))  # (Y) + (K)
        c12 = self.event.get(
            "c12", SlsHelper.mix_2to1(sls, c6, c4, darkf)
        )  # (C + M) + (Y)
        c13 = self.event.get(
            "c13", SlsHelper.mix_2to1(sls, c6, c5, darkf)
        )  # (C + M) + (K)
        c14 = self.event.get(
            "c14", SlsHelper.mix_2to1(sls, c7, c5, darkf)
        )  # (C + Y) + (K)
        c15 = self.event.get(
            "c15", SlsHelper.mix_2to1(sls, c9, c5, darkf)
        )  # (M + Y) + (K)
        c16 = self.event.get(
            "c16", SlsHelper.mix_3to1(sls, c12, c5, darkf)
        )  # (C + M + Y) + (K)

        chunksize = self.event.get("chunk", 100)

        jd.update({"elapsed-1-init": self.get_elapsed_time()})

        # --- 1. PREDICT TOWER ---

        # [[M, CM],  [MY, CMY], [MYK, CMYK],[MK, MCK]]
        edges_T = [[c3, c6], [c9, c12], [c15, c16], [c10, c13]]
        # [[W, C],   [Y, CY],  [YK, CYK],  [K, CK]]
        edges_D = [[c1, c2], [c4, c7], [c11, c14], [c5, c8]]

        interpolated_edges_T = [
            SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges_T
        ]
        interpolated_edges_D = [
            SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges_D
        ]

        jd.update({"elapsed-2a-interpolated_edges": self.get_elapsed_time()})
        jd.update({"elapsed-2a-interpolated_edges-length": len(interpolated_edges_T[0])})

        ENTRY_LENGTH = len(interpolated_edges_T[0])
        EDGES_LENGTH = len(interpolated_edges_T)
        tower = [[] for _ in range(EDGES_LENGTH)]

        for i in range(ENTRY_LENGTH):
            for j in range(EDGES_LENGTH):
                tower[j].extend(
                    SlsHelper.mix_all(
                        sls,
                        interpolated_edges_D[j][i],
                        interpolated_edges_T[j][i],
                        STEPS,
                    )
                )

        jd.update({"elapsed-2b-tower": self.get_elapsed_time()})
        jd.update({"elapsed-2b-tower-length": len(tower[0])})

        # --- 2. PREDICT RAIL ---
        
        towers_01 = [tower[0], tower[1]]
        towers_23 = [tower[3], tower[2]]
        
        """         
        jd.update({"elapsed-3-rail": towers_01})
        jd.update({"elapsed-3-rail-length": len(towers_01[0])})
        return self.get_common_response(jd) 
        """
                                        
        tower_chunksize = len(towers_01[0]) if len(towers_01[0]) < 30 else 30
        
        towers_chunked_01 = self.chunk_rail_numpy(towers_01, tower_chunksize)
        rail_elapsed_01, rail_01 = self.excute_lambda_rails(debug, space, preci, toler, STEPS, towers_chunked_01)
        
        towers_chunked_23 = self.chunk_rail_numpy(towers_23, tower_chunksize)
        rail_elapsed_23, rail_23 = self.excute_lambda_rails(debug, space, preci, toler, STEPS, towers_chunked_23)
        
        jd.update({"elapsed-2-rail_elapsed_01": rail_elapsed_01})
        jd.update({"elapsed-2-rail_elapsed_23": rail_elapsed_23})
        
        tower_rail = [
            [item["snm"] for item in rail_01],
            [item["snm"] for item in rail_23],
        ]

        jd.update({"elapsed-3-rail": self.get_elapsed_time()})
        jd.update({"elapsed-3-rail-length": len(tower_rail[0])})

        # --- 3. PREDICT COLORS ---

        rail_chunked = self.chunk_rail_numpy(tower_rail, chunksize)
        flattened_elapsed, flattened_results = self.excute_lambda_rails(debug, space, preci, toler, STEPS, rail_chunked)

        # Process results
        jd.update({"color": flattened_results})
        jd.update({"color_length": len(flattened_results)})
        jd.update({"elapsed-5-parallel": flattened_elapsed})
        jd.update({"elapsed-5": self.get_elapsed_time()})

        # Add DCS values to each color
        dcs_gradient = np.linspace(0, 1, STEPS)
        dcs = GradientMixGenerator.generate_dcs_tolist(dcs_gradient, 4, scale=100)
        for color, dcs_value in zip(jd["color"], dcs):
            color["dcs"] = dcs_value

        # Final response
        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)


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

        SLS_PARAMS = (debug, space, preci, toler, steps)

        jd = {"id": id}

        try:
            jd.update({"color": SlsHelper.process_colors_batch(SLS_PARAMS, rail)})
        except Exception as e:
            return self.get_error_response(str(e))

        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)


# -- DELETE -- DELETE -- DELETE -- DELETE -- DELETE -- DELETE -- DELETE -- DELETE --


class Predict_SynHyperFourV4_Parallel_Handler_BACKUP001(BaseLambdaHandler):

    @staticmethod
    def chunk_rail_numpy(rail, chunk_size):
        """Splits both sub-arrays of rail into chunks using NumPy and converts them to lists for JSON serialization."""
        rail_np = np.array(rail)  # Convert input to NumPy array
        chunked = np.array_split(
            rail_np, range(chunk_size, len(rail[0]), chunk_size), axis=1
        )

        # Convert each chunk to a nested Python list
        return [chunk.tolist() for chunk in chunked]

        # Example Usage
        """
        chunksize = 50
        rail_chunked_json = chunk_rail_numpy(rail, chunksize)
        """

    @staticmethod
    def invoke_aws(payload):
        """Invokes an AWS Lambda function and handles possible errors."""
        try:
            lambda_client = boto3.client("lambda")

            response = lambda_client.invoke(
                FunctionName="mars-colorpy-predict-interpolate-pairs",
                InvocationType="RequestResponse",  # Wait for response
                Payload=json.dumps(payload),
            )

            # Read the payload response
            response_payload = json.loads(response["Payload"].read())

            # Check if Lambda returned an error
            if "errorMessage" in response_payload:
                raise RuntimeError(f"Lambda error: {response_payload['errorMessage']}")

            return response_payload

        except Exception as e:
            print(f"Error invoking Lambda: {e}")
            return {"error": str(e)}

    def handle(self):

        (debug, space, preci, toler, STEPS, darkf) = SlsHelper.evalEvent(self.event)

        # SLS_PARAMS = (debug, space, preci, toler, STEPS)

        sls = SlsHelper.initClass(debug, space, preci, toler)

        jd = {}

        c1 = self.event.get("c1")  # (W)
        c2 = self.event.get("c2")  # (C)
        c3 = self.event.get("c3")  # (M)
        c4 = self.event.get("c4")  # (Y)
        c5 = self.event.get("c5")  # (K)
        c6 = self.event.get("c6", SlsHelper.mix_1to1(sls, c2, c3, darkf))  # (C) + (M)
        c7 = self.event.get("c7", SlsHelper.mix_1to1(sls, c2, c4, darkf))  # (C) + (Y)
        c8 = self.event.get("c8", SlsHelper.mix_1to1(sls, c2, c5, darkf))  # (C) + (K)
        c9 = self.event.get("c9", SlsHelper.mix_1to1(sls, c3, c4, darkf))  # (M) + (Y)
        c10 = self.event.get("c10", SlsHelper.mix_1to1(sls, c3, c5, darkf))  # (M) + (K)
        c11 = self.event.get("c11", SlsHelper.mix_1to1(sls, c4, c5, darkf))  # (Y) + (K)
        c12 = self.event.get(
            "c12", SlsHelper.mix_2to1(sls, c6, c4, darkf)
        )  # (C + M) + (Y)
        c13 = self.event.get(
            "c13", SlsHelper.mix_2to1(sls, c6, c5, darkf)
        )  # (C + M) + (K)
        c14 = self.event.get(
            "c14", SlsHelper.mix_2to1(sls, c7, c5, darkf)
        )  # (C + Y) + (K)
        c15 = self.event.get(
            "c15", SlsHelper.mix_2to1(sls, c9, c5, darkf)
        )  # (M + Y) + (K)
        c16 = self.event.get(
            "c16", SlsHelper.mix_3to1(sls, c12, c5, darkf)
        )  # (C + M + Y) + (K)

        chunksize = self.event.get("chunk", 100)

        jd.update({"elapsed-1-init": self.get_elapsed_time()})

        # --- 1. PREDICT TOWER ---

        # [[M, CM],  [MY, CMY], [MYK, CMYK],[MK, MCK]]
        edges_T = [[c3, c6], [c9, c12], [c15, c16], [c10, c13]]
        # [[W, C],   [Y, CY],  [YK, CYK],  [K, CK]]
        edges_D = [[c1, c2], [c4, c7], [c11, c14], [c5, c8]]

        interpolated_edges_T = [
            SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges_T
        ]
        interpolated_edges_D = [
            SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges_D
        ]

        jd.update({"elapsed-2a-interpolated_edges": self.get_elapsed_time()})
        jd.update(
            {"elapsed-2a-interpolated_edges-length": len(interpolated_edges_T[0])}
        )

        ENTRY_LENGTH = len(interpolated_edges_T[0])
        EDGES_LENGTH = len(interpolated_edges_T)
        tower = [[] for _ in range(EDGES_LENGTH)]

        for i in range(ENTRY_LENGTH):
            for j in range(EDGES_LENGTH):
                tower[j].extend(
                    SlsHelper.mix_all(
                        sls,
                        interpolated_edges_D[j][i],
                        interpolated_edges_T[j][i],
                        STEPS,
                    )
                )

        jd.update({"elapsed-2b-tower": self.get_elapsed_time()})
        jd.update({"elapsed-2b-tower-length": len(tower[0])})

        # --- 2. PREDICT RAIL ---

        TG_LENGTH = len(tower[0])
        rail = [
            [
                item
                for i in range(TG_LENGTH)
                for item in SlsHelper.mix_all(sls, tower[0][i], tower[1][i], STEPS)
            ],
            [
                item
                for i in range(TG_LENGTH)
                for item in SlsHelper.mix_all(sls, tower[3][i], tower[2][i], STEPS)
            ],
        ]

        # jd.update({ "rail": rail })
        jd.update({"elapsed-3-rail": self.get_elapsed_time()})
        jd.update({"elapsed-3-rail-length": len(rail[0])})

        # --- 3. PREDICT COLORS ---

        # --> SPLIT in multiple Lambda Functions <--

        rail_chunked_list = self.chunk_rail_numpy(rail, chunksize)

        # --> SPLIT in multiple Lambda Functions <--

        # Payloads with unique IDs
        payloads = [
            {
                "id": i,
                "rail": rail_chunked_list[i],
                "debug": debug,
                "space": space,
                "preci": preci,
                "toler": toler,
                "steps": STEPS,
            }
            for i in range(len(rail_chunked_list))
        ]

        # Using ThreadPoolExecutor to submit multiple requests in parallel
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.invoke_aws, payload): payload["id"]
                for payload in payloads
            }

            # Collect results as tuples (id, response) and sort them immediately
            results = sorted(
                (
                    (futures[future], future.result())
                    for future in as_completed(futures)
                ),
                key=lambda x: x[0],
            )

        # Extract and flatten the relevant fields
        # flattened_results = [color for _, result in results for color in result["body"]["color"]]
        flattened_results = list(
            chain.from_iterable(result["body"]["color"] for _, result in results)
        )
        # flattened_elapsed = [result["body"]["elapsed"] for _, result in results]
        flattened_elapsed = list(
            chain.from_iterable(result["body"]["elapsed"] for _, result in results)
        )

        # Process results
        jd.update({"color": flattened_results})
        jd.update({"color_length": len(flattened_results)})
        jd.update({"elapsed-5-parallel": flattened_elapsed})
        jd.update({"elapsed": self.get_elapsed_time()})

        dcs_gradient = np.linspace(0, 1, STEPS)
        dcs = GradientMixGenerator.generate_dcs_tolist(dcs_gradient, 4)
        for color, dcs_value in zip(jd["color"], dcs):
            color["dcs"] = dcs_value

        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)


class Predict_SynHyperFourV4_Parallel_Handler_BACKUP000(BaseLambdaHandler):

    @staticmethod
    def chunk_rail_numpy(rail, chunk_size):
        """Splits both sub-arrays of rail into chunks using NumPy and converts them to lists for JSON serialization."""
        rail_np = np.array(rail)  # Convert input to NumPy array
        chunked = np.array_split(
            rail_np, range(chunk_size, len(rail[0]), chunk_size), axis=1
        )

        # Convert each chunk to a nested Python list
        return [chunk.tolist() for chunk in chunked]

        # Example Usage
        """
        chunksize = 50
        rail_chunked_json = chunk_rail_numpy(rail, chunksize)
        """

    def handle(self):

        (debug, space, preci, toler, STEPS, darkf) = SlsHelper.evalEvent(self.event)

        SLS_PARAMS = (debug, space, preci, toler, STEPS)

        sls = SlsHelper.initClass(debug, space, preci, toler)

        jd = {}

        c1 = self.event.get("c1")  # (W)
        c2 = self.event.get("c2")  # (C)
        c3 = self.event.get("c3")  # (M)
        c4 = self.event.get("c4")  # (Y)
        c5 = self.event.get("c5")  # (K)
        c6 = self.event.get("c6", SlsHelper.mix_1to1(sls, c2, c3, darkf))  # (C) + (M)
        c7 = self.event.get("c7", SlsHelper.mix_1to1(sls, c2, c4, darkf))  # (C) + (Y)
        c8 = self.event.get("c8", SlsHelper.mix_1to1(sls, c2, c5, darkf))  # (C) + (K)
        c9 = self.event.get("c9", SlsHelper.mix_1to1(sls, c3, c4, darkf))  # (M) + (Y)
        c10 = self.event.get("c10", SlsHelper.mix_1to1(sls, c3, c5, darkf))  # (M) + (K)
        c11 = self.event.get("c11", SlsHelper.mix_1to1(sls, c4, c5, darkf))  # (Y) + (K)
        c12 = self.event.get(
            "c12", SlsHelper.mix_2to1(sls, c6, c4, darkf)
        )  # (C + M) + (Y)
        c13 = self.event.get(
            "c13", SlsHelper.mix_2to1(sls, c6, c5, darkf)
        )  # (C + M) + (K)
        c14 = self.event.get(
            "c14", SlsHelper.mix_2to1(sls, c7, c5, darkf)
        )  # (C + Y) + (K)
        c15 = self.event.get(
            "c15", SlsHelper.mix_2to1(sls, c9, c5, darkf)
        )  # (M + Y) + (K)
        c16 = self.event.get(
            "c16", SlsHelper.mix_3to1(sls, c12, c5, darkf)
        )  # (C + M + Y) + (K)

        chunksize = self.event.get("chunk", 100)

        jd.update({"elapsed-1-init": self.get_elapsed_time()})

        # --- 1. PREDICT TOWER ---

        # [[M, CM],  [MY, CMY], [MYK, CMYK],[MK, MCK]]
        edges_T = [[c3, c6], [c9, c12], [c15, c16], [c10, c13]]
        # [[W, C],   [Y, CY],  [YK, CYK],  [K, CK]]
        edges_D = [[c1, c2], [c4, c7], [c11, c14], [c5, c8]]

        interpolated_edges_T = [
            SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges_T
        ]
        interpolated_edges_D = [
            SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges_D
        ]

        jd.update({"elapsed-2a-interpolated_edges": self.get_elapsed_time()})
        jd.update(
            {"elapsed-2a-interpolated_edges-length": len(interpolated_edges_T[0])}
        )

        ENTRY_LENGTH = len(interpolated_edges_T[0])
        EDGES_LENGTH = len(interpolated_edges_T)
        tower = [[] for _ in range(EDGES_LENGTH)]

        for i in range(ENTRY_LENGTH):
            for j in range(EDGES_LENGTH):
                tower[j].extend(
                    SlsHelper.mix_all(
                        sls,
                        interpolated_edges_D[j][i],
                        interpolated_edges_T[j][i],
                        STEPS,
                    )
                )

        jd.update({"elapsed-2b-tower": self.get_elapsed_time()})
        jd.update({"elapsed-2b-tower-length": len(tower[0])})

        # --- 2. PREDICT RAIL ---

        TG_LENGTH = len(tower[0])
        rail = [
            [
                item
                for i in range(TG_LENGTH)
                for item in SlsHelper.mix_all(sls, tower[0][i], tower[1][i], STEPS)
            ],
            [
                item
                for i in range(TG_LENGTH)
                for item in SlsHelper.mix_all(sls, tower[3][i], tower[2][i], STEPS)
            ],
        ]

        # jd.update({ "rail": rail })
        jd.update({"elapsed-3-rail": self.get_elapsed_time()})
        jd.update({"elapsed-3-rail-length": len(rail[0])})

        # --- 3. PREDICT COLORS ---

        # --> SPLIT in multiple Lambda Functions <--

        """
        # Convert to a NumPy array for easier manipulation
        rail_np = np.array(rail)
        # Let's say we want to split the arrays along the first axis into chunks of size 1 for simplicity
        chunk_size = 50
        # Splitting the arrays
        rail_chunks = [rail_np[i:i + chunk_size] for i in range(0, len(rail_np), chunk_size)]
        # Convert np.ndarray to regular Python list to make it JSON serializable
        rail_chunked_list = [chunk.tolist() for chunk in rail_chunks]
        """

        rail_chunked_list = self.chunk_rail_numpy(rail, chunksize)

        # jd.update({ "rail_chunked_list": rail_chunked_list })

        # use boto3 to invoke the lambda function "mars-colorpy-predict-interpolate-pairs"
        import boto3  # type: ignore
        import json
        from concurrent.futures import ThreadPoolExecutor

        lambda_client = boto3.client("lambda")

        def invoke_aws(payload):
            """Invokes an AWS Lambda function and handles possible errors."""
            try:
                response = lambda_client.invoke(
                    FunctionName="mars-colorpy-predict-interpolate-pairs",
                    InvocationType="RequestResponse",  # Wait for response
                    Payload=json.dumps(payload),
                )

                # Read the payload response
                response_payload = json.loads(response["Payload"].read())

                # Check if Lambda returned an error
                if "errorMessage" in response_payload:
                    raise RuntimeError(
                        f"Lambda error: {response_payload['errorMessage']}"
                    )

                return response_payload

            except Exception as e:
                print(f"Error invoking Lambda: {e}")
                return {"error": str(e)}

        # Payloads with unique IDs
        payloads = [
            {
                "id": i,
                "rail": rail_chunked_list[i],
                "debug": debug,
                "space": space,
                "preci": preci,
                "toler": toler,
                "steps": STEPS,
            }
            for i in range(len(rail_chunked_list))
        ]

        """ jd.update({ "payloads": payloads })
        return self.get_common_response(jd) """

        # Using ThreadPoolExecutor to submit multiple requests
        """         
        with ThreadPoolExecutor() as executor:
            # Submit tasks and store futures
            futures = {executor.submit(invoke_aws, payload): payload["id"] for payload in payloads}

            # Collect results, ensuring the responses are in the correct order
            results = []
            for future in futures:
                result = future.result()
                #results.append((futures[future], result))  # Map ID to the response
                results.append(result["body"]["color"])  # Extract just the results, not the IDs 
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(invoke_aws, payload): payload["id"]
                for payload in payloads
            }

            results = []
            for future in as_completed(
                futures
            ):  # As soon as a future completes, process it
                result = future.result()
                results.append((futures[future], result))

                # results.extend(result["body"]["color"])

        sorted_results = sorted(results, key=lambda x: x[0])

        flattened_results_tmp = [
            result["body"]["color"] for _, result in sorted_results
        ]
        flattened_results = [
            item for sublist in flattened_results_tmp for item in sublist
        ]

        flattened_elapsed = [result["body"]["elapsed"] for _, result in sorted_results]
        # Sort results by ID (if needed)
        # results.sort(key=lambda x: x[0])  # Sort based on ID

        # Process results
        """ jd.update({ "id": [r[0] for r in results] })  # Extract just the IDs """
        jd.update({"color": flattened_results})  # Extract just the results, not the IDs
        jd.update({"color_length": len(flattened_results)})
        jd.update({"elapsed-5-parallel": flattened_elapsed})
        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)

        """ response_payload = json.loads(response['Payload'].read())
        jd.update({ "color": response_payload.get('color', []) }) """

        # --> Sort responses by ID <--
        """ jd["color"].sort(key=lambda x: x.get("id", 0)) """

        dcs_gradient = np.linspace(0, 1, STEPS)
        dcs = GradientMixGenerator.generate_dcs_tolist(dcs_gradient, 4)
        for color, dcs_value in zip(jd["color"], dcs):
            color["dcs"] = dcs_value

        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)
