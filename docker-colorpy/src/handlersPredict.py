from typing import Any
from src.handlers import BaseLambdaHandler
from src.code.predict.linearization.linearInterpolation import LinearInterpolation
from src.code.predict.linearization.synlinV4a import SynLinSolidV4a
#from src.code.space.colorConverter import Cs_Spectral2Multi, ColorTrafo

from src.code.space.colorConverterNumpy import ColorTrafoNumpy

#from src.lambda_prewarm import LambdaPrewarmer

import numpy as np  # type: ignore
#import itertools
#import math

import boto3  # type: ignore
#import json

#import time

from botocore.config import Config # type: ignore

from src.code.files.botox import Botox

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
import orjson # type: ignore

from src.code.files.jsonToCgats import JsonToCgats

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

import string
import random
    
    
PRECISION = 100
TOLERANCE = 0.00025

class Helper:
    
    @staticmethod
    def random_id_string(length):
        '''
        Generate a random string of fixed length
        '''

        return ''.join(random.choices(string.digits + string.ascii_uppercase, k=length))
    
    @staticmethod
    def random_id(blocks=4, block_length=5, separator='-'):
        '''
        Generate random String with the following pattern xxxxx-xxxxx-xxxxx-xxxxx, containing only numbers and uppercase letters
        '''
        
        return separator.join(Helper.random_id_string(block_length) for _ in range(blocks))
    
    
class GradientMixGenerator:

    @staticmethod
    def generate_dcs(gradient, dimension):
        """
        Generate a list of mixtures for a given gradient and dimension using NumPy.

        :param gradient: A list or array of gradient points (values from 0 to 1).
        :param dimension: The dimension of the space (1D, 2D, 3D, etc.).
        :return: A 2D NumPy array of mixtures where each row represents a mixture.
        """
        
        # Ensure gradient is a NumPy array
        gradient = np.asarray(gradient)

        # Generate meshgrid for all dimensions
        grids = np.meshgrid(*[gradient] * dimension, indexing="ij")

        # Reshape into a 2D array where each row is a combination
        result = np.stack(grids, axis=-1).reshape(-1, dimension)

        return result

    @staticmethod
    def generate_dcs_tolist(gradient, dimension):
        return GradientMixGenerator.generate_dcs(gradient, dimension).tolist()
    
    @staticmethod
    def generate_dcs_multilin_tolist(gradient: np.ndarray, dimension: int):
        """
        Generate a list of mixtures for a given gradient and dimension using NumPy.
        For each dimension, create a sequence where that dimension varies across the gradient values
        while other dimensions are 0.
        
        e.g. gradient = [0, 50, 100] and diemnsion = 3
        result = [
            [0, 0, 0],
            [50, 0, 0],
            [100, 0, 0],
            [0, 0, 0],
            [0, 50, 0],
            [0, 100, 0],
            [0, 0, 0],
            [0, 0, 50],
            [0, 0, 100],
        ]   
        """   
        
        result = []
        gradient = gradient.tolist()  # Convert to list if it's a NumPy array

        # For each dimension
        for dim in range(dimension):
            base = np.zeros(dimension)  # Create a base array of zeros
            
            # For each gradient value
            for g in gradient:
                mix = base.copy()
                mix[dim] = g
                result.append(mix.tolist())  # Convert to list before appending

        return result
        
        """ 
        result = []
        
        gradient = gradient.tolist()
        
        # For each dimension
        for dim in range(dimension):
            # Create a base array of zeros with length = dimension 
            base = np.zeros(dimension)
            
            # For each gradient value
            for g in gradient:
                # Create copy of base array
                mix = base.copy()
                # Set current dimension to gradient value
                mix[dim] = g
                # Add to result
                result.append(mix)
                
        # Convert numpy arrays to regular lists
        return result
        """
class SlsHelper:

    """ @staticmethod
    def apply_dynamic_factor(values, alpha=0.75, value_range=(0, 100)):
        min_val, max_val = value_range
        scaled = (np.array(values) - min_val) / (max_val - min_val)
        adjusted = scaled**alpha
        result = adjusted * (max_val - min_val) + min_val
        return result
    """
    
    @staticmethod
    def evalEvent(event):
        steps = int(event.get("steps", 5))
        toler = float(event.get("tolerance", TOLERANCE))
        preci = int(event.get("precision", PRECISION))
        debug = event.get("debug", False)
        space = event.get("space", "XYZ")
        darkf = float(event.get("darken", 0.0))
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
            result = ColorTrafoNumpy().Cs_SNM2LAB(np.array(snm)).tolist()
            l_normalized = result[0] / 100
            factor = (1 - l_normalized) ** (darken)
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

        dcs_gradient = np.linspace(0, 100, STEPS)
        dcs = GradientMixGenerator.generate_dcs_tolist(dcs_gradient, 1)
        for color, dcs_value in zip(jd["color"], dcs):
            color["dcs"] = dcs_value

        jd.update({"elapsed": self.get_elapsed_time()})
        return self.get_common_response(jd)
    
class Predict_SynlinV4Multi_Handler(BaseLambdaHandler):

    def handle(self):

        (debug, space, preci, toler, STEPS, darkf) = SlsHelper.evalEvent(self.event)

        SLS_PARAMS = (debug, space, preci, toler, STEPS)

        jd = {}

        """
        1.	No ink (substrate color)
        2.	C (Cyan)
        """

        cAll = self.event["cAll"]  # (W)
        
        
        # 1a. Generate all possible pairs
        corners = []
        for i in range(1, len(cAll)):
            sub_corner = [cAll[0], cAll[i]]
            corners.append(sub_corner)
            
        # 1b. Generate the DCS for pairs
        dcs_gradient = np.linspace(0, 100, STEPS)
        dcs = GradientMixGenerator.generate_dcs_multilin_tolist(dcs_gradient, len(corners))
        
        # --- 2. PREDICT RAIL ---
        rails = []
        sls = SlsHelper.initClass(debug, space, preci, toler)
        
        for corner in corners:
            rail = SlsHelper.mix_all(sls, corner[0], corner[1], STEPS)
            rails.append(rail)

        rails_flattened = list(chain.from_iterable(rails))
        
        trafo = ColorTrafoNumpy()
        rails_colors = trafo.Cs_SNM2MULTI(rails_flattened, {"SNM": True, "LCH": True, "HEX": True})
        
        for color, dcs_value in zip(rails_colors, dcs):
                color["dcs"] = dcs_value
    
        # --- 3. Final Response ---

        jd.update({
            "color": rails_colors,
            "elapsed": self.get_elapsed_time()
        })
        return self.get_common_response(jd)


class Predict_SynAreaV4_Handler(BaseLambdaHandler):

    def handle(self):

        (debug, space, preci, toler, STEPS, darkf) = SlsHelper.evalEvent(self.event)

        SLS_PARAMS = (debug, space, preci, toler, STEPS)

        sls = SlsHelper.initClass(debug, space, preci, toler)

        jd = {}

        jd.update({ "usr-darken": darkf })

        c1 = self.event.get("c1")  # (W)
        c2 = self.event.get("c2")  # (C)
        c3 = self.event.get("c3")  # (M)
        c4 = self.event.get("c4", SlsHelper.mix_1to1(sls, c2, c3, darkf))  # (C) + (M)

        # --- 1. PREDICT TOWER ---

        edges = [[c1, c3], [c2, c4]]

        # --- 2. PREDICT RAIL ---

        rail = [
            SlsHelper.mix_all(sls, edges[0][0], edges[1][0], STEPS),
            SlsHelper.mix_all(sls, edges[0][1], edges[1][1], STEPS),
        ]

        # --- 3. PREDICT COLORS ---

        jd.update({"color": SlsHelper.process_colors_batch(SLS_PARAMS, rail)})

        dcs_gradient = np.linspace(0, 100, STEPS)
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

        dcs_gradient = np.linspace(0, 100, STEPS)
        dcs = GradientMixGenerator.generate_dcs_tolist(dcs_gradient, 3)
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
            response = client.invoke(
                FunctionName="mars-colorpy-predict-interpolate-pairs",
                InvocationType="RequestResponse",
                Payload=orjson.dumps(payload),
            )
            return orjson.loads(response["Payload"].read())

        except Exception as e:
            return {"error": str(e)}




    def prewarm_lambda(self, amount=20):
        
        payloads = [{"id": i} for i in range(amount)]
        
        config = Config(
            connect_timeout=1,
            read_timeout=1,
            retries={"max_attempts": 2},
            max_pool_connections=50
        )
        lambda_client = boto3.client('lambda', config=config)

        with ThreadPoolExecutor(max_workers=50) as executor:
            for payload in payloads:
                executor.submit(self.invoke_aws, lambda_client, payload)
                
            


    def excute_lambda_rails(self, sls_params, rail_chunked_list, destination_types={"SNM": True, "LCH": True, "HEX": True}):
        
        debug, space, preci, toler, STEPS = sls_params
        
        payloads = [
            {
                "id": i,
                "rail": rail_chunked_list[i].tolist(),
                "debug": debug,
                "space": space,
                "preci": preci,
                "toler": toler,
                "steps": STEPS,
                "destination_types": destination_types
            }
            for i in range(len(rail_chunked_list))
        ]

        
        unflattened_results = [None] * len(payloads)
        flattened_elapsed = []

        #lambda_client = boto3.client("lambda")
        #with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        # Configure boto3 client with larger connection pool
        config = Config(
            connect_timeout=5,
            read_timeout=70,
            retries={"max_attempts": 2},
            max_pool_connections=50
        )
        lambda_client = boto3.client('lambda', config=config)

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = {
                executor.submit(self.invoke_aws, lambda_client, payload): payload["id"]
                for payload in payloads
            }

            for future in as_completed(futures):
                result = future.result()
                unflattened_results[result["body"]["id"]] = result["body"]["color"]
                flattened_elapsed.append(result["body"]["elapsed"])

        flattened_results = list(chain.from_iterable(unflattened_results))
        return flattened_elapsed, flattened_results

    def optimal_chunksize(self, color_count: int, steps: int, max_color_count_per_chunk: int=500, max_chunks: int=50):

        """ total_colors = color_count * steps
        total_chunks = total_colors / max_color_count_per_chunk

        return math.ceil( min(max_chunks, total_chunks) * steps ) """
        
        return 30
        
        """ 
        return max_color_count_per_chunk # opt_chunksize
        """

    def handle(self):
        
        jd = {"0-start-elapsed": self.get_elapsed_time()}
        
        (debug, space, preci, toler, STEPS, darkf) = SlsHelper.evalEvent(self.event)
        
        SLS_PARAMS = (debug, space, preci, toler, STEPS)

        sls = SlsHelper.initClass(debug, space, preci, toler)
        
        # PREWARM
        """ 
        x1 = time.time()
        
        lambda_prewarmer = LambdaPrewarmer("mars-colorpy-predict-interpolate-pairs", 50)
        lambda_prewarmer.prewarm_lambda()
        
        x2 = time.time()
        jd.update({"prewarm-duration": x2-x1}) 
        """

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
        c12 = self.event.get("c12", SlsHelper.mix_2to1(sls, c6, c4, darkf))  # (C + M) + (Y)
        c13 = self.event.get("c13", SlsHelper.mix_2to1(sls, c6, c5, darkf))  # (C + M) + (K)
        c14 = self.event.get("c14", SlsHelper.mix_2to1(sls, c7, c5, darkf))  # (C + Y) + (K)
        c15 = self.event.get("c15", SlsHelper.mix_2to1(sls, c9, c5, darkf))  # (M + Y) + (K)
        c16 = self.event.get("c16", SlsHelper.mix_3to1(sls, c12, c5, darkf))  # (C + M + Y) + (K)

        chunksize = 30 #self.event.get("chunk", 45)
        
        jd.update({
            "1-init-user-chunksize": chunksize,
            "1-init-elapsed": self.get_elapsed_time()
        })


        # --- 1. PREDICT TOWER ---

        #         [[M, CM],  [MY, CMY], [MYK, CMYK],[MK, MCK]]
        edges_T = [[c3, c6], [c9, c12], [c15, c16], [c10, c13]]
        #         [[W, C],   [Y, CY],  [YK, CYK],  [K, CK]]
        edges_D = [[c1, c2], [c4, c7], [c11, c14], [c5, c8]]

        interpolated_edges_T = [SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges_T]
        interpolated_edges_D = [SlsHelper.mix_all(sls, edge[0], edge[1], STEPS) for edge in edges_D]

        jd.update({
            "2a-interpolated_edges-elapsed": self.get_elapsed_time(),
            "2a-interpolated_edges-length": len(interpolated_edges_T[0]),
            "2a-interpolated_edgesT-count": len(interpolated_edges_T),
            "2a-interpolated_edgesD-count": len(interpolated_edges_D)
        })

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
        
        jd.update({
            "2b-tower-elapsed": self.get_elapsed_time(),
            "2b-tower-length": len(tower[0])
        })


        # --- 2. PREDICT RAIL ---
        
        towerA = [tower[0], tower[1]]
        towerB = [tower[3], tower[2]]
        
        towerA_chunked = self.chunk_rail_numpy(towerA, chunksize)
        towerA_flattened_elapsed, towerA_flattened = self.excute_lambda_rails(SLS_PARAMS, towerA_chunked, {"SNM": True})
                
        towerB_chunked = self.chunk_rail_numpy(towerB, chunksize)
        towerB_flattened_elapsed, towerB_flattened = self.excute_lambda_rails(SLS_PARAMS, towerB_chunked, {"SNM": True})
        
        jd.update({
            "2c-railA-elapsed": towerA_flattened_elapsed,
            "2c-railA-length": len(towerA_flattened),
            "2c-railB-elapsed": towerB_flattened_elapsed,
            "2c-railB-length": len(towerB_flattened)
        })
        
        
        # --- 3. PREDICT COLORS ---
        
        rail = [
            [item["snm"] for item in towerA_flattened],
            [item["snm"] for item in towerB_flattened]
        ]

        rail_chunked = self.chunk_rail_numpy(rail, chunksize)
        rail_flattened_elapsed, rail_flattened = self.excute_lambda_rails(SLS_PARAMS, rail_chunked)

        jd.update({
            "4-parallel-elapsed": rail_flattened_elapsed,
            "4-elapsed": self.get_elapsed_time(),
            "5-color_length": len(rail_flattened)
        })


        # Add DCS 
        dcs_gradient = np.linspace(0, 100, STEPS)
        dcs = GradientMixGenerator.generate_dcs_tolist(dcs_gradient, 4)
        for color, dcs_value in zip(rail_flattened, dcs):
            color["dcs"] = dcs_value


        # Object name
        object_name = Helper.random_id()
        
        # Store in bucket -- JSON
        datastore_json = Botox("mars-predicted-data")
        datastore_json_result = datastore_json.store_S3(
            object_name, 
            orjson.dumps(rail_flattened),
            f"data/{object_name}.json"
        )
        
        # Store in bucket -- CGATS        
        datastore_cgats = Botox("mars-predicted-data")
        datastore_cgats_result = datastore_cgats.store_S3(
            object_name, 
            JsonToCgats(rail_flattened).convert(),
            f"data/{object_name}.txt"
        )
        
        
        jd.update({
            "UPID": datastore_cgats_result["UPID"],
            "bytes": datastore_cgats_result["bytes"],
            "color": rail_flattened[:122],
            "elapsed": self.get_elapsed_time()
        })
        
        return self.get_common_response(jd)


class InterpolatePairs(BaseLambdaHandler):

    def handle(self):

        rail = self.event.get("rail", [])
        if len(rail) == 0:
            return self.get_error_response("No pairs provided.")

        debug = self.event.get("debug", False)
        space = self.event.get("space", "XYZ")
        preci = self.event.get("precision", PRECISION)
        toler = self.event.get("tolerance", TOLERANCE)
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


# -- DELETE -- DELETE -- DELETE -- DELETE -- DELETE -- DELETE -- DELETE -- DELETE --
