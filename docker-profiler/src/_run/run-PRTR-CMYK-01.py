import sys
sys.path.append("/Users/heikopieper/Documents/GitHub/mars-AWS/docker-profiler")

from src.code.icctools.IccV4_Helper import Helper
from src.code.icctools.Interpolation import VectorTranslate, VectorScale
from src.code.icctools.ColorAnalyser import ColorAnalyser
from src.code.icctools.LutInvers import build_btoa_lut, LUTInverter
from src.code.icctools.LutInvers_DS import BToA_LUT_Generator, LUTInverter_DS
from src.handlers_prtr import PrtrHandler
from src._run.RunHelper import save_bytes_to_file
from src._run.input.mdata import mdata

import numpy as np


if __name__ == "__main__":
    
    name = "Tokio 1881_V06"
    
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")
    print(name)
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")

    labs = [x["lab"] for x in mdata]
    xyzs = [x["xyz"] for x in mdata]
    dcss = [x["dcs"] for x in mdata]

    xyz_white = xyzs[0]
    wtpt = Helper.scale_to_range(xyz_white, 0, 100, 0, 1)
    xyz_black = xyzs[len(xyzs) - 1]
    bktp = Helper.scale_to_range(xyz_black, 0, 100, 0, 1)
    
    relative_lut = VectorTranslate(labs[0], [100,0,0]).apply(labs)
    #print("Relative LUT: ", relative_lut)
    
    # find index with the lighthest LAB L value
    lighthest_index = max(range(len(relative_lut)), key=lambda i: labs[i][0])
    lighthest_lab = relative_lut[lighthest_index]
    print("RELATIVE\tLighthest- Index: ", lighthest_index, "\tLAB: ", lighthest_lab)

    # find index with the darkest LAB L value
    darkest_index = min(range(len(relative_lut)), key=lambda i: labs[i][0])
    darkest_lab = relative_lut[darkest_index]
    print("RELATIVE\tDarkest. - Index: ", darkest_index, "\tLAB: ", darkest_lab)


    # Perceptual
    """ lut = LutMaker()
    perceptual_lut = lut.perceptual_compression(
        labs,
        (labs[len(labs) - 1][0], labs[len(labs) - 1][1], labs[len(labs) - 1][2]),
        (0.0, 0.0, 0.0),
        (labs[0][0], labs[0][1], labs[0][2]),
        (100.0, 0.0, 0.0),
        True,
        0.8,
        0.7,
        5.0
    ) """
    
    """ lut = LutMakerHybrid()
    perceptual_lut = lut.hybrid_perceptual_lut(
        labs,
        (labs[len(labs) - 1][0], labs[len(labs) - 1][1], labs[len(labs) - 1][2]),
        (labs[0][0], labs[0][1], labs[0][2]),
        (0.0, 0.0, 0.0),
        (100.0, 0.0, 0.0),
        0.8,
        0.7,
        5.0,
        True
    ) """
    
    #vecmap = VectorMapper()
    """ mx = vecmap.vec_rot_mat(
        (lighthest_lab[0]-0.1, 0.0, 0.0),
        (darkest_lab[0], 0.0, 0.0),
        (100.0, 0.0, 0.0),
        (0.0, 0.0, 0.0)
    ) """

    perceptual_lut = VectorScale().apply_l_only(relative_lut, darkest_lab[0], 100)
    print("PERCEPTUAL\tLighthest: ", "\t", "\tLAB: ", round(perceptual_lut[lighthest_index][0], 5))
    print("PERCEPTUAL\tDarkest: ", "\t", "\tLAB: ", round(perceptual_lut[darkest_index][0], 5))

    #print("Perceptual LUT: ", perceptual_lut)
    
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")

    """ atob_lut = []
    for i in range(len(labs)):
        atob_lut.append((dcss[i], relative_lut[i])) """
        
    atob_lut = list(zip(dcss, relative_lut))
    #test = build_btoa_lut(atob_lut, lab_grid_shape=(7, 7, 7))
    
    inverter = LUTInverter(atob_lut, cmyk_grid_size=4, lab_grid_shape=(13, 13, 13))
    # Build BToA LUT
    test = inverter.build_btoa_lut()
    # Optionally smooth the result
    #test = LUTInverter.smooth_lut_2(btoa, sigma=1.0)
    
    """ inverter = LUTInverter_DS(atob_lut, 4, lab_grid_shape=(11, 11, 11))
    
    # Faster version (more approximations)
    #fast_lut = inverter.build_btoa_lut(fast_approx_threshold=10.0)

    # Higher quality version (fewer approximations)
    quality_lut = inverter.build_btoa_lut(fast_approx_threshold=2.0)

    # With smoothing
    # smoothed_lut = inverter.smooth_lut(quality_lut, sigma=0.7)
    
    test = quality_lut
     """
    """ inverter = BToA_LUT_Generator(atob_lut, cmyk_grid_size=4, lab_grid_shape=(11, 11, 11))
    test = inverter.generate_btoa_lut(smooth_sigma=0) """

    #print("Table: ", test)
    print("Shape: ", test.shape)
    print("Length: ", test.size / test.shape[3])
    print("First: ", test[0][0][0])


    btoa1_uint16 = (test * 65535.0).astype(np.uint16).tolist()
    #print("Flattened test: ", flat_test)

    # flat_uint16_test = Helper.flatten_list(uint16_test)

    #uint16_test = Helper.lab_to_uint16(flat_test.tolist())
    #print("uint16_test: ", flat_uint16_test)
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")



    
    """ vecmap = VectorMapper()
    mx = vecmap.vec_rot_mat(
        (labs[0][0], labs[0][1], labs[0][2]),
        (labs[len(labs) - 1][0], labs[len(labs) - 1][1], labs[len(labs) - 1][2]),
        (100.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    #print("s0: ", (labs[0][0], labs[0][1], labs[0][2]))
    #print("t0: ", (labs[len(labs) - 1][0], labs[len(labs) - 1][1], labs[len(labs) - 1][2]))
    #print("Matrix: ", mx)
    perceptual_lut = vecmap.apply(labs)
    #print("Perceptual LUT: ", perceptual_lut) """

    # Perceptual
    clut_atob0 = Helper.lab_to_uint16(perceptual_lut)
    clut_btoa0 = Helper.flatten_list(btoa1_uint16)

    # Media-relative colorimetric
    clut_atob1 = Helper.lab_to_uint16(relative_lut)
    clut_btoa1 = Helper.flatten_list(btoa1_uint16)
    
    # Saturation
    clut_atob2 = Helper.lab_to_uint16(labs)
    clut_btoa2 = Helper.flatten_list(uint16_test)
    
    
    # name = "Tokio 1874"

    # Sample request Body for PrtrHandler
    # !!! "includeBytes" is set to True, only for testing purposes !!!

    """ 
    "curves": [
            [0, 12000, 65534],
            [65534, 32768, 0],
            [0, 32768, 65534],
            [0, 32768, 65534]
        ],
    """

    event = {
        "wtpt": wtpt,  # White Point XYZ
        "bktp": bktp,  # Black Point XYZ
        "atob0_clut": Helper.flatten_list(clut_atob0),
        "atob1_clut": Helper.flatten_list(clut_atob1),
        "atob2_clut": Helper.flatten_list(clut_atob2),
        "btoa0_clut": clut_btoa0,
        "btoa1_clut": clut_btoa1,
        "btoa2_clut": clut_btoa2,


        "link_name": name,
        "includeBytes": True,
    }
    context = {}

    handler = PrtrHandler(event, context)
    response = handler.handle()

    body = response["body"]

    # print response without “bytes"
    result = {k: v for k, v in body.items() if k != "bytes"}
    print(result)
    print("")
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")

    save_bytes_to_file(body["bytes"], f"PRTR-{name}-{body['fileID']}.icc")
    
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")
