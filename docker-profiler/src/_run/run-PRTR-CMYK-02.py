import sys
sys.path.append("/Users/heikopieper/Documents/GitHub/mars-AWS/docker-profiler")

from src.code.icctools.IccV4_Helper import Helper
from src.code.icctools.Interpolation import VectorTranslate, VectorScale
from src.code.icctools.LutInvers import LUTInverter
from src.handlers_prtr import PrtrHandler
from src._run.RunHelper import save_bytes_to_file
from src._run.input.mdata import mdata

import numpy as np


if __name__ == "__main__":
    
    grid_size = 11
    
    name = f"Tokio 1884_G{grid_size}_V01"
    
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
    perceptual_lut = VectorScale().apply_l_only(relative_lut, darkest_lab[0], 100)
    print("PERCEPTUAL\tLighthest: ", "\t", "\tLAB: ", round(perceptual_lut[lighthest_index][0], 5))
    print("PERCEPTUAL\tDarkest: ", "\t", "\tLAB: ", round(perceptual_lut[darkest_index][0], 5))

    #print("Perceptual LUT: ", perceptual_lut)
    
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")
    
    atob_lut = list(zip(dcss, relative_lut))
    #test = build_btoa_lut(atob_lut, lab_grid_shape=(7, 7, 7))
    
    inverter = LUTInverter(atob_lut, cmyk_grid_size=4, lab_grid_shape=(grid_size, grid_size, grid_size))
    # Build BToA LUT
    test = inverter.build_btoa_lut()
    # Optionally smooth the result
    #test = LUTInverter.smooth_lut_2(btoa, sigma=1.0)

    #print("Table: ", test)
    print("Shape: ", test.shape)
    print("Length: ", test.size / test.shape[3])
    print("First: ", test[0][0][0])


    btoa1_uint16 = (test * 65535.0).astype(np.uint16).tolist()

    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")

    atob0_lut = list(zip(dcss, perceptual_lut))
    #test = build_btoa_lut(atob_lut, lab_grid_shape=(7, 7, 7))

    inverter_atob0 = LUTInverter(atob0_lut, cmyk_grid_size=4, lab_grid_shape=(grid_size, grid_size, grid_size))
    # Build BToA LUT
    test_atob0 = inverter_atob0.build_btoa_lut()
    # Optionally smooth the result
    #test = LUTInverter.smooth_lut_2(btoa, sigma=1.0)

    #print("Table: ", test)
    print("Shape: ", test_atob0.shape)
    print("Length: ", test_atob0.size / test_atob0.shape[3])
    print("First: ", test_atob0[0][0][0])


    btoa0_uint16 = (test_atob0 * 65535.0).astype(np.uint16).tolist()

    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")


    # Perceptual
    clut_atob0 = Helper.lab_to_uint16(perceptual_lut)
    clut_btoa0 = Helper.flatten_list(btoa0_uint16)

    # Media-relative colorimetric
    clut_atob1 = Helper.lab_to_uint16(relative_lut)
    clut_btoa1 = Helper.flatten_list(btoa1_uint16)
    
    # Saturation
    clut_atob2 = Helper.lab_to_uint16(labs)
    clut_btoa2 = Helper.flatten_list(btoa0_uint16)
    
    
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
