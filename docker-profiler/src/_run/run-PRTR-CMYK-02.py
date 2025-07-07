import sys
sys.path.append("/Users/heikopieper/Documents/GitHub/mars-AWS/docker-profiler")

from src.code.icctools.IccV4_Helper import Helper, ColorTrafo
from src.code.icctools.MakeXtoX import MakeAtoB, MakeBtoA, MakeCurve
from src.handlers_prtr import PrtrHandler
from src._run.RunHelper import save_bytes_to_file
# from src._run.input.mdata03 import mdata
from src._run.input.mdata_example2401 import mdata
# from src._run.input.mdataBeton import mdata

from concurrent.futures import ThreadPoolExecutor, as_completed

# import numpy as np

if __name__ == "__main__":


    name = "Ceton 017-01_KDtree_3scale"
    grid_size = 17

    grid_size_atob0 = 13
    grid_size_atob1 = 17
    grid_size_atob2 = 9
    
    grid_size_btoa0 = 17
    grid_size_btoa1 = 23
    grid_size_btoa2 = 11


    link_name = f"{name}_A0-{grid_size_atob0}-{grid_size_btoa0}_A1-{grid_size_atob1}-{grid_size_btoa1}_A2-{grid_size_atob2}-{grid_size_btoa2}"
    # link_name = "EX-1600C_7"
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")
    print(link_name)
    print("")
    print("Input Data: ", len(mdata))
    print("Channel Count: ", len(mdata[0]["dcs"]))
    #print("Grid Size: ", grid_size)
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")

    dcss = [x["dcs"] for x in mdata]

    xyzs_abs = Helper.divide_by_100([x["xyz"] for x in mdata])
    labs_abs = ColorTrafo.xyz_to_lab(xyzs_abs)
    
    
    
    """ # --- UPSCALED RELATIVE ---
    make_atob = MakeAtoB(labs_abs)
    dense_dcs, dense_lab = make_atob._interpolate_grid(dcss, labs_abs, 11)
    #dense_lab_uint16 = Helper.lab_to_uint16(dense_lab.tolist())
    #dense_lab_uint16_flatted = Helper.flatten_list(dense_lab_uint16)
    # relative_lut = dense_lab_uint16
    #print("DENSE LUT: ", dense_lab_uint16[0:10])
    
    labs_abs = dense_lab.tolist()
    #idx_dark = len(labs_rel) - 1
    xyzs_abs = ColorTrafo.lab_to_xyz(labs_abs) """
    
    

    idx_light = 0
    idx_dark = len(xyzs_abs) - 1
    """ 
    darkest_index = min(range(len(relative_lut)), key=lambda i: labs_abs[i][0])
    lighthest_index = max(range(len(relative_lut)), key=lambda i: labs_abs[i][0]) 
    """
    WTPT = xyzs_abs[idx_light]
    BKTP = xyzs_abs[idx_dark]


    xyzs_rel = ColorTrafo.media_relative(xyzs_abs, WTPT)
    labs_rel = ColorTrafo.xyz_to_lab(xyzs_rel)
    
    
    
    print("- + - AtoB Tables - + -")
    
    # --- ABSOLUTE ---
    print("ABSOLUTE\tLightest - Index: ", idx_light, "\t\tLAB: ", Helper.round_list(labs_abs[idx_light], 4))
    print("ABSOLUTE\tDarkest. - Index: ", idx_dark, "\tLAB: ", Helper.round_list(labs_abs[idx_dark], 4))
    print("")
    
    
    
    # --- AtoB1 RELATIVE ---
    make_atob1 = MakeAtoB(labs_rel)
    make_atob1.index_lightest = idx_light
    make_atob1.index_darkest = idx_dark
    make_atob1.generate_relative()
    relative_lut = make_atob1.clut
    make_atob1.report("RELATIVE")
    
    # --- UPSCALED RELATIVE ---
    dense_dcs_rel, dense_lab_rel = MakeAtoB.interpolate_grid(dcss, labs_rel, grid_size_atob1)
    a2b1_scaled = Helper.lab_to_uint16(dense_lab_rel.tolist())


    # --- AtoB0 PERCEPTUAL ---
    make_atob0 = MakeAtoB(labs_rel)
    make_atob0.index_lightest = idx_light
    make_atob0.index_darkest = idx_dark
    make_atob0.generate_perceptual()
    perceptual_lut = make_atob0.clut
    make_atob0.report("PERCEPTUAL")

    # --- UPSCALED PERCEPTUAL ---
    dense_dcs_sat, dense_lab_sat = MakeAtoB.interpolate_grid(dcss, perceptual_lut, grid_size_atob0)
    a2b0_scaled = Helper.lab_to_uint16(dense_lab_sat.tolist())
    
    
    # --- AtoB2 SATURATION ---
    make_atob2 = MakeAtoB(labs_rel)
    make_atob2.index_lightest = idx_light
    make_atob2.index_darkest = idx_dark
    make_atob2.generate_saturation()
    saturation_lut = make_atob2.clut
    make_atob2.report("SATURATION")
    
    # --- UPSCALED SATURATION ---
    dense_dcs_pcl, dense_lab_pcl = MakeAtoB.interpolate_grid(dcss, saturation_lut, grid_size_atob2)
    a2b2_scaled = Helper.lab_to_uint16(dense_lab_pcl.tolist())

    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")

    # --- BtoA0 PERCEPTUAL ---
    # make_btoa0 = MakeBtoA(dcss, perceptual_lut, grid_size_atob0)
    make_btoa0 = MakeBtoA(dense_dcs_pcl.tolist(), dense_lab_pcl.tolist(), grid_size_atob0)
    clut_btoa0 = make_btoa0.generate()
    make_btoa0.report("PERCEPTUAL")
    
    """ perc_C = MakeCurve.createZigZagCurve_LAB(0, 99.0 / 100)
    perc_M = MakeCurve.createZigZagCurve_LAB(0, 99.9 / 100)
    perc_Y = MakeCurve.createZigZagCurve_LAB(0, 99.7 / 100)
    perc_K = MakeCurve.createZigZagCurve_LAB(0, 100 / 100) """

    # --- BtoA1 RELATIVE ---
    #make_btoa1 = MakeBtoA(dcss, relative_lut, grid_size_atob1)
    make_btoa1 = MakeBtoA(dense_dcs_rel.tolist(), dense_lab_rel.tolist(), grid_size_atob1)
    clut_btoa1 = make_btoa1.generate()
    make_btoa1.report("RELATIVE")

    #curve_btoa1_L = MakeCurve.createZigZagCurve_LAB(labs_abs[idx_light][0] / 100, labs_abs[idx_dark][0] / 100)
    """ 
    curve_btoa1_A = [int((i / 255 * 255)* 255) for i in range(256)]
    curve_btoa1_B = [int((i / 255 * 255)* 255) for i in range(256)]
    print(curve_btoa1_L)
    print(curve_btoa1_A)
    print(curve_btoa1_B) 
    """


    # --- BtoA2 SATURATION ---
    # make_btoa2 = MakeBtoA(dcss, saturation_lut, grid_size_atob2)
    make_btoa2 = MakeBtoA(dense_dcs_sat.tolist(), dense_lab_sat.tolist(), grid_size_atob2)
    clut_btoa2 = make_btoa2.generate()
    make_btoa2.report("SATURATION")
    
    # # --- BtoA generation in parallel ---
    # def make_btoa(args):
    #     name, dcs, lab, grid_size, report_name = args
    #     btoa = MakeBtoA(dcs, lab, grid_size)
    #     btoa.generate()
    #     #btoa.report(report_name)
    #     return name, btoa

    # btoa_tasks = [
    #     ("btoa0", dense_dcs_pcl.tolist(), dense_lab_pcl.tolist(), grid_size_atob0, "PERCEPTUAL"),
    #     ("btoa1", dense_dcs_rel.tolist(), dense_lab_rel.tolist(), grid_size_atob1, "RELATIVE"),
    #     ("btoa2", dense_dcs_sat.tolist(), dense_lab_sat.tolist(), grid_size_atob2, "SATURATION"),
    # ]

    # results = {}
    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     futures = {executor.submit(make_btoa, args): args[0] for args in btoa_tasks}
    #     for future in as_completed(futures):
    #         name, btoa = future.result()
    #         results[name] = btoa

    # make_btoa0 = results["btoa0"]
    # make_btoa1 = results["btoa1"]
    # make_btoa2 = results["btoa2"]

    # - + - + - + - + - + - + - + - + - + - + -

    event = {
        "wtpt": WTPT,
        "bktp": BKTP,
        "atob0_clut": Helper.flatten_list(a2b0_scaled), #make_atob0.clut_as_uint16,
        "atob1_clut": Helper.flatten_list(a2b1_scaled), #make_atob1.clut_as_uint16,
        "atob2_clut": Helper.flatten_list(a2b2_scaled), #make_atob2.clut_as_uint16,

        "btoa0_clut": make_btoa0.clut_as_uint16,
        #""" "btoa1_output_table": [
        #    perc_C,
        #    perc_M,
        #    perc_Y,
        #    perc_K
        #],
        # """
        "btoa1_clut": make_btoa1.clut_as_uint16,
        #"btoa1_input_table": [
        #    [0, 32768, 65534],
        #    [0, 32768, 65534],
        #    [0, 32768, 65534]
        #],
        #"btoa1_output_table": [
        #    curve_btoa1_L,
        #    curve_btoa1_L,
        #    curve_btoa1_L,
        #    curve_btoa1_L
        #],
        
        "btoa2_clut": make_btoa2.clut_as_uint16,

        "link_name": link_name,
        "includeBytes": True,
    }
    context = {}

    handler = PrtrHandler(event, context)
    response = handler.handle()

    body = response["body"]

    # print response without “bytes"
    result = {k: v for k, v in body.items() if k != "bytes"}
    print(result)
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")

    save_bytes_to_file(body["bytes"], f"PRTR-{link_name}-{body['fileID']}.icc")
    
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")
