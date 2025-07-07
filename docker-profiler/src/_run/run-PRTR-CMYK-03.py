import sys
sys.path.append("/Users/heikopieper/Documents/GitHub/mars-AWS/docker-profiler")

from src.code.icctools.IccV4_Helper import Helper, ColorTrafo
from src.code.icctools.MakeXtoX import MakeAtoB, MakeBtoA, MakeCurve
from src.handlers_prtr import PrtrHandler
from src._run.RunHelper import save_bytes_to_file
# from src._run.input.mdata03 import mdata
# from src._run.input.mdata_example2401 import mdata
from src._run.input.mdataBeton import mdata


if __name__ == "__main__":


    name = "Ceton 017-30--"
    grid_size = 17

    grid_size_atob0 = 11
    grid_size_atob1 = 11
    grid_size_atob2 = 9
    
    grid_size_btoa0 = 17
    grid_size_btoa1 = 21
    grid_size_btoa2 = 11


    link_name = f"{name}_A0-{grid_size_atob0}-{grid_size_btoa0}_A1-{grid_size_atob1}-{grid_size_btoa1}_A2-{grid_size_atob2}-{grid_size_btoa2}"

    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")
    print(link_name)
    print("")
    print("Input Data: ", len(mdata))
    print("Channel Count: ", len(mdata[0]["dcs"]))
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")

    dcss = [x["dcs"] for x in mdata]

    xyzs_abs = Helper.divide_by_100([x["xyz"] for x in mdata])
    labs_abs = ColorTrafo.xyz_to_lab(xyzs_abs)
    

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
    make_atob1 = MakeAtoB(labs_rel, dcss)
    make_atob1.index_lightest = idx_light
    make_atob1.index_darkest = idx_dark
    make_atob1.generate_relative()
    make_atob1.interpolate_clut(grid_size_atob1)
    make_atob1.report("RELATIVE")

    # --- AtoB0 PERCEPTUAL ---
    make_atob0 = MakeAtoB(labs_rel, dcss)
    make_atob0.index_lightest = idx_light
    make_atob0.index_darkest = idx_dark
    make_atob0.generate_perceptual()
    make_atob0.interpolate_clut(grid_size_atob0)
    make_atob0.report("PERCEPTUAL")

    # --- AtoB2 SATURATION ---
    make_atob2 = MakeAtoB(labs_rel, dcss)
    make_atob2.index_lightest = idx_light
    make_atob2.index_darkest = idx_dark
    make_atob2.generate_saturation()
    make_atob2.interpolate_clut(grid_size_atob2)
    make_atob2.report("SATURATION")

    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")

    # --- BtoA0 PERCEPTUAL ---
    make_btoa0 = MakeBtoA(make_atob0._dcs, make_atob0._clut, grid_size_btoa0)
    make_btoa0.generate()
    make_btoa0.report("PERCEPTUAL")

    # --- BtoA1 RELATIVE ---
    make_btoa1 = MakeBtoA(make_atob1._dcs, make_atob1._clut, grid_size_btoa1)
    make_btoa1.generate()
    make_btoa1.report("RELATIVE")
    
    # --- BtoA2 SATURATION ---
    make_btoa2 = MakeBtoA(make_atob2._dcs, make_atob2._clut, grid_size_btoa2)
    make_btoa2.generate()
    make_btoa2.report("SATURATION")
    
    # - + - + - + - + - + - + - + - + - + - + -

    event = {
        "wtpt": WTPT,
        "bktp": BKTP,
        "atob0_clut": make_atob0.clut_as_uint16,
        "atob1_clut": make_atob1.clut_as_uint16,
        "atob2_clut": make_atob2.clut_as_uint16,

        "btoa0_clut": make_btoa0.clut_as_uint16,
        "btoa1_clut": make_btoa1.clut_as_uint16,
        "btoa2_clut": make_btoa2.clut_as_uint16,

        "link_name": link_name,
        "includeBytes": True,
    }
    context = {}

    handler = PrtrHandler(event, context)
    response = handler.handle()

    body = response["body"]

    print({k: v for k, v in body.items() if k != "bytes"})
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")

    save_bytes_to_file(body["bytes"], f"PRTR-{link_name}-{body['fileID']}.icc")
    
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")
