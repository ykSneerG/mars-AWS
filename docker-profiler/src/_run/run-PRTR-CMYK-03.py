import sys
sys.path.append("/Users/heikopieper/Documents/GitHub/mars-AWS/docker-profiler")

from src.code.icctools.IccV4_Helper import Helper
from src.handlers_prtr import PrtrHandler
from src._run.RunHelper import save_bytes_to_file

# from src._run.input.mdata03 import mdata
# from src._run.input.mdata_example2401 import mdata
# from src._run.input.mdataBeton import mdata
# from src._run.input.mdataRoland import mdata
from src._run.input.mdataEpson import mdata

if __name__ == "__main__":

    mdata_dcs = [x["dcs"] for x in mdata]
    grid_len = int(pow(len(mdata_dcs), 1 / len(mdata_dcs[0])))

    # print(len(mdata_dcs))
    # print(f"Grid length: {grid_len}")

    event = {
        "xyzs_src": Helper.divide_by_100([x["xyz"] for x in mdata]),
        "dcss_src": [x["dcs"] for x in mdata],
        "link_name": f"{"Test FIRST 23a09-0x4"}",
        "grid_size": {
            "atob": [11, 13, 11],
            # "atob": [grid_len, grid_len, grid_len],
            "btoa": [21, 21, 17],
        },
        "includeBytes": True,
    }
    context = {}

    handler = PrtrHandler(event, context)
    response = handler.handle()

    body = response["body"]

    print({k: v for k, v in body.items() if k != "bytes" and k != "gcr_test"})
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")
    # save_bytes_to_file(body["bytes"], f"PRTR-{event.get('link_name')}-{body['fileID']}.icc")
    save_bytes_to_file(body["bytes"], f"PRTR-{event.get('link_name')}.icc")
    print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -")
    
    
    # Plotten
    import matplotlib.pyplot as plt

    L = body["gcr_test"]["L"]
    C = body["gcr_test"]["C"]
    M = body["gcr_test"]["M"]
    Y = body["gcr_test"]["Y"]
    K = body["gcr_test"]["K"]

    plt.figure(figsize=(5, 5))
    plt.plot(L, C, label='Cyan', color='cyan')
    plt.plot(L, M, label='Magenta', color='magenta')
    plt.plot(L, Y, label='Yellow', color='gold')
    plt.plot(L, K, label='Black', color='black')

    plt.xlabel("Ink Coverage (%)")
    plt.ylabel("L* (Lightness)")
    plt.title("CMYK-Anteil in Abhängigkeit von LAB-Helligkeit (L*)")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()  # L* 100 oben, 0 unten
    plt.tight_layout()
    plt.show()
