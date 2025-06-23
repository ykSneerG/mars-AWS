import sys
sys.path.append('/Users/heikopieper/Documents/GitHub/mars-AWS/docker-profiler')

from src.handlers_link import SwapLinkHandler
from src._run.RunHelper import save_bytes_to_file


if __name__ == "__main__":

    name = "Amsterdam 1803"

    # Sample request Body for SwapLinkHandler
    # !!! "includeBytes" is set to True, only for testing purposes !!!
    event = {
        "orderOld": [0, 1, 2, 3],
        "orderNew": [2, 3, 1, 0],
        "link_name": name,
        "includeBytes": True
    }
    context = {}

    handler = SwapLinkHandler(event, context)
    response = handler.handle()

    body = response["body"]

    # print response without “bytes"
    result = {k: v for k, v in body.items() if k != "bytes"}
    print(result)
    
    save_bytes_to_file(body["bytes"], f"SWAP-{name}-{body['fileID']}.icc")    
