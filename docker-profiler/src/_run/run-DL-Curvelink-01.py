import sys

sys.path.append("/Users/heikopieper/Documents/GitHub/mars-AWS/docker-profiler")

from src.handlers_link import CurveLinkHandler
from src._run.RunHelper import save_bytes_to_file


if __name__ == "__main__":

    name = "New York 1815"

    # Sample request Body for SwapLinkHandler
    # !!! "includeBytes" is set to True, only for testing purposes !!!
    event = {
        "curves": [
            [0, 12000, 65534],
            [65534, 32768, 0],
            [0, 32768, 65534],
            [0, 32768, 65534]
        ],
        "link_name": name,
        "includeBytes": True
    }
    context = {}

    handler = CurveLinkHandler(event, context)
    response = handler.handle()

    body = response["body"]

    # print response without “bytes"
    result = {k: v for k, v in body.items() if k != "bytes"}
    print(result)

    save_bytes_to_file(body["bytes"], f"CURVE-{name}-{body['fileID']}.icc")
