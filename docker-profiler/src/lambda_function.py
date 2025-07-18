def lambda_handler_curvelink(event, context):
    from src.handlers_link import CurveLinkHandler
    handler = CurveLinkHandler(event, context)
    return handler.handle()

def lambda_handler_swaplink(event, context):
    from src.handlers_link import SwapLinkHandler
    handler = SwapLinkHandler(event, context)
    return handler.handle()

def lambda_handler_spylink(event, context):
    from src.handlers_link import SpyLinkHandler
    handler = SpyLinkHandler(event, context)
    return handler.handle()

def lambda_handler_download(event, context):
    from src.handlers_base import DownloadHandler
    handler = DownloadHandler(event, context)
    return handler.handle()

def lambda_handler_printerlink(event, context):
    from src.handlers_prtr import PrtrHandler
    handler = PrtrHandler(event, context)
    return handler.handle()