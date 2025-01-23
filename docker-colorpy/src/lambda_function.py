""" from src.handlers import CurveLinkHandler, SwapLinkHandler, SpyLinkHandler, DownloadHandler

def lambda_handler_curvelink(event, context):
    handler = CurveLinkHandler(event, context)
    return handler.handle()

def lambda_handler_swaplink(event, context):
    handler = SwapLinkHandler(event, context)
    return handler.handle()

def lambda_handler_spylink(event, context):
    handler = SpyLinkHandler(event, context)
    return handler.handle()

def lambda_handler_download(event, context):
    handler = DownloadHandler(event, context)
    return handler.handle() """

from src.code.predict.linearization.synlin import SynLinSolid
#from src.code.linear.sctv import SCTV

def lh_predict_linearization(event, context):
    """ handler = CurveLinkHandler(event, context)
    return handler.handle() """
    
    media = event['media']
    solid = event['solid']
    steps = event.get('steps', 5)
    iters = int(event.get('iterations', 1)) 
    round = int(event.get('round', 3)) 
    toler = event.get('tolerance', 2)

    sls = SynLinSolid()
    sls.set_places = round
    sls.set_media(media)
    sls.set_solid(solid)
    sls.set_gradient_by_steps(steps)
    #sls.set_gradient([0.0, 50.0, 100.0])
    sls.tolerance = toler
    sls.calculate_loops(iters)
    res = sls.get_prediction()
    
    # ----
                
    return {
        'statusCode': 200,
        'result': res
    }
