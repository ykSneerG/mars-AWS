""" 
from src.handlers import CurveLinkHandler

def lambda_handler_curvelink(event, context):
    handler = CurveLinkHandler(event, context)
    return handler.handle()
"""

import time
from src.error import status_error

from src.code.predict.linearization.synlin import SynLinSolid
def lh_predict_linearization(event, context):
    """ handler = CurveLinkHandler(event, context)
    return handler.handle() """
    
    start_time = time.time()
    
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
    
    res['time'] = time.time() - start_time
    
    # ----
                
    return {
        'statusCode': 200,
        'result': res
    }
    
    
from src.code.predict.linearization.synlinV2 import SynLinSolidV2
def lh_predict_linearization_v2(event, context):
    
    start_time = time.time()
    
    media = event['media']
    solid = event['solid']
    steps = int(event.get('steps', 5))
    iters = int(event.get('iterations', 1)) 
    toler = float(event.get('tolerance', 0.004))

    sls = SynLinSolidV2()
    """ sls.set_places = int(event.get('round', 3))  """
    sls.set_media(media)
    sls.set_solid(solid)
    
    err = sls.set_gradient_by_steps(steps)
    if err: return status_error(400, err)
    
    #sls.set_gradient([0.0, 50.0, 100.0])
    sls.tolerance = toler
    sls.set_max_loops = iters
    res = sls.start()
    
    res['time'] = time.time() - start_time
    
    # ----
                
    return {
        'statusCode': 200,
        'result': res
    }


from src.code.predict.linearization.linearInterpolation import LinearInterpolation
def lh_predict_linearinterpolation(event, context):
    
    start_time = time.time()
    
    media = event['media']
    solid = event['solid']
    steps = int(event.get('steps', 5))
    iters = int(event.get('iterations', 1)) 
    toler = float(event.get('tolerance', 0.004))

    sls = LinearInterpolation()
    """ sls.set_places = int(event.get('round', 3))  """
    sls.set_media(media)
    sls.set_solid(solid)
    
    err = sls.set_gradient_by_steps(steps)
    if err: return status_error(400, err)
    
    #sls.set_gradient([0.0, 50.0, 100.0])
    sls.tolerance = toler
    sls.set_max_loops = iters
    res = sls.start()
    
    res['time'] = time.time() - start_time
    
    # ----
                
    return {
        'statusCode': 200,
        'result': res
    }


from src.code.predict.linearization.synlinV3 import SynLinSolidV3
def lh_predict_linearization_v3(event, context):
    
    start_time = time.time()
    
    media = event['media']
    solid = event['solid']
    steps = int(event.get('steps', 5))
    iters = int(event.get('iterations', 1)) 
    toler = float(event.get('tolerance', 0.004))
    cfact = float(event.get('correction', 4.5))

    sls = SynLinSolidV3()
    """ sls.set_places = int(event.get('round', 3))  """
    sls.set_media(media)
    sls.set_solid(solid)
    
    err = sls.set_gradient_by_steps(steps)
    if err: return status_error(400, err)
    
    #sls.set_gradient([0.0, 50.0, 100.0])
    sls.tolerance = toler
    sls.set_max_loops = iters
    sls.setCorrection(cfact)
    res = sls.start()
    
    res['time'] = time.time() - start_time
    
    # ----
                
    return {
        'statusCode': 200,
        'result': res
    }
