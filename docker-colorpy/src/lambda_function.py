from src.handlersPredict import Predict_SynlinV4_Handler, Predict_LinearInterpolation_Handler, Predict_SynAreaV4_Handler, Predict_SynVolumeV4_Handler, Predict_SynHyperFourV4_Handler
from src.handlersFiles import Files_CgatsToJson_Handler
from src.handlersSpace import Space_SampleSpectral_Handler


import time
from src.error import status_error

# - - - PREDICT - - -

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



def lh_predict_linearinterpolation(event, context):
    handler = Predict_LinearInterpolation_Handler(event, context)
    return handler.handle()
    
def lh_predict_line_v4(event, context):
    handler = Predict_SynlinV4_Handler(event, context)
    return handler.handle()

def lh_predict_aera_v4(event, context):
    handler = Predict_SynAreaV4_Handler(event, context)
    return handler.handle()

def lh_predict_volume_v4(event, context):
    handler = Predict_SynVolumeV4_Handler(event, context)
    return handler.handle()

def lh_predict_4dimensional_v4(event, context):
    handler = Predict_SynHyperFourV4_Handler(event, context)
    return handler.handle()



# - - - FILE - - -

def lf_files_cgats2json(event, context):
    handler = Files_CgatsToJson_Handler(event, context)
    return handler.handle()


# - - - SAMPLE - - -

def lh_sample_color_spectral(event, context):
    handler = Space_SampleSpectral_Handler(event, context)
    return handler.handle()
