import time
from src.error import status_error

# - - - PREDICT - - -

# from src.code.predict.linearization.synlin import SynLinSolid
# def lh_predict_linearization(event, context):
#     """ handler = CurveLinkHandler(event, context)
#     return handler.handle() """
    
#     start_time = time.time()
    
#     media = event['media']
#     solid = event['solid']
#     steps = event.get('steps', 5)
#     iters = int(event.get('iterations', 1)) 
#     round = int(event.get('round', 3)) 
#     toler = event.get('tolerance', 2)

#     sls = SynLinSolid()
#     sls.set_places = round
#     sls.set_media(media)
#     sls.set_solid(solid)
#     sls.set_gradient_by_steps(steps)
#     #sls.set_gradient([0.0, 50.0, 100.0])
#     sls.tolerance = toler
#     sls.calculate_loops(iters)
#     res = sls.get_prediction()
    
#     res['time'] = time.time() - start_time
    
#     # ----
                
#     return {
#         'statusCode': 200,
#         'result': res
#     }
    
# from src.code.predict.linearization.synlinV2 import SynLinSolidV2
# def lh_predict_linearization_v2(event, context):
    
#     start_time = time.time()
    
#     media = event['media']
#     solid = event['solid']
#     steps = int(event.get('steps', 5))
#     iters = int(event.get('iterations', 1)) 
#     toler = float(event.get('tolerance', 0.004))

#     sls = SynLinSolidV2()
#     """ sls.set_places = int(event.get('round', 3))  """
#     sls.set_media(media)
#     sls.set_solid(solid)
    
#     err = sls.set_gradient_by_steps(steps)
#     if err: return status_error(400, err)
    
#     #sls.set_gradient([0.0, 50.0, 100.0])
#     sls.tolerance = toler
#     sls.set_max_loops = iters
#     res = sls.start()
    
#     res['time'] = time.time() - start_time
    
#     # ----
                
#     return {
#         'statusCode': 200,
#         'result': res
#     }



def lh_predict_linearinterpolation(event, context):
    from src.handlersPredict import Predict_LinearInterpolation_Handler
    handler = Predict_LinearInterpolation_Handler(event, context)
    return handler.handle()

# color api -- can be deleted
# def lh_predict_line_v4(event, context):
#     from src.handlersPredict import Predict_SynlinV4_Handler
#     handler = Predict_SynlinV4_Handler(event, context)
#     return handler.handle()

# mars api -- NEW !
def lh_predict_1D(event, context):
    # from src.handlersPredict import Predict_SynlinV4_Handler
    # handler = Predict_SynlinV4_Handler(event, context)
    # return handler.handle()
    from src.handlersPredict import Predict_SynlinV4Multi_Handler
    handler = Predict_SynlinV4Multi_Handler(event, context)
    return handler.handle()

# def lh_predict_line_multi_v4(event, context):
#     from src.handlersPredict import Predict_SynlinV4Multi_Handler
#     handler = Predict_SynlinV4Multi_Handler(event, context)
#     return handler.handle()

#def lh_predict_aera_v4(event, context):
def lh_predict_2D(event, context):
    from src.handlersPredict import Predict_SynAreaV4_Handler
    handler = Predict_SynAreaV4_Handler(event, context)
    return handler.handle()


#def lh_predict_volume_v4(event, context):
def lh_predict_3D(event, context):
    from src.handlersPredict import Predict_SynVolumeV4_Handler
    handler = Predict_SynVolumeV4_Handler(event, context)
    return handler.handle()

#def lh_predict_4dimensional_v4(event, context):
def lh_predict_4D(event, context):
    from src.handlersPredict import Predict_SynHyperFourV4_Parallel_Handler
    handler = Predict_SynHyperFourV4_Parallel_Handler(event, context)
    return handler.handle()

def lh_predict_interpolate_pairs(event, context):
    from src.handlersPredict import InterpolatePairs
    handler = InterpolatePairs(event, context)
    return handler.handle()

# - - - FILE - - -

def lf_files_cgats2json(event, context):
    from src.handlersFiles import Files_CgatsToJson_Handler
    handler = Files_CgatsToJson_Handler(event, context)
    return handler.handle()


# - - - SAMPLE - - -

def lh_sample_color_spectral(event, context):
    from src.handlersSpace import Space_SampleSpectral_Handler
    handler = Space_SampleSpectral_Handler(event, context)
    return handler.handle()


# - - - INTERPOLATE TARGET - - -

def lh_interpolate_target_rbf(event, context):
    from src.handlersPredict import InterpolateTarget_RBF_Handler
    handler = InterpolateTarget_RBF_Handler(event, context)
    return handler.handle()

def lh_interpolate_target_optrbf(event, context):
    from src.handlersPredict import InterpolateTarget_OptRBF_Handler
    handler = InterpolateTarget_OptRBF_Handler(event, context)
    return handler.handle()

def lh_interpolate_target_modernrbf(event, context):
    from src.handlersPredict import InterpolateTarget_modernRBF_Handler
    handler = InterpolateTarget_modernRBF_Handler(event, context)
    return handler.handle()

def lh_interpolate_target_modernchs(event, context):
    from src.handlersPredict import InterpolateTarget_cubicHermiteSpline_Handler
    handler = InterpolateTarget_cubicHermiteSpline_Handler(event, context)
    return handler.handle()


# - - - INTERPOLATE TARGET - - -

def lh_file_upload_cgats(event, context):
    from src.handlersFiles import File_UploadCgats_Handler
    handler = File_UploadCgats_Handler(event, context)
    return handler.handle()

def lh_file_uploadedcgats_to_json(event, context):
    from src.handlersFiles import File_UploadedToJson_Handler
    handler = File_UploadedToJson_Handler(event, context)
    return handler.handle()

def lh_file_uploaded_interpolate_target(event, context):
    from src.handlersPredict import InterpolateTarget_Handler
    handler = InterpolateTarget_Handler(event, context)
    return handler.handle()

# NOT USED !!!
def lh_file_uploaded_interpolate_target_chs(event, context):
    from src.handlersPredict import InterpolateTarget_cubicHermiteSpline_Handler
    handler = InterpolateTarget_cubicHermiteSpline_Handler(event, context)
    return handler.handle()


# - - - COLOR DIFFERENCE - - -

def lh_trafo_delta(event, context):
    from src.handlersTrafo import Trafo_Delta_Handler
    handler = Trafo_Delta_Handler(event, context)
    return handler.handle()
    # -- USED: MARSAPI --

def lh_trafo_convert_spectral(event, context):
    from src.handlersTrafo import Trafo_ConvertSpectral_Handler
    handler = Trafo_ConvertSpectral_Handler(event, context)
    return handler.handle()
    # -- USED: MARSAPI --