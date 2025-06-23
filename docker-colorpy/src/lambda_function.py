
# - - - PREDICT - - -

def lh_predict_linearinterpolation(event, context):
    from src.handlersPredict import Predict_LinearInterpolation_Handler
    handler = Predict_LinearInterpolation_Handler(event, context)
    return handler.handle()


# mars api -- NEW !
def lh_predict_1D(event, context):
    from src.handlersPredict import Predict_SynlinV4Multi_Handler
    handler = Predict_SynlinV4Multi_Handler(event, context)
    return handler.handle()

def lh_predict_2D(event, context):
    from src.handlersPredict import Predict_SynAreaV4_Handler
    handler = Predict_SynAreaV4_Handler(event, context)
    return handler.handle()

def lh_predict_3D(event, context):
    from src.handlersPredict import Predict_SynVolumeV4_Handler
    handler = Predict_SynVolumeV4_Handler(event, context)
    return handler.handle()

def lh_predict_4D(event, context):
    from src.handlersPredict import Predict_SynHyperFourV4_Parallel_Handler
    handler = Predict_SynHyperFourV4_Parallel_Handler(event, context)
    return handler.handle()

def lh_predict_interpolate_pairs(event, context):
    from src.handlersPredict import InterpolatePairs
    handler = InterpolatePairs(event, context)
    return handler.handle()


def lh_predict_mix(event, context):
    from src.handlersPredict import PredictYNSN_Handler
    handler = PredictYNSN_Handler(event, context)
    return handler.handle()

def lh_predict_mix_cell(event, context):
    from src.handlersPredict import PredictCYNSN_Handler
    handler = PredictCYNSN_Handler(event, context)
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

def lh_interpolate_target_modernrbf(event, context):
    from src.handlersPredict import InterpolateTarget_modernRBF_Handler
    handler = InterpolateTarget_modernRBF_Handler(event, context)
    return handler.handle()


# - - - FILE - - -

def lh_file_upload_cgats(event, context):
    from src.handlersFiles import File_UploadCgats_Handler
    handler = File_UploadCgats_Handler(event, context)
    return handler.handle()

def lh_file_uploadedcgats_to_json(event, context):
    from src.handlersFiles import File_UploadedToJson_Handler
    handler = File_UploadedToJson_Handler(event, context)
    return handler.handle()
    # -- USED: MARSAPI --

def lh_file_uploaded_interpolate_target(event, context):
    from src.handlersPredict import InterpolateTarget_Handler
    handler = InterpolateTarget_Handler(event, context)
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
    
def lh_delta_intersection(event, context):
    from src.handlersDelta import IntersectionDelta_Handler
    handler = IntersectionDelta_Handler(event, context)
    return handler.handle()
    # -- USED: MARSAPI --
    

# - - - BLENDING - - -

def lh_blend_spectral(event, context):
    from src.handlersPredict import BlendSpectral_Handler
    handler = BlendSpectral_Handler(event, context)
    return handler.handle()
    # -- USED: MARSAPI --
    

# - - - PROFILE - - -

def lh_profile_printer(event, context):
    from src.handlersProfile import Profile_Handler
    handler = Profile_Handler(event, context)
    return handler.handle()
    # -- USED: MARSAPI -- NEW ! UNDER DEVELOPMENT