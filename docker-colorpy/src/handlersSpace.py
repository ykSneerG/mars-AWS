from src.code.space.colorSample import SampleSpectral
from src.handlers import BaseLambdaHandler

class Space_SampleSpectral_Handler(BaseLambdaHandler):

    def handle(self):

        sample_spectral = SampleSpectral()
        
        list_samples = sample_spectral.get_list()
        
        jd = {}
        jd.update({"samples": list_samples})
        jd.update({"elapsed": self.get_elapsed_time()})
        
        return self.get_common_response(jd)
