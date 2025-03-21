from src.code.space.colorSample import SampleSpectral
from src.handlers import BaseLambdaHandler

class Space_SampleSpectral_Handler(BaseLambdaHandler):

    def handle(self):
        
        id = self.event.get("id", 0)
        # filepath = self.event.get("filepath", None)

        sam = SampleSpectral(None, id)
                
        jd = {
            "samples": sam.get_spectral(),
            "name": sam.get_name(),
            "description": sam.get_description(),
            "id": id,
            "elapsed": self.get_elapsed_time()
        }
        
        return self.get_common_response(jd)
