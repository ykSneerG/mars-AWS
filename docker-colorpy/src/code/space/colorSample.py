import numpy as np  # type: ignore
import json

from src.code.space.colorConverterNumpy import ColorTrafoNumpy


class SampleBase:
    def __init__(self, filepath: str, id: int = 0):
        self.id = id
        self.dataset_path = filepath
        self.dataset = self.load(id)
        
    def load(self, id: int = 0):
        with open(self.dataset_path) as f:
            samples_base = json.load(f)
            return samples_base["samples"][id]
        
    def get_name(self):
        return self.dataset["name"]
        
    def get_description(self):
        return self.dataset["description"]



class SampleSpectral(SampleBase):
    
    def __init__(self, filepath: str, id: int = 0):
        DATAFILE = 'src/code/space/colorSamples/snm.json' if not filepath else filepath
        super().__init__(DATAFILE, id)

    def get_spectral(self):
        trafo = ColorTrafoNumpy()
        return trafo.Cs_SNM2MULTI(np.array(self.dataset['data']), {"HEX": True})


class SampleDevice4C(SampleBase):

    def __init__(self, filepath: str, id: int = 0):
        DATAFILE = 'src/code/space/colorSamples/dcs.json' if not filepath else filepath
        super().__init__(DATAFILE, id)

    def get_data(self):
        return self.dataset['data']
    
    def get_rowlength(self):
        return self.dataset['rowlength']
