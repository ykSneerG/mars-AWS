from src.code.icctools.IccV4_Helper import Helper
from src.code.icctools.IccV4_Elements import DataColourSpaceSignatures as dcss
from src.handlers_base import BaseLambdaHandler
from src.code.Printerlink import Profile_Printer

# Specify the S3 bucket for all uploads
BUCKET_NAME = 'clrtsplt-uploads'


class PrtrHandler(BaseLambdaHandler):
    
    @staticmethod
    def include_params(event, params, tag: str):
        entry = event.get(tag, None)
        if entry is not None:
            params[tag] = entry

    
    def handle(self):
        
        jd = {
            'fileID': Helper.random_id(4, 5, '-'),
        }
        
        channel_count = 4
                        
        colormode = dcss.get_dict_from_num(channel_count)
        jd['colormode'] = colormode["sig"].strip()
        
        inclBytes = self.event.get('includeBytes', False)
        link_name = self.event.get('link_name', f"{jd['colormode']}_PRTR_{jd['fileID']}")
        
        params = {
            "name": link_name,
            "input_type": colormode,
            "output_type": colormode,
            
            'xyzs_src': self.event.get('xyzs_src', None),
            'dcss_src': self.event.get('dcss_src', None),
            'grid_size': self.event.get('grid_size', None),
        }
        
        dl = Profile_Printer(params)
        dl.create()

        jd.update({ "gcr_test": dl.gcr_result })
        

        metadata = {
            "type": "PrinterLink",
            "colormodeA": colormode,
            "colormodeB": colormode,
            "fileID": jd['fileID'],
            "elapsed": self.get_elapsed_time(),
            "filesize": dl.filesize()
        }
        self.add_metadata(metadata)
                
        if inclBytes:
            jd['bytes'] = dl.result()
        else:
            err = self.store_S3(f"{jd['fileID']}-PRTR.icc", dl.result())
            if err: return err
        
        jd.update({
            'filesize': dl.filesize(),
            'elapsed': self.get_elapsed_time(),
            'metadata': self.get_metadata()
        })
        
        return self.get_common_response(jd)
