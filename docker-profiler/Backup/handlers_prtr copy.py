from src.code.icctools.IccV4_Helper import Helper
from src.code.icctools.IccV4_Elements import DataColourSpaceSignatures as dcss
from src.handlers_base import BaseLambdaHandler


# Specify the S3 bucket for all uploads
BUCKET_NAME = 'clrtsplt-uploads'


class PrtrHandler(BaseLambdaHandler):
    ''' 
    # Sample request Body
    {
        "curves": [
            [0, 12000, 65534],
            [65534, 32768, 0],
            [0, 32768, 65534],
            [0, 32768, 65534]
        ],
        "link_name": name,
        "includeBytes": True
    }
    '''
    
    @staticmethod
    def include_params(event, params, tag: str):
        # Remove "atob0_input_table" if it is None
        entry = event.get(tag, None)
        if entry is not None:
            params[tag] = entry

    
    def handle(self):
        
        from src.code.Printerlink import Profile_Printer
        
        jd = {
            'curves': self.event.get('curves', None),
            'fileID': Helper.random_id(4, 5, '-')
        }
        
        channel_count = 4
                        
        colormode = dcss.get_dict_from_num(channel_count)
        jd['colormode'] = colormode["sig"].strip()
        
        inclBytes = self.event.get('includeBytes', False)
        link_name = self.event.get('link_name', f"{jd['colormode']}_CRVDL_{jd['fileID']}")
        
        """ params = {
            "name": link_name,
            "input_type": colormode,
            "output_type": colormode,
            "output_table": jd['curves'],
            
            
            "wtpt": self.event.get('wtpt', None),
            "bktp": self.event.get('bktp', None),
            "atob0_clut": self.event.get('atob0_clut', None),
            
            "atob1_clut": self.event.get('atob1_clut', None),
            "atob1_input_table": self.event.get('atob1_input_table', None),
            "atob1_output_table": self.event.get('atob1_output_table', None),

            "atob2_clut": self.event.get('atob2_clut', None),

            "btoa0_clut": self.event.get('btoa0_clut', None),
            
            "btoa1_clut": self.event.get('btoa1_clut', None),
            "btoa1_input_table": self.event.get('btoa1_input_table', None),
            "btoa1_output_table": self.event.get('btoa1_output_table', None),
            
            "btoa2_clut": self.event.get('btoa2_clut', None)
        } """
        """ "output_table": jd['curves'], """
        params = {
            "name": link_name,
            "input_type": colormode,
            "output_type": colormode,
            
            "wtpt": self.event.get('wtpt', None),
            "bktp": self.event.get('bktp', None),
            
            "atob0_clut": self.event.get('atob0_clut', None),
            #"atob0_input_table": self.event.get('atob0_input_table', None),
            #"atob0_output_table": self.event.get('atob0_output_table', None),

            "atob1_clut": self.event.get('atob1_clut', None),
            #"atob1_input_table": self.event.get('atob1_input_table', None),
            #"atob1_output_table": self.event.get('atob1_output_table', None),

            "atob2_clut": self.event.get('atob2_clut', None),
            #"atob2_input_table": self.event.get('atob2_input_table', None),
            #"atob2_output_table": self.event.get('atob2_output_table', None),

            "btoa0_clut": self.event.get('btoa0_clut', None),
            #"btoa0_input_table": self.event.get('btoa0_input_table', None),
            #"btoa0_output_table": self.event.get('btoa0_output_table', None),
            
            "btoa1_clut": self.event.get('btoa1_clut', None),
            #"btoa1_input_table": self.event.get('btoa1_input_table', None),
            #"btoa1_output_table": self.event.get('btoa1_output_table', None),
            
            "btoa2_clut": self.event.get('btoa2_clut', None),
            #"btoa2_input_table": self.event.get('btoa2_input_table', None),
            #"btoa2_output_table": self.event.get('btoa2_output_table', None)
        }
        self.include_params(self.event, params, 'atob0_input_table')
        self.include_params(self.event, params, 'atob1_input_table')
        self.include_params(self.event, params, 'atob2_input_table')
        
        self.include_params(self.event, params, 'btoa0_input_table')
        self.include_params(self.event, params, 'btoa1_input_table')
        self.include_params(self.event, params, 'btoa2_input_table')
        
        self.include_params(self.event, params, 'atob0_output_table')
        self.include_params(self.event, params, 'atob1_output_table')
        self.include_params(self.event, params, 'atob2_output_table')
        
        self.include_params(self.event, params, 'btoa0_output_table')
        self.include_params(self.event, params, 'btoa1_output_table')
        self.include_params(self.event, params, 'btoa2_output_table')
        
        #print(f"Params: {params}")

        dl = Profile_Printer(params)
        dl.create()

        metadata = {
            "type": "CurveLink",
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
            err = self.store_S3(f"{jd['fileID']}-CRVDL.icc", dl.result())
            if err: return err
        
        jd.update({
            'filesize': dl.filesize(),
            'elapsed': self.get_elapsed_time(),
            'metadata': self.get_metadata()
        })
        
        return self.get_common_response(jd)
