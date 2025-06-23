from src.code.icctools.IccV4_Helper import Helper
from src.code.icctools.IccV4_Elements import DataColourSpaceSignatures as dcss
from src.handlers_base import BaseLambdaHandler


# Specify the S3 bucket for all uploads
BUCKET_NAME = 'clrtsplt-uploads'


class CurveLinkHandler(BaseLambdaHandler):
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
    
    def handle(self):
        
        from src.code.Devicelink import Devicelink_OutputCurves
        
        jd = {
            'curves': self.event.get('curves', None),
            'fileID': Helper.random_id(4, 5, '-')
        }
                        
        colormode = dcss.get_dict_from_num(len(jd['curves']))
        jd['colormode'] = colormode["sig"].strip()
        
        inclBytes = self.event.get('includeBytes', False)
        link_name = self.event.get('link_name', f"{jd['colormode']}_CRVDL_{jd['fileID']}")
        
        params = {
            "name": link_name,
            "input_type": colormode,
            "output_type": colormode,
            "output_table": jd['curves']
        }
        dl = Devicelink_OutputCurves(params)
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


class SwapLinkHandler(BaseLambdaHandler):
    ''' 
    # Sample request Body with integers
    {
        "orderOld": [1, 2, 3, 4],
        "orderNew": [4, 1, 2, 3]
    }
    
    # Sample request Body with strings --- NOT SURE IF THIS WORKS
    {
        "orderOld": ["cyan", "color2", "my3", "last"],
        "orderNew": ["last", "cyan", "color2", "my3"]
    }
    '''
        
    def handle(self):
        
        from src.code.Devicelink import Devicelink_Redirect
        
        jd = {
            'orderOld': self.event.get('orderOld', None),
            'orderNew': self.event.get('orderNew', None),
            'fileID': Helper.random_id(4, 5, '-')
        }

        check = Helper.can_resort_to_equal(jd['orderOld'], jd['orderNew'])
        err = self.check_not_none(check, 'Input lists are not the same or not unique')
        if err: return err

        colormode = dcss.get_dict_from_num(len(jd['orderNew']))                    
        signature: str = colormode["sig"].strip()
        
        inclBytes = self.event.get('includeBytes', False)
        link_name = self.event.get('link_name', f"{signature}_SWAP_{jd['orderNew']}")
                    
        params = {
            "name": link_name,
            "input_type": colormode,
            "output_type": colormode,
            "channel_order": jd['orderNew']
        }            
        dl = Devicelink_Redirect(params)
        dl.create()
        

        metadata = {
            "type": "SwapLink",
            "colormodeA": colormode,
            "colormodeB": colormode,
            "fileID": jd['fileID'],
            "orderNew": jd['orderNew'],
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
            'elapsed': self.get_elapsed_time(),
            'filesize': dl.filesize(),
            'orderUse': jd['orderNew'],
            'colormode': signature
        })
        
        return self.get_common_response(jd)


class SpyLinkHandler(BaseLambdaHandler):
    
    def handle(self):
        
        from src.code.Devicelink import Devicelink_Dcsdata
        from src.code.icctools.SpyImage import SpyImage
        from src.code.icctools.botoX import BucketMan as bm

        jd = {        
            'imageID': self.event.get('fileID', None),
            'name': self.event.get('name', None)
        }
        jd['fileID'] = jd["name"] or jd["imageID"]  
        
        image_bytes = bm.get_tif_from_s3(BUCKET_NAME, f'{jd["imageID"]}.tif')
        err = self.check_not_none(image_bytes, 'Error retrieving image from S3')
        if err: return err
        
        sp = SpyImage(image_bytes)
        
        # It is unclear how to detect the colormode_A ???? TBD
        colormode_A: dcss = dcss.CMYK
        colormode_B: dcss = dcss.get_dict_from_num(sp.channels)
        
        err = self.check_not_none(colormode_B, 'Unsupported number of channels')
        if err: return err
        
        inclBytes = self.event.get('includeBytes', False)
        link_name = self.event.get('link_name', jd['fileID'])

        params = {
            "name": link_name,
            "input_type": colormode_A.value,
            "output_type": colormode_B,
            "gridpoints": sp.gridpoints
        }                        
        dl = Devicelink_Dcsdata(params)
        dl.create()
        
        
        metadata = {
            "type": "SpyLink",
            "colormodeA": colormode_A.value,
            "colormodeB": colormode_B,
            "fileID": jd['fileID'],
            "elapsed": self.get_elapsed_time(),
            "gridpoints": len(sp.gridpoints),
            "filesize": dl.filesize()
        }
        self.add_metadata(metadata)
                
        
        if inclBytes:
            jd['bytes'] = dl.result()
        else:
            err = self.store_S3(f"{jd['fileID']}-CRVDL.icc", dl.result())
            if err: return err
        
        jd.update({
            'input_type': colormode_A.value["sig"].strip(),
            'output_type': colormode_B["sig"].strip(),    
            'elapsed': self.get_elapsed_time(),
            'filesize': dl.filesize()
        })
        return self.get_common_response(jd)
