from datetime import datetime
from src.code.botoX import BucketMan as bm
import time


# Specify the S3 bucket for all uploads
BUCKET_NAME = 'clrtsplt-uploads'


class BaseLambdaHandler:
    def __init__(self, event, context):
        self.event = event
        self.context = context
        self.start_time = time.time()
        
        # add metadata to the object
        self.meta = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_elapsed_time(self):
        return time.time() - self.start_time
    
    def get_elapsed_time_str(self):
        elapsedtime = self.get_elapsed_time()
        return f"{elapsedtime:.6f} sec."

    def get_common_response(self, body):
        return {
            'statusCode': 200,
            'body': body
        }

    def get_error_response(self, error_message):
        return {
            'statusCode': 400,
            'body': {
                'error': error_message
            }
        }
        
    def check_upload(self, res):
        if res is False: 
            return {
                'statusCode': 500,
                'body': 'Error uploading file.'
            }
            
    def check_not_none(self, value, message):
        if value is None:
            return {
                'statusCode': 400,
                'body': {
                    'error': f'{message}'
                }
            }
        
    
    # adding metadata --- 
    def add_metadata(self, meta):
        self.meta.update(meta)
        
    def get_metadata(self):
        return { "jptr": json.dumps(self.meta) }
    
class CurveLinkHandler(BaseLambdaHandler):
    def handle(self):
        
        jd = {
            'curves': self.event.get('curves', None),
            'fileID': Helper.random_id(4, 5, '-')
        }
                        
        colormode = dcss.get_dict_from_num(len(jd['curves']))
        jd['colormode'] = colormode["sig"].strip()
        
        params = {
            "name": f"{jd['colormode']}_CRVDL_{jd['fileID']}",
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
                
        
        res = bm.put_object(BUCKET_NAME, f"{jd['fileID']}-CRVDL.icc", dl.result(), self.get_metadata())
        err = self.check_upload(res)
        if err: return err
        
        jd.update({
            'filesize': dl.filesize(),
            'elapsed': self.get_elapsed_time(),
            'metadata': self.get_metadata()
        })
        return self.get_common_response(jd)

class SwapLinkHandler(BaseLambdaHandler):
    def handle(self):
        ''' 
        # Sample request Body with integers
        {
            "orderOld": [1, 2, 3, 4],
            "orderNew": [4, 1, 2, 3]
        }
        
        # Sample request Body with strings
        {
            "orderOld": ["cyan", "color2", "my3", "last"],
            "orderNew": ["last", "cyan", "color2", "my3"]
        }
        '''
        
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
                    
        params = {
            "name": f"{signature}_SWAP_{jd['orderNew']}",
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
          
        
        res = bm.put_object(BUCKET_NAME, f"{jd['fileID']}-SWPDL.icc", dl.result(), self.get_metadata())
        err = self.check_upload(res)
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

        params = {
            "name": jd['fileID'],
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
                
        
        res = bm.put_object(BUCKET_NAME, f'{jd['fileID']}-SPYDL.icc', dl.result(), self.get_metadata())
        err = self.check_upload(res)
        if err: return err
        
        jd.update({
            'input_type': colormode_A.value["sig"].strip(),
            'output_type': colormode_B["sig"].strip(),    
            'elapsed': self.get_elapsed_time(),
            'filesize': dl.filesize()
        })
        return self.get_common_response(jd)


import json, base64

class DownloadHandler():
    
    def __init__(self, event, context):
        
        print(event)
        
        # get fileID and typeID from query string
        self.fileID = event.get('queryStringParameters', {}).get('fileID', None)
        self.typeID = event.get('queryStringParameters', {}).get('type', None)
        
    def handle(self):
        try:
            
            if not self.fileID or not self.typeID:
                return {
                    'statusCode': 400,
                    'body': json.dumps('Missing fileID or typeID')
                }
                        
            object_key = f'{self.fileID}-{self.typeID}.icc'
            
            response = bm.get_object_from_s3(BUCKET_NAME, object_key)
            respdata = base64.b64encode(response).decode('utf-8')
            
            return {
                'statusCode': 200,
                'body': respdata,
                'isBase64Encoded': True
            }

        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps(f'Internal Server Error: {str(e)}')
            }
