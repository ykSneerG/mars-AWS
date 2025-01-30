from datetime import datetime
#from src.code.botoX import BucketMan as bm
import time, json


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
    
    
    
    
# class CurveLinkHandler(BaseLambdaHandler):
#     def handle(self):
        
#         jd = {
#             'curves': self.event.get('curves', None),
#             'fileID': Helper.random_id(4, 5, '-')
#         }
                        
#         colormode = dcss.get_dict_from_num(len(jd['curves']))
#         jd['colormode'] = colormode["sig"].strip()
        
#         params = {
#             "name": f"{jd['colormode']}_CRVDL_{jd['fileID']}",
#             "input_type": colormode,
#             "output_type": colormode,
#             "output_table": jd['curves']
#         }
#         dl = Devicelink_OutputCurves(params)
#         dl.create()    
        

#         metadata = {
#             "type": "CurveLink",
#             "colormodeA": colormode,
#             "colormodeB": colormode,
#             "fileID": jd['fileID'],
#             "elapsed": self.get_elapsed_time(),
#             "filesize": dl.filesize()
#         }
#         self.add_metadata(metadata)
                
        
#         res = bm.put_object(BUCKET_NAME, f"{jd['fileID']}-CRVDL.icc", dl.result(), self.get_metadata())
#         err = self.check_upload(res)
#         if err: return err
        
#         jd.update({
#             'filesize': dl.filesize(),
#             'elapsed': self.get_elapsed_time(),
#             'metadata': self.get_metadata()
#         })
#         return self.get_common_response(jd)

# class DownloadHandler():
    
#     def __init__(self, event, context):
        
#         print(event)
        
#         # get fileID and typeID from query string
#         self.fileID = event.get('queryStringParameters', {}).get('fileID', None)
#         self.typeID = event.get('queryStringParameters', {}).get('type', None)
        
#     def handle(self):
#         try:
            
#             if not self.fileID or not self.typeID:
#                 return {
#                     'statusCode': 400,
#                     'body': json.dumps('Missing fileID or typeID')
#                 }
                        
#             object_key = f'{self.fileID}-{self.typeID}.icc'
            
#             response = bm.get_object_from_s3(BUCKET_NAME, object_key)
#             respdata = base64.b64encode(response).decode('utf-8')
            
#             return {
#                 'statusCode': 200,
#                 'body': respdata,
#                 'isBase64Encoded': True
#             }

#         except Exception as e:
#             return {
#                 'statusCode': 500,
#                 'body': json.dumps(f'Internal Server Error: {str(e)}')
#             }
