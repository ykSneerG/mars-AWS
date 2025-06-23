import time
import json, base64
from datetime import datetime


# Specify the S3 bucket for all uploads
BUCKET_NAME = 'clrtsplt-uploads'


class BaseLambdaHandler:
    """ Base class for Lambda handlers, providing common functionality."""
    
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
    
    def store_S3(self, filename, data):
        from src.code.icctools.botoX import BucketMan
        res = BucketMan.put_object(BUCKET_NAME, filename, data, self.get_metadata())
        err = self.check_upload(res)
        return err
        

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
            
            from src.code.icctools.botoX import BucketMan
            response = BucketMan.get_object_from_s3(BUCKET_NAME, object_key)
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
