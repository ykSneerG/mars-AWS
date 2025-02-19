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
