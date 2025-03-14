import boto3    # type: ignore
import json

class Botox:
    def __init__(self, bucket_name):
        self.client = boto3.client("s3")
        self.resource = boto3.resource("s3")
        self.bucket_name = bucket_name
        
        self.metadata = {}
        
    def update_metadata(self, key, value):
        # Add x-amz-meta- prefix if not already present
        if not key.startswith('x-amz-meta-'):
            key = f'x-amz-meta-{key}'
            
        self.metadata.update({
            key: value
        })
        
    
    def store_S3(self, object_name, object_data, object_key):

        # Metadata must be a dictionary, not a JSON string
        #object_metadata = metadata if metadata else None
        """ Example metadata format:
            metadata={
                'x-amz-meta-FileContentHash': metadata.get('x-amz-meta-FileContentHash', '1234567890'),
            }
        """

        try:
            #object_key = f"data/{object_name}.json"
            self.client.put_object(
                Bucket=self.bucket_name, 
                Key=object_key, 
                Body=object_data,
                # Custom metadata keys must be prefixed with 'x-amz-meta-'
                # Pass metadata as a dict, not JSON string
                Metadata=self.metadata if self.metadata != {} else None
            )
            
            return {
                "upload_status": "success", 
                "bytes": len(object_data),
                "UPID": object_name
            }
            
        except Exception as e:
            return {
                "upload_status": "failed", 
                "error": str(e)
            }

    def hash_exists(self, hash_value):
        # Check if hash-value exists in S3 bucket and subfolders
        # check if hash-value exists in metadata
        # if not, return False
        # if yes, return True
        try:
            # get all objects in bucket
            objects = self.client.list_objects_v2(Bucket=self.bucket_name)
            for obj in objects.get('Contents', []):
                # Get the object's metadata
                obj_metadata = self.client.head_object(Bucket=self.bucket_name, Key=obj['Key'])
                if obj_metadata['Metadata'].get('x-amz-meta-filecontenthash') == hash_value:
                    # return object key or filename
                    found_key = obj['Key']
                    return found_key.split('/')[1].split('.')[0]
            return False
        except Exception as e:
            return {
                "error": str(e)
            }

    def load_S3(self, object_key):
        try:
            obj = self.client.get_object(Bucket=self.bucket_name, Key=object_key)
            return obj['Body'].read().decode('utf-8')
        except Exception as e:
            return {
                "error": str(e)
            }

    # @staticmethod
    # def invoke_aws(client, payload):
    #     """Invokes an AWS Lambda function and handles possible errors."""
    #     try:
    #         # lambda_client = boto3.client('lambda')

    #         response = client.invoke(
    #             FunctionName="mars-colorpy-predict-interpolate-pairs",
    #             InvocationType="RequestResponse",  # Wait for response
    #             Payload=orjson.dumps(payload),
    #         )

    #         # Read the payload response
    #         response_payload = orjson.loads(response["Payload"].read())

    #         # Check if Lambda returned an error
    #         if "errorMessage" in response_payload:
    #             raise RuntimeError(f"Lambda error: {response_payload['errorMessage']}")

    #         return response_payload

    #     except Exception as e:
    #         print(f"Error invoking Lambda: {e}")
    #         return {"error": str(e)}