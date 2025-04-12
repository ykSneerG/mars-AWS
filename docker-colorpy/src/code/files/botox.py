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

    # def hash_exists(self, hash_value):
    #     # Check if hash-value exists in S3 bucket and subfolders
    #     # check if hash-value exists in metadata
    #     # if not, return False
    #     # if yes, return True
        
    #     res = "notFound"
        
    #     """ try: """
    #         # get all objects in bucket with pagination to avoid timeout
    #     paginator = self.client.get_paginator('list_objects_v2')
    #     for obj in paginator.paginate(Bucket=self.bucket_name).search('Contents'):
    #         # Get the object's metadata
    #         obj_metadata = self.client.head_object(Bucket=self.bucket_name, Key=obj['Key'])
    #         if obj_metadata['Metadata'].get('x-amz-meta-filecontenthash') == hash_value:
    #             # Extract filename from key
    #             return obj['Key'].split('/')[1].split('.')[0]
                
    #     return res
    #     """ except Exception as e:
    #         return {
    #             "error": str(e)
    #         } """
    
    # def hash_exists(self, hash_value):
    #     """Check if hash_value exists in S3 metadata; return filename if found, else 'notFound'."""

    #     paginator = self.client.get_paginator('list_objects_v2')

    #     for page in paginator.paginate(Bucket=self.bucket_name):
    #         for obj in page.get('Contents', []):
    #             if 'Key' in obj:
    #                 metadata = self.client.head_object(Bucket=self.bucket_name, Key=obj['Key']).get('Metadata', {})
                    
    #                 if metadata.get('x-amz-meta-filecontenthash') == hash_value:
    #                     return obj['Key'].rsplit('/', 1)[-1].split('.')[0]  # Extract filename
        
    #     return "notFound"
    
    def hash_exists(self, hash_value):
        """Check if a file with the given hash already exists in S3 metadata."""
        
        paginator = self.client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket_name):
            objects = page.get('Contents', [])
            
            # Collect object keys to minimize API calls
            object_keys = [obj['Key'] for obj in objects]

            # Fetch metadata for each object in batches (reduce API calls)
            for key in object_keys:
                try:
                    metadata = self.client.head_object(Bucket=self.bucket_name, Key=key).get('Metadata', {})
                    if metadata.get('x-amz-meta-filecontenthash') == hash_value:
                        return key.rsplit('/', 1)[-1].split('.')[0]  # Extract filename
                except self.client.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        continue  # Object not found (shouldn't happen)
                    raise  # Other errors should be handled properly

        return "notFound"

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