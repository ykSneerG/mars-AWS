import boto3    # type: ignore

class Botox:
    def __init__(self, bucket_name):
        self.client = boto3.client("s3")
        self.resource = boto3.resource("s3")
        self.bucket_name = bucket_name
    
    def store_S3(self, object_name, object_data, object_key):

        try:
            #object_key = f"data/{object_name}.json"
            self.client.put_object(
                Bucket=self.bucket_name, 
                Key=object_key, 
                Body=object_data
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