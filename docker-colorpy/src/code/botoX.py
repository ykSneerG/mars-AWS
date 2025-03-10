import boto3 # type: ignore
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError # type: ignore

class BucketMan:
    def __init__(self):
        # self.s3 = boto3.client('s3')
        pass
 
    @staticmethod
    def object_exists(bucket_name, object_key) -> bool:

        s3 = boto3.client('s3')
        
        try:
            s3.head_object(Bucket=bucket_name, Key=object_key)
            return True
        except ClientError as e:
            return False

    @staticmethod
    def get_metadata(bucket_name, object_key):

        s3 = boto3.client('s3')

        try:
            return s3.head_object(Bucket=bucket_name, Key=object_key)
        except ClientError as e:
            return None
        
    @staticmethod
    def put_object(bucket_name, object_key, body, metadata=None) -> bool:

        s3 = boto3.client('s3')

        try:
            if metadata:
                s3.put_object(Bucket=bucket_name, Key=object_key, Body=body, Metadata=metadata)
            else:
                s3.put_object(Bucket=bucket_name, Key=object_key, Body=body)
            #s3.put_object(Bucket=bucket_name, Key=object_key, Body=body)
            return True
        except ClientError as e:
            return False
        
        

    # --- NEW SPYLINK --- CAN BE OPTIMZED ---

    @staticmethod
    def get_tif_from_s3(bucket_name, object_key) -> bytes:
        
        s3 = boto3.client('s3')
        
        # Get the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        # Read the object's content
        object_data = response['Body'].read()
        
        return object_data

    @staticmethod
    def upload_image_to_s3(bucket_name, object_key, my_data: bytearray):
        
        s3 = boto3.client('s3')
        
        # Convert bytearray to bytes
        byte_data = bytes(my_data)
        # Upload the image to S3
        s3.put_object(Bucket=bucket_name, Key=object_key, Body=byte_data)

    @staticmethod
    def get_object_from_s3(bucket_name, object_key) -> bytes:

        s3 = boto3.client('s3')

        # Get the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        # Read the object's content
        object_data = response['Body'].read()

        return object_data
    
    @staticmethod
    def get_txt_from_s3(bucket_name, object_key) -> str:

        s3 = boto3.client('s3')

        # Get the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        # Read the object's content
        object_data = response['Body'].read().decode('utf-8')

        return object_data
    
