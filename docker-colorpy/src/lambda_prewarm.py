import boto3 # type: ignore
import json
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config # type: ignore
from botocore.exceptions import BotoCoreError, ClientError # type: ignore

class LambdaPrewarmer:
    def __init__(self, function_name, amount=50):
        """
        :param function_name: The name of the Lambda function to prewarm
        :param amount: Number of concurrent invocations
        """
        self.function_name = function_name
        self.amount = amount

        # Optimized AWS Lambda client
        self.config = Config(
            connect_timeout=1,
            read_timeout=2,  # Slightly increased for reliability
            retries={"max_attempts": 2},
            max_pool_connections=50
        )
        self.lambda_client = boto3.client('lambda', config=self.config)

    def invoke_aws(self, payload):
        """Asynchronously invokes the AWS Lambda function for pre-warming."""
        try:
            self.lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType="Event",  # Asynchronous (no waiting)
                Payload=json.dumps(payload),
            )
            return {"status": "invoked"}
        except (BotoCoreError, ClientError) as e:
            print(f"Error prewarming {self.function_name}: {e}")
            return {"error": str(e)}

    def prewarm_lambda(self):
        """Pre-warms the Lambda function 50 times in parallel."""
        payloads = [{"warmup": True, "id": i} for i in range(self.amount)]  # Unique warm-up calls

        with ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(self.invoke_aws, payloads))

        return results

# Usage Example
""" prewarmer = LambdaPrewarmer(function_name="my_lambda_function", amount=50)  # 50 pre-warm calls
prewarmer.prewarm_lambda() """