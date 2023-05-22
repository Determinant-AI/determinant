from botocore.exceptions import ClientError
from ray import serve
from event import Event
import os
import boto3


@serve.deployment(num_replicas=1)
class EventHandler:
    def __init__(self, **kwargs):
        pass

    def handle(self, event: Event):
        raise NotImplementedError(
            "handle method not implemented on {self.__class__.__name__}")


class SQSPublisher(EventHandler):
    def __init__(self, aws_region: str = "us-east-1", queue_name="determinant-alpha"):
        # Create an SQS client
        self.sqs = boto3.client('sqs',
                                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                                region_name=aws_region)
        self.queue_url = self.sqs.get_queue_url(
            QueueName=queue_name)['QueueUrl']

    def get_secret(self, secret_name: str = "sqs-producer-ci-user", region_name: str = "us-west-2"):
        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        try:
            get_secret_value_response = client.get_secret_value(
                SecretId=secret_name
            )
        except ClientError as e:
            # For a list of exceptions thrown, see
            # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
            raise e

        # Decrypts secret using the associated KMS key.
        secret = get_secret_value_response['SecretString']
        return secret

    def send_message(self, message: str):
        """
        Sample response:
        {'MD5OfMessageBody': 'fd087ec3a396840105c2fefe76670061',
        'MessageId': 'd4720e13-dda7-4708-9505-01365bcf0bf7',
        'ResponseMetadata': {'RequestId': '9171264b-b4ab-53f1-8640-7aae5b45e24f',
        'HTTPStatusCode': 200,
        'HTTPHeaders': {'x-amzn-requestid': '9171264b-b4ab-53f1-8640-7aae5b45e24f',
        'date': 'Mon, 22 May 2023 22:33:18 GMT',
        'content-type': 'application/x-amz-json-1.0',
        'content-length': '106'},
        'RetryAttempts': 0}}
        """
        # Send message to SQS queue
        response = self.sqs.send_message(
            QueueUrl=self.queue_url,
            DelaySeconds=10,
            # https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-message-metadata.html#sqs-message-attributes
            MessageAttributes={
            },
            MessageBody=(
                message
            )
        )
        return response
