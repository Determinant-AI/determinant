import boto3


class Ingestor(object):
    def __init__(self):
        pass

# KnowledgeIngestor ingests knowledge from relatively "static" sources e.g. confluence, wiki, etc.
# On demand or on a schedule in the background.


class KnowledgeIngestor(Ingestor):
    def __init__(self):
        pass


# ChatIngestor ingests the chat messages from Slack, Teams, etc, on real time
class MessageIngestor(Ingestor):
    def __init__(self):
        pass

    def ingest(self):
        pass


class SQSIngestor(MessageIngestor):
    def __init__(self, aws_region: str):

        # Create an SQS client
        self.sqs = boto3.client('sqs', region_name=aws_region)
        self.queue_url = self.sqs.get_queue_url(QueueName='test')['QueueUrl']

    def ingest(self, message: str):
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


def ingest(self):
    pass
