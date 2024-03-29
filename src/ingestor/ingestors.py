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

    def send_message(self):
        pass
