
"""
ingestor class:
- slack channel -> topics -> relevant
- slack emojis -> "expert validator"
- slack threads between human -> "expert demonstration" -> exclude bot

storage class:
- GCS
- gs://channel=slack_channel/date=2023-04-04/...
"""
import os
import json
import shutil
from collections import defaultdict
from datetime import datetime
from typing import List

import slack
from logger import create_logger
from slack_sdk.errors import SlackApiError

log = create_logger(__name__)


class SlackIngestor:
    def __init__(self, channel_list: List[str], token: str, bots: List[str]):
        # Create a Slack client.
        self.client = slack.WebClient(token=token)
        self.channels = channel_list
        self.bots = bots

    def fetch_messages(self, channel_id: str):
        try:
            result = self.client.conversations_history(channel=channel_id)
            return result['messages']
        except SlackApiError as e:
            log.error(f"Error fetching messages from {channel_id}: {e}")
            return []

    def normalize_message(self, message: str):
        """
        normalize the messages to make it digestable

        e.g. exclude bot's messages, images; process emojis
        """
        pass

    def group_messages_by_date(self, messages: list, channel_id: str):
        grouped_messages = defaultdict(list)
        for message in messages:
            # Unix epoch time format with sub-second precision e.g., '1649200000.000200'
            timestamp = float(message['ts'])
            dt = datetime.fromtimestamp(timestamp)
            key = channel_id, dt.strftime('%Y-%m-%d')
            grouped_messages[key].append(message)
        return grouped_messages

    def ingest_slack_messages(self):
        """Ingests Slack messages into a database.

        Returns:
            None
        """
        channels = self.client.channels_list()

        # For each channel, get a list of all the messages in the channel.
        for channel in channels:
            messages = self.fetch_messages(channel)
            grouped_msg = self.group_messages_by_date(messages, channel)
            local_dir = self.write_messages(grouped_msg)
            self.upload_to_gcs(local_dir, "")

    def write_messages(self, grouped_msg: dict, output_dir: str):
        """Inserts a Slack message into the local file.

        Args:
            message (slack.Message): The Slack message.

        Returns:
            None
        """
        # Insert the message ID.
        os.makedirs(output_dir, exist_ok=True)
        for key, messages in grouped_msg.items():
            channel, date = key
            output_file = os.path.join(output_dir, channel, f"{date}.json")
            with open(output_file, 'w') as f:
                json.dump(messages, f, indent=2)

        # TODO: check correctness
        zip_path = shutil.make_archive(output_dir, 'zip', output_dir)
        return zip_path

    def upload_to_gcs(self, local_source: str, destination: str):
        pass


if __name__ == "__main__":
    # Get the Slack API token.
    token = os.environ["SLACK_API_TOKEN"]
    channel_list = ["core"]
    bots = ["determinant"]

    ingestor = SlackIngestor(channel_list, token, bots)

    # Ingest Slack messages.
    ingestor.ingest_slack_messages()
