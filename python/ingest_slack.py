
"""
ingestor class:
- slack channel -> topics -> relevant
- slack emojis -> "expert validator"
- slack threads between human -> "expert demonstration" -> exclude bot

storage class:
- GCS or Gdrive?
- gs://channel=slack_channel/date=2023-04-04/...
"""
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

import slack
from logger import create_logger
from slack_sdk.errors import SlackApiError

log = create_logger(__name__)

chann_test = 'C0526HLKN3G'
chann_general = 'C04SL2P0X5M'
chann_core = 'C04TD7RCG8H'
DEFAULT_CHANNEL_LIST = {"test": chann_test,
                        "general": chann_general,
                        "core": chann_core}

# Note: each slack channel has another type of bot for notifications e.g. user is added.
determinant_bot = 'B04UWAWQVQC'
chatbot = 'B04UT1HM21H'


class SlackIngestor:
    def __init__(self, token: str, channel_name_list: List[str], folder: str = None, bots: List[str] = None):
        self.client = slack.WebClient(token=token)
        self.channels = channel_name_list
        self.bots = bots
        self.folder = folder

    def fetch_message_history(self, channel_id: str, limit: int = 10):
        try:
            result = self.client.conversations_history(
                channel=channel_id, limit=limit)
            return result['messages']
        except SlackApiError as e:
            log.error(f"Error fetching messages from {channel_id}: {e}")
            return []

    def chat_report_finished(self, channel_id: str, text: str = "finished!"):
        self.client.chat_postMessage(
            channel=channel_id, text=text, as_user=True)

    def normalize_message(self, message: str):
        """
        normalize the messages to make it digestable

        e.g. exclude bot's messages, images; process emojis
        """
        pass

    def group_messages_by_date(self, messages: list, chann_name: str):
        grouped_messages = defaultdict(list)
        for message in messages:
            # Unix epoch time format with sub-second precision e.g., '1649200000.000200'
            timestamp = float(message['ts'])
            dt = datetime.fromtimestamp(timestamp)
            key = chann_name, dt.strftime('%Y-%m-%d')
            grouped_messages[key].append(message)
        return grouped_messages

    def ingest_slack_messages(self):
        """Ingests Slack messages into a database.

        Returns:
            None
        """
        # For each channel, get a list of all the messages in the channel.
        for chann_name in self.channels:
            chann_id = DEFAULT_CHANNEL_LIST[chann_name]
            messages = self.fetch_message_history(chann_id)
            grouped_msg = self.group_messages_by_date(messages, chann_name)
            zip_folder = self.write_messages(grouped_msg)
            # self.upload_to_gcs(local_dir, "")
            self.chat_report_finished(channel_id=chann_id)

    def write_messages(self, grouped_msg: dict):
        """Inserts a Slack message into the local file.

        Args:
            message (slack.Message): The Slack message.

        Returns:
            None
        """
        # Insert the message ID.
        os.makedirs(self.folder, exist_ok=True)
        for key, messages in grouped_msg.items():
            channel, date = key
            sub_folder = os.path.join(self.folder, channel)
            Path(sub_folder).mkdir(parents=True, exist_ok=True)
            file_path = os.path.join(sub_folder, f"{date}.json")
            with open(file_path, 'a+') as f:
                json.dump(messages, f, indent=2)
        log.info(
            f"wrote the grouped messages into local directory: {self.folder}")

        zip_path = shutil.make_archive(self.folder, 'zip', self.folder)
        log.info(
            f"zipped the messages into local file: {zip_path}")
        return zip_path

    def upload_to_gcs(self, local_source: str, destination: str):
        pass


if __name__ == "__main__":
    # Get the Slack API token.
    token = os.environ["SLACK_BOT_TOKEN"]
    ingestor = SlackIngestor(token, channel_name_list=[
                             "general"], folder="./test-slack-ingest/")

    # Ingest Slack messages.
    ingestor.ingest_slack_messages()
