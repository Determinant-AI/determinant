import sqlite3
import datetime
import random

class SQLMemoryManager:
    def __init__(self, db_file='conversation_history', strategy='token_limit'):
        self.conn = sqlite3.connect(db_file)
        self.c = self.conn.cursor()
        self.strategies_mapping = {
            'token_limit': self.get_conversations_token_limit,
            'time_period': self.get_conversations_time_period,
            'random': self.get_random_conversations
        }
        self.strategy = strategy

    def __del__(self):
        self.conn.close()

    def get_conversations_token_limit(self, token_limit):
        # Execute the query to fetch conversations with running length
        self.c.execute('''SELECT ch.conversation_id, ch.timestamp, ch.handle, ch.message,
                                SUM(LENGTH(ch.message)) OVER (PARTITION BY ch.conversation_id ORDER BY ch.timestamp DESC) AS running_length,
                                LENGTH(ch.message) AS message_length
                         FROM conversation_history AS ch
                         ORDER BY ch.conversation_id, ch.timestamp DESC''')

        # Fetch the conversations with running length and length
        rows = self.c.fetchall()

        # Process the conversations with running length and length
        prev_conversation_id = None
        prompt_parts = []
        for row in rows:
            conversation_id = row[0]
            timestamp = row[1]
            handle = row[2]
            message = row[3]
            running_length = row[4]
            message_length = row[5]

            # Check if the conversation ID changes
            if conversation_id != prev_conversation_id:
                prompt_parts.append(f"Conversation ID: {conversation_id}")

            # Check if the running length is smaller than the threshold
            if running_length < token_limit:
                prompt_parts.append(f"    Timestamp: {timestamp}, Handle: {handle}, Message: {message}, Running Length: {running_length}, Message Length: {message_length}")

            # Update the previous conversation ID
            prev_conversation_id = conversation_id

        # Concatenate the prompt parts into a single string
        prompt = "\n".join(prompt_parts)

        return prompt

    def get_conversations_time_period(self, time_period):
        current_timestamp = datetime.datetime.now()
        start_timestamp = current_timestamp - time_period

        # Execute the query to fetch conversations within the time period
        self.c.execute('''SELECT ch.conversation_id, ch.timestamp, ch.handle, ch.message
                         FROM conversation_history AS ch
                         WHERE ch.timestamp >= ?
                         ORDER BY ch.timestamp DESC''', (start_timestamp,))

        # Fetch the conversations within the time period
        rows = self.c.fetchall()

        # Process the conversations within the time period
        prompt_parts = []
        for row in rows:
            conversation_id = row[0]
            timestamp = row[1]
            handle = row[2]
            message = row[3]

            prompt_parts.append(f"Conversation ID: {conversation_id}, Timestamp: {timestamp}, Handle: {handle}, Message: {message}")

        # Concatenate the prompt parts into a single string
        prompt = "\n".join(prompt_parts)

        return prompt

    def get_random_conversations(self, num_conversations):
        # Execute the query to fetch all conversations
        self.c.execute('''SELECT ch.conversation_id, ch.timestamp, ch.handle, ch.message
                         FROM conversation_history AS ch
                         ORDER BY RANDOM()''')

        # Fetch the random conversations
        rows = self.c.fetchmany(num_conversations)

        # Process the random conversations
        prompt_parts = []
        for row in rows:
            conversation_id = row[0]
            timestamp = row[1]
            handle = row[2]
            message = row[3]

            prompt_parts.append(f"Conversation ID: {conversation_id}, Timestamp: {timestamp}, Handle: {handle}, Message: {message}")

        # Concatenate the prompt parts into a single string
        prompt = "\n".join(prompt_parts)

        return prompt

    def get_conversations(self, **kwargs):
        # Check if the strategy name is valid
        if self.strategy not in self.strategies_mapping:
            raise ValueError(f"Invalid strategy: {self.strategy}")

        # Get the corresponding strategy method based on the strategy name
        strategy_method = self.strategies[self.strategy]

        # Call the strategy method with the provided arguments
        return strategy_method(**kwargs)
