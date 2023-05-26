import sqlite3
import datetime
import random

# Generate sample workplace discussion data
conversations = [
    {
        'timestamp': datetime.datetime(2023, 5, 25, 9, 0),
        'conversation_id': 1,
        'thread_ts': 'T1234567890',
        'handle': 'user1',
        'message': "Good morning, everyone! I hope you all had a great weekend. Just wanted to check in and see if everyone is ready for the upcoming project meeting."
    },
    {
        'timestamp': datetime.datetime(2023, 5, 25, 9, 5),
        'conversation_id': 1,
        'thread_ts': 'T1234567890',
        'handle': 'user2',
        'message': "Good morning! Yes, I'm prepared for the meeting. Looking forward to discussing the project progress."
    },
    {
        'timestamp': datetime.datetime(2023, 5, 25, 9, 10),
        'conversation_id': 1,
        'thread_ts': 'T1234567890',
        'handle': 'user3',
        'message': "Morning, everyone. I have a few questions regarding the project timeline. Can we discuss that during the meeting?"
    },
    {
        'timestamp': datetime.datetime(2023, 5, 25, 9, 15),
        'conversation_id': 1,
        'thread_ts': 'T1234567890',
        'handle': 'bot',
        'message': "Good morning! Sure, we can definitely address the project timeline during the meeting. I will provide an update on the timeline and any adjustments required."
    },
    {
        'timestamp': datetime.datetime(2023, 5, 25, 10, 0),
        'conversation_id': 2,
        'thread_ts': 'T0987654321',
        'handle': 'user4',
        'message': "Hello everyone, any updates on the project? We have an important deadline coming up, so it's crucial to stay on track."
    },
    {
        'timestamp': datetime.datetime(2023, 5, 25, 10, 5),
        'conversation_id': 2,
        'thread_ts': 'T0987654321',
        'handle': 'user1',
        'message': "Hi! I have made some progress on my assigned tasks. I will share the updates shortly and address any concerns during the meeting."
    },
    {
        'timestamp': datetime.datetime(2023, 5, 25, 10, 10),
        'conversation_id': 2,
        'thread_ts': 'T0987654321',
        'handle': 'user3',
        'message': "Great! I'm glad to hear that. Let's make sure to discuss the timeline and any adjustments required to meet the deadline."
    },
    {
        'timestamp': datetime.datetime(2023, 5, 25, 10, 15),
        'conversation_id': 2,
        'thread_ts': 'T0987654321',
        'handle': 'bot',
        'message': "Absolutely! I will go through the timeline and provide necessary updates during the meeting to ensure we meet the deadline successfully."
    },
    {
        'timestamp': datetime.datetime(2023, 5, 25, 10, 20),
        'conversation_id': 2,
        'thread_ts': 'T0987654321',
        'handle': 'user2',
        'message': "I agree. It's crucial to ensure that we allocate enough resources and time to meet the deadline successfully."
    },
    {
        'timestamp': datetime.datetime(2023, 5, 25, 10, 25),
        'conversation_id': 2,
        'thread_ts': 'T0987654321',
        'handle': 'bot',
        'message': "Absolutely! Resource allocation and effective time management are key to meeting project deadlines. Let's discuss this further in the meeting."
    },
]


# Connect to the database
conn = sqlite3.connect('conversation_history.db')
c = conn.cursor()

# Create the conversation_history table
c.execute('''CREATE TABLE IF NOT EXISTS conversation_history
                (timestamp TEXT, message TEXT, conversation_id INTEGER, thread_ts TEXT, handle TEXT)''')


for conversation in conversations:
    timestamp = conversation['timestamp']
    conversation_id = conversation['conversation_id']
    thread_ts = conversation['thread_ts']
    handle = conversation['handle']
    message = conversation['message']
    c.execute('''INSERT INTO conversation_history (timestamp, conversation_id, thread_ts, handle, message)
              VALUES (?, ?, ?, ?, ?)''', (timestamp, conversation_id, thread_ts, handle, message))

    
# Execute the query to fetch conversations with running length and length
c.execute('''SELECT ch.conversation_id, ch.timestamp, ch.handle, ch.message,
                    SUM(LENGTH(ch.message)) OVER (PARTITION BY ch.conversation_id ORDER BY ch.timestamp DESC) AS running_length,
                    LENGTH(ch.message) AS message_length
             FROM conversation_history AS ch
             ORDER BY ch.conversation_id, ch.timestamp DESC''')

# Fetch the conversations with running length and length
rows = c.fetchall()
threshold = 500

# Process the conversations with running length and length
prev_conversation_id = None
for row in rows:
    conversation_id = row[0]
    timestamp = row[1]
    handle = row[2]
    message = row[3]
    running_length = row[4]
    message_length = row[5]

    # Check if the conversation ID changes
    if conversation_id != prev_conversation_id:
        print(f"Conversation ID: {conversation_id}")

    # Check if the running length is smaller than the threshold
    if running_length < threshold:
        print(f"    Timestamp: {timestamp}, Handle: {handle}, Message: {message}, Running Length: {running_length}, Message Length: {message_length}")

    # Update the previous conversation ID
    prev_conversation_id = conversation_id

# Conversation ID: 1
#     Timestamp: 2023-05-25 09:15:00, Handle: bot, Message: Good morning! Sure, we can definitely address the project timeline during the meeting. I will provide an update on the timeline and any adjustments required., Running Length: 157, Message Length: 157
#     Timestamp: 2023-05-25 09:10:00, Handle: user3, Message: Morning, everyone. I have a few questions regarding the project timeline. Can we discuss that during the meeting?, Running Length: 270, Message Length: 113
#     Timestamp: 2023-05-25 09:05:00, Handle: user2, Message: Good morning! Yes, I'm prepared for the meeting. Looking forward to discussing the project progress., Running Length: 370, Message Length: 100
# Conversation ID: 2
#     Timestamp: 2023-05-25 10:25:00, Handle: bot, Message: Absolutely! Resource allocation and effective time management are key to meeting project deadlines. Let's discuss this further in the meeting., Running Length: 142, Message Length: 142
#     Timestamp: 2023-05-25 10:20:00, Handle: user2, Message: I agree. It's crucial to ensure that we allocate enough resources and time to meet the deadline successfully., Running Length: 251, Message Length: 109
#     Timestamp: 2023-05-25 10:15:00, Handle: bot, Message: Absolutely! I will go through the timeline and provide necessary updates during the meeting to ensure we meet the deadline successfully., Running Length: 387, Message Length: 136
