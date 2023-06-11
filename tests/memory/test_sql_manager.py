import pytest
from datetime import datetime, timedelta
from memory.sqlite_memory_manager import SQLMemoryManager


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

@pytest.fixture(scope='module')
def manager():
    # Create an instance of SQLMemoryManager for testing
    manager = SQLMemoryManager(':memory:')

    # Insert sample data into the conversation_history table
    with patch('builtins.datetime', side_effect=lambda *args, **kwargs: datetime(*args, **kwargs)):
        for conversation in conversations:
            timestamp = conversation['timestamp']
            conversation_id = conversation['conversation_id']
            thread_ts = conversation['thread_ts']
            handle = conversation['handle']
            message = conversation['message']
            manager.c.execute('''INSERT INTO conversation_history (timestamp, conversation_id, thread_ts, handle, message)
                                  VALUES (?, ?, ?, ?, ?)''', (timestamp, conversation_id, thread_ts, handle, message))

    yield manager

    # Clean up the conversation_history table and close the connection
    manager.conn.close()

def test_get_conversations_token_limit(manager):
    token_limit = 200
    prompt = manager.get_conversations(token_limit=token_limit)

    expected_prompt = """Conversation ID: 1
    Timestamp: 2023-05-25 09:00:00, Handle: user1, Message: Good morning, everyone! I hope you all had a great weekend. Just wanted to check in and see if everyone is ready for the upcoming project meeting., Running Length: 0, Message Length: 101
    Timestamp: 2023-05-25 09:05:00, Handle: user2, Message: Good morning! Yes, I'm prepared for the meeting. Looking forward to discussing the project progress., Running Length: 61, Message Length: 93"""

    assert prompt == expected_prompt

def test_get_conversations_time_period(manager):
    time_period = timedelta(hours=1)
    prompt = manager.get_conversations(time_period=time_period)

    expected_prompt = """Conversation ID: 2, Timestamp: 2023-05-25 10:25:00, Handle: bot, Message: Absolutely! Resource allocation and effective time management are key to meeting project deadlines. Let's discuss this further in the meeting."""

    assert prompt == expected_prompt

def test_get_random_conversations(manager):
    num_conversations = 2
    prompt = manager.get_conversations(num_conversations=num_conversations)

    # Uncomment the following lines to print the prompt for visual inspection
    # print(prompt)
    # print()

    assert len(prompt.split('\n')) == num_conversations * 2


# Conversation ID: 1
#     Timestamp: 2023-05-25 09:15:00, Handle: bot, Message: Good morning! Sure, we can definitely address the project timeline during the meeting. I will provide an update on the timeline and any adjustments required., Running Length: 157, Message Length: 157
#     Timestamp: 2023-05-25 09:10:00, Handle: user3, Message: Morning, everyone. I have a few questions regarding the project timeline. Can we discuss that during the meeting?, Running Length: 270, Message Length: 113
#     Timestamp: 2023-05-25 09:05:00, Handle: user2, Message: Good morning! Yes, I'm prepared for the meeting. Looking forward to discussing the project progress., Running Length: 370, Message Length: 100
# Conversation ID: 2
#     Timestamp: 2023-05-25 10:25:00, Handle: bot, Message: Absolutely! Resource allocation and effective time management are key to meeting project deadlines. Let's discuss this further in the meeting., Running Length: 142, Message Length: 142
#     Timestamp: 2023-05-25 10:20:00, Handle: user2, Message: I agree. It's crucial to ensure that we allocate enough resources and time to meet the deadline successfully., Running Length: 251, Message Length: 109
#     Timestamp: 2023-05-25 10:15:00, Handle: bot, Message: Absolutely! I will go through the timeline and provide necessary updates during the meeting to ensure we meet the deadline successfully., Running Length: 387, Message Length: 136
