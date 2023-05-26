import sqlite3

# Connect to the database
conn = sqlite3.connect('conversation_history.db')
c = conn.cursor()

# Create the conversation_history table
c.execute('''CREATE TABLE IF NOT EXISTS conversation_history
                (timestamp TEXT, user_message TEXT, bot_response TEXT, conversation_id INTEGER)''')

# Generate sample data
sample_data = [
    ('2023-05-25 10:00:00', 'Hello', 'Hi', 1),
    ('2023-05-25 10:05:00', 'How are you? I hope you are doing well.', 'I\'m fine, thank you!', 1),
    ('2023-05-25 10:10:00', 'Bye', 'Goodbye', 2),
    ('2023-05-25 10:15:00', 'Can you help me with some information about programming languages?', 'Of course! I have expertise in multiple programming languages.', 2),
    ('2023-05-25 10:20:00', 'Thank you so much for your assistance.', 'You\'re welcome! Feel free to ask if you have any more questions.', 2),
    ('2023-05-25 10:25:00', 'I have another query. How can I secure my database?', 'Securing a database involves various aspects such as access controls, encryption, and regular updates.', 3),
    ('2023-05-25 10:30:00', 'Alright, I will keep that in mind. Thank you!', 'You\'re welcome! Good luck with your database security.', 3),
    ('2023-05-25 10:35:00', 'Hi', 'Hello', 4),
    ('2023-05-25 10:40:00', 'What is the meaning of life?', 'The meaning of life is subjective and can vary for each individual.', 4)
]

# Insert sample data into the table
c.executemany("INSERT INTO conversation_history VALUES (?, ?, ?, ?)", sample_data)

# Define the threshold value
threshold = 100

# Execute the query to fetch conversations with running length and length
c.execute('''SELECT ch.conversation_id, ch.timestamp, ch.user_message, ch.bot_response,
                    SUM(LENGTH(ch.user_message)) OVER (PARTITION BY ch.conversation_id ORDER BY ch.timestamp DESC) AS running_length,
                    LENGTH(ch.user_message) AS message_length
             FROM conversation_history AS ch
             ORDER BY ch.conversation_id, ch.timestamp DESC''')

# Fetch the conversations with running length and length
rows = c.fetchall()

# Process the conversations with running length and length
prev_conversation_id = None
for row in rows:
    conversation_id = row[0]
    timestamp = row[1]
    user_message = row[2]
    bot_response = row[3]
    running_length = row[4]
    message_length = row[5]

    # Check if the conversation ID changes
    if conversation_id != prev_conversation_id:
        print(f"Conversation ID: {conversation_id}")

    # Check if the running length is smaller than the threshold
    if running_length < threshold:
        print(f"    Timestamp: {timestamp}, User Message: {user_message}, Bot Response: {bot_response}, Running Length: {running_length}, Message Length: {message_length}")

    # Update the previous conversation ID
    prev_conversation_id = conversation_id

# Close the connection
conn.close()

# output
# Conversation ID: 1
#     Timestamp: 2023-05-25 10:05:00, User Message: How are you? I hope you are doing well., Bot Response: I'm fine, thank you!, Running Length: 39, Message Length: 39
#     Timestamp: 2023-05-25 10:00:00, User Message: Hello, Bot Response: Hi, Running Length: 44, Message Length: 5
# Conversation ID: 2
#     Timestamp: 2023-05-25 10:20:00, User Message: Thank you so much for your assistance., Bot Response: You're welcome! Feel free to ask if you have any more questions., Running Length: 38, Message Length: 38
# Conversation ID: 3
#     Timestamp: 2023-05-25 10:30:00, User Message: Alright, I will keep that in mind. Thank you!, Bot Response: You're welcome! Good luck with your database security., Running Length: 45, Message Length: 45
#     Timestamp: 2023-05-25 10:25:00, User Message: I have another query. How can I secure my database?, Bot Response: Securing a database involves various aspects such as access controls, encryption, and regular updates., Running Length: 96, Message Length: 51
# Conversation ID: 4
#     Timestamp: 2023-05-25 10:40:00, User Message: What is the meaning of life?, Bot Response: The meaning of life is subjective and can vary for each individual., Running Length: 28, Message Length: 28
#     Timestamp: 2023-05-25 10:35:00, User Message: Hi, Bot Response: Hello, Running Length: 30, Message Length: 2

