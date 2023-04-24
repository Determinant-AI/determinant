import pytest
import fakeredis
from typing import List, Optional
from memory.redis_manager import RedisManager 

# Define a fixture to create a RedisManager instance with a fakeredis connection
@pytest.fixture
def redis_manager():
    fake_redis_conn = fakeredis.FakeRedis()
    return RedisManager(redis_conn=fake_redis_conn)

def test_insert_list_strings(redis_manager):
    # Define sample data
    list_key = 'test_list_key'
    strings_to_insert = ['apple', 'banana', 'cherry']

    # Call the insert_list_strings method and check the result
    assert redis_manager.insert_list_strings(list_key, strings_to_insert)

    # Retrieve the list from the fake Redis server and check its contents
    result = redis_manager.redis_conn.lrange(list_key, 0, -1)
    expected_result = [item.encode() for item in strings_to_insert]
    assert result == expected_result

def test_get_list_strings(redis_manager):
    # Define sample data
    list_key = 'test_list_key'
    list_contents = ['apple', 'banana', 'cherry']

    # Insert the sample data into the fake Redis server
    redis_manager.redis_conn.rpush(list_key, *list_contents)

    # Call the get_list_strings method and check the result
    result = redis_manager.get_list_strings(list_key)
    expected_result = list_contents
    assert result == expected_result

# You can add additional test functions for other methods of the RedisManager class

