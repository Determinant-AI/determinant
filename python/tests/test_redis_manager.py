import pytest
import fakeredis
from typing import List, Optional
from memory.redis_manager import RedisManager
from redis.exceptions import RedisError

# Create a fixture that sets up and tears down a mock Redis server
@pytest.fixture
def redis_manager():
    fake_redis = fakeredis.FakeStrictRedis()
    manager = RedisManager(redis_conn=fake_redis)
    yield manager
    fake_redis.flushall()

def test_insert_list_strings_success(redis_manager):
    # Mock successful rpush operation
    # redis_manager.redis_conn.rpush.return_value = 2
    # Test the method
    assert redis_manager.insert_list_strings('fruits', ['apple', 'banana']) == 2

def test_insert_list_strings_failure(redis_manager):
    # Mock RedisError
    # redis_manager.redis_conn.rpush.side_effect = RedisError('Failed to insert list')
    # Test the method
    assert not redis_manager.insert_list_strings('fruits', ['apple', 'banana'])

def test_get_list_strings_success(redis_manager):
    # Mock successful lrange operation
    redis_manager.redis_conn.lrange.return_value = [b'apple', b'banana']
    # Test the method
    assert redis_manager.get_list_strings('fruits') == ['apple', 'banana']

def test_get_list_strings_failure(redis_manager):
    # Mock RedisError
    redis_manager.redis_conn.lrange.side_effect = RedisError('Failed to get list contents')
    # Test the method
    assert redis_manager.get_list_strings('fruits') == []

