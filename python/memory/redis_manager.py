import redis
from redis.exceptions import RedisError

import subprocess
import time
import logging
from typing import Optional, List

# Configure the logger.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RedisManager:
    def __init__(self, host:str ='localhost', port:int =6379, *args, **kwargs):
        if 'redis_conn' in kwargs:
            self.redis_conn = kwargs['redis_conn']
            logger.info("A Redis server is already running on host:{}, port:{}."
                        .format(host, port))
        else:
            self.redis_conn = self.start_redis_server(host, port)

    # Define a function to start the Redis server as a child process.
    def start_redis_server(self, host, port) -> Optional[redis.Redis]:
        # Check if Redis is already running by attempting to establish a connection.
        try:
            redis_conn = redis.Redis()
            # If we can successfully ping the Redis server, it means that it is running.
            if redis_conn.ping():
                logger.info("A Redis server is already running.")
                return redis.Redis(host=host, port=port)
        except redis.exceptions.ConnectionError as e:
            # ConnectionError indicates that a Redis server is not running.
            logger.error(f"Redis server is not connected: {e}")
            pass

        try:
            # Start a new Redis server as a child process.
            redis_process = subprocess.Popen(['redis-server'], 
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Redis server started successfully.")
        except Exception as e:
            logger.error(f"Failed to start Redis server: {e}")
            return None
        
        return redis.Redis(host=host, port=port)
    
    def insert_list_strings(self, list_key: str, strings_to_insert: List[str]) -> bool:
        try:
            # Use * to unpack the list of strings as separate arguments to rpush
            return self.redis_conn.rpush(list_key, *strings_to_insert) > 0
        except redis.exceptions.RedisError as e:
            logger.error(f"Failed to insert list of strings: {e}")
            return False
                        
    def get_list_strings(self, list_key: str) -> Optional[List[str]]:
        try:
            list_contents = self.redis_conn.lrange(list_key, 0, -1)
            return [item.decode() for item in list_contents]
        except redis.exceptions.RedisError as e:
            logger.error(f"Failed to get list contents: {e}")
            return []
      
    def show_all_keys_and_values(self) -> None:
        try:
            # Get a list of all keys in Redis (use pattern '*' to match all keys)
            keys = self.redis_conn.keys('*')
            # Iterate over each key and retrieve its value based on its data type
            for key in keys:
                key_str = key.decode()
                data_type = self.redis_conn.type(key_str).decode()
                if data_type == 'string':
                    value = self.redis_conn.get(key_str)
                    value = value.decode() if isinstance(value, bytes) else value
                elif data_type == 'list':
                    value = self.redis_conn.lrange(key_str, 0, -1)
                    value = [item.decode() for item in value]
                elif data_type == 'hash':
                    value = self.redis_conn.hgetall(key_str)
                    value = {k.decode(): v.decode() for k, v in value.items()}
                # Handle additional data types (e.g., 'set', 'zset', 'stream') here if needed
                else:
                    value = None
                # Print the key-value pair
                print(f"{key_str}: {value}")
        except redis.exceptions.RedisError as e:
            print(f"Failed to show all keys and values: {e}")

    def shutdown(self) -> None:
        """
        Gracefully shut down the Redis server.
        """
        try:
            self.redis_conn.shutdown()
            print("Redis server has been shut down gracefully.")
        except Exception as e:
            print(f"Failed to shut down Redis server: {e}")