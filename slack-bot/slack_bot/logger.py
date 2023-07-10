import logging
import sys

DEBUG = logging.DEBUG
INFO = logging.INFO


def create_logger(name, log_level=INFO, log_format=None):
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter(log_format)

    # Create console handler and set its log level and format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger
