import logging
from os import getenv

FILEPATH: str = "mov_be_llm.log"
FORMAT: str = "[{name}]-[%(levelname)s]-[%(asctime)s]-[%(message)s]"
TO_FILE: bool = getenv("LOG_TO_FILE", "False").lower() == "true"
TO_TERMINAL: bool = getenv("LOG_TO_TERMINAL", "True").lower() == "true"
FILE_LEVEL: int = logging.DEBUG
TERMINAL_LEVEL: int = logging.INFO


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(FORMAT.format(name=name))
    if TO_FILE:
        file_handler = logging.FileHandler(str(FILEPATH), mode="a")
        file_handler.setLevel(FILE_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if TO_TERMINAL:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(TERMINAL_LEVEL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
