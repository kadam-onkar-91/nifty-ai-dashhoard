"""Central structured logging for the NIFTY AI Dashboard."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.getenv("NIFTY_AI_LOG_DIR", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(os.getenv("NIFTY_AI_LOG_LEVEL", "INFO").upper())
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = RotatingFileHandler(
        os.path.join(_LOG_DIR, "nifty_ai.log"), maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    if os.getenv("NIFTY_AI_CONSOLE_LOG", "0") == "1":
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return _build_logger(name)
