import sys
import logging
from logging.handlers import RotatingFileHandler

def setup_environment(log_file="trading.log"):
    """
    Configures the environment for the Trading AI application.
    Forces UTF-8 encoding for stdout on Windows to support emojis and sets up standard logging.
    """
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, Exception):
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

