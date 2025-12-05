# base/star_logger.py
import logging
import os
from datetime import datetime

# Default verbose setting (will be updated after CONFIG is loaded)
verbose = True

# Create logs directory if it doesn't exist
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename with timestamp
LOG_FILE = os.path.join(LOG_DIR, f"space_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger("space_state.dynamics")
if not logger.handlers:
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with more detailed format
    file_handler = logging.FileHandler(LOG_FILE)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

logger.setLevel(logging.DEBUG if verbose else logging.INFO)
level_name = "DEBUG" if verbose else "INFO"

logger.debug(f"Logging level set to {level_name}")
logger.info(f"Log file: {LOG_FILE}")
