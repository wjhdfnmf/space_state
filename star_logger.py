# star_logger.py
import logging

verbose = True

logger = logging.getLogger("space_state.dynamics")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.DEBUG if verbose else logging.INFO)
level_name = "DEBUG" if verbose else "INFO"

logger.debug(f"Logging level set to {level_name}")
