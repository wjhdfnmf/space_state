# base/star_config.py
from typing import NamedTuple
from astropy.time import Time

from .star_logger import logger

class SimulationConfig(NamedTuple):
    SECONDS_PER_DAY = 86400
    g0 = 9.80665  # Standard gravity in m/s²
    rtol: float = 1e-9
    atol: float = 1e-9
    max_step: float = 60 / SECONDS_PER_DAY # days (60 seconds)
    verbose = True
    start_time = Time('2025-11-14 00:00:00', scale='utc')
    all_bodies = ["sun", "earth", "moon", "mars barycenter", "jupiter barycenter", "venus barycenter", "saturn barycenter"]
    earth_bodies = ["earth", "moon"]

# Default global config instance
CONFIG = SimulationConfig()

# ------------------- LOG CONFIGURATION -------------------
def log_config() -> None:
    """Log the current simulation configuration."""
    logger.info("base.star_config.py - Configuration:")

    for field in CONFIG._fields:
        value = getattr(CONFIG, field)
        logger.info(f"  CONFIG.{field}: {value}")
    
log_config()