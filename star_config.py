# star_config.py
from typing import NamedTuple
from astropy.time import Time
from star_logger import logger

SECONDS_PER_DAY = 86400
g0 = 9.80665  # Standard gravity in m/s²

# Altitude of common orbits
EARTH_LEO = 400  # km
EARTH_GSO = 35786  # km
MOON_ORBIT = 100  # km

class SimulationConfig(NamedTuple):
    rtol: float = 1e-9
    atol: float = 1e-9
    max_step: float = 60 / SECONDS_PER_DAY # days (60 seconds)
    verbose = True
    start_time = Time('2025-11-14 00:00:00', scale='utc')
    all_bodies = ["sun", "earth", "moon", "mars barycenter", "jupiter barycenter", "venus barycenter", "saturn barycenter"]
    earth_bodies = ["earth", "moon"]

# Default global config instance
DEFAULT_CONFIG = SimulationConfig()

# ------------------- LOG CONFIGURATION -------------------
def log_config() -> None:
    """Log the current simulation configuration."""
    logger.info("star_config.py - Configuration:")
    logger.info(f"  SECONDS_PER_DAY: {SECONDS_PER_DAY}")
    logger.info(f"  g0 (m/s²): {g0}")
    logger.info(f"  EARTH_LEO (km): {EARTH_LEO}")
    logger.info(f"  EARTH_GSO (km): {EARTH_GSO}")
    logger.info(f"  MOON_ORBIT (km): {MOON_ORBIT}")

    for field in DEFAULT_CONFIG._fields:
        value = getattr(DEFAULT_CONFIG, field)
        logger.info(f"  DEFAULT_CONFIG.{field}: {value}")
    
log_config()