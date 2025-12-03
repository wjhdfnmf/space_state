# base/star_config.py
from typing import NamedTuple
from astropy.time import Time

from .star_logger import logger
from .star_utils import sec_to_days


class SimulationConfig(NamedTuple):

    log_verbose: bool = True
    integrator: str = 'DOP853'  # 'RK45', 'RK23', 'DOP853', 'LSODA'
    progress_bar: bool = True
    rtol: float = 1e-9
    atol: float = 1e-9
    max_step: float = sec_to_days(60) # days (60 seconds)
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