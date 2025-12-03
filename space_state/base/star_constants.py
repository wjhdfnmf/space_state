# base/star_constants.py

from skyfield.api import PlanetaryConstants, load
from astropy.time import Time

from .star_logger import logger
from .star_utils import convert_gm, km_to_au, mean_radius_km
from .star_config import CONFIG

epoch_jd = CONFIG.start_time.tt.jd
ts = load.timescale()

EPH = load('./ephemeris/de440.bsp') # Auto download
pc = PlanetaryConstants()
pc.read_text(load('./ephemeris/gm_de440.tpc')) # https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc
pc.read_text(load('./ephemeris/pck00011.tpc')) # https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc
c = pc.variables

# ------------------- CONSTANTS -------------------
g0 = 9.80665  # Standard gravity in m/s²

# Gravitational parameters and radii for celestial bodies
BODIES = {
    'sun': {
        "name": "Sun",
        "skyfield": EPH['sun'],
        "mu": convert_gm(c['BODY10_GM']),
        "mu_km3_s2": c['BODY10_GM'],
        "radius": None
    },
    'earth': {
        "name": "Earth",
        "skyfield": EPH['earth'],
        "mu": convert_gm(c['BODY399_GM']),
        "mu_km3_s2": c['BODY399_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY399_RADII']))
    },
    'moon': {
        "name": "Moon",
        "skyfield": EPH['moon'],
        "mu": convert_gm(c['BODY301_GM']),
        "mu_km3_s2": c['BODY301_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY301_RADII']))
    },
    'mars barycenter': {
        "name": "Mars Barycenter",
        "skyfield": EPH['mars barycenter'],
        "mu": convert_gm(c['BODY4_GM']),
        "mu_km3_s2": None,
        "radius": None
    },
    'jupiter barycenter': {
        "name": "Jupiter Barycenter",
        "skyfield": EPH['jupiter barycenter'],
        "mu": convert_gm(c['BODY5_GM']),
        "mu_km3_s2": None,
        "radius": None
    },
    'venus barycenter': {
        "name": "Venus Barycenter",
        "skyfield": EPH['venus barycenter'],
        "mu": convert_gm(c['BODY2_GM']),
        "mu_km3_s2": None,
        "radius": None
    },
    'saturn barycenter': {
        "name": "Saturn Barycenter",
        "skyfield": EPH['saturn barycenter'],
        "mu": convert_gm(c['BODY6_GM']),
        "mu_km3_s2": None,
        "radius": None
    }
}

# ------------------- TIME UTILITIES -------------------
def world_time(time: float) -> Time:
    """Convert simulation time (days since epoch) to a Time object."""
    return ts.tt_jd(epoch_jd + time)

# ------------------- TEST OUTPUT -------------------

def log_config() -> None:
    logger.info("star_constants.py - Constants:")
    for body_name, body_data in BODIES.items():
        mu = body_data["mu"]
        mu_k = body_data["mu_km3_s2"]
        radius = body_data["radius"]
        logger.info(f"  BODIES.{body_name}.mu: {mu:.6e} AU^3/day^2")
        if mu_k is not None:
            logger.info(f"  BODIES.{body_name}.mu_km3_s2: {mu_k:.6e} km^3/s^2")
        if radius is not None:
            logger.info(f"  BODIES.{body_name}.radius: {radius:.6e} AU")
    for orbit_name, altitude_km in ORBITS.items():
        logger.info(f"  ORBITS.{orbit_name}: {altitude_km:.1f} km")

log_config()