# star_constants.py

from skyfield.api import PlanetaryConstants, load

from star_logger import logger
from star_utils import convert_gm, km_to_au, mean_radius_km
from star_config import DEFAULT_CONFIG

epoch_jd = DEFAULT_CONFIG.start_time.tt.jd
ts = load.timescale() 

EPH = load('./ephemeris/de440.bsp') # Auto download
pc = PlanetaryConstants()
pc.read_text(load('./ephemeris/gm_de440.tpc')) # https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc
pc.read_text(load('./ephemeris/pck00011.tpc')) # https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc
c = pc.variables

BODIES = {
    'sun': {
        "skyfield": EPH['sun'],
        "mu": convert_gm(c['BODY10_GM']),
        "radius": None
    },
    'earth': {
        "skyfield": EPH['earth'],
        "mu": convert_gm(c['BODY399_GM']),
        "radius": km_to_au(mean_radius_km(c['BODY399_RADII']))
    },
    'moon': {
        "skyfield": EPH['moon'],
        "mu": convert_gm(c['BODY301_GM']),
        "radius": km_to_au(mean_radius_km(c['BODY301_RADII']))
    },
    'mars barycenter': {
        "skyfield": EPH['mars barycenter'],
        "mu": convert_gm(c['BODY4_GM']),
        "radius": None
    },
    'jupiter barycenter': {
        "skyfield": EPH['jupiter barycenter'],
        "mu": convert_gm(c['BODY5_GM']),
        "radius": None
    },
    'venus barycenter': {
        "skyfield": EPH['venus barycenter'],
        "mu": convert_gm(c['BODY2_GM']),
        "radius": None
    },
    'saturn barycenter': {
        "skyfield": EPH['saturn barycenter'],
        "mu": convert_gm(c['BODY6_GM']),
        "radius": None
    }
}

# ------------------- TEST OUTPUT -------------------

def log_config() -> None:
    logger.info("star_constants.py - Constants:")
    for body_name, body_data in BODIES.items():
        mu = body_data["mu"]
        radius = body_data["radius"]
        logger.info(f"  BODIES.{body_name}.mu: {mu:.6e} AU^3/day^2")
        if radius is not None:
            logger.info(f"  BODIES.{body_name}.radius: {radius:.6e} AU")

log_config()