# base/star_constants.py

from pathlib import Path
from skyfield.api import PlanetaryConstants, load
from astropy.time import Time

from .star_logger import logger
from .star_utils import convert_gm, km_to_au, mean_radius_km
from .star_config import CONFIG

# Ephemeris path relative to this file (space_state/base/../.. = space_state parent dir)
_EPH_DIR = Path(__file__).parent.parent.parent / 'ephemeris'

epoch_jd = CONFIG.start_time.tt.jd
ts = load.timescale()

EPH = load(str(_EPH_DIR / 'de440.bsp'))  # JPL DE440 ephemeris
pc = PlanetaryConstants()
pc.read_text(load(str(_EPH_DIR / 'gm_de440.tpc')))  # https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc
pc.read_text(load(str(_EPH_DIR / 'pck00011.tpc')))  # https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc
c = pc.variables

# ------------------- CONSTANTS -------------------
g0 = 9.80665  # Standard gravity in m/s²

# Unit conversions
AU_TO_KM = 1.495978707e8    # Astronomical unit in km
DAY_TO_SEC = 86400.0        # Day in seconds
KM_TO_AU = 1.0 / AU_TO_KM   # km to AU

# Gravitational parameters and radii for celestial bodies
# Body codes: Sun=10, Mercury=199, Venus=299, Earth=399, Moon=301, Mars=499
# Jupiter=599, Saturn=699, Uranus=799, Neptune=899, Pluto=999
# Barycenters: 1-9 (include planet + moons mass)
BODIES = {
    # ==================== STAR ====================
    'sun': {
        "name": "Sun",
        "skyfield": EPH['sun'],
        "mu": convert_gm(c['BODY10_GM']),
        "mu_km3_s2": c['BODY10_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY10_RADII']))
    },
    # ==================== INNER PLANETS ====================
    'mercury': {
        "name": "Mercury",
        "skyfield": EPH['mercury'],  # Code 199
        "mu": convert_gm(c['BODY199_GM']),
        "mu_km3_s2": c['BODY199_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY199_RADII']))
    },
    'mercury barycenter': {
        "name": "Mercury Barycenter",
        "skyfield": EPH['mercury barycenter'],  # Code 1
        "mu": convert_gm(c['BODY1_GM']),
        "mu_km3_s2": c['BODY1_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY199_RADII']))
    },
    'venus': {
        "name": "Venus",
        "skyfield": EPH['venus'],  # Code 299
        "mu": convert_gm(c['BODY299_GM']),
        "mu_km3_s2": c['BODY299_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY299_RADII']))
    },
    'venus barycenter': {
        "name": "Venus Barycenter",
        "skyfield": EPH['venus barycenter'],  # Code 2
        "mu": convert_gm(c['BODY2_GM']),
        "mu_km3_s2": c['BODY2_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY299_RADII']))
    },
    'earth': {
        "name": "Earth",
        "skyfield": EPH['earth'],  # Code 399
        "mu": convert_gm(c['BODY399_GM']),
        "mu_km3_s2": c['BODY399_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY399_RADII']))
    },
    'moon': {
        "name": "Moon",
        "skyfield": EPH['moon'],  # Code 301
        "mu": convert_gm(c['BODY301_GM']),
        "mu_km3_s2": c['BODY301_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY301_RADII']))
    },
    # ==================== MARS SYSTEM ====================
    'mars': {
        "name": "Mars",
        "skyfield": EPH['mars barycenter'],  # DE440 Mars at barycenter
        "mu": convert_gm(c['BODY499_GM']),  # Mars planet GM (not including moons)
        "mu_km3_s2": c['BODY499_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY499_RADII']))
    },
    'mars barycenter': {
        "name": "Mars Barycenter",
        "skyfield": EPH['mars barycenter'],  # Code 4
        "mu": convert_gm(c['BODY4_GM']),  # Includes Phobos + Deimos
        "mu_km3_s2": c['BODY4_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY499_RADII']))
    },
    'phobos': {
        "name": "Phobos",
        "skyfield": EPH['mars barycenter'],
        "mu": convert_gm(c['BODY401_GM']),
        "mu_km3_s2": c['BODY401_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY401_RADII'])),
        "sma_km": 9376.0
    },
    'deimos': {
        "name": "Deimos",
        "skyfield": EPH['mars barycenter'],
        "mu": convert_gm(c['BODY402_GM']),
        "mu_km3_s2": c['BODY402_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY402_RADII'])),
        "sma_km": 23463.0
    },
    # ==================== JUPITER SYSTEM ====================
    'jupiter': {
        "name": "Jupiter",
        "skyfield": EPH['jupiter barycenter'],
        "mu": convert_gm(c['BODY599_GM']),
        "mu_km3_s2": c['BODY599_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY599_RADII']))
    },
    'jupiter barycenter': {
        "name": "Jupiter Barycenter",
        "skyfield": EPH['jupiter barycenter'],  # Code 5
        "mu": convert_gm(c['BODY5_GM']),
        "mu_km3_s2": c['BODY5_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY599_RADII']))
    },
    'io': {
        "name": "Io",
        "skyfield": EPH['jupiter barycenter'],
        "mu": convert_gm(c['BODY501_GM']),
        "mu_km3_s2": c['BODY501_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY501_RADII'])),
        "sma_km": 421700.0
    },
    'europa': {
        "name": "Europa",
        "skyfield": EPH['jupiter barycenter'],
        "mu": convert_gm(c['BODY502_GM']),
        "mu_km3_s2": c['BODY502_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY502_RADII'])),
        "sma_km": 671100.0
    },
    'ganymede': {
        "name": "Ganymede",
        "skyfield": EPH['jupiter barycenter'],
        "mu": convert_gm(c['BODY503_GM']),
        "mu_km3_s2": c['BODY503_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY503_RADII'])),
        "sma_km": 1070400.0
    },
    'callisto': {
        "name": "Callisto",
        "skyfield": EPH['jupiter barycenter'],
        "mu": convert_gm(c['BODY504_GM']),
        "mu_km3_s2": c['BODY504_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY504_RADII'])),
        "sma_km": 1882700.0
    },
    # ==================== SATURN SYSTEM ====================
    'saturn': {
        "name": "Saturn",
        "skyfield": EPH['saturn barycenter'],
        "mu": convert_gm(c['BODY699_GM']),
        "mu_km3_s2": c['BODY699_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY699_RADII']))
    },
    'saturn barycenter': {
        "name": "Saturn Barycenter",
        "skyfield": EPH['saturn barycenter'],  # Code 6
        "mu": convert_gm(c['BODY6_GM']),
        "mu_km3_s2": c['BODY6_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY699_RADII']))
    },
    'titan': {
        "name": "Titan",
        "skyfield": EPH['saturn barycenter'],
        "mu": convert_gm(c['BODY606_GM']),
        "mu_km3_s2": c['BODY606_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY606_RADII'])),
        "sma_km": 1221870.0
    },
    'enceladus': {
        "name": "Enceladus",
        "skyfield": EPH['saturn barycenter'],
        "mu": convert_gm(c['BODY602_GM']),
        "mu_km3_s2": c['BODY602_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY602_RADII'])),
        "sma_km": 238020.0
    },
    # ==================== URANUS SYSTEM ====================
    'uranus': {
        "name": "Uranus",
        "skyfield": EPH['uranus barycenter'],
        "mu": convert_gm(c['BODY799_GM']),
        "mu_km3_s2": c['BODY799_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY799_RADII']))
    },
    'uranus barycenter': {
        "name": "Uranus Barycenter",
        "skyfield": EPH['uranus barycenter'],  # Code 7
        "mu": convert_gm(c['BODY7_GM']),
        "mu_km3_s2": c['BODY7_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY799_RADII']))
    },
    # ==================== NEPTUNE SYSTEM ====================
    'neptune': {
        "name": "Neptune",
        "skyfield": EPH['neptune barycenter'],
        "mu": convert_gm(c['BODY899_GM']),
        "mu_km3_s2": c['BODY899_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY899_RADII']))
    },
    'neptune barycenter': {
        "name": "Neptune Barycenter",
        "skyfield": EPH['neptune barycenter'],  # Code 8
        "mu": convert_gm(c['BODY8_GM']),
        "mu_km3_s2": c['BODY8_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY899_RADII']))
    },
    'triton': {
        "name": "Triton",
        "skyfield": EPH['neptune barycenter'],
        "mu": convert_gm(c['BODY801_GM']),
        "mu_km3_s2": c['BODY801_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY801_RADII'])),
        "sma_km": 354759.0
    },
    # ==================== DWARF PLANETS ====================
    'pluto': {
        "name": "Pluto",
        "skyfield": EPH['pluto barycenter'],
        "mu": convert_gm(c['BODY999_GM']),
        "mu_km3_s2": c['BODY999_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY999_RADII']))
    },
    'pluto barycenter': {
        "name": "Pluto Barycenter",
        "skyfield": EPH['pluto barycenter'],  # Code 9
        "mu": convert_gm(c['BODY9_GM']),
        "mu_km3_s2": c['BODY9_GM'],
        "radius": km_to_au(mean_radius_km(c['BODY999_RADII']))
    },
}

# ------------------- STATE VECTOR INDEXING -------------------
# State vector: [r_x, r_y, r_z, v_x, v_y, v_z, m, q0, q1, q2, q3, ωx, ωy, ωz]
IDX_R = slice(0, 3)       # Position [r_x, r_y, r_z] (AU)
IDX_V = slice(3, 6)       # Velocity [v_x, v_y, v_z] (AU/day)
IDX_M = 6                 # Total mass (kg)
IDX_Q = slice(7, 11)      # Quaternion [q0, q1, q2, q3] (scalar-first)
IDX_OMEGA = slice(11, 14) # Angular velocity [ωx, ωy, ωz] (rad/day)

# ------------------- ATTITUDE CONTROL HARDWARE -------------------
# Reaction Wheel System (4-wheel tetrahedral pyramid)
# Sized for Starship-class spacecraft (I ~ 500,000 kg·m²)
# Based on CMG-class actuators for large spacecraft
RW_MAX_MOMENTUM = 5000.0  # N·m·s per wheel (100x typical satellite RW)
RW_MAX_TORQUE = 500.0     # N·m per wheel (high-torque CMG-class)
RW_INERTIA = 50.0         # kg·m² wheel rotor inertia

# RCS Thruster System (16 thrusters in 4 quads)
RCS_THRUST = 1200.0       # N per thruster
RCS_ISP = 290.0           # seconds (hypergolic propellant)
RCS_MIN_PULSE = 0.04      # seconds (40ms minimum pulse width)

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

log_config()