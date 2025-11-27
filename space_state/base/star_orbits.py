# base/star_constants.py
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Orbit:
    """Named orbit with altitude (km) and optional description."""
    altitude_km: float
    description: str = ""

# ———————————————————— EARTH ORBITS ————————————————————
EARTH_ORBITS = {
    # Low Earth Orbits
    'leo_200':      Orbit(200,   "Lowest sustainable orbit (e.g. early Starlink)"),
    'leo_iss':      Orbit(400,   "International Space Station / Starship LEO"),
    'leo_550':      Orbit(550,   "Starlink operational shell"),
    
    # Medium Earth Orbits
    'meo_gps':      Orbit(20200, "GPS constellation"),
    
    # Geosynchronous & Geostationary
    'gto_apo':      Orbit(35786, "Typical GTO apogee"),
    'geo':          Orbit(35786, "Geostationary Earth Orbit (0° inclination)"),
    'gso':          Orbit(35786, "Geosynchronous (any inclination)"),
    
    # Special & Historical
    'tli_apo':      Orbit(384400 - 1737, "Typical Trans-Lunar Injection apoapsis (~100 km lunar orbit)"),
    'molniya_peri': Orbit(500,   "Molniya perigee"),
    'molniya_apo':  Orbit(40000, "Molniya apogee"),
    
    # Escape & Transfers
    'c3_zero':      Orbit(float('inf'), "Earth escape (C3 = 0)"),
    'moon_transfer':Orbit(384400 - 100, "Lunar free-return or low-energy transfer apoapsis"),
}

# ———————————————————— MOON ORBITS ————————————————————
MOON_ORBITS = {
    'lunar_100km':     Orbit(100,    "Standard low lunar orbit (Apollo, Artemis)"),
    'lunar_50km':      Orbit(50,     "Very low lunar orbit"),
    'lunar_15km':      Orbit(15,     "Lunar Gateway NRHO perigee (approx)"),
    'nrho_55000km':    Orbit(55000,  "NRHO apolune distance from Moon"),
    'dro':             Orbit(60000,  "Distant Retrograde Orbit apolune"),
    'lunar_frozen':    Orbit(1737,   "Theoretical frozen orbit altitude"),
}

# Combine for convenience
ALL_ORBITS = EARTH_ORBITS | MOON_ORBITS