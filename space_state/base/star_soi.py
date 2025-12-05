# base/star_soi.py
"""
Sphere of Influence (SOI) and Orbital Regime Detection

Determines which celestial body's gravity dominates for a spacecraft,
enabling proper reference frame selection and atmospheric drag application.

SOI Radius: r_SOI ≈ a * (m_planet / m_sun)^(2/5)
where a = semi-major axis of planet's orbit
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum

from .star_constants import BODIES, ts, epoch_jd
from .star_logger import logger


class OrbitalRegime(Enum):
    """Orbital regime classification."""
    HELIOCENTRIC = "heliocentric"  # Sun-dominated
    EARTH_ORBIT = "earth_orbit"     # Earth SOI
    LUNAR_ORBIT = "lunar_orbit"     # Moon SOI (within Earth SOI)
    MARS_ORBIT = "mars_orbit"
    VENUS_ORBIT = "venus_orbit"
    JUPITER_ORBIT = "jupiter_orbit"
    SATURN_ORBIT = "saturn_orbit"
    ATMOSPHERIC = "atmospheric"     # Within atmosphere of body


@dataclass
class SOIInfo:
    """Information about sphere of influence crossing."""
    body: str
    regime: OrbitalRegime
    distance_au: float           # Distance from body
    altitude_km: float           # Altitude above surface (if applicable)
    in_atmosphere: bool          # True if within atmosphere
    soi_radius_au: float         # SOI radius of current body
    relative_velocity_kmps: float  # Velocity relative to body


# SOI radii in AU (pre-computed for performance)
# r_SOI = a * (m_planet / m_sun)^(2/5)
SOI_RADII = {
    'earth': 0.00617,      # ~924,000 km
    'moon': 0.000434,      # ~65,000 km (from Earth, within Earth SOI)
    'mars barycenter': 0.00386,   # ~578,000 km
    'venus barycenter': 0.00410,  # ~614,000 km
    'jupiter barycenter': 0.322,  # ~48,200,000 km
    'saturn barycenter': 0.366,   # ~54,800,000 km
}

# Atmospheric interface altitudes (km)
# Below these altitudes, drag becomes significant
ATMOSPHERE_INTERFACE = {
    'earth': 120.0,    # Karman line is 100 km, we use 120 for margin
    'mars barycenter': 125.0,
    'venus barycenter': 250.0,  # Venus has thick atmosphere
    'moon': 0.0,       # No atmosphere
}


def compute_soi_radius(body_name: str) -> float:
    """
    Compute sphere of influence radius for a body.

    Returns:
        SOI radius in AU
    """
    if body_name in SOI_RADII:
        return SOI_RADII[body_name]

    # Compute from body parameters
    body = BODIES.get(body_name)
    if body is None:
        return 0.0

    mu_body = body.get('mu', 0.0)
    mu_sun = BODIES['sun']['mu']

    # Estimate semi-major axis from current position
    # For simplicity, use 1 AU for unspecified bodies
    a = 1.0

    # SOI radius
    r_soi = a * (mu_body / mu_sun) ** 0.4
    return r_soi


def get_body_position(body_name: str, time: float) -> np.ndarray:
    """Get body position at given time."""
    body = BODIES.get(body_name)
    if body is None:
        return np.zeros(3)

    skyfield_time = ts.tt_jd(epoch_jd + time)
    return body['skyfield'].at(skyfield_time).position.au


def get_body_velocity(body_name: str, time: float) -> np.ndarray:
    """Get body velocity at given time."""
    body = BODIES.get(body_name)
    if body is None:
        return np.zeros(3)

    skyfield_time = ts.tt_jd(epoch_jd + time)
    return body['skyfield'].at(skyfield_time).velocity.au_per_d


def determine_soi(
    r_sc: np.ndarray,
    time: float,
    bodies_to_check: List[str] = None,
) -> Tuple[str, OrbitalRegime, float]:
    """
    Determine which body's SOI the spacecraft is in.

    Args:
        r_sc: Spacecraft position (AU, SSB frame)
        time: Current time (days from epoch)
        bodies_to_check: List of body names to check (default: all)

    Returns:
        (body_name, regime, distance_au) - dominant body and distance from it
    """
    if bodies_to_check is None:
        bodies_to_check = ['earth', 'moon', 'mars barycenter', 'venus barycenter',
                          'jupiter barycenter', 'saturn barycenter']

    # Check each body's SOI
    for body_name in bodies_to_check:
        r_body = get_body_position(body_name, time)
        distance = np.linalg.norm(r_sc - r_body)

        soi_radius = SOI_RADII.get(body_name, 0.0)
        if soi_radius > 0 and distance < soi_radius:
            # Special case: Moon is within Earth's SOI
            if body_name == 'moon':
                return body_name, OrbitalRegime.LUNAR_ORBIT, distance
            elif body_name == 'earth':
                # Check if also in Moon's SOI
                r_moon = get_body_position('moon', time)
                d_moon = np.linalg.norm(r_sc - r_moon)
                if d_moon < SOI_RADII['moon']:
                    return 'moon', OrbitalRegime.LUNAR_ORBIT, d_moon
                return body_name, OrbitalRegime.EARTH_ORBIT, distance
            elif body_name == 'mars barycenter':
                return body_name, OrbitalRegime.MARS_ORBIT, distance
            elif body_name == 'venus barycenter':
                return body_name, OrbitalRegime.VENUS_ORBIT, distance
            elif body_name == 'jupiter barycenter':
                return body_name, OrbitalRegime.JUPITER_ORBIT, distance
            elif body_name == 'saturn barycenter':
                return body_name, OrbitalRegime.SATURN_ORBIT, distance

    # Not in any body's SOI - heliocentric
    r_sun = np.zeros(3)  # Sun at SSB origin (approximately)
    distance_sun = np.linalg.norm(r_sc - r_sun)
    return 'sun', OrbitalRegime.HELIOCENTRIC, distance_sun


def get_soi_info(
    r_sc: np.ndarray,
    v_sc: np.ndarray,
    time: float,
) -> SOIInfo:
    """
    Get detailed SOI information for spacecraft.

    Args:
        r_sc: Spacecraft position (AU, SSB frame)
        v_sc: Spacecraft velocity (AU/day, SSB frame)
        time: Current time (days from epoch)

    Returns:
        SOIInfo with full orbital regime details
    """
    body_name, regime, distance_au = determine_soi(r_sc, time)

    # Get body properties
    body = BODIES.get(body_name, {})
    body_radius_au = body.get('radius', 0.0)

    # Altitude above surface
    altitude_km = (distance_au - body_radius_au) * 1.496e8 if body_radius_au > 0 else -1

    # Check if in atmosphere
    atm_interface = ATMOSPHERE_INTERFACE.get(body_name, 0.0)
    in_atmosphere = altitude_km >= 0 and altitude_km < atm_interface

    # Relative velocity
    v_body = get_body_velocity(body_name, time)
    v_rel = v_sc - v_body
    v_rel_kmps = np.linalg.norm(v_rel) * 1731.46  # AU/day to km/s

    return SOIInfo(
        body=body_name,
        regime=regime,
        distance_au=distance_au,
        altitude_km=altitude_km,
        in_atmosphere=in_atmosphere,
        soi_radius_au=SOI_RADII.get(body_name, 0.0),
        relative_velocity_kmps=v_rel_kmps,
    )


# ------------------- INTEGRATION EVENTS -------------------

def create_soi_events(body_name: str, direction: int = 0):
    """
    Create event function for SOI crossing.

    Args:
        body_name: Body whose SOI to monitor
        direction: 0 = both, 1 = entry only, -1 = exit only

    Returns:
        Event function for solve_ivp
    """
    soi_radius = SOI_RADII.get(body_name, 0.0)

    def soi_event(t, y):
        r_sc = y[:3]
        r_body = get_body_position(body_name, t)
        distance = np.linalg.norm(r_sc - r_body)
        return distance - soi_radius

    soi_event.terminal = False
    soi_event.direction = direction

    return soi_event


def create_periapsis_event(body_name: str):
    """
    Create event function for periapsis passage.

    Periapsis occurs when d(distance)/dt = 0 and distance is minimum.
    """
    def periapsis_event(t, y):
        r_sc, v_sc = y[:3], y[3:6]
        r_body = get_body_position(body_name, t)
        v_body = get_body_velocity(body_name, t)

        r_rel = r_sc - r_body
        v_rel = v_sc - v_body

        # d(distance)/dt = (r_rel · v_rel) / |r_rel|
        return np.dot(r_rel, v_rel)

    periapsis_event.terminal = False
    periapsis_event.direction = 1  # Increasing = passing periapsis

    return periapsis_event


def create_apoapsis_event(body_name: str):
    """Create event function for apoapsis passage."""
    def apoapsis_event(t, y):
        r_sc, v_sc = y[:3], y[3:6]
        r_body = get_body_position(body_name, t)
        v_body = get_body_velocity(body_name, t)

        r_rel = r_sc - r_body
        v_rel = v_sc - v_body

        return np.dot(r_rel, v_rel)

    apoapsis_event.terminal = False
    apoapsis_event.direction = -1  # Decreasing = passing apoapsis

    return apoapsis_event


def create_atmosphere_entry_event(body_name: str):
    """Create event for atmospheric entry (crossing interface altitude)."""
    atm_interface = ATMOSPHERE_INTERFACE.get(body_name, 120.0)
    body = BODIES.get(body_name, {})
    body_radius_au = body.get('radius', 0.0)
    interface_radius_au = body_radius_au + atm_interface / 1.496e8

    def atm_event(t, y):
        r_sc = y[:3]
        r_body = get_body_position(body_name, t)
        distance = np.linalg.norm(r_sc - r_body)
        return distance - interface_radius_au

    atm_event.terminal = False
    atm_event.direction = -1  # Descending into atmosphere

    return atm_event


def create_altitude_event(body_name: str, altitude_km: float, direction: int = 0):
    """
    Create event for crossing specific altitude.

    Args:
        body_name: Central body
        altitude_km: Target altitude (km)
        direction: 0 = both, 1 = ascending, -1 = descending
    """
    body = BODIES.get(body_name, {})
    body_radius_au = body.get('radius', 0.0)
    target_radius_au = body_radius_au + altitude_km / 1.496e8

    def altitude_event(t, y):
        r_sc = y[:3]
        r_body = get_body_position(body_name, t)
        distance = np.linalg.norm(r_sc - r_body)
        return distance - target_radius_au

    altitude_event.terminal = False
    altitude_event.direction = direction

    return altitude_event


def create_impact_event(body_name: str):
    """Create terminal event for surface impact."""
    body = BODIES.get(body_name, {})
    body_radius_au = body.get('radius', 0.0)

    def impact_event(t, y):
        r_sc = y[:3]
        r_body = get_body_position(body_name, t)
        distance = np.linalg.norm(r_sc - r_body)
        return distance - body_radius_au

    impact_event.terminal = True  # Stop integration on impact
    impact_event.direction = -1   # Descending

    return impact_event
