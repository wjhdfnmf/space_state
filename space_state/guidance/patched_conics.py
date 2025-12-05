# guidance/patched_conics.py
"""
Patched Conics Trajectory Module

Implements the patched conics approximation for multi-body trajectory analysis.
This method treats the trajectory as a series of two-body problems, switching
reference frames at sphere of influence (SOI) boundaries.

Key Concepts:
- Inside planet SOI: Planet-centered two-body problem
- Outside planet SOI: Sun-centered two-body problem
- Transition: State vector transformation at SOI boundary

Reference:
- Battin, "An Introduction to the Mathematics and Methods of Astrodynamics"
- Vallado, "Fundamentals of Astrodynamics and Applications"
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto

from base.star_constants import BODIES
from base.star_logger import logger


class ReferenceFrame(Enum):
    """Reference frame for trajectory segment."""
    HELIOCENTRIC = auto()   # Sun-centered inertial
    PLANETOCENTRIC = auto() # Planet-centered rotating or inertial


@dataclass
class TrajectorySegment:
    """A single two-body trajectory segment."""
    frame: ReferenceFrame
    central_body: str           # Body name (e.g., 'sun', 'earth')
    r_initial: np.ndarray       # Initial position in frame
    v_initial: np.ndarray       # Initial velocity in frame
    t_start: float              # Start time
    t_end: float                # End time (or None if unbounded)
    mu: float                   # Gravitational parameter

    def __repr__(self) -> str:
        return (f"TrajectorySegment({self.frame.name}, body={self.central_body}, "
                f"t=[{self.t_start:.4f}, {self.t_end:.4f}])")


def sphere_of_influence_radius(a_planet: float, m_planet: float, m_sun: float) -> float:
    """
    Calculate sphere of influence radius using Laplace's formula.

    r_SOI = a * (m_planet / m_sun)^(2/5)

    Parameters
    ----------
    a_planet : float
        Semi-major axis of planet's orbit around Sun
    m_planet : float
        Planet mass (or GM)
    m_sun : float
        Sun mass (or GM)

    Returns
    -------
    float
        SOI radius in same units as a_planet
    """
    return a_planet * (m_planet / m_sun) ** 0.4


def get_soi_radius(body_name: str) -> float:
    """
    Get SOI radius for a body in AU.

    Parameters
    ----------
    body_name : str
        Body name (e.g., 'earth', 'mars')

    Returns
    -------
    float
        SOI radius in AU
    """
    if body_name == 'sun':
        return np.inf

    body = BODIES.get(body_name)
    if body is None:
        raise ValueError(f"Unknown body: {body_name}")

    sun = BODIES['sun']

    # Get approximate semi-major axis (distance from sun at epoch)
    # In a real implementation, this would come from orbital elements
    # For now, use typical values in AU
    sma_au = {
        'mercury': 0.387,
        'venus': 0.723,
        'earth': 1.000,
        'moon': 1.000,  # Orbits Earth which orbits Sun at 1 AU
        'mars': 1.524,
        'jupiter': 5.203,
        'saturn': 9.537,
        'uranus': 19.19,
        'neptune': 30.07,
        'pluto': 39.48,
    }

    a = sma_au.get(body_name.split()[0], 1.0)  # Handle 'mars barycenter' etc.
    mu_body = body['mu']
    mu_sun = sun['mu']

    return sphere_of_influence_radius(a, mu_body, mu_sun)


# Pre-computed SOI radii for major bodies (AU)
SOI_RADII_AU = {
    'mercury': 0.0007,      # ~112,000 km
    'venus': 0.0041,        # ~616,000 km
    'earth': 0.0062,        # ~929,000 km
    'moon': 0.00044,        # ~66,000 km (relative to Earth)
    'mars': 0.0039,         # ~577,000 km
    'jupiter': 0.322,       # ~48 million km
    'saturn': 0.365,        # ~55 million km
    'uranus': 0.346,        # ~52 million km
    'neptune': 0.580,       # ~87 million km
    'pluto': 0.022,         # ~3.3 million km
}


def heliocentric_to_planetocentric(
    r_helio: np.ndarray,
    v_helio: np.ndarray,
    r_planet: np.ndarray,
    v_planet: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform state from heliocentric to planet-centered frame.

    Parameters
    ----------
    r_helio : np.ndarray
        Spacecraft position in heliocentric frame
    v_helio : np.ndarray
        Spacecraft velocity in heliocentric frame
    r_planet : np.ndarray
        Planet position in heliocentric frame
    v_planet : np.ndarray
        Planet velocity in heliocentric frame

    Returns
    -------
    r_planet_centered : np.ndarray
        Position relative to planet
    v_planet_centered : np.ndarray
        Velocity relative to planet
    """
    r_planet_centered = np.asarray(r_helio) - np.asarray(r_planet)
    v_planet_centered = np.asarray(v_helio) - np.asarray(v_planet)
    return r_planet_centered, v_planet_centered


def planetocentric_to_heliocentric(
    r_planet_centered: np.ndarray,
    v_planet_centered: np.ndarray,
    r_planet: np.ndarray,
    v_planet: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform state from planet-centered to heliocentric frame.

    Parameters
    ----------
    r_planet_centered : np.ndarray
        Position relative to planet
    v_planet_centered : np.ndarray
        Velocity relative to planet
    r_planet : np.ndarray
        Planet position in heliocentric frame
    v_planet : np.ndarray
        Planet velocity in heliocentric frame

    Returns
    -------
    r_helio : np.ndarray
        Spacecraft position in heliocentric frame
    v_helio : np.ndarray
        Spacecraft velocity in heliocentric frame
    """
    r_helio = np.asarray(r_planet_centered) + np.asarray(r_planet)
    v_helio = np.asarray(v_planet_centered) + np.asarray(v_planet)
    return r_helio, v_helio


def check_soi_entry(
    r_helio: np.ndarray,
    r_planet: np.ndarray,
    body_name: str
) -> bool:
    """
    Check if spacecraft is inside a body's SOI.

    Parameters
    ----------
    r_helio : np.ndarray
        Spacecraft heliocentric position
    r_planet : np.ndarray
        Planet heliocentric position
    body_name : str
        Name of the body to check

    Returns
    -------
    bool
        True if inside SOI
    """
    r_rel = np.asarray(r_helio) - np.asarray(r_planet)
    distance = np.linalg.norm(r_rel)
    soi_radius = SOI_RADII_AU.get(body_name, 0)
    return distance < soi_radius


def find_soi_crossing(
    r_helio: np.ndarray,
    v_helio: np.ndarray,
    r_planet: np.ndarray,
    v_planet: np.ndarray,
    body_name: str,
    dt_max: float = 365.0,
    entry: bool = True
) -> Optional[float]:
    """
    Find time to SOI entry or exit using bisection.

    Parameters
    ----------
    r_helio : np.ndarray
        Spacecraft heliocentric position
    v_helio : np.ndarray
        Spacecraft velocity
    r_planet : np.ndarray
        Planet heliocentric position
    v_planet : np.ndarray
        Planet velocity
    body_name : str
        Body name
    dt_max : float
        Maximum time to search (days)
    entry : bool
        True to find entry, False for exit

    Returns
    -------
    float or None
        Time to crossing, or None if not found
    """
    soi_radius = SOI_RADII_AU.get(body_name, 0)
    if soi_radius == 0:
        return None

    # Simple linear propagation search
    # In production, use proper Kepler propagation
    dt = 0.001  # 1/1000 day step
    t = 0.0

    r_rel_prev = np.asarray(r_helio) - np.asarray(r_planet)
    inside_prev = np.linalg.norm(r_rel_prev) < soi_radius

    while t < dt_max:
        t += dt
        # Simple linear extrapolation (should use Kepler propagation)
        r_sc = np.asarray(r_helio) + np.asarray(v_helio) * t
        r_pl = np.asarray(r_planet) + np.asarray(v_planet) * t

        r_rel = r_sc - r_pl
        inside = np.linalg.norm(r_rel) < soi_radius

        if entry and not inside_prev and inside:
            return t
        if not entry and inside_prev and not inside:
            return t

        inside_prev = inside

    return None


@dataclass
class PatchedConicTrajectory:
    """Complete patched conic trajectory with multiple segments."""
    segments: List[TrajectorySegment]
    soi_crossings: List[Dict[str, Any]]
    total_time: float

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["Patched Conic Trajectory Summary", "=" * 40]
        for i, seg in enumerate(self.segments):
            lines.append(f"\nSegment {i+1}: {seg.frame.name}")
            lines.append(f"  Central body: {seg.central_body}")
            lines.append(f"  Duration: {seg.t_end - seg.t_start:.4f} days")
        lines.append(f"\nTotal time: {self.total_time:.4f} days")
        lines.append(f"SOI crossings: {len(self.soi_crossings)}")
        return "\n".join(lines)


def compute_c3(v_inf: float) -> float:
    """
    Compute characteristic energy C3 = v∞².

    Parameters
    ----------
    v_inf : float
        Hyperbolic excess velocity

    Returns
    -------
    float
        C3 in velocity² units
    """
    return v_inf ** 2


def v_infinity_from_c3(c3: float) -> float:
    """
    Compute v-infinity from C3.

    Parameters
    ----------
    c3 : float
        Characteristic energy

    Returns
    -------
    float
        V-infinity magnitude
    """
    if c3 < 0:
        raise ValueError("C3 must be non-negative for escape trajectory")
    return np.sqrt(c3)


def departure_v_infinity(
    v_helio_sc: np.ndarray,
    v_helio_planet: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Calculate departure V-infinity and C3.

    Parameters
    ----------
    v_helio_sc : np.ndarray
        Spacecraft heliocentric velocity after departure
    v_helio_planet : np.ndarray
        Planet heliocentric velocity

    Returns
    -------
    v_inf : np.ndarray
        V-infinity vector
    v_inf_mag : float
        V-infinity magnitude
    c3 : float
        Characteristic energy
    """
    v_inf = np.asarray(v_helio_sc) - np.asarray(v_helio_planet)
    v_inf_mag = np.linalg.norm(v_inf)
    c3 = v_inf_mag ** 2
    return v_inf, v_inf_mag, c3


def arrival_conditions(
    v_helio_sc: np.ndarray,
    v_helio_planet: np.ndarray,
    mu_planet: float
) -> Dict[str, float]:
    """
    Calculate arrival conditions at target body.

    Parameters
    ----------
    v_helio_sc : np.ndarray
        Spacecraft heliocentric velocity at arrival
    v_helio_planet : np.ndarray
        Target planet heliocentric velocity
    mu_planet : float
        Target planet gravitational parameter

    Returns
    -------
    dict
        Arrival parameters including v_inf, C3, capture delta-V estimates
    """
    v_inf = np.asarray(v_helio_sc) - np.asarray(v_helio_planet)
    v_inf_mag = np.linalg.norm(v_inf)
    c3 = v_inf_mag ** 2

    # Estimate capture delta-V to low orbit
    # Assuming capture to altitude of 1.1 * body radius
    # (would need actual radius for precise calculation)

    return {
        'v_infinity': v_inf_mag,
        'v_infinity_vector': v_inf,
        'c3': c3,
    }


def interplanetary_transfer_type(c3_departure: float, c3_arrival: float) -> str:
    """
    Classify interplanetary transfer type.

    Parameters
    ----------
    c3_departure : float
        Departure C3
    c3_arrival : float
        Arrival C3

    Returns
    -------
    str
        Transfer classification
    """
    if c3_departure < 10 and c3_arrival < 10:
        return "Low-energy transfer"
    elif c3_departure < 30 and c3_arrival < 30:
        return "Medium-energy transfer"
    else:
        return "High-energy transfer"


def estimate_transfer_delta_v(
    body_departure: str,
    body_arrival: str,
    parking_orbit_alt_km: float = 200.0
) -> Dict[str, float]:
    """
    Estimate delta-V for an interplanetary transfer using patched conics.

    Uses Hohmann-like assumptions for a rough estimate.

    Parameters
    ----------
    body_departure : str
        Departure body name
    body_arrival : str
        Arrival body name
    parking_orbit_alt_km : float
        Parking orbit altitude at departure body

    Returns
    -------
    dict
        Delta-V estimates for different mission phases
    """
    # Get body data
    dep_body = BODIES.get(body_departure)
    arr_body = BODIES.get(body_arrival)
    sun = BODIES['sun']

    if dep_body is None or arr_body is None:
        raise ValueError(f"Unknown body: {body_departure} or {body_arrival}")

    # Approximate orbital radii (AU)
    orbital_radii = {
        'mercury': 0.387, 'venus': 0.723, 'earth': 1.000,
        'mars': 1.524, 'jupiter': 5.203, 'saturn': 9.537,
        'uranus': 19.19, 'neptune': 30.07, 'pluto': 39.48
    }

    r1 = orbital_radii.get(body_departure, 1.0)
    r2 = orbital_radii.get(body_arrival, 1.524)

    mu_sun = sun['mu']  # AU³/day²
    mu_dep = dep_body['mu']

    # Hohmann transfer semi-major axis
    a_transfer = (r1 + r2) / 2

    # Heliocentric velocities
    v_circ_dep = np.sqrt(mu_sun / r1)
    v_circ_arr = np.sqrt(mu_sun / r2)

    # Transfer orbit velocities
    v_dep_helio = np.sqrt(mu_sun * (2/r1 - 1/a_transfer))
    v_arr_helio = np.sqrt(mu_sun * (2/r2 - 1/a_transfer))

    # V-infinity at departure and arrival
    v_inf_dep = abs(v_dep_helio - v_circ_dep)
    v_inf_arr = abs(v_arr_helio - v_circ_arr)

    # Convert to km/s for readability
    au_per_day_to_km_s = 1.495978707e8 / 86400

    return {
        'v_inf_departure_km_s': v_inf_dep * au_per_day_to_km_s,
        'v_inf_arrival_km_s': v_inf_arr * au_per_day_to_km_s,
        'c3_departure': (v_inf_dep * au_per_day_to_km_s) ** 2,
        'c3_arrival': (v_inf_arr * au_per_day_to_km_s) ** 2,
        'transfer_time_days': np.pi * np.sqrt(a_transfer**3 / mu_sun),
    }
