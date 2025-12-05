# guidance/station_keeping.py
"""
Station Keeping Module

Provides station keeping calculations for:
- Geostationary orbit (GEO) maintenance
- Lagrange point orbits (L1, L2, halo orbits)
- Low-thrust propulsion maneuvers

GEO Station Keeping:
- North-South (inclination) corrections
- East-West (longitude) corrections
- Eccentricity control

Lagrange Points:
- Circular Restricted Three-Body Problem (CR3BP)
- Halo orbit maintenance
- Station-keeping delta-V budgets
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum, auto

from base.star_logger import logger


# Constants
MU_EARTH = 398600.4418  # km³/s²
MU_SUN = 1.32712440041e11  # km³/s²
MU_MOON = 4902.8  # km³/s²
R_EARTH = 6378.137  # km
R_GEO = 42164.0  # km (GEO radius)
SIDEREAL_DAY = 86164.0905  # seconds


class StationKeepingMode(Enum):
    """Station keeping operational modes."""
    NORTH_SOUTH = auto()  # Inclination control
    EAST_WEST = auto()    # Longitude control
    ECCENTRICITY = auto() # Eccentricity control
    COMBINED = auto()     # All corrections


@dataclass
class GEOStationKeepingBudget:
    """Annual delta-V budget for GEO station keeping."""
    north_south: float      # m/s per year (inclination)
    east_west: float        # m/s per year (longitude)
    eccentricity: float     # m/s per year
    total: float            # m/s per year
    maneuvers_per_year: int # Typical number of maneuvers

    def __repr__(self) -> str:
        return (f"GEOStationKeepingBudget(N-S={self.north_south:.1f} m/s/yr, "
                f"E-W={self.east_west:.1f} m/s/yr, total={self.total:.1f} m/s/yr)")


@dataclass
class LagrangePoint:
    """Lagrange point location and properties."""
    name: str           # L1, L2, L3, L4, L5
    x: float            # x position in rotating frame (normalized)
    y: float            # y position in rotating frame (normalized)
    stable: bool        # True if naturally stable (L4, L5)
    distance_km: float  # Distance from primary body


# ============================================================================
# GEO Station Keeping
# ============================================================================

def geo_inclination_drift(
    initial_inclination: float,
    years: float = 1.0
) -> float:
    """
    Calculate inclination drift at GEO due to solar/lunar perturbations.

    The inclination of a GEO satellite drifts toward ~15° over ~26 years
    due to gravitational perturbations from the Sun and Moon.

    Parameters
    ----------
    initial_inclination : float
        Current inclination (degrees)
    years : float
        Time period (years)

    Returns
    -------
    float
        Final inclination (degrees)
    """
    # Inclination drift rate is approximately 0.85°/year toward 15°
    drift_rate = 0.85  # deg/year
    equilibrium = 14.67  # degrees

    # Simplified linear drift model
    if initial_inclination < equilibrium:
        return min(initial_inclination + drift_rate * years, equilibrium)
    else:
        return max(initial_inclination - drift_rate * years, equilibrium)


def geo_north_south_delta_v(
    inclination_change: float
) -> float:
    """
    Calculate delta-V for North-South station keeping (inclination control).

    ΔV ≈ 2 * V_geo * sin(Δi/2)

    For small angles: ΔV ≈ V_geo * Δi (radians)

    Parameters
    ----------
    inclination_change : float
        Required inclination change (degrees)

    Returns
    -------
    float
        Delta-V required (m/s)
    """
    v_geo = np.sqrt(MU_EARTH / R_GEO) * 1000  # m/s (~3075 m/s)
    delta_i_rad = np.radians(inclination_change)

    return 2 * v_geo * np.sin(delta_i_rad / 2)


def geo_east_west_delta_v(
    longitude_drift_rate: float,
    correction_period: float = 14.0
) -> float:
    """
    Calculate delta-V for East-West station keeping (longitude control).

    Longitude drift is caused by Earth's triaxiality (non-spherical shape).
    Satellites drift toward stable longitudes at 75°E and 105°W.

    Parameters
    ----------
    longitude_drift_rate : float
        Drift rate in degrees per day
    correction_period : float
        Days between corrections

    Returns
    -------
    float
        Delta-V required per maneuver (m/s)
    """
    # Longitude drift corresponds to semi-major axis error
    # For GEO: 1 m/s ΔV changes drift rate by ~0.003 °/day

    drift_correction = abs(longitude_drift_rate) * correction_period
    # Empirical relationship
    delta_v = drift_correction * 0.34  # m/s per degree

    return delta_v


def geo_eccentricity_delta_v(
    eccentricity: float,
    target_eccentricity: float = 0.0001
) -> float:
    """
    Calculate delta-V for eccentricity control.

    Solar radiation pressure causes eccentricity to grow.
    Annual delta-V depends on spacecraft area-to-mass ratio.

    Parameters
    ----------
    eccentricity : float
        Current eccentricity
    target_eccentricity : float
        Target eccentricity

    Returns
    -------
    float
        Delta-V required (m/s)
    """
    v_geo = np.sqrt(MU_EARTH / R_GEO) * 1000  # m/s
    delta_e = abs(eccentricity - target_eccentricity)

    return v_geo * delta_e


def estimate_geo_station_keeping_budget(
    inclination_deadband: float = 0.05,
    longitude_deadband: float = 0.1,
    area_to_mass: float = 0.02
) -> GEOStationKeepingBudget:
    """
    Estimate annual GEO station keeping delta-V budget.

    Parameters
    ----------
    inclination_deadband : float
        Maximum allowed inclination (degrees)
    longitude_deadband : float
        Maximum allowed longitude error (degrees)
    area_to_mass : float
        Spacecraft area-to-mass ratio (m²/kg)

    Returns
    -------
    GEOStationKeepingBudget
        Annual delta-V requirements
    """
    # North-South: ~50 m/s/year for typical 0.05° deadband
    # This is the dominant cost for GEO station keeping
    ns_budget = geo_north_south_delta_v(0.85)  # Annual drift is ~0.85°

    # East-West: depends on longitude, typically 2-6 m/s/year
    # Unstable longitudes require more delta-V
    ew_budget = 4.0  # m/s/year (typical average)

    # Eccentricity: depends on area-to-mass, typically 1-3 m/s/year
    # For A/m = 0.02 m²/kg, about 2 m/s/year
    ecc_budget = 100.0 * area_to_mass  # m/s/year

    total = ns_budget + ew_budget + ecc_budget

    return GEOStationKeepingBudget(
        north_south=ns_budget,
        east_west=ew_budget,
        eccentricity=ecc_budget,
        total=total,
        maneuvers_per_year=24  # Typical bi-weekly maneuvers
    )


# ============================================================================
# Lagrange Points
# ============================================================================

def compute_lagrange_points(
    m1: float,
    m2: float,
    distance: float
) -> Dict[str, LagrangePoint]:
    """
    Compute Lagrange point locations in the circular restricted 3-body problem.

    Parameters
    ----------
    m1 : float
        Mass of primary body (larger)
    m2 : float
        Mass of secondary body (smaller)
    distance : float
        Distance between bodies (km)

    Returns
    -------
    dict
        Dictionary of LagrangePoint objects for L1-L5
    """
    # Mass ratio
    mu = m2 / (m1 + m2)

    # L1, L2, L3 are on the line connecting the bodies
    # L4, L5 are at equilateral triangle points

    # L1: between bodies, closer to smaller mass
    # Approximate: r_L1 = d * (mu/3)^(1/3)
    r_L1 = distance * (mu / 3) ** (1/3)
    x_L1 = 1 - mu - r_L1 / distance

    # L2: beyond smaller mass
    r_L2 = distance * (mu / 3) ** (1/3)
    x_L2 = 1 - mu + r_L2 / distance

    # L3: beyond larger mass
    x_L3 = -1 - 5 * mu / 12

    # L4 and L5: equilateral triangle points
    x_L4 = 0.5 - mu
    y_L4 = np.sqrt(3) / 2

    x_L5 = 0.5 - mu
    y_L5 = -np.sqrt(3) / 2

    return {
        'L1': LagrangePoint('L1', x_L1, 0, False, (1 - x_L1) * distance),
        'L2': LagrangePoint('L2', x_L2, 0, False, (x_L2 - 1 + mu) * distance),
        'L3': LagrangePoint('L3', x_L3, 0, False, abs(x_L3) * distance),
        'L4': LagrangePoint('L4', x_L4, y_L4, True, distance),  # Leading
        'L5': LagrangePoint('L5', x_L5, y_L5, True, distance),  # Trailing
    }


def sun_earth_lagrange_points() -> Dict[str, LagrangePoint]:
    """
    Compute Sun-Earth Lagrange points.

    Returns distances in km from Earth.
    """
    AU_KM = 149597870.7
    points = compute_lagrange_points(MU_SUN, MU_EARTH, AU_KM)

    # Adjust distances to be Earth-centric
    points['L1'].distance_km = AU_KM * (MU_EARTH / (3 * MU_SUN)) ** (1/3)
    points['L2'].distance_km = AU_KM * (MU_EARTH / (3 * MU_SUN)) ** (1/3)

    return points


def earth_moon_lagrange_points() -> Dict[str, LagrangePoint]:
    """
    Compute Earth-Moon Lagrange points.

    Returns distances in km from Earth.
    """
    MOON_DISTANCE = 384400.0  # km
    return compute_lagrange_points(MU_EARTH, MU_MOON, MOON_DISTANCE)


def lagrange_station_keeping_budget(
    point: str,
    system: str = 'sun-earth'
) -> float:
    """
    Estimate annual station keeping delta-V for a Lagrange point orbit.

    Parameters
    ----------
    point : str
        Lagrange point (L1, L2, L3, L4, L5)
    system : str
        'sun-earth' or 'earth-moon'

    Returns
    -------
    float
        Annual delta-V (m/s/year)
    """
    # Typical values from mission heritage
    budgets = {
        'sun-earth': {
            'L1': 5.0,   # SOHO, DSCOVR (~5 m/s/year)
            'L2': 5.0,   # JWST, Gaia (~1-10 m/s/year)
            'L3': 50.0,  # Unstable, rarely used
            'L4': 0.0,   # Stable, minimal maintenance
            'L5': 0.0,   # Stable, minimal maintenance
        },
        'earth-moon': {
            'L1': 100.0,  # Gateway station concept
            'L2': 100.0,
            'L3': 200.0,
            'L4': 10.0,   # Potentially stable
            'L5': 10.0,
        }
    }

    return budgets.get(system, {}).get(point, 50.0)


# ============================================================================
# Low-Thrust Station Keeping
# ============================================================================

@dataclass
class LowThrustBudget:
    """Low-thrust propulsion budget."""
    thrust_n: float         # Thrust level (N)
    isp_s: float           # Specific impulse (s)
    delta_v_per_year: float # m/s
    propellant_per_year: float  # kg


def low_thrust_delta_v(
    thrust: float,
    mass: float,
    burn_time: float
) -> float:
    """
    Calculate delta-V from low-thrust burn.

    Parameters
    ----------
    thrust : float
        Thrust level (N)
    mass : float
        Spacecraft mass (kg)
    burn_time : float
        Burn duration (seconds)

    Returns
    -------
    float
        Delta-V (m/s)
    """
    acceleration = thrust / mass  # m/s²
    return acceleration * burn_time


def estimate_low_thrust_budget(
    thrust_n: float,
    isp_s: float,
    spacecraft_mass: float,
    delta_v_requirement: float
) -> LowThrustBudget:
    """
    Estimate low-thrust propulsion budget.

    Parameters
    ----------
    thrust_n : float
        Thrust level (N)
    isp_s : float
        Specific impulse (s)
    spacecraft_mass : float
        Spacecraft mass (kg)
    delta_v_requirement : float
        Annual delta-V requirement (m/s)

    Returns
    -------
    LowThrustBudget
        Propulsion budget
    """
    g0 = 9.80665  # m/s²
    v_exhaust = isp_s * g0

    # Tsiolkovsky equation for propellant mass
    mass_ratio = np.exp(delta_v_requirement / v_exhaust)
    propellant_mass = spacecraft_mass * (mass_ratio - 1)

    return LowThrustBudget(
        thrust_n=thrust_n,
        isp_s=isp_s,
        delta_v_per_year=delta_v_requirement,
        propellant_per_year=propellant_mass
    )


# ============================================================================
# Orbit Maintenance Maneuvers
# ============================================================================

def hohmann_correction(
    current_sma: float,
    target_sma: float,
    mu: float
) -> Tuple[float, float]:
    """
    Calculate delta-V for Hohmann transfer correction.

    Parameters
    ----------
    current_sma : float
        Current semi-major axis (km)
    target_sma : float
        Target semi-major axis (km)
    mu : float
        Gravitational parameter (km³/s²)

    Returns
    -------
    dv1, dv2 : tuple
        Delta-V for each burn (km/s)
    """
    r1 = current_sma
    r2 = target_sma
    a_transfer = (r1 + r2) / 2

    v1 = np.sqrt(mu / r1)
    v_transfer_1 = np.sqrt(mu * (2/r1 - 1/a_transfer))
    dv1 = abs(v_transfer_1 - v1)

    v2 = np.sqrt(mu / r2)
    v_transfer_2 = np.sqrt(mu * (2/r2 - 1/a_transfer))
    dv2 = abs(v2 - v_transfer_2)

    return dv1, dv2


def plane_change_delta_v(
    velocity: float,
    angle_change: float
) -> float:
    """
    Calculate delta-V for plane change maneuver.

    Parameters
    ----------
    velocity : float
        Orbital velocity (km/s)
    angle_change : float
        Plane change angle (degrees)

    Returns
    -------
    float
        Delta-V (km/s)
    """
    angle_rad = np.radians(angle_change)
    return 2 * velocity * np.sin(angle_rad / 2)


def combined_maneuver_delta_v(
    velocity: float,
    altitude_change_km: float,
    plane_change_deg: float,
    mu: float = MU_EARTH
) -> float:
    """
    Calculate delta-V for combined altitude and plane change.

    It's more efficient to combine maneuvers at apoapsis.

    Parameters
    ----------
    velocity : float
        Current orbital velocity (km/s)
    altitude_change_km : float
        Altitude change (km)
    plane_change_deg : float
        Plane change (degrees)
    mu : float
        Gravitational parameter

    Returns
    -------
    float
        Total delta-V (km/s)
    """
    # Simplified: vector sum of required changes
    dv_altitude = abs(altitude_change_km) * 0.001  # Rough approximation
    dv_plane = plane_change_delta_v(velocity, plane_change_deg)

    # Combined is less than sum due to vector addition
    return np.sqrt(dv_altitude**2 + dv_plane**2)
