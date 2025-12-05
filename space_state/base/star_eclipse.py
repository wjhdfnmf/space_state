# base/star_eclipse.py
"""
Eclipse and Occultation Modeling

Provides calculations for:
- Solar eclipses (umbra and penumbra)
- Planetary occultations
- Communication blackouts
- Eclipse duration and frequency

Eclipse Types:
- Umbra: Total shadow (no direct sunlight)
- Penumbra: Partial shadow (some direct sunlight)
- Antumbra: Beyond umbra tip (annular eclipses)

Applications:
- Power system sizing (battery capacity)
- Thermal control design
- Communication link planning
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto

from .star_constants import BODIES, ts, epoch_jd
from .star_logger import logger


# Constants
AU_TO_KM = 149597870.7
R_SUN = 695700.0  # km
R_EARTH = 6378.137  # km
R_MOON = 1737.4  # km


class EclipseType(Enum):
    """Type of eclipse/shadow."""
    NONE = auto()      # Full sunlight
    PENUMBRA = auto()  # Partial shadow
    UMBRA = auto()     # Total shadow
    ANTUMBRA = auto()  # Beyond umbra (annular)


@dataclass
class EclipseState:
    """Current eclipse state for a spacecraft."""
    eclipse_type: EclipseType
    illumination_fraction: float  # 0.0 = umbra, 1.0 = full sun
    eclipse_body: Optional[str]   # Body causing eclipse
    shadow_depth: float           # Distance into shadow (km)
    time_to_exit: Optional[float] # Time to exit shadow (seconds)

    def __repr__(self) -> str:
        return (f"EclipseState({self.eclipse_type.name}, "
                f"illumination={self.illumination_fraction:.2%}, "
                f"body={self.eclipse_body})")


@dataclass
class EclipsePrediction:
    """Predicted eclipse event."""
    start_time: float       # Simulation time (days)
    end_time: float         # Simulation time (days)
    duration: float         # Duration (seconds)
    eclipse_type: EclipseType
    eclipsing_body: str
    max_shadow_depth: float # km


def shadow_cone_geometry(
    r_sun: float,
    r_body: float,
    distance_sun_body: float
) -> Tuple[float, float, float]:
    """
    Calculate shadow cone geometry.

    Parameters
    ----------
    r_sun : float
        Sun radius (km)
    r_body : float
        Eclipsing body radius (km)
    distance_sun_body : float
        Distance from Sun to body (km)

    Returns
    -------
    umbra_length : float
        Length of umbra cone (km)
    umbra_angle : float
        Half-angle of umbra cone (radians)
    penumbra_angle : float
        Half-angle of penumbra cone (radians)
    """
    # Umbra cone (converging shadow)
    umbra_angle = np.arcsin((r_sun - r_body) / distance_sun_body)
    umbra_length = r_body / np.tan(umbra_angle) if umbra_angle > 0 else np.inf

    # Penumbra cone (diverging shadow)
    penumbra_angle = np.arcsin((r_sun + r_body) / distance_sun_body)

    return umbra_length, umbra_angle, penumbra_angle


def cylindrical_shadow_check(
    r_sc: np.ndarray,
    r_body: np.ndarray,
    r_sun: np.ndarray,
    body_radius: float
) -> EclipseType:
    """
    Check if spacecraft is in cylindrical shadow (simplified model).

    This is a fast approximation that treats the shadow as a cylinder.

    Parameters
    ----------
    r_sc : np.ndarray
        Spacecraft position (km)
    r_body : np.ndarray
        Eclipsing body position (km)
    r_sun : np.ndarray
        Sun position (km)
    body_radius : float
        Eclipsing body radius (km)

    Returns
    -------
    EclipseType
        Shadow type (NONE, PENUMBRA, UMBRA)
    """
    # Vector from body to spacecraft
    r_body_sc = r_sc - r_body

    # Vector from body to sun
    r_body_sun = r_sun - r_body
    r_body_sun_unit = r_body_sun / np.linalg.norm(r_body_sun)

    # Project spacecraft position onto sun direction
    proj_length = np.dot(r_body_sc, r_body_sun_unit)

    # If spacecraft is between body and sun, no eclipse
    if proj_length > 0:
        return EclipseType.NONE

    # Perpendicular distance from shadow axis
    r_perp = r_body_sc - proj_length * r_body_sun_unit
    perp_distance = np.linalg.norm(r_perp)

    # Check if inside cylindrical shadow
    if perp_distance < body_radius:
        return EclipseType.UMBRA
    elif perp_distance < body_radius * 1.1:  # Simplified penumbra
        return EclipseType.PENUMBRA

    return EclipseType.NONE


def conical_shadow_check(
    r_sc: np.ndarray,
    r_body: np.ndarray,
    r_sun: np.ndarray,
    body_radius: float,
    sun_radius: float = R_SUN
) -> Tuple[EclipseType, float]:
    """
    Check if spacecraft is in conical shadow (accurate model).

    Parameters
    ----------
    r_sc : np.ndarray
        Spacecraft position (km)
    r_body : np.ndarray
        Eclipsing body position (km)
    r_sun : np.ndarray
        Sun position (km)
    body_radius : float
        Eclipsing body radius (km)
    sun_radius : float
        Sun radius (km)

    Returns
    -------
    eclipse_type : EclipseType
        Type of eclipse
    illumination : float
        Fraction of sun visible (0-1)
    """
    # Vectors
    r_body_sc = r_sc - r_body
    r_body_sun = r_sun - r_body

    d_body_sun = np.linalg.norm(r_body_sun)
    d_body_sc = np.linalg.norm(r_body_sc)

    # Unit vector from body to sun
    s_hat = r_body_sun / d_body_sun

    # Project spacecraft onto shadow axis
    proj = np.dot(r_body_sc, s_hat)

    # If on sun side of body, no eclipse
    if proj > 0:
        return EclipseType.NONE, 1.0

    # Distance along shadow axis (behind body)
    d_shadow = -proj

    # Shadow cone geometry
    umbra_length, umbra_angle, penumbra_angle = shadow_cone_geometry(
        sun_radius, body_radius, d_body_sun
    )

    # Perpendicular distance from axis
    r_perp_vec = r_body_sc - proj * s_hat
    r_perp = np.linalg.norm(r_perp_vec)

    # Umbra radius at this distance
    if d_shadow < umbra_length:
        umbra_radius = body_radius - d_shadow * np.tan(umbra_angle)
    else:
        umbra_radius = 0  # Beyond umbra tip

    # Penumbra radius at this distance
    penumbra_radius = body_radius + d_shadow * np.tan(penumbra_angle)

    # Determine eclipse type
    if r_perp < umbra_radius and umbra_radius > 0:
        return EclipseType.UMBRA, 0.0
    elif r_perp < penumbra_radius:
        # Linear interpolation for partial shadow
        if umbra_radius > 0:
            illumination = (r_perp - umbra_radius) / (penumbra_radius - umbra_radius)
        else:
            illumination = r_perp / penumbra_radius
        return EclipseType.PENUMBRA, illumination
    else:
        return EclipseType.NONE, 1.0


def compute_eclipse_state(
    r_sc: np.ndarray,
    time: float,
    check_bodies: Optional[List[str]] = None
) -> EclipseState:
    """
    Compute current eclipse state for spacecraft.

    Parameters
    ----------
    r_sc : np.ndarray
        Spacecraft position in SSB frame (AU)
    time : float
        Simulation time (days since epoch)
    check_bodies : list, optional
        Bodies to check for eclipses. Default: ['earth', 'moon']

    Returns
    -------
    EclipseState
        Current eclipse state
    """
    if check_bodies is None:
        check_bodies = ['earth', 'moon']

    # Get sun position
    t_sky = ts.tt_jd(epoch_jd + time)
    sun_pos = BODIES['sun']['skyfield'].at(t_sky).position.au
    sun_pos_km = sun_pos * AU_TO_KM

    r_sc_km = np.asarray(r_sc) * AU_TO_KM

    worst_eclipse = EclipseType.NONE
    worst_illumination = 1.0
    eclipsing_body = None
    shadow_depth = 0.0

    for body_name in check_bodies:
        if body_name not in BODIES:
            continue

        body = BODIES[body_name]
        body_pos = body['skyfield'].at(t_sky).position.au
        body_pos_km = body_pos * AU_TO_KM
        body_radius = body['radius'] * AU_TO_KM

        eclipse_type, illumination = conical_shadow_check(
            r_sc_km, body_pos_km, sun_pos_km, body_radius
        )

        if eclipse_type != EclipseType.NONE:
            if illumination < worst_illumination:
                worst_eclipse = eclipse_type
                worst_illumination = illumination
                eclipsing_body = body_name

                # Calculate shadow depth
                r_body_sc = r_sc_km - body_pos_km
                r_body_sun = sun_pos_km - body_pos_km
                s_hat = r_body_sun / np.linalg.norm(r_body_sun)
                shadow_depth = -np.dot(r_body_sc, s_hat) - body_radius

    return EclipseState(
        eclipse_type=worst_eclipse,
        illumination_fraction=worst_illumination,
        eclipse_body=eclipsing_body,
        shadow_depth=shadow_depth,
        time_to_exit=None  # Would require orbit propagation
    )


def estimate_eclipse_duration_leo(
    altitude_km: float,
    inclination_deg: float,
    beta_angle_deg: float = 0.0
) -> float:
    """
    Estimate eclipse duration for LEO orbit.

    Parameters
    ----------
    altitude_km : float
        Orbital altitude (km)
    inclination_deg : float
        Orbital inclination (degrees)
    beta_angle_deg : float
        Beta angle - angle between orbital plane and sun vector (degrees)

    Returns
    -------
    float
        Maximum eclipse duration (minutes)
    """
    r = R_EARTH + altitude_km
    T = 2 * np.pi * np.sqrt(r**3 / 398600.4418) / 60  # Period in minutes

    # Earth's angular radius as seen from orbit
    rho = np.arcsin(R_EARTH / r)

    beta = np.radians(abs(beta_angle_deg))

    # If beta > 90° - rho, no eclipse (sun-synchronous dawn/dusk)
    if beta > np.pi/2 - rho:
        return 0.0

    # Eclipse fraction of orbit
    cos_eta = np.sqrt(np.sin(beta)**2 + np.cos(beta)**2 * np.cos(rho)**2)
    eta = np.arccos(np.clip(cos_eta, -1, 1))

    eclipse_fraction = eta / np.pi
    eclipse_duration = eclipse_fraction * T

    return eclipse_duration


def estimate_eclipses_per_orbit_geo() -> Dict[str, Any]:
    """
    Estimate eclipse characteristics for GEO orbit.

    Returns
    -------
    dict
        Eclipse statistics for GEO
    """
    return {
        'eclipse_seasons_per_year': 2,
        'season_duration_days': 45,
        'max_eclipse_duration_minutes': 72,
        'eclipses_per_day_during_season': 1,
        'total_eclipse_hours_per_year': 90,
    }


def predict_eclipses(
    trajectory_positions: np.ndarray,
    trajectory_times: np.ndarray,
    check_bodies: Optional[List[str]] = None
) -> List[EclipsePrediction]:
    """
    Predict eclipse events along a trajectory.

    Parameters
    ----------
    trajectory_positions : np.ndarray
        Spacecraft positions in SSB frame (AU), shape (3, N)
    trajectory_times : np.ndarray
        Times (days since epoch), shape (N,)
    check_bodies : list, optional
        Bodies to check for eclipses

    Returns
    -------
    list
        List of EclipsePrediction events
    """
    predictions = []
    in_eclipse = False
    eclipse_start = None
    current_body = None
    current_type = EclipseType.NONE
    max_depth = 0.0

    for i in range(len(trajectory_times)):
        state = compute_eclipse_state(
            trajectory_positions[:, i],
            trajectory_times[i],
            check_bodies
        )

        if state.eclipse_type != EclipseType.NONE and not in_eclipse:
            # Eclipse entry
            in_eclipse = True
            eclipse_start = trajectory_times[i]
            current_body = state.eclipse_body
            current_type = state.eclipse_type
            max_depth = state.shadow_depth

        elif state.eclipse_type != EclipseType.NONE and in_eclipse:
            # Still in eclipse
            max_depth = max(max_depth, state.shadow_depth)
            if state.eclipse_type == EclipseType.UMBRA:
                current_type = EclipseType.UMBRA

        elif state.eclipse_type == EclipseType.NONE and in_eclipse:
            # Eclipse exit
            in_eclipse = False
            duration = (trajectory_times[i] - eclipse_start) * 86400  # seconds

            predictions.append(EclipsePrediction(
                start_time=eclipse_start,
                end_time=trajectory_times[i],
                duration=duration,
                eclipse_type=current_type,
                eclipsing_body=current_body,
                max_shadow_depth=max_depth
            ))

    return predictions


def communication_blackout_check(
    r_sc: np.ndarray,
    r_ground: np.ndarray,
    time: float,
    check_bodies: Optional[List[str]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if communication is blocked by a celestial body.

    Parameters
    ----------
    r_sc : np.ndarray
        Spacecraft position (AU)
    r_ground : np.ndarray
        Ground station position (AU)
    time : float
        Time (days since epoch)
    check_bodies : list, optional
        Bodies to check for occultation

    Returns
    -------
    blocked : bool
        True if communication is blocked
    blocking_body : str or None
        Name of blocking body
    """
    if check_bodies is None:
        check_bodies = ['earth', 'moon', 'sun']

    r_sc_km = np.asarray(r_sc) * AU_TO_KM
    r_ground_km = np.asarray(r_ground) * AU_TO_KM

    # Line of sight vector
    los = r_sc_km - r_ground_km
    los_length = np.linalg.norm(los)
    los_unit = los / los_length

    t_sky = ts.tt_jd(epoch_jd + time)

    for body_name in check_bodies:
        if body_name not in BODIES:
            continue

        body = BODIES[body_name]
        body_pos = body['skyfield'].at(t_sky).position.au * AU_TO_KM
        body_radius = body['radius'] * AU_TO_KM

        # Vector from ground to body
        r_ground_body = body_pos - r_ground_km

        # Project body center onto line of sight
        proj = np.dot(r_ground_body, los_unit)

        # Check if body is between ground and spacecraft
        if proj < 0 or proj > los_length:
            continue

        # Perpendicular distance to line of sight
        closest_point = r_ground_km + proj * los_unit
        perp_distance = np.linalg.norm(body_pos - closest_point)

        if perp_distance < body_radius:
            return True, body_name

    return False, None
