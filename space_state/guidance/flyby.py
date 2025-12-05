# guidance/flyby.py
"""
Gravity Assist and Flyby Trajectory Module

Provides tools for calculating gravity assist maneuvers, including:
- V-infinity vectors and hyperbolic excess velocity
- B-plane targeting parameters
- Flyby trajectory design
- Delta-V from powered gravity assists

Reference frames:
- V-infinity: Velocity relative to body at SOI boundary
- B-plane: Plane perpendicular to incoming V-infinity, passing through body center

Key Equations:
- Turn angle: δ = 2 * arcsin(1/e_hyp)
- B-plane radius: b = a * sqrt(e² - 1)
- Periapsis: rp = a * (e - 1) = μ/v∞² * (sqrt(1 + (b*v∞²/μ)²) - 1)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from base.star_logger import logger


@dataclass
class FlybyGeometry:
    """Container for flyby geometric parameters."""
    v_inf_in: np.ndarray      # Incoming V-infinity vector (km/s or AU/day)
    v_inf_out: np.ndarray     # Outgoing V-infinity vector
    v_inf_mag: float          # V-infinity magnitude
    turn_angle: float         # Total turn angle (radians)
    periapsis: float          # Closest approach distance
    b_parameter: float        # B-plane impact parameter
    e_hyperbolic: float       # Hyperbolic eccentricity
    a_hyperbolic: float       # Hyperbolic semi-major axis (negative)
    delta_v_gain: float       # Heliocentric delta-V gain from flyby

    def __repr__(self) -> str:
        return (f"FlybyGeometry(v∞={self.v_inf_mag:.3f}, δ={np.degrees(self.turn_angle):.2f}°, "
                f"rp={self.periapsis:.1f}, Δv_gain={self.delta_v_gain:.3f})")


@dataclass
class BPlaneTarget:
    """B-plane targeting parameters."""
    b_dot_t: float    # B·T component (in orbital plane)
    b_dot_r: float    # B·R component (perpendicular to orbital plane)
    b_mag: float      # Total B magnitude
    theta: float      # B-plane angle (radians)

    @classmethod
    def from_components(cls, b_t: float, b_r: float) -> 'BPlaneTarget':
        """Create from B·T and B·R components."""
        b_mag = np.sqrt(b_t**2 + b_r**2)
        theta = np.arctan2(b_r, b_t)
        return cls(b_dot_t=b_t, b_dot_r=b_r, b_mag=b_mag, theta=theta)

    @classmethod
    def from_periapsis(cls, rp: float, v_inf: float, mu: float, theta: float = 0.0) -> 'BPlaneTarget':
        """
        Create B-plane target from desired periapsis.

        Parameters
        ----------
        rp : float
            Desired periapsis radius
        v_inf : float
            V-infinity magnitude
        mu : float
            Body gravitational parameter
        theta : float
            B-plane angle (0 = in-plane, π/2 = out of plane)
        """
        # From periapsis, calculate B magnitude
        # rp = a(e-1) = (μ/v∞²)(sqrt(1 + (bv∞²/μ)²) - 1)
        # Solving for b: b = rp * sqrt(1 + 2μ/(rp*v∞²))
        b_mag = rp * np.sqrt(1 + 2 * mu / (rp * v_inf**2))
        b_t = b_mag * np.cos(theta)
        b_r = b_mag * np.sin(theta)
        return cls(b_dot_t=b_t, b_dot_r=b_r, b_mag=b_mag, theta=theta)


def compute_v_infinity(
    v_spacecraft: np.ndarray,
    v_body: np.ndarray
) -> np.ndarray:
    """
    Compute V-infinity (hyperbolic excess velocity) vector.

    Parameters
    ----------
    v_spacecraft : np.ndarray
        Spacecraft heliocentric velocity
    v_body : np.ndarray
        Body heliocentric velocity

    Returns
    -------
    np.ndarray
        V-infinity vector (relative velocity at infinity)
    """
    return np.asarray(v_spacecraft) - np.asarray(v_body)


def flyby_turn_angle(v_inf: float, rp: float, mu: float) -> float:
    """
    Calculate the turn angle for a flyby.

    δ = 2 * arcsin(1/e)

    Parameters
    ----------
    v_inf : float
        V-infinity magnitude
    rp : float
        Periapsis radius
    mu : float
        Body gravitational parameter

    Returns
    -------
    float
        Turn angle in radians
    """
    # Semi-major axis (negative for hyperbola)
    a = -mu / v_inf**2

    # Eccentricity
    e = 1 - rp / a

    if e <= 1:
        raise ValueError(f"Eccentricity {e} <= 1, not a hyperbolic trajectory")

    # Turn angle
    delta = 2 * np.arcsin(1 / e)

    return delta


def flyby_periapsis_from_turn_angle(
    v_inf: float,
    delta: float,
    mu: float
) -> float:
    """
    Calculate required periapsis for a desired turn angle.

    Parameters
    ----------
    v_inf : float
        V-infinity magnitude
    delta : float
        Desired turn angle (radians)
    mu : float
        Body gravitational parameter

    Returns
    -------
    float
        Required periapsis radius
    """
    # From δ = 2*arcsin(1/e), get e = 1/sin(δ/2)
    e = 1 / np.sin(delta / 2)

    # From e = 1 - rp/a and a = -μ/v∞²
    a = -mu / v_inf**2
    rp = a * (1 - e)

    return rp


def b_plane_from_periapsis(rp: float, v_inf: float, mu: float) -> float:
    """
    Calculate B-plane impact parameter from periapsis.

    b = rp * sqrt(1 + 2μ/(rp*v∞²))

    Parameters
    ----------
    rp : float
        Periapsis radius
    v_inf : float
        V-infinity magnitude
    mu : float
        Body gravitational parameter

    Returns
    -------
    float
        B-plane impact parameter
    """
    return rp * np.sqrt(1 + 2 * mu / (rp * v_inf**2))


def periapsis_from_b_plane(b: float, v_inf: float, mu: float) -> float:
    """
    Calculate periapsis from B-plane impact parameter.

    rp = (μ/v∞²) * (sqrt(1 + (b*v∞²/μ)²) - 1)

    Parameters
    ----------
    b : float
        B-plane impact parameter
    v_inf : float
        V-infinity magnitude
    mu : float
        Body gravitational parameter

    Returns
    -------
    float
        Periapsis radius
    """
    x = b * v_inf**2 / mu
    return (mu / v_inf**2) * (np.sqrt(1 + x**2) - 1)


def rotate_v_infinity(
    v_inf_in: np.ndarray,
    turn_angle: float,
    plane_normal: np.ndarray
) -> np.ndarray:
    """
    Rotate V-infinity vector by turn angle around plane normal.

    Parameters
    ----------
    v_inf_in : np.ndarray
        Incoming V-infinity vector
    turn_angle : float
        Turn angle (radians)
    plane_normal : np.ndarray
        Unit normal to the flyby plane

    Returns
    -------
    np.ndarray
        Outgoing V-infinity vector
    """
    v_inf_in = np.asarray(v_inf_in)
    plane_normal = np.asarray(plane_normal)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Rodrigues' rotation formula
    cos_d = np.cos(turn_angle)
    sin_d = np.sin(turn_angle)

    v_inf_out = (v_inf_in * cos_d +
                 np.cross(plane_normal, v_inf_in) * sin_d +
                 plane_normal * np.dot(plane_normal, v_inf_in) * (1 - cos_d))

    return v_inf_out


def compute_flyby(
    v_inf_in: np.ndarray,
    rp: float,
    mu: float,
    flyby_plane_normal: Optional[np.ndarray] = None
) -> FlybyGeometry:
    """
    Compute complete flyby geometry.

    Parameters
    ----------
    v_inf_in : np.ndarray
        Incoming V-infinity vector
    rp : float
        Periapsis radius (must be > body radius!)
    mu : float
        Body gravitational parameter
    flyby_plane_normal : np.ndarray, optional
        Normal to the flyby plane. If None, defaults to ecliptic normal [0,0,1]

    Returns
    -------
    FlybyGeometry
        Complete flyby parameters
    """
    v_inf_in = np.asarray(v_inf_in, dtype=float)
    v_inf_mag = np.linalg.norm(v_inf_in)

    if v_inf_mag < 1e-10:
        raise ValueError("V-infinity magnitude is too small")

    # Hyperbolic elements
    a = -mu / v_inf_mag**2  # Negative for hyperbola
    e = 1 - rp / a

    if e <= 1:
        raise ValueError(f"Eccentricity {e:.4f} <= 1. Check periapsis distance.")

    # Turn angle
    turn_angle = 2 * np.arcsin(1 / e)

    # B-plane parameter
    b = b_plane_from_periapsis(rp, v_inf_mag, mu)

    # Default flyby plane normal
    if flyby_plane_normal is None:
        # Use ecliptic normal, but ensure it's not parallel to v_inf
        flyby_plane_normal = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(v_inf_in / v_inf_mag, flyby_plane_normal)) > 0.99:
            flyby_plane_normal = np.array([0.0, 1.0, 0.0])

    # Compute outgoing V-infinity
    v_inf_out = rotate_v_infinity(v_inf_in, turn_angle, flyby_plane_normal)

    # Delta-V gain (change in heliocentric velocity)
    # This is the magnitude of the change in V-infinity direction * V-infinity
    delta_v_gain = np.linalg.norm(v_inf_out - v_inf_in)

    return FlybyGeometry(
        v_inf_in=v_inf_in,
        v_inf_out=v_inf_out,
        v_inf_mag=v_inf_mag,
        turn_angle=turn_angle,
        periapsis=rp,
        b_parameter=b,
        e_hyperbolic=e,
        a_hyperbolic=a,
        delta_v_gain=delta_v_gain
    )


def powered_flyby_delta_v(
    v_inf_in: float,
    v_inf_out: float,
    rp: float,
    mu: float
) -> float:
    """
    Calculate delta-V required for a powered gravity assist.

    A powered flyby applies thrust at periapsis to change the outgoing
    V-infinity magnitude. This is the most efficient point for maneuvers
    (Oberth effect).

    Parameters
    ----------
    v_inf_in : float
        Incoming V-infinity magnitude
    v_inf_out : float
        Desired outgoing V-infinity magnitude
    rp : float
        Periapsis radius
    mu : float
        Body gravitational parameter

    Returns
    -------
    float
        Delta-V required at periapsis
    """
    # Velocity at periapsis for incoming trajectory
    vp_in = np.sqrt(v_inf_in**2 + 2 * mu / rp)

    # Velocity at periapsis for outgoing trajectory
    vp_out = np.sqrt(v_inf_out**2 + 2 * mu / rp)

    # Delta-V at periapsis
    return abs(vp_out - vp_in)


def maximum_turn_angle(v_inf: float, rp_min: float, mu: float) -> float:
    """
    Calculate maximum possible turn angle for given minimum periapsis.

    Parameters
    ----------
    v_inf : float
        V-infinity magnitude
    rp_min : float
        Minimum allowed periapsis (body radius + altitude)
    mu : float
        Body gravitational parameter

    Returns
    -------
    float
        Maximum turn angle (radians)
    """
    return flyby_turn_angle(v_inf, rp_min, mu)


def flyby_delta_v_vector(
    v_body: np.ndarray,
    flyby: FlybyGeometry
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate heliocentric velocity change from flyby.

    Parameters
    ----------
    v_body : np.ndarray
        Body heliocentric velocity
    flyby : FlybyGeometry
        Flyby parameters

    Returns
    -------
    v_helio_in : np.ndarray
        Incoming heliocentric velocity
    v_helio_out : np.ndarray
        Outgoing heliocentric velocity
    """
    v_body = np.asarray(v_body)
    v_helio_in = v_body + flyby.v_inf_in
    v_helio_out = v_body + flyby.v_inf_out

    return v_helio_in, v_helio_out


def design_flyby_for_target(
    v_inf_in: np.ndarray,
    target_direction: np.ndarray,
    mu: float,
    body_radius: float,
    min_altitude: float = 100.0
) -> Optional[FlybyGeometry]:
    """
    Design a flyby to achieve a target outgoing direction.

    Parameters
    ----------
    v_inf_in : np.ndarray
        Incoming V-infinity vector
    target_direction : np.ndarray
        Desired outgoing direction (unit vector)
    mu : float
        Body gravitational parameter
    body_radius : float
        Body radius
    min_altitude : float
        Minimum flyby altitude above surface

    Returns
    -------
    FlybyGeometry or None
        Flyby parameters if achievable, None if not possible
    """
    v_inf_in = np.asarray(v_inf_in, dtype=float)
    target_direction = np.asarray(target_direction, dtype=float)
    target_direction = target_direction / np.linalg.norm(target_direction)

    v_inf_mag = np.linalg.norm(v_inf_in)
    v_inf_in_unit = v_inf_in / v_inf_mag

    # Required turn angle
    cos_turn = np.dot(v_inf_in_unit, target_direction)
    if cos_turn > 1 - 1e-10:  # Already pointing in target direction
        return None

    required_turn = np.arccos(np.clip(cos_turn, -1, 1))

    # Check if turn is achievable
    rp_min = body_radius + min_altitude
    max_turn = maximum_turn_angle(v_inf_mag, rp_min, mu)

    if required_turn > max_turn:
        logger.warning(f"Required turn {np.degrees(required_turn):.1f}° exceeds "
                      f"maximum {np.degrees(max_turn):.1f}°")
        return None

    # Find periapsis for required turn
    rp = flyby_periapsis_from_turn_angle(v_inf_mag, required_turn, mu)

    # Flyby plane normal (perpendicular to both v_inf_in and target)
    plane_normal = np.cross(v_inf_in_unit, target_direction)
    plane_normal_mag = np.linalg.norm(plane_normal)

    if plane_normal_mag < 1e-10:
        # Vectors are parallel - any plane works
        plane_normal = np.array([0, 0, 1])
    else:
        plane_normal = plane_normal / plane_normal_mag

    return compute_flyby(v_inf_in, rp, mu, plane_normal)


def gravity_assist_sequence_delta_v(
    body_sequence: list,
    v_inf_sequence: list,
    rp_sequence: list,
    mu_sequence: list
) -> Dict[str, Any]:
    """
    Calculate total delta-V benefit from a gravity assist sequence.

    Parameters
    ----------
    body_sequence : list
        Names of bodies in flyby sequence
    v_inf_sequence : list
        V-infinity magnitudes at each body
    rp_sequence : list
        Periapsis distances at each body
    mu_sequence : list
        Gravitational parameters of each body

    Returns
    -------
    dict
        Sequence analysis including total delta-V gain
    """
    results = {
        'bodies': body_sequence,
        'flybys': [],
        'total_delta_v_gain': 0.0
    }

    for i, (body, v_inf, rp, mu) in enumerate(zip(
        body_sequence, v_inf_sequence, rp_sequence, mu_sequence
    )):
        turn = flyby_turn_angle(v_inf, rp, mu)
        # Delta-V gain approximation: 2 * v_inf * sin(δ/2)
        dv_gain = 2 * v_inf * np.sin(turn / 2)

        flyby_info = {
            'body': body,
            'v_infinity': v_inf,
            'periapsis': rp,
            'turn_angle_deg': np.degrees(turn),
            'delta_v_gain': dv_gain
        }
        results['flybys'].append(flyby_info)
        results['total_delta_v_gain'] += dv_gain

    return results
