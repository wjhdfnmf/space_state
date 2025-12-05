# base/star_elements.py
"""
Orbital Element Conversions

Provides bidirectional conversion between Cartesian state vectors and
Keplerian orbital elements, plus utility functions for orbital mechanics.

Coordinate Systems:
- Cartesian: [x, y, z, vx, vy, vz] in reference frame (typically J2000/ICRF)
- Keplerian: [a, e, i, Ω, ω, ν] (classical orbital elements)

Element Definitions:
- a: Semi-major axis (same units as position)
- e: Eccentricity (dimensionless, 0=circular, <1=ellipse, =1=parabola, >1=hyperbola)
- i: Inclination (radians, 0 to π)
- Ω: Right ascension of ascending node (RAAN) (radians, 0 to 2π)
- ω: Argument of periapsis (radians, 0 to 2π)
- ν: True anomaly (radians, 0 to 2π)

Alternative anomalies:
- M: Mean anomaly (radians)
- E: Eccentric anomaly (radians)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .star_logger import logger


# Small number for numerical comparisons
EPS = 1e-12


@dataclass
class OrbitalElements:
    """Container for classical Keplerian orbital elements."""
    a: float          # Semi-major axis
    e: float          # Eccentricity
    i: float          # Inclination (rad)
    raan: float       # Right ascension of ascending node (rad)
    omega: float      # Argument of periapsis (rad)
    nu: float         # True anomaly (rad)

    # Optional derived quantities (computed on request)
    mu: Optional[float] = None  # Gravitational parameter used

    @property
    def period(self) -> Optional[float]:
        """Orbital period (same time unit as sqrt(a³/μ))."""
        if self.mu is None or self.e >= 1.0:
            return None
        return 2 * np.pi * np.sqrt(self.a**3 / self.mu)

    @property
    def mean_motion(self) -> Optional[float]:
        """Mean motion n = sqrt(μ/a³)."""
        if self.mu is None or self.a <= 0:
            return None
        return np.sqrt(self.mu / abs(self.a)**3)

    @property
    def energy(self) -> Optional[float]:
        """Specific orbital energy ε = -μ/(2a)."""
        if self.mu is None:
            return None
        return -self.mu / (2 * self.a)

    @property
    def h(self) -> Optional[float]:
        """Specific angular momentum magnitude h = sqrt(μ * a * (1-e²))."""
        if self.mu is None:
            return None
        return np.sqrt(self.mu * self.a * (1 - self.e**2))

    @property
    def periapsis(self) -> float:
        """Periapsis distance rp = a(1-e)."""
        return self.a * (1 - self.e)

    @property
    def apoapsis(self) -> Optional[float]:
        """Apoapsis distance ra = a(1+e). None for hyperbolic orbits."""
        if self.e >= 1.0:
            return None
        return self.a * (1 + self.e)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'a': self.a,
            'e': self.e,
            'i': np.degrees(self.i),
            'raan': np.degrees(self.raan),
            'omega': np.degrees(self.omega),
            'nu': np.degrees(self.nu),
            'period': self.period,
            'periapsis': self.periapsis,
            'apoapsis': self.apoapsis,
        }

    def __repr__(self) -> str:
        orbit_type = "elliptic" if self.e < 1 else "hyperbolic" if self.e > 1 else "parabolic"
        return (f"OrbitalElements({orbit_type}: a={self.a:.6g}, e={self.e:.6f}, "
                f"i={np.degrees(self.i):.2f}°, Ω={np.degrees(self.raan):.2f}°, "
                f"ω={np.degrees(self.omega):.2f}°, ν={np.degrees(self.nu):.2f}°)")


def cartesian_to_keplerian(
    r: np.ndarray,
    v: np.ndarray,
    mu: float
) -> OrbitalElements:
    """
    Convert Cartesian state vector to Keplerian orbital elements.

    Parameters
    ----------
    r : np.ndarray
        Position vector [x, y, z] (any consistent units)
    v : np.ndarray
        Velocity vector [vx, vy, vz] (consistent with r and mu)
    mu : float
        Gravitational parameter μ = GM (consistent units)

    Returns
    -------
    OrbitalElements
        Classical Keplerian orbital elements

    Notes
    -----
    Uses the standard algorithm from Vallado, "Fundamentals of Astrodynamics"
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    # Magnitudes
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    if r_mag < EPS:
        raise ValueError("Position vector magnitude is zero")

    # Specific angular momentum
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    if h_mag < EPS:
        raise ValueError("Angular momentum is zero (rectilinear orbit)")

    # Node vector (points to ascending node)
    k = np.array([0, 0, 1])  # Z-axis unit vector
    n = np.cross(k, h)
    n_mag = np.linalg.norm(n)

    # Eccentricity vector
    e_vec = ((v_mag**2 - mu/r_mag) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)

    # Specific orbital energy
    energy = v_mag**2 / 2 - mu / r_mag

    # Semi-major axis
    if abs(1 - e) < EPS:  # Parabolic
        a = np.inf
    else:
        a = -mu / (2 * energy)

    # Inclination (0 to π)
    i = np.arccos(np.clip(h[2] / h_mag, -1, 1))

    # Right Ascension of Ascending Node (RAAN)
    if n_mag < EPS:  # Equatorial orbit
        raan = 0.0
    else:
        raan = np.arccos(np.clip(n[0] / n_mag, -1, 1))
        if n[1] < 0:
            raan = 2 * np.pi - raan

    # Argument of periapsis
    if n_mag < EPS:  # Equatorial orbit
        if e < EPS:  # Circular equatorial
            omega = 0.0
        else:
            # Use longitude of periapsis
            omega = np.arccos(np.clip(e_vec[0] / e, -1, 1))
            if e_vec[1] < 0:
                omega = 2 * np.pi - omega
    else:
        if e < EPS:  # Circular inclined
            omega = 0.0
        else:
            omega = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * e), -1, 1))
            if e_vec[2] < 0:
                omega = 2 * np.pi - omega

    # True anomaly
    if e < EPS:  # Circular orbit
        if n_mag < EPS:  # Circular equatorial
            # Use true longitude
            nu = np.arccos(np.clip(r[0] / r_mag, -1, 1))
            if r[1] < 0:
                nu = 2 * np.pi - nu
        else:
            # Use argument of latitude
            nu = np.arccos(np.clip(np.dot(n, r) / (n_mag * r_mag), -1, 1))
            if r[2] < 0:
                nu = 2 * np.pi - nu
    else:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1, 1))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu

    return OrbitalElements(a=a, e=e, i=i, raan=raan, omega=omega, nu=nu, mu=mu)


def keplerian_to_cartesian(
    elements: OrbitalElements,
    mu: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Keplerian orbital elements to Cartesian state vector.

    Parameters
    ----------
    elements : OrbitalElements
        Classical Keplerian orbital elements
    mu : float, optional
        Gravitational parameter (uses elements.mu if not provided)

    Returns
    -------
    r : np.ndarray
        Position vector [x, y, z]
    v : np.ndarray
        Velocity vector [vx, vy, vz]
    """
    if mu is None:
        mu = elements.mu
    if mu is None:
        raise ValueError("Gravitational parameter mu must be provided")

    a = elements.a
    e = elements.e
    i = elements.i
    raan = elements.raan
    omega = elements.omega
    nu = elements.nu

    # Semi-latus rectum
    if e < 1.0:
        p = a * (1 - e**2)
    else:
        p = a * (e**2 - 1)  # For hyperbolic orbits

    # Position and velocity in perifocal frame (PQW)
    r_pqw = p / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0])

    # Rotation matrix from perifocal to inertial frame
    # R = Rz(-Ω) @ Rx(-i) @ Rz(-ω)
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)

    R = np.array([
        [cos_raan * cos_omega - sin_raan * sin_omega * cos_i,
         -cos_raan * sin_omega - sin_raan * cos_omega * cos_i,
         sin_raan * sin_i],
        [sin_raan * cos_omega + cos_raan * sin_omega * cos_i,
         -sin_raan * sin_omega + cos_raan * cos_omega * cos_i,
         -cos_raan * sin_i],
        [sin_omega * sin_i,
         cos_omega * sin_i,
         cos_i]
    ])

    r = R @ r_pqw
    v = R @ v_pqw

    return r, v


def true_to_eccentric_anomaly(nu: float, e: float) -> float:
    """
    Convert true anomaly to eccentric anomaly.

    Parameters
    ----------
    nu : float
        True anomaly (radians)
    e : float
        Eccentricity

    Returns
    -------
    E : float
        Eccentric anomaly (radians) for elliptic orbits
        Hyperbolic anomaly H for hyperbolic orbits
    """
    if e < 1.0:  # Elliptic
        E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
        if E < 0:
            E += 2 * np.pi
        return E
    elif e > 1.0:  # Hyperbolic
        # H = 2 * arctanh(sqrt((e-1)/(e+1)) * tan(nu/2))
        tan_half_nu = np.tan(nu / 2)
        arg = np.sqrt((e - 1) / (e + 1)) * tan_half_nu
        H = 2 * np.arctanh(np.clip(arg, -0.99999, 0.99999))
        return H
    else:  # Parabolic (e = 1)
        D = np.tan(nu / 2)
        return D


def eccentric_to_true_anomaly(E: float, e: float) -> float:
    """
    Convert eccentric anomaly to true anomaly.

    Parameters
    ----------
    E : float
        Eccentric anomaly (radians) for elliptic
        Hyperbolic anomaly H for hyperbolic
    e : float
        Eccentricity

    Returns
    -------
    nu : float
        True anomaly (radians)
    """
    if e < 1.0:  # Elliptic
        nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
        if nu < 0:
            nu += 2 * np.pi
        return nu
    elif e > 1.0:  # Hyperbolic
        nu = 2 * np.arctan(np.sqrt((e + 1) / (e - 1)) * np.tanh(E / 2))
        return nu
    else:  # Parabolic
        nu = 2 * np.arctan(E)
        return nu


def mean_to_eccentric_anomaly(
    M: float,
    e: float,
    tol: float = 1e-12,
    max_iter: int = 50
) -> float:
    """
    Convert mean anomaly to eccentric anomaly using Newton-Raphson iteration.

    Parameters
    ----------
    M : float
        Mean anomaly (radians)
    e : float
        Eccentricity
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    E : float
        Eccentric anomaly (radians)

    Raises
    ------
    RuntimeError
        If iteration does not converge
    """
    # Normalize M to [0, 2π]
    M = M % (2 * np.pi)

    if e < 1.0:  # Elliptic - Kepler's equation: M = E - e*sin(E)
        # Initial guess
        if e < 0.8:
            E = M
        else:
            E = np.pi

        for _ in range(max_iter):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            E_new = E - f / f_prime

            if abs(E_new - E) < tol:
                return E_new
            E = E_new

        raise RuntimeError(f"Kepler's equation did not converge for M={M}, e={e}")

    elif e > 1.0:  # Hyperbolic - M = e*sinh(H) - H
        # Initial guess
        H = M / (e - 1) if abs(M) < np.pi else np.sign(M) * np.log(2 * abs(M) / e)

        for _ in range(max_iter):
            f = e * np.sinh(H) - H - M
            f_prime = e * np.cosh(H) - 1
            H_new = H - f / f_prime

            if abs(H_new - H) < tol:
                return H_new
            H = H_new

        raise RuntimeError(f"Hyperbolic Kepler's equation did not converge")

    else:  # Parabolic (e = 1) - Barker's equation
        # M = D + D³/3
        # Solve cubic: D³ + 3D - 3M = 0
        W = 3 * M
        D = (W + np.sqrt(W**2 + 1))**(1/3) - (np.sqrt(W**2 + 1) - W)**(1/3)
        return D


def eccentric_to_mean_anomaly(E: float, e: float) -> float:
    """
    Convert eccentric anomaly to mean anomaly.

    Parameters
    ----------
    E : float
        Eccentric anomaly (radians)
    e : float
        Eccentricity

    Returns
    -------
    M : float
        Mean anomaly (radians)
    """
    if e < 1.0:  # Elliptic
        M = E - e * np.sin(E)
    elif e > 1.0:  # Hyperbolic
        M = e * np.sinh(E) - E
    else:  # Parabolic
        M = E + E**3 / 3

    return M


def true_to_mean_anomaly(nu: float, e: float) -> float:
    """Convert true anomaly to mean anomaly."""
    E = true_to_eccentric_anomaly(nu, e)
    return eccentric_to_mean_anomaly(E, e)


def mean_to_true_anomaly(M: float, e: float) -> float:
    """Convert mean anomaly to true anomaly."""
    E = mean_to_eccentric_anomaly(M, e)
    return eccentric_to_true_anomaly(E, e)


def propagate_kepler(
    elements: OrbitalElements,
    dt: float,
    mu: Optional[float] = None
) -> OrbitalElements:
    """
    Propagate orbital elements forward in time using Kepler's equation.

    Parameters
    ----------
    elements : OrbitalElements
        Initial orbital elements
    dt : float
        Time step (same units as sqrt(a³/μ))
    mu : float, optional
        Gravitational parameter

    Returns
    -------
    OrbitalElements
        Propagated orbital elements (only ν changes for unperturbed motion)
    """
    if mu is None:
        mu = elements.mu
    if mu is None:
        raise ValueError("Gravitational parameter mu must be provided")

    # Mean motion
    n = np.sqrt(mu / abs(elements.a)**3)

    # Current mean anomaly
    M0 = true_to_mean_anomaly(elements.nu, elements.e)

    # Propagate mean anomaly
    M = M0 + n * dt

    # Convert back to true anomaly
    nu_new = mean_to_true_anomaly(M, elements.e)

    return OrbitalElements(
        a=elements.a,
        e=elements.e,
        i=elements.i,
        raan=elements.raan,
        omega=elements.omega,
        nu=nu_new,
        mu=mu
    )


def time_since_periapsis(elements: OrbitalElements, mu: Optional[float] = None) -> float:
    """
    Calculate time since periapsis passage.

    Parameters
    ----------
    elements : OrbitalElements
        Orbital elements
    mu : float, optional
        Gravitational parameter

    Returns
    -------
    float
        Time since periapsis (same units as sqrt(a³/μ))
    """
    if mu is None:
        mu = elements.mu
    if mu is None:
        raise ValueError("Gravitational parameter mu must be provided")

    M = true_to_mean_anomaly(elements.nu, elements.e)
    n = np.sqrt(mu / abs(elements.a)**3)

    return M / n


def time_to_true_anomaly(
    elements: OrbitalElements,
    nu_target: float,
    mu: Optional[float] = None
) -> float:
    """
    Calculate time to reach a specific true anomaly.

    Parameters
    ----------
    elements : OrbitalElements
        Current orbital elements
    nu_target : float
        Target true anomaly (radians)
    mu : float, optional
        Gravitational parameter

    Returns
    -------
    float
        Time to reach target (positive, may be > 1 period)
    """
    if mu is None:
        mu = elements.mu
    if mu is None:
        raise ValueError("Gravitational parameter mu must be provided")

    M_current = true_to_mean_anomaly(elements.nu, elements.e)
    M_target = true_to_mean_anomaly(nu_target, elements.e)

    dM = M_target - M_current
    if dM < 0:
        dM += 2 * np.pi

    n = np.sqrt(mu / abs(elements.a)**3)

    return dM / n


def vis_viva(r: float, a: float, mu: float) -> float:
    """
    Calculate orbital velocity using vis-viva equation.

    v² = μ(2/r - 1/a)

    Parameters
    ----------
    r : float
        Current radius
    a : float
        Semi-major axis
    mu : float
        Gravitational parameter

    Returns
    -------
    float
        Orbital velocity magnitude
    """
    return np.sqrt(mu * (2/r - 1/a))


def circular_velocity(r: float, mu: float) -> float:
    """
    Calculate circular orbital velocity at radius r.

    Parameters
    ----------
    r : float
        Orbital radius
    mu : float
        Gravitational parameter

    Returns
    -------
    float
        Circular orbital velocity
    """
    return np.sqrt(mu / r)


def escape_velocity(r: float, mu: float) -> float:
    """
    Calculate escape velocity at radius r.

    Parameters
    ----------
    r : float
        Distance from central body
    mu : float
        Gravitational parameter

    Returns
    -------
    float
        Escape velocity
    """
    return np.sqrt(2 * mu / r)


def orbital_period(a: float, mu: float) -> float:
    """
    Calculate orbital period for an elliptic orbit.

    Parameters
    ----------
    a : float
        Semi-major axis
    mu : float
        Gravitational parameter

    Returns
    -------
    float
        Orbital period
    """
    return 2 * np.pi * np.sqrt(a**3 / mu)


def flight_path_angle(elements: OrbitalElements) -> float:
    """
    Calculate flight path angle (angle between velocity and local horizontal).

    Parameters
    ----------
    elements : OrbitalElements
        Orbital elements

    Returns
    -------
    float
        Flight path angle (radians), positive when climbing
    """
    e = elements.e
    nu = elements.nu

    tan_gamma = e * np.sin(nu) / (1 + e * np.cos(nu))
    return np.arctan(tan_gamma)


def radius_at_true_anomaly(a: float, e: float, nu: float) -> float:
    """
    Calculate orbital radius at a given true anomaly.

    Parameters
    ----------
    a : float
        Semi-major axis
    e : float
        Eccentricity
    nu : float
        True anomaly (radians)

    Returns
    -------
    float
        Orbital radius
    """
    p = a * (1 - e**2) if e < 1 else a * (e**2 - 1)
    return p / (1 + e * np.cos(nu))


# =============================================================================
# Plane Change Maneuvers
# =============================================================================

def plane_change_delta_v(v: float, delta_i: float) -> float:
    """
    Compute delta-V for a simple plane change maneuver.

    Plane change is most efficient at nodes (ascending/descending) and at
    apoapsis where velocity is lowest.

    Parameters
    ----------
    v : float
        Orbital velocity at maneuver point
    delta_i : float
        Inclination change (radians)

    Returns
    -------
    float
        Required delta-V (same units as v)

    Notes
    -----
    Formula: Δv = 2v sin(Δi/2)
    This is the "pure" plane change - only changes inclination, no altitude change.
    """
    return 2 * v * np.sin(abs(delta_i) / 2)


def combined_plane_change_delta_v(
    v1: float,
    v2: float,
    delta_i: float
) -> float:
    """
    Compute delta-V for a combined plane change with velocity change.

    More efficient than separate plane change and speed change maneuvers.
    Used when changing both orbit altitude and inclination simultaneously.

    Parameters
    ----------
    v1 : float
        Initial velocity magnitude
    v2 : float
        Final velocity magnitude
    delta_i : float
        Inclination change (radians)

    Returns
    -------
    float
        Required delta-V

    Notes
    -----
    Formula: Δv = √(v1² + v2² - 2v1v2cos(Δi))
    This is the law of cosines applied to the velocity triangle.
    """
    return np.sqrt(v1**2 + v2**2 - 2*v1*v2*np.cos(delta_i))


def optimal_plane_change_location(elements: OrbitalElements) -> dict:
    """
    Find optimal locations for plane change maneuvers.

    Plane changes are cheapest at:
    1. Ascending/descending nodes (where orbit crosses reference plane)
    2. Points of lowest velocity (apoapsis for elliptic orbits)

    The optimal point is typically the node closest to apoapsis.

    Parameters
    ----------
    elements : OrbitalElements
        Current orbital elements

    Returns
    -------
    dict
        Dictionary containing:
        - 'ascending_node_nu': True anomaly at ascending node (rad)
        - 'descending_node_nu': True anomaly at descending node (rad)
        - 'optimal_nu': True anomaly at optimal plane change point (rad)
        - 'optimal_type': 'ascending' or 'descending'
    """
    # Ascending node is at argument of latitude u = 0
    # True anomaly at ascending node: ν = -ω (or 2π - ω)
    nu_ascending = (-elements.omega) % (2 * np.pi)

    # Descending node is at argument of latitude u = π
    # True anomaly at descending node: ν = π - ω
    nu_descending = (np.pi - elements.omega) % (2 * np.pi)

    # For elliptic orbits, prefer the node closest to apoapsis (ν = π)
    if elements.e < 1.0:
        # Distance from apoapsis (ν = π)
        dist_asc = min(abs(nu_ascending - np.pi), 2*np.pi - abs(nu_ascending - np.pi))
        dist_desc = min(abs(nu_descending - np.pi), 2*np.pi - abs(nu_descending - np.pi))

        if dist_asc <= dist_desc:
            optimal_nu = nu_ascending
            optimal_type = 'ascending'
        else:
            optimal_nu = nu_descending
            optimal_type = 'descending'
    else:
        # For hyperbolic, use whichever node is accessible
        optimal_nu = nu_ascending
        optimal_type = 'ascending'

    return {
        'ascending_node_nu': nu_ascending,
        'descending_node_nu': nu_descending,
        'optimal_nu': optimal_nu,
        'optimal_type': optimal_type
    }


def plane_change_direction(
    elements: OrbitalElements,
    target_inclination: float,
    at_ascending: bool = True
) -> np.ndarray:
    """
    Compute the thrust direction for a plane change maneuver.

    Parameters
    ----------
    elements : OrbitalElements
        Current orbital elements
    target_inclination : float
        Target inclination (radians)
    at_ascending : bool
        True if at ascending node, False if at descending node

    Returns
    -------
    np.ndarray
        Unit vector in inertial frame for thrust direction

    Notes
    -----
    At the ascending node, to increase inclination, thrust is in the +Z direction
    (normal to the orbital plane in the direction of the angular momentum vector
    for prograde orbits).
    At the descending node, the thrust direction is reversed.
    """
    delta_i = target_inclination - elements.i

    # Orbit normal direction (angular momentum direction)
    cos_i = np.cos(elements.i)
    sin_i = np.sin(elements.i)
    cos_raan = np.cos(elements.raan)
    sin_raan = np.sin(elements.raan)

    # Angular momentum direction for the current orbit
    h_hat = np.array([
        sin_i * sin_raan,
        -sin_i * cos_raan,
        cos_i
    ])

    # For plane change, we thrust normal to the orbit
    # At ascending node: +h direction increases inclination
    # At descending node: -h direction increases inclination
    if at_ascending:
        thrust_dir = np.sign(delta_i) * h_hat
    else:
        thrust_dir = -np.sign(delta_i) * h_hat

    return thrust_dir / np.linalg.norm(thrust_dir)


def compute_inclination_from_state(
    r: np.ndarray,
    v: np.ndarray,
    reference_normal: np.ndarray = None
) -> float:
    """
    Compute orbital inclination from position and velocity vectors.

    Parameters
    ----------
    r : np.ndarray
        Position vector
    v : np.ndarray
        Velocity vector
    reference_normal : np.ndarray, optional
        Normal to reference plane (default: [0, 0, 1] for equatorial)

    Returns
    -------
    float
        Inclination in radians (0 to π)
    """
    if reference_normal is None:
        reference_normal = np.array([0, 0, 1])

    reference_normal = reference_normal / np.linalg.norm(reference_normal)

    # Angular momentum vector
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    if h_mag < EPS:
        return 0.0

    # Inclination is angle between h and reference normal
    cos_i = np.dot(h, reference_normal) / h_mag
    return np.arccos(np.clip(cos_i, -1, 1))


def hohmann_transfer(
    r1: float,
    r2: float,
    mu: float
) -> dict:
    """
    Compute Hohmann transfer orbit parameters.

    Parameters
    ----------
    r1 : float
        Initial circular orbit radius
    r2 : float
        Final circular orbit radius
    mu : float
        Gravitational parameter

    Returns
    -------
    dict
        Transfer parameters including:
        - 'dv1': Delta-V for first burn
        - 'dv2': Delta-V for second burn
        - 'dv_total': Total delta-V
        - 'transfer_time': Time of flight
        - 'a_transfer': Transfer orbit semi-major axis
    """
    # Transfer orbit semi-major axis
    a_transfer = (r1 + r2) / 2

    # Velocities in initial and final circular orbits
    v_circ1 = circular_velocity(r1, mu)
    v_circ2 = circular_velocity(r2, mu)

    # Velocities at periapsis and apoapsis of transfer orbit
    v_transfer_peri = vis_viva(r1, a_transfer, mu)
    v_transfer_apo = vis_viva(r2, a_transfer, mu)

    # Delta-V for each burn
    dv1 = abs(v_transfer_peri - v_circ1)
    dv2 = abs(v_circ2 - v_transfer_apo)

    # Transfer time (half the period of transfer orbit)
    transfer_time = np.pi * np.sqrt(a_transfer**3 / mu)

    return {
        'dv1': dv1,
        'dv2': dv2,
        'dv_total': dv1 + dv2,
        'transfer_time': transfer_time,
        'a_transfer': a_transfer,
        'v_transfer_peri': v_transfer_peri,
        'v_transfer_apo': v_transfer_apo
    }


def bielliptic_transfer(
    r1: float,
    r2: float,
    rb: float,
    mu: float
) -> dict:
    """
    Compute bi-elliptic transfer orbit parameters.

    Bi-elliptic transfer can be more efficient than Hohmann for r2/r1 > 11.94.

    Parameters
    ----------
    r1 : float
        Initial circular orbit radius
    r2 : float
        Final circular orbit radius
    rb : float
        Intermediate apoapsis radius (rb > max(r1, r2))
    mu : float
        Gravitational parameter

    Returns
    -------
    dict
        Transfer parameters
    """
    # First transfer ellipse: r1 to rb
    a1 = (r1 + rb) / 2
    v1_circ = circular_velocity(r1, mu)
    v1_transfer = vis_viva(r1, a1, mu)
    dv1 = abs(v1_transfer - v1_circ)

    # At rb (apoapsis of first transfer, periapsis of second)
    v_rb_1 = vis_viva(rb, a1, mu)

    # Second transfer ellipse: rb to r2
    a2 = (rb + r2) / 2
    v_rb_2 = vis_viva(rb, a2, mu)
    dv2 = abs(v_rb_2 - v_rb_1)

    # Final circularization at r2
    v2_transfer = vis_viva(r2, a2, mu)
    v2_circ = circular_velocity(r2, mu)
    dv3 = abs(v2_circ - v2_transfer)

    # Transfer times
    t1 = np.pi * np.sqrt(a1**3 / mu)
    t2 = np.pi * np.sqrt(a2**3 / mu)

    return {
        'dv1': dv1,
        'dv2': dv2,
        'dv3': dv3,
        'dv_total': dv1 + dv2 + dv3,
        'transfer_time': t1 + t2,
        'a1': a1,
        'a2': a2,
        'rb': rb
    }
