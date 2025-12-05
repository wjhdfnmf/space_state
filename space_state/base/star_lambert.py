# base/star_lambert.py
"""
Lambert Problem Solver

Solves Lambert's problem: Given two position vectors and time of flight,
find the orbit connecting them (i.e., the initial and final velocities).

This module provides:
- Universal variable Lambert solver (handles all conic sections)
- Multi-revolution Lambert solutions
- Porkchop plot generation for mission design

Algorithm based on:
- Battin, R.H., "An Introduction to the Mathematics and Methods of Astrodynamics"
- Vallado, D.A., "Fundamentals of Astrodynamics and Applications"

Units:
- Position: Any consistent unit (AU, km)
- Time: Any consistent unit (days, seconds)
- mu: Must be consistent with position^3 / time^2
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, NamedTuple, List
from dataclasses import dataclass
from enum import Enum

from .star_logger import logger


# =============================================================================
# Constants and Types
# =============================================================================

EPS = 1e-12  # Small number for numerical comparisons
MAX_ITERATIONS = 50  # Maximum Newton-Raphson iterations


class TransferType(Enum):
    """Lambert transfer geometry type."""
    SHORT_WAY = 1   # Transfer angle < 180° (Type I)
    LONG_WAY = -1   # Transfer angle > 180° (Type II)


@dataclass
class LambertSolution:
    """Solution to Lambert's problem."""
    v1: np.ndarray          # Initial velocity vector
    v2: np.ndarray          # Final velocity vector
    tof: float              # Time of flight
    transfer_angle: float   # Transfer angle (radians)
    transfer_type: TransferType
    n_revs: int             # Number of complete revolutions
    converged: bool         # Whether solver converged
    iterations: int         # Iterations taken

    @property
    def delta_v1(self) -> float:
        """Magnitude of v1."""
        return np.linalg.norm(self.v1)

    @property
    def delta_v2(self) -> float:
        """Magnitude of v2."""
        return np.linalg.norm(self.v2)

    def __repr__(self) -> str:
        return (f"LambertSolution(|v1|={self.delta_v1:.4f}, |v2|={self.delta_v2:.4f}, "
                f"tof={self.tof:.4f}, type={self.transfer_type.name}, "
                f"n_revs={self.n_revs}, converged={self.converged})")


# =============================================================================
# Stumpff Functions
# =============================================================================

def stumpff_c(z: float) -> float:
    """
    Stumpff function c(z) = (1 - cos(sqrt(z))) / z for z > 0
                          = (cosh(sqrt(-z)) - 1) / (-z) for z < 0
                          = 1/2 for z = 0

    Used in universal variable formulation.
    """
    if z > EPS:
        sqrt_z = np.sqrt(z)
        return (1 - np.cos(sqrt_z)) / z
    elif z < -EPS:
        sqrt_neg_z = np.sqrt(-z)
        return (np.cosh(sqrt_neg_z) - 1) / (-z)
    else:
        # Taylor series for small z: c(z) ≈ 1/2 - z/24 + z²/720 - ...
        return 0.5 - z/24 + z**2/720


def stumpff_s(z: float) -> float:
    """
    Stumpff function s(z) = (sqrt(z) - sin(sqrt(z))) / sqrt(z)^3 for z > 0
                          = (sinh(sqrt(-z)) - sqrt(-z)) / sqrt(-z)^3 for z < 0
                          = 1/6 for z = 0

    Used in universal variable formulation.
    """
    if z > EPS:
        sqrt_z = np.sqrt(z)
        return (sqrt_z - np.sin(sqrt_z)) / (sqrt_z**3)
    elif z < -EPS:
        sqrt_neg_z = np.sqrt(-z)
        return (np.sinh(sqrt_neg_z) - sqrt_neg_z) / (sqrt_neg_z**3)
    else:
        # Taylor series for small z: s(z) ≈ 1/6 - z/120 + z²/5040 - ...
        return 1/6 - z/120 + z**2/5040


def stumpff_c_prime(z: float) -> float:
    """Derivative of Stumpff c(z) with respect to z."""
    if abs(z) < EPS:
        return -1/24 + z/360
    return (1/(2*z)) * (1 - z*stumpff_s(z) - 2*stumpff_c(z))


def stumpff_s_prime(z: float) -> float:
    """Derivative of Stumpff s(z) with respect to z."""
    if abs(z) < EPS:
        return -1/120 + z/2520
    return (1/(2*z)) * (stumpff_c(z) - 3*stumpff_s(z))


# =============================================================================
# Core Lambert Solver
# =============================================================================

def solve_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    transfer_type: TransferType = TransferType.SHORT_WAY,
    n_revs: int = 0,
    tol: float = 1e-10,
    max_iter: int = MAX_ITERATIONS,
) -> LambertSolution:
    """
    Solve Lambert's problem using universal variables.

    Given two position vectors and time of flight, find the transfer orbit.

    Parameters
    ----------
    r1 : np.ndarray
        Initial position vector [x, y, z]
    r2 : np.ndarray
        Final position vector [x, y, z]
    tof : float
        Time of flight (must be positive)
    mu : float
        Gravitational parameter
    transfer_type : TransferType
        SHORT_WAY (< 180°) or LONG_WAY (> 180°)
    n_revs : int
        Number of complete revolutions (0 for direct transfer)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    LambertSolution
        Solution containing initial and final velocity vectors

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., tof <= 0, collinear vectors for LONG_WAY)
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)

    if tof <= 0:
        raise ValueError(f"Time of flight must be positive, got {tof}")

    # Position magnitudes
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    if r1_mag < EPS or r2_mag < EPS:
        raise ValueError("Position vectors must be non-zero")

    # Cross product for orbit normal
    cross = np.cross(r1, r2)
    cross_mag = np.linalg.norm(cross)

    # Transfer angle (0 to 2π)
    cos_theta = np.clip(np.dot(r1, r2) / (r1_mag * r2_mag), -1, 1)

    # Determine transfer angle based on direction
    if transfer_type == TransferType.SHORT_WAY:
        # Short way: θ < π
        if cross[2] >= 0:
            theta = np.arccos(cos_theta)
        else:
            theta = 2*np.pi - np.arccos(cos_theta)
        tm = 1.0
    else:
        # Long way: θ > π
        if cross[2] < 0:
            theta = np.arccos(cos_theta)
        else:
            theta = 2*np.pi - np.arccos(cos_theta)
        tm = -1.0

    # Check for nearly collinear vectors (problematic for long way)
    if cross_mag < EPS and transfer_type == TransferType.LONG_WAY:
        logger.warning("Nearly collinear vectors for long-way transfer")

    # Chord and semi-perimeter
    c = np.sqrt(r1_mag**2 + r2_mag**2 - 2*r1_mag*r2_mag*cos_theta)
    s = (r1_mag + r2_mag + c) / 2

    # Minimum energy semi-major axis
    a_min = s / 2

    # Parameter A from universal variable formulation
    A = tm * np.sqrt(r1_mag * r2_mag * (1 + cos_theta))

    if abs(A) < EPS:
        raise ValueError("Transfer angle results in degenerate geometry (A ≈ 0)")

    # Initial guess for z (universal variable squared / semi-major axis)
    # z > 0: elliptic, z = 0: parabolic, z < 0: hyperbolic
    if n_revs == 0:
        # For direct transfer, start with elliptic guess
        z = 0.0  # Start near parabolic
    else:
        # Multi-rev: need better initial guess based on revolution count
        z = (2 * n_revs * np.pi)**2

    # Newton-Raphson iteration
    converged = False
    for iteration in range(max_iter):
        C = stumpff_c(z)
        S = stumpff_s(z)

        # y parameter
        y = r1_mag + r2_mag + A * (z*S - 1) / np.sqrt(C)

        if y < 0:
            # y must be positive for physical solution
            # Adjust z to find valid region
            z *= 0.5
            continue

        # Universal variable x
        x = np.sqrt(y / C)

        # Time of flight from current z
        tof_calc = (x**3 * S + A * np.sqrt(y)) / np.sqrt(mu)

        # Check convergence
        if abs(tof_calc - tof) < tol:
            converged = True
            break

        # Derivatives for Newton-Raphson
        C_prime = stumpff_c_prime(z)
        S_prime = stumpff_s_prime(z)

        # dy/dz
        if abs(C) > EPS:
            dy_dz = A * (S - z*S_prime) / np.sqrt(C) - A * (z*S - 1) * C_prime / (2*C*np.sqrt(C))
        else:
            dy_dz = A * S / np.sqrt(C)

        # dtof/dz (chain rule)
        dx_dz = (dy_dz * C - y * C_prime) / (2 * C * np.sqrt(y * C))
        dtof_dz = (3*x**2*S*dx_dz + x**3*S_prime + A*dy_dz/(2*np.sqrt(y))) / np.sqrt(mu)

        # Newton step
        if abs(dtof_dz) > EPS:
            z_new = z - (tof_calc - tof) / dtof_dz
        else:
            # Fall back to bisection-like step
            z_new = z * 1.1 if tof_calc < tof else z * 0.9

        z = z_new

    if not converged:
        logger.warning(f"Lambert solver did not converge after {max_iter} iterations")

    # Compute velocity vectors from converged solution
    C = stumpff_c(z)
    S = stumpff_s(z)
    y = r1_mag + r2_mag + A * (z*S - 1) / np.sqrt(C)

    # Lagrange coefficients
    f = 1 - y / r1_mag
    g = A * np.sqrt(y / mu)
    g_dot = 1 - y / r2_mag

    # Velocity vectors
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g

    return LambertSolution(
        v1=v1,
        v2=v2,
        tof=tof,
        transfer_angle=theta,
        transfer_type=transfer_type,
        n_revs=n_revs,
        converged=converged,
        iterations=iteration + 1
    )


def solve_lambert_izzo(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    prograde: bool = True,
    low_path: bool = True,
    n_revs: int = 0,
    tol: float = 1e-10,
    max_iter: int = MAX_ITERATIONS,
) -> LambertSolution:
    """
    Solve Lambert's problem using Izzo's algorithm.

    This is an alternative formulation that's often more robust.
    Uses Householder iteration for faster convergence.

    Parameters
    ----------
    r1, r2 : np.ndarray
        Position vectors
    tof : float
        Time of flight
    mu : float
        Gravitational parameter
    prograde : bool
        True for prograde transfer, False for retrograde
    low_path : bool
        True for low-energy path, False for high-energy
    n_revs : int
        Number of complete revolutions
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    LambertSolution
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)

    if tof <= 0:
        raise ValueError(f"Time of flight must be positive, got {tof}")

    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # Chord vector
    c_vec = r2 - r1
    c = np.linalg.norm(c_vec)

    # Semi-perimeter
    s = (r1_mag + r2_mag + c) / 2

    # Unit vectors
    ir1 = r1 / r1_mag
    ir2 = r2 / r2_mag
    ih = np.cross(ir1, ir2)
    ih_mag = np.linalg.norm(ih)

    if ih_mag < EPS:
        # Collinear - use arbitrary normal
        ih = np.array([0, 0, 1])
    else:
        ih = ih / ih_mag

    # Check prograde/retrograde
    if (ih[2] < 0 and prograde) or (ih[2] >= 0 and not prograde):
        ih = -ih

    # Transfer angle
    cos_theta = np.dot(ir1, ir2)
    sin_theta = np.linalg.norm(np.cross(ir1, ir2))

    if np.dot(ih, np.cross(r1, r2)) < 0:
        theta = 2*np.pi - np.arccos(np.clip(cos_theta, -1, 1))
    else:
        theta = np.arccos(np.clip(cos_theta, -1, 1))

    # Lambda parameter
    lambda_sq = 1 - c/s
    lambda_param = np.sqrt(lambda_sq)
    if theta > np.pi:
        lambda_param = -lambda_param

    # Non-dimensional time of flight
    T = np.sqrt(2*mu/s**3) * tof

    # Initial guess for x (depends on multi-rev)
    if n_revs == 0:
        # Single revolution
        T0 = np.arccos(lambda_param) + lambda_param * np.sqrt(1 - lambda_sq)
        T1 = 2/3 * (1 - lambda_param**3)

        if T >= T0:
            x0 = (T0/T)**(2/3) - 1
        elif T < T1:
            x0 = 5/2 * T1/T * (T1 - T) / (1 - lambda_param**5) + 1
        else:
            x0 = (T0/T)**(np.log(T1/T0)/np.log(2)) - 1
    else:
        # Multi-revolution - more complex initial guess needed
        x0 = 0.0  # Start from parabolic

    # Householder iteration
    x = x0
    converged = False

    for iteration in range(max_iter):
        # Compute hypergeometric-like series
        y = 1 - lambda_sq * (1 - x)
        if y < 0:
            y = abs(y)  # Force positive for continuation

        xi = np.sqrt(y)
        eta = x * y

        # Semi-major axis (non-dimensional)
        if abs(eta) > EPS:
            psi = _compute_psi(eta)
        else:
            psi = 0.0

        # Time of flight for current x
        T_x = _lambert_tof(x, lambda_param, n_revs)

        # Check convergence
        if abs(T_x - T) < tol:
            converged = True
            break

        # Derivatives for Householder
        dT = _lambert_tof_derivative(x, T_x, lambda_param)

        if abs(dT) > EPS:
            x_new = x - (T_x - T) / dT
        else:
            x_new = x + 0.1 * np.sign(T - T_x)

        # Bound x to valid range
        x_new = np.clip(x_new, -1 + EPS, 1 - EPS)
        x = x_new

    if not converged:
        logger.warning(f"Izzo Lambert solver did not converge after {max_iter} iterations")

    # Compute velocities from x
    y = 1 - lambda_sq * (1 - x)
    gamma = np.sqrt(mu * s / 2)

    rho = (r1_mag - r2_mag) / c
    sigma = np.sqrt(1 - rho**2)

    # Velocity components
    v_r1 = gamma * ((lambda_param * y - x) - rho * (lambda_param * y + x)) / r1_mag
    v_r2 = -gamma * ((lambda_param * y - x) + rho * (lambda_param * y + x)) / r2_mag
    v_t1 = gamma * sigma * (y + lambda_param * x) / r1_mag
    v_t2 = gamma * sigma * (y + lambda_param * x) / r2_mag

    # Transverse unit vectors
    it1 = np.cross(ih, ir1)
    it2 = np.cross(ih, ir2)

    v1 = v_r1 * ir1 + v_t1 * it1
    v2 = v_r2 * ir2 + v_t2 * it2

    transfer_type = TransferType.SHORT_WAY if theta <= np.pi else TransferType.LONG_WAY

    return LambertSolution(
        v1=v1,
        v2=v2,
        tof=tof,
        transfer_angle=theta,
        transfer_type=transfer_type,
        n_revs=n_revs,
        converged=converged,
        iterations=iteration + 1
    )


def _compute_psi(eta: float) -> float:
    """Compute psi parameter for Izzo algorithm."""
    if eta > 0:
        return np.arccos(1 - 2*eta) / np.sqrt(eta)
    elif eta < 0:
        return np.arccosh(1 - 2*eta) / np.sqrt(-eta)
    else:
        return 2.0


def _lambert_tof(x: float, lambda_param: float, n_revs: int) -> float:
    """Compute non-dimensional time of flight for Izzo algorithm."""
    y = 1 - lambda_param**2 * (1 - x)

    if y < 0:
        return float('inf')

    xi = np.sqrt(y)
    eta = x * y

    if eta > 0:
        psi = np.arccos(x * xi - lambda_param * (1 - x))
    elif eta < 0:
        psi = np.arccosh(x * xi - lambda_param * (1 - x))
    else:
        psi = 0.0

    T = ((psi - lambda_param * xi + n_revs * np.pi) /
         np.sqrt(abs(eta)) if abs(eta) > EPS else 0.0)

    return T


def _lambert_tof_derivative(x: float, T: float, lambda_param: float) -> float:
    """Compute derivative of TOF with respect to x for Izzo algorithm."""
    y = 1 - lambda_param**2 * (1 - x)

    if y <= 0 or abs(1 - x**2) < EPS:
        return 1.0

    return (3*T*x - 2 + 2*lambda_param**3*x/y) / (1 - x**2)


# =============================================================================
# High-Level Interface
# =============================================================================

def lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    prograde: bool = True,
    n_revs: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified Lambert solver interface.

    Parameters
    ----------
    r1 : np.ndarray
        Initial position vector
    r2 : np.ndarray
        Final position vector
    tof : float
        Time of flight
    mu : float
        Gravitational parameter
    prograde : bool
        True for prograde orbit (default)
    n_revs : int
        Number of complete revolutions

    Returns
    -------
    v1 : np.ndarray
        Initial velocity vector
    v2 : np.ndarray
        Final velocity vector

    Example
    -------
    >>> # Earth to Mars transfer (simplified)
    >>> r_earth = np.array([1.0, 0.0, 0.0])  # AU
    >>> r_mars = np.array([0.0, 1.5, 0.0])   # AU
    >>> tof = 200.0  # days
    >>> mu_sun = 2.959e-4  # AU³/day²
    >>> v1, v2 = lambert(r_earth, r_mars, tof, mu_sun)
    """
    # Determine transfer type from geometry
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)

    cross = np.cross(r1, r2)

    if prograde:
        transfer_type = TransferType.SHORT_WAY if cross[2] >= 0 else TransferType.LONG_WAY
    else:
        transfer_type = TransferType.LONG_WAY if cross[2] >= 0 else TransferType.SHORT_WAY

    try:
        solution = solve_lambert(r1, r2, tof, mu, transfer_type, n_revs)
    except Exception as e:
        logger.warning(f"Primary Lambert solver failed: {e}, trying Izzo")
        solution = solve_lambert_izzo(r1, r2, tof, mu, prograde, True, n_revs)

    return solution.v1, solution.v2


def lambert_all_solutions(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    max_revs: int = 0,
) -> List[LambertSolution]:
    """
    Find all Lambert solutions including multi-revolution transfers.

    Parameters
    ----------
    r1, r2 : np.ndarray
        Position vectors
    tof : float
        Time of flight
    mu : float
        Gravitational parameter
    max_revs : int
        Maximum number of revolutions to consider

    Returns
    -------
    List[LambertSolution]
        All valid solutions sorted by total delta-v
    """
    solutions = []

    for n_revs in range(max_revs + 1):
        for transfer_type in [TransferType.SHORT_WAY, TransferType.LONG_WAY]:
            try:
                sol = solve_lambert(r1, r2, tof, mu, transfer_type, n_revs)
                if sol.converged:
                    solutions.append(sol)
            except Exception:
                pass  # Skip invalid configurations

    # Sort by total delta-v magnitude
    solutions.sort(key=lambda s: s.delta_v1 + s.delta_v2)

    return solutions


# =============================================================================
# Porkchop Plot Generation
# =============================================================================

@dataclass
class PorkchopData:
    """Data for porkchop plot visualization."""
    departure_dates: np.ndarray   # Julian dates or days
    arrival_dates: np.ndarray     # Julian dates or days
    c3_departure: np.ndarray      # C3 at departure (km²/s²)
    c3_arrival: np.ndarray        # C3 at arrival (km²/s²)
    delta_v_total: np.ndarray     # Total delta-v (km/s)
    tof_grid: np.ndarray          # Time of flight grid (days)


def generate_porkchop(
    get_body_state: callable,
    departure_body: str,
    arrival_body: str,
    departure_dates: np.ndarray,
    arrival_dates: np.ndarray,
    mu: float,
    departure_alt_km: float = 200.0,
    arrival_alt_km: float = 200.0,
    mu_departure: float = None,
    mu_arrival: float = None,
) -> PorkchopData:
    """
    Generate porkchop plot data for interplanetary transfer.

    Parameters
    ----------
    get_body_state : callable
        Function(body_name, date) -> (r, v) returning position and velocity
    departure_body : str
        Name of departure body (e.g., 'earth')
    arrival_body : str
        Name of arrival body (e.g., 'mars')
    departure_dates : np.ndarray
        Array of departure dates (days from epoch)
    arrival_dates : np.ndarray
        Array of arrival dates (days from epoch)
    mu : float
        Gravitational parameter of central body (e.g., Sun)
    departure_alt_km : float
        Parking orbit altitude at departure body (km)
    arrival_alt_km : float
        Target orbit altitude at arrival body (km)
    mu_departure : float
        Gravitational parameter of departure body (for C3 calculation)
    mu_arrival : float
        Gravitational parameter of arrival body

    Returns
    -------
    PorkchopData
        Grid of transfer parameters for plotting
    """
    n_dep = len(departure_dates)
    n_arr = len(arrival_dates)

    c3_departure = np.full((n_dep, n_arr), np.nan)
    c3_arrival = np.full((n_dep, n_arr), np.nan)
    delta_v_total = np.full((n_dep, n_arr), np.nan)
    tof_grid = np.full((n_dep, n_arr), np.nan)

    for i, t_dep in enumerate(departure_dates):
        for j, t_arr in enumerate(arrival_dates):
            tof = t_arr - t_dep

            if tof <= 0:
                continue  # Invalid: arrival before departure

            try:
                # Get body states
                r1, v1_body = get_body_state(departure_body, t_dep)
                r2, v2_body = get_body_state(arrival_body, t_arr)

                # Solve Lambert problem
                v1_transfer, v2_transfer = lambert(r1, r2, tof, mu, prograde=True)

                # Compute v-infinity at departure and arrival
                v_inf_dep = v1_transfer - v1_body
                v_inf_arr = v2_transfer - v2_body

                v_inf_dep_mag = np.linalg.norm(v_inf_dep)
                v_inf_arr_mag = np.linalg.norm(v_inf_arr)

                # C3 = v_inf²
                c3_departure[i, j] = v_inf_dep_mag**2
                c3_arrival[i, j] = v_inf_arr_mag**2

                # Total delta-v (simplified: just v_inf magnitudes)
                # For more accurate: include escape/capture burns
                delta_v_total[i, j] = v_inf_dep_mag + v_inf_arr_mag

                tof_grid[i, j] = tof

            except Exception:
                # Lambert failed - leave as NaN
                pass

    return PorkchopData(
        departure_dates=departure_dates,
        arrival_dates=arrival_dates,
        c3_departure=c3_departure,
        c3_arrival=c3_arrival,
        delta_v_total=delta_v_total,
        tof_grid=tof_grid,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def compute_c3(v_inf: np.ndarray) -> float:
    """
    Compute characteristic energy C3.

    C3 = v_infinity² = 2 * specific_orbital_energy

    Parameters
    ----------
    v_inf : np.ndarray
        Hyperbolic excess velocity vector

    Returns
    -------
    float
        C3 in (velocity units)²
    """
    return np.dot(v_inf, v_inf)


def compute_delta_v_escape(v_inf: float, r_parking: float, mu: float) -> float:
    """
    Compute delta-v for hyperbolic escape from parking orbit.

    Parameters
    ----------
    v_inf : float
        Hyperbolic excess velocity (target C3 = v_inf²)
    r_parking : float
        Parking orbit radius
    mu : float
        Gravitational parameter of body

    Returns
    -------
    float
        Delta-v required for escape burn
    """
    v_parking = np.sqrt(mu / r_parking)  # Circular orbit velocity
    v_escape = np.sqrt(v_inf**2 + 2*mu/r_parking)  # Hyperbolic velocity at periapsis
    return v_escape - v_parking


def compute_delta_v_capture(v_inf: float, r_target: float, mu: float) -> float:
    """
    Compute delta-v for capture from hyperbolic approach.

    Parameters
    ----------
    v_inf : float
        Hyperbolic excess velocity (approach C3 = v_inf²)
    r_target : float
        Target orbit radius (periapsis of hyperbola)
    mu : float
        Gravitational parameter of body

    Returns
    -------
    float
        Delta-v required for capture burn
    """
    v_hyperbolic = np.sqrt(v_inf**2 + 2*mu/r_target)  # Velocity at periapsis
    v_circular = np.sqrt(mu / r_target)  # Target circular velocity
    return v_hyperbolic - v_circular
