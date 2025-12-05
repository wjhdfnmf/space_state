# guidance/lambert.py
"""
Lambert Problem Solver

Solves the two-point boundary value problem for Keplerian orbits:
Given r1, r2, and time of flight (TOF), find the transfer orbit.

Uses the universal variable formulation for robustness across all orbit types.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from base.star_logger import logger


@dataclass
class LambertSolution:
    """Solution to Lambert's problem."""
    v1: np.ndarray  # Departure velocity (AU/day)
    v2: np.ndarray  # Arrival velocity (AU/day)
    a: float        # Semi-major axis (AU)
    e: float        # Eccentricity
    tof: float      # Time of flight (days)
    dv1: float      # Departure delta-V magnitude (AU/day)
    dv2: float      # Arrival delta-V magnitude (AU/day)
    dv_total: float # Total delta-V (AU/day)
    prograde: bool  # True if prograde transfer
    converged: bool # True if solution converged


def stumpff_c(z: float) -> float:
    """Stumpff function C(z)."""
    if z > 1e-6:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < -1e-6:
        return (1 - np.cosh(np.sqrt(-z))) / z
    else:
        return 0.5 - z/24 + z**2/720


def stumpff_s(z: float) -> float:
    """Stumpff function S(z)."""
    if z > 1e-6:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / (sz**3)
    elif z < -1e-6:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / (sz**3)
    else:
        return 1/6 - z/120 + z**2/5040


def lambert_universal(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    prograde: bool = True,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Optional[LambertSolution]:
    """
    Solve Lambert's problem using universal variables.

    Args:
        r1: Initial position vector (AU)
        r2: Final position vector (AU)
        tof: Time of flight (days)
        mu: Gravitational parameter (AU³/day²)
        prograde: True for prograde transfer, False for retrograde
        max_iter: Maximum Newton-Raphson iterations
        tol: Convergence tolerance

    Returns:
        LambertSolution or None if failed to converge
    """
    # Magnitudes
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # Check for degenerate cases
    if r1_mag < 1e-12 or r2_mag < 1e-12:
        return None
    if tof <= 0:
        return None

    # Cross product for transfer angle
    cross = np.cross(r1, r2)
    cross_mag = np.linalg.norm(cross)

    # Dot product
    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)

    # Handle near-180° transfers (degenerate plane)
    # When cross product is near zero, we need to define the transfer plane
    if cross_mag < 1e-10:
        # Near 0° or 180° transfer - define an arbitrary normal
        if cos_dnu > 0:
            # Near 0° - same position, degenerate
            return None
        else:
            # Near 180° - choose z-axis as normal for prograde
            if prograde:
                cross = np.array([0.0, 0.0, 1.0])
            else:
                cross = np.array([0.0, 0.0, -1.0])

    # Determine transfer angle based on prograde/retrograde
    if prograde:
        if cross[2] >= 0:
            dnu = np.arccos(cos_dnu)
        else:
            dnu = 2*np.pi - np.arccos(cos_dnu)
    else:
        if cross[2] < 0:
            dnu = np.arccos(cos_dnu)
        else:
            dnu = 2*np.pi - np.arccos(cos_dnu)

    # Auxiliary parameter A
    # Use sin(dnu/2) formulation to avoid 1-cos_dnu issues near 0°
    sin_dnu = np.sin(dnu)
    if abs(sin_dnu) < 1e-10:
        # Use limit form for near-0° or near-180°
        sin_dnu = 1e-10 if sin_dnu >= 0 else -1e-10

    # Standard A computation with safe denominator
    denom = 1 - cos_dnu
    if abs(denom) < 1e-12:
        denom = 1e-12

    A = sin_dnu * np.sqrt(r1_mag * r2_mag / denom)

    # Initial guess for z based on transfer geometry
    # Estimate semi-major axis for better initial guess
    c = np.sqrt(r1_mag**2 + r2_mag**2 - 2*r1_mag*r2_mag*cos_dnu)  # chord
    s = (r1_mag + r2_mag + c) / 2  # semi-perimeter
    a_min = s / 2  # minimum energy ellipse

    # Initial guess: start with parabolic (z=0) unless very long TOF
    # Estimate parabolic TOF
    tof_parabolic = np.sqrt(2) / 3 * (s**1.5 - np.sign(sin_dnu) * (s - c)**1.5) / np.sqrt(mu)

    if tof > tof_parabolic * 1.2:
        z = 0.1  # Elliptic (positive z)
    elif tof < tof_parabolic * 0.8:
        z = -0.1  # Hyperbolic (negative z)
    else:
        z = 0.0

    # Newton-Raphson iteration with bisection fallback
    z_low = -4 * np.pi**2  # Lower bound (hyperbolic limit)
    z_high = 4 * np.pi**2  # Upper bound (elliptic, less than one revolution)

    for iteration in range(max_iter):
        C = stumpff_c(z)
        S = stumpff_s(z)

        # y function
        y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)

        if y < 0:
            # y must be positive - adjust search bounds
            if z > 0:
                z_high = z
            z = (z + z_low) / 2
            continue

        # chi (universal variable)
        chi = np.sqrt(y / C)

        # Time of flight from current z
        tof_calc = (chi**3 * S + A * np.sqrt(y)) / np.sqrt(mu)

        # Check convergence on TOF
        if abs(tof_calc - tof) < tol * tof:
            break

        # Derivatives for Newton-Raphson
        if abs(z) > 1e-6:
            dC = (0.5 - z*S - C) / z
            dS = (C - 3*S) / (2*z)
        else:
            dC = -1/12 + z/240
            dS = -1/60 + z/2520

        dy = A * (S + z*dS) / np.sqrt(C) - A * (z*S - 1) * dC / (2*C*np.sqrt(C))
        dchi = 0.5 * dy / (C * chi) - chi * dC / (2 * C)
        dtof = (3*chi**2 * S * dchi + chi**3 * dS + A * dy / (2*np.sqrt(y))) / np.sqrt(mu)

        # Newton-Raphson update with bounds checking
        if abs(dtof) < 1e-14:
            # Derivative too small - use bisection
            if tof_calc > tof:
                z_low = z
            else:
                z_high = z
            z = (z_low + z_high) / 2
            continue

        z_new = z + (tof - tof_calc) / dtof

        # Bound the update
        if z_new < z_low:
            z_new = (z + z_low) / 2
        if z_new > z_high:
            z_new = (z + z_high) / 2

        # Update bounds based on TOF comparison
        if tof_calc > tof:
            z_low = z
        else:
            z_high = z

        # Check convergence
        if abs(z_new - z) < tol:
            z = z_new
            break

        z = z_new
    else:
        # Failed to converge
        return None

    # Compute final values
    C = stumpff_c(z)
    S = stumpff_s(z)
    y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)

    # Lagrange coefficients
    f = 1 - y / r1_mag
    g = A * np.sqrt(y / mu)
    g_dot = 1 - y / r2_mag

    # Velocity vectors
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g

    # Orbital elements
    h = np.cross(r1, v1)
    h_mag = np.linalg.norm(h)

    # Semi-major axis from vis-viva
    v1_mag = np.linalg.norm(v1)
    energy = 0.5 * v1_mag**2 - mu / r1_mag
    if abs(energy) > 1e-12:
        a = -mu / (2 * energy)
    else:
        a = np.inf  # Parabolic

    # Eccentricity
    e_vec = np.cross(v1, h) / mu - r1 / r1_mag
    e = np.linalg.norm(e_vec)

    return LambertSolution(
        v1=v1,
        v2=v2,
        a=a,
        e=e,
        tof=tof,
        dv1=0.0,  # Will be computed by caller with initial/target velocities
        dv2=0.0,
        dv_total=0.0,
        prograde=prograde,
        converged=True,
    )


def lambert_solve(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    v1_initial: Optional[np.ndarray] = None,
    v2_target: Optional[np.ndarray] = None,
    prograde: bool = True,
) -> Optional[LambertSolution]:
    """
    Solve Lambert's problem and compute delta-V requirements.

    Args:
        r1: Departure position (AU)
        r2: Arrival position (AU)
        tof: Time of flight (days)
        mu: Gravitational parameter (AU³/day²)
        v1_initial: Initial velocity at r1 (optional, for delta-V calculation)
        v2_target: Target velocity at r2 (optional, for delta-V calculation)
        prograde: True for prograde, False for retrograde

    Returns:
        LambertSolution with delta-V computed if velocities provided
    """
    solution = lambert_universal(r1, r2, tof, mu, prograde)

    if solution is None:
        return None

    # Compute delta-V if initial/target velocities provided
    if v1_initial is not None:
        dv1_vec = solution.v1 - v1_initial
        solution.dv1 = np.linalg.norm(dv1_vec)
    else:
        solution.dv1 = np.linalg.norm(solution.v1)

    if v2_target is not None:
        dv2_vec = v2_target - solution.v2
        solution.dv2 = np.linalg.norm(dv2_vec)
    else:
        solution.dv2 = np.linalg.norm(solution.v2)

    solution.dv_total = solution.dv1 + solution.dv2

    return solution


def lambert_multirev(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    max_revs: int = 0,
    prograde: bool = True,
) -> list:
    """
    Solve Lambert's problem including multi-revolution solutions.

    Args:
        r1: Departure position (AU)
        r2: Arrival position (AU)
        tof: Time of flight (days)
        mu: Gravitational parameter (AU³/day²)
        max_revs: Maximum number of complete revolutions (0 for direct)
        prograde: True for prograde, False for retrograde

    Returns:
        List of LambertSolution objects (may include multiple solutions)
    """
    solutions = []

    # Direct transfer (0 revolutions)
    sol = lambert_universal(r1, r2, tof, mu, prograde)
    if sol is not None:
        solutions.append(sol)

    # Multi-revolution solutions would go here (more complex)
    # For now, only direct transfers are supported

    return solutions
