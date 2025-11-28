# guidance/star_tools.py

import numpy as np

from base.star_logger import logger

# ————————————————————————
# Time to Apoapsis Calculation
# From Current State to Apoapsis relative to Central Body
# ————————————————————————
def time_to_apoapsis(r_current, v_current, mu, tol=1e-12):
    """
    Time from current state to next apoapsis (days).
    Returns:
        - Positive finite value     : time to apoapsis (elliptical orbit)
        - np.inf                    : parabolic or hyperbolic (no apoapsis)
        - Negative value            : you are past apoapsis, time since apoapsis
    Units: r [AU], v [AU/day], mu [AU³/day²]
    """
    r_vec = np.asarray(r_current)
    v_vec = np.asarray(v_current)
    r = np.linalg.norm(r_vec)
    v2 = np.dot(v_vec, v_vec)

    # Specific energy
    energy = v2 / 2 - mu / r

    # Angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Eccentricity vector and magnitude
    e_vec = (np.cross(v_vec, h_vec) / mu) - r_vec / r
    e = np.linalg.norm(e_vec)

    # Radial velocity
    vr = np.dot(r_vec, v_vec) / r

    if e < 1.0 - tol:                              # Elliptical (bound)
        a = -mu / (2 * energy)                     # positive
        n = np.sqrt(mu / a**3)

        # True anomaly (0 at peri, π at apo)
        cos_nu = np.dot(e_vec, r_vec) / (e * r)
        cos_nu = np.clip(cos_nu, -1.0, 1.0)
        nu = np.arccos(cos_nu)
        if vr < 0:                                 # descending part
            nu = 2*np.pi - nu

        # Time from periapsis to current position
        E = 2 * np.arctan2(
            np.sqrt(1 - e) * np.sin(nu/2),
            np.sqrt(1 + e) * np.cos(nu/2)
        )
        M = E - e * np.sin(E)                      # mean anomaly

        M_apo = np.pi                              # apoapsis is always at M = π
        delta_M = M_apo - M
        if delta_M < 0:
            delta_M += 2*np.pi

        return delta_M / n

    elif abs(e - 1.0) <= tol:                      # Parabolic – no apoapsis
        return np.inf

    else:                                          # Hyperbolic (e > 1)
        # No bounded apoapsis → return +inf for "next", or negative time since vertex
        a = -mu / (2 * energy)                     # a < 0 for hyperbola
        # True anomaly of vertex (analogous to apoapsis)
        cos_nu_vertex = -1.0 / e
        nu_vertex = np.arccos(cos_nu_vertex)

        cos_nu = np.dot(e_vec, r_vec) / (e * r)
        cos_nu = np.clip(cos_nu, -1.0, 1.0)
        nu = np.arccos(cos_nu)
        if vr < 0:
            nu = 2*np.pi - nu

        # Hyperbolic anomaly at vertex = 0
        # Time since vertex is positive before, negative after
        # But conventionally we say "no apoapsis" → return +inf
        return np.inf

# ————————————————————————
# Exact Apoapsis State Calculation
# ————————————————————————
def get_apoapsis_state2(r_rel, v_rel, mu):
    r = np.linalg.norm(r_rel)
    v2 = np.dot(v_rel, v_rel)
    h_vec = np.cross(r_rel, v_rel)
    h = np.linalg.norm(h_vec)
    
    energy = 0.5 * v2 - mu / r
    if energy >= 0:
        raise ValueError("Not bound orbit")
    
    a = -mu / (2 * energy)
    e_vec = np.cross(v_rel, h_vec) / mu - r_rel / r
    e = np.linalg.norm(e_vec)
    
    if e >= 1.0:
        raise ValueError("Eccentricity >= 1")
    
    # Exact radius at apoapsis
    r_apo = a * (1 + e)
    
    # Position vector at apoapsis: scale current radius vector
    r_apo_vec = r_rel * (r_apo / r)
    
    # Velocity at apoapsis: perpendicular to radius
    v_apo_mag = np.sqrt(mu * (1 - e) / (a * (1 + e)))
    h_unit = h_vec / h
    v_apo_vec = np.cross(h_unit, r_apo_vec)
    v_apo_vec = v_apo_vec / np.linalg.norm(v_apo_vec) * v_apo_mag
    
    return {
        "r_apo": r_apo_vec,
        "v_apo": v_apo_vec,
        "r_apo_norm": r_apo,
        "a": a,
        "e": e
    }


# ————————————————————————
# 1. BURN DURATION ESTIMATION
# ————————————————————————
def estimate_burn_duration(
    dv_req: float,          # in m/s
    propellant_mass: float,  # in kg
    dry_mass: float,        # in kg
    max_thrust: float,     # in N
    max_ve: float,         # in m/s
    max_mdot: float,      # in kg/s
    verbose: bool = False
) -> float:
    """
    Estimate finite burn duration in **seconds** using the rocket equation.
    Returns: Burn time in seconds. Returns 0.0 if no burn needed, np.inf if impossible.
    """

    if dv_req <= 0:
        if verbose: logger.debug("No Δv required; burn duration = 0.")
        return 0.0

    if propellant_mass <= 0:
        if verbose: logger.warning("No propellant available (m0 ≤ dry_mass).")
        return 0.0

    if max_thrust <= 0:
        if verbose: logger.error("Thrust ≤ 0; infinite burn time.")
        return np.inf

    # Tsiolkovsky ideal rocket equation
    mass = dry_mass + propellant_mass
    mf = mass * np.exp(-dv_req / max_ve)

    if mf < dry_mass:
        if verbose: logger.warning("Insufficient propellant for Δv; infinite burn time.")
        return np.inf

    delta_m = mass - mf
    if delta_m <= 0:
        if verbose: logger.debug("Δv already achieved; no burn needed.")
        return 0.0

    burn_time_sec = delta_m / max_mdot
    if verbose: logger.debug(f"Estimated burn time: {burn_time_sec:.2f} seconds")
    return burn_time_sec
