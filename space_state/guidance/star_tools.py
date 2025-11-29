# guidance/star_tools.py

import numpy as np

from base.star_logger import logger
from base.star_ship import Spacecraft

# ————————————————————————
# Inclination Calculation
# ————————————————————————
def inclination(r_vec, v_vec, axis: int = 0) -> dict:
    """
    Calculate inclination from position and velocity vectors relative to central body
    Args:
        r_vec: Position vector [AU] shape (3,) or (N, 3)
        v_vec: Velocity vector [AU/day] matching shape
        axis: 0 for single state, 1 for time series
    Returns:
        Inclination in radians (0, π), degrees (0, 180), and signed degrees (-180, 180)
        Scalars if axis=0, arrays if axis=1
    """
    if axis == 1:
        # time series
        h_vec = np.cross(r_vec, v_vec, axis=1)
        h_norm = np.linalg.norm(h_vec, axis=1)
        hz = h_vec[:, 2]
    else:
        # scalar
        h_vec = np.cross(r_vec, v_vec)              # Angular momentum
        h_norm = np.linalg.norm(h_vec)
        hz = h_vec[2]                               # Or np.dot(h_vec, [0, 0, 1])
    
    if np.any(np.isclose(h_norm, 0)):
        raise ValueError("Angular momentum is zero; cannot determine inclination")
    
    cos_i = np.abs(hz / h_norm)                     # abs to ensure [0, π]
    incl = np.arccos(np.clip(cos_i, -1.0, 1.0))     # Avoid numerical issues like acos(1.0000001)
    incl_deg = np.degrees(incl)
    incl_deg_signed = np.sign(hz) * incl_deg

    return {
        'i_rad': incl,
        'i_deg': incl_deg,
        'i_deg_signed': incl_deg_signed,
    }

# ————————————————————————
# Eccentricity Calculation
# ————————————————————————
def eccentricity(r_vec, v_vec, mu, axis: int = 0) -> dict:
    """
    Calculate eccentricity vector and magnitude from position and velocity vectors relative to central body
    Args:
        r_vec: Position vector [AU] shape (3,) or (N, 3)
        v_vec: Velocity vector [AU/day] matching shape
        mu: Standard gravitational parameter of central body [AU^3/day^2]
        axis: 0 for single state, 1 for time series
    Returns:
        Eccentricity vector (3,) or (N, 3)
        Eccentricity magnitude scalar or (N,)
    """
    if np.isclose(np.linalg.norm(r_vec), 0.0):
        raise ValueError("Position vector is zero; cannot determine eccentricity")
    
    if axis == 1:
        # time series
        h_vec = np.cross(r_vec, v_vec, axis=1)
        r = np.linalg.norm(r_vec, axis=1)
        e_vec = np.cross(v_vec, h_vec, axis=1) / mu - (r_vec.T / r).T
        e_mag = np.linalg.norm(e_vec, axis=1)
    else:
        # scalar
        h_vec = np.cross(r_vec, v_vec)                      # Angular momentum
        r = np.linalg.norm(r_vec)                     
        e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r     # Eccentricity
        e = np.linalg.norm(e_vec)

    return {
        'e_vec': e_vec,
        'e_mag': e_mag,
    }

# ————————————————————————
# Specific Energy
# ————————————————————————
def specific_energy(r_vec, v_vec, mu, axis: int = 0) -> np.ndarray:
    """
    Calculate specific orbital energy ε = v²/2 - μ/r [AU²/day²]
    Args:
        r_vec: Position vector(s) [AU] shape (3,) or (N, 3)
        v_vec: Velocity vector(s) [AU/day] matching shape
        mu: Standard gravitational parameter of central body [AU^3/day^2]
        axis: 0 for single state, 1 for time series
    Returns:
        Specific orbital energy ε (scalar or (N,)) [AU^2/day^2]
        negative = bound orbit, zero = parabolic, positive = unbound
    """
    if axis == 1:
        # time series
        r = np.linalg.norm(r_vec, axis=1)
        v2 = np.sum(v_vec**2, axis=1)
    else:
        # scalar
        r = np.linalg.norm(r_vec)
        v2 = np.dot(v_vec, v_vec)
    specific_energy = v2 /2 - mu / r                        # Specific energy
    return specific_energy

# ————————————————————————
# Semi-Major Axis
# ————————————————————————
def semi_major_axis(r_vec, v_vec, mu, axis: int = 0) -> np.ndarray:
    """
    Calculate semi-major axis a [AU]
    Args:
        r_vec: Position vector(s) [AU] shape (3,) or (N, 3)
        v_vec: Velocity vector(s) [AU/day] matching shape
        mu: Standard gravitational parameter of central body [AU^3/day^2]
        axis: 0 for single state, 1 for time series
    Returns:
        Semi-major axis a (scalar or (N,)) [AU]
        positive = elliptical, negative = hyperbolic
    """
    energy = specific_energy(r_vec, v_vec, mu, axis)
    a = -mu / (2 * (energy + 1e-20 * np.sign(energy)))      # Semi-major axis
    return a

################################################################################
################################################################################

# ————————————————————————
# Time to Apoapsis Calculation
# ————————————————————————
def time_to_apoapsis(r_vec, v_vec, mu, tol=1e-12):
    """
    Time from current state to next apoapsis (days) relative to central body.
    Args:
        r_vec: Current position vector [AU] shape (3,)
        v_vec: Current velocity vector [AU/day] shape (3,)
        mu: Standard gravitational parameter of central body [AU^3/day^2]
        tol: Tolerance for eccentricity comparison
    Returns:
        - Positive finite value     : time to apoapsis (elliptical orbit)
        - np.inf                    : parabolic or hyperbolic (no apoapsis)
    Units: r [AU], v [AU/day], mu [AU³/day²]
    """
    # Eccentricity vector and magnitude
    eccentricity = eccentricity(r_vec, v_vec, mu)
    e_vec = eccentricity['e_vec']
    e = eccentricity['e_mag']

    if e < 1.0 - tol:                              # Elliptical (bound)
        energy = specific_energy(r_vec, v_vec, mu)
        a = semi_major_axis(r_vec, v_vec, mu)
        n = np.sqrt(mu / a**3)

        # True anomaly (0 at peri, π at apo)
        r = np.linalg.norm(r_vec)
        cos_nu = np.dot(e_vec, r_vec) / (e * r)
        cos_nu = np.clip(cos_nu, -1.0, 1.0)
        nu = np.arccos(cos_nu)

        # Radial velocity
        vr = np.dot(r_vec, v_vec) / r
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

    else: return np.inf                            # Hyperbolic – no apoapsis

# ————————————————————————
# Apoapsis State Calculation
# ————————————————————————
def get_apoapsis_state(r_vec, v_vec, mu) -> dict:
    """
    Apoapsis state for elliptical orbit (e < 1, ε < 0) relative to central body.
    ASSUMPTIONS: Near periapsis
    Args:
        r_vec: Position vector(s) [AU] shape (3,)
        v_vec: Velocity vector(s) [AU/day] shape (3,)
        mu: Gravitational parameter [AU³/day²]
    Returns:
        Dict with r_apo, v_apo, r_apo_norm, a, e
    """
    energy = specific_energy(r_vec, v_vec, mu)
    if energy >= 0:
        raise ValueError("Unbound orbit (ε ≥ 0); no apoapsis")
    
    a = semi_major_axis(r_vec, v_vec, mu)
    eccentricity = eccentricity(r_vec, v_vec, mu)
    e = eccentricity['e_mag']
    
    if e >= 1.0:
        raise ValueError("Eccentricity ≥ 1; no apoapsis")
    
    # Exact radius at apoapsis
    r_apo = a * (1 + e)
    
    # Position vector at apoapsis: scale current radius vector
    r = np.linalg.norm(r_vec)
    r_apo_vec = (r_vec / r) * r_apo
    
    # Velocity at apoapsis: perpendicular to radius
    v_apo_mag = np.sqrt(mu * (1 - e) / (a * (1 + e)))
    h_vec = np.cross(r_vec, v_vec)
    h_unit = h_vec / np.linalg.norm(h_vec)
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
# Burn Duration Estimation
# ————————————————————————
def estimate_burn_duration(dv_req: float, ship: Spacecraft, verbose: bool = False) -> float:
    """
    Estimate finite burn duration using the rocket equation.
    Returns: Returns 0.0 (no burn), np.inf (impossible), or seconds.
    """

    ship_copy = ship.copy()     # Non-mutating copy of the ship
    ship_copy.throttle = 1.0    # Full throttle for estimation

    # Extract parameters
    propellant_mass = ship_copy.propellant_mass
    dry_mass = ship_copy.dry_mass
    thrust = ship_copy.thrust
    mdot = ship_copy.mdot
    ve = ship_copy.ve

    # Sanity checks
    if dv_req <= 0:
        if verbose: logger.debug("No Δv required; burn duration = 0.")
        return 0.0

    if propellant_mass <= 0:
        if verbose: logger.warning("No propellant available (m0 ≤ dry_mass).")
        return 0.0

    if thrust <= 0:
        if verbose: logger.error("Thrust ≤ 0; infinite burn time.")
        return np.inf

    # Tsiolkovsky ideal rocket equation
    mass = dry_mass + propellant_mass
    mf = mass * np.exp(-dv_req / ve)

    if mf < dry_mass:
        if verbose: logger.warning("Insufficient propellant for Δv; infinite burn time.")
        return np.inf

    # Mass to be expelled
    delta_m = mass - mf

    if delta_m <= 0:
        if verbose: logger.debug("Δv already achieved; no burn needed.")
        return 0.0

    # Burn time based on mass flow rate
    burn_time_sec = delta_m / mdot

    return burn_time_sec