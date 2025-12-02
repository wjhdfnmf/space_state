# guidance/star_tools.py

import numpy as np
from astropy.coordinates import earth_orientation
from astropy.time import Time
from scipy.integrate import solve_ivp

from base.star_constants import BODIES, ts, epoch_jd
from base.star_logger import logger
from base.star_ship import Spacecraft
from base.star_constants import world_time
from base.star_utils import body_position, body_velocity, sec_to_days, au_to_km, ssb_to_body_centered, day_to_sec, apd_kps
from base.star_propagator import Propagator

from visual.star_plane_vis import plot_orbital_plane

# ————————————————————————
# Ecliptic Normal Vector
# ————————————————————————
def ecliptic_normal(t: float) -> np.ndarray:
    """
    Get ecliptic normal vector at given time t (days)
    Args:
        t: time in days
    Returns:
        Ecliptic normal vector [3,] in equatorial frame
    """
    t_ast = Time(epoch_jd + t, format='jd', scale='tt')  # Align scales
    eps_deg = earth_orientation.obliquity(t_ast.jd)
    eps_rad = np.radians(eps_deg)
    return np.array([0.0, -np.sin(eps_rad), np.cos(eps_rad)])


# ————————————————————————
# Inclination Calculation
# ————————————————————————
def inclination(r_vec, v_vec, axis: int = 0, ref_normal: np.ndarray = None) -> dict:
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
    ref_normal = np.array([0, 0, 1]) if ref_normal is None else ref_normal / np.linalg.norm(ref_normal)
    if axis == 1:
        # time series
        h_vec = np.cross(r_vec, v_vec, axis=1)
        h_norm = np.linalg.norm(h_vec, axis=1)
        hz = np.dot(h_vec, ref_normal)
    else:
        # scalar
        h_vec = np.cross(r_vec, v_vec)              # Angular momentum
        h_norm = np.linalg.norm(h_vec)
        hz = np.dot(h_vec, ref_normal)
    
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
        e_mag = np.linalg.norm(e_vec)

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

# ———————————————————————
# Vis-Viva Equation
# ————————————————————————
def vis_viva(mu: float, r: float, a: float) -> float:
    """
    Calculate orbital speed using the vis-viva equation
    Args:
        mu: Standard gravitational parameter of central body [AU^3/day^2]
        r: Current orbital radius [AU]
        a: Semi-major axis [AU]
    Returns:
        Orbital speed v [AU/day]
    """
    if a == 0:  # Parabolic
        return np.sqrt(2 * mu / r)
    if a == r:  # Circular
        return np.sqrt(mu / r)
    v = np.sqrt(mu * (2 / r - 1 / a))
    return v

# ————————————————————————
# Node Finding
# ————————————————————————
def find_nodes(r1: np.ndarray, v1: np.ndarray, r2: np.ndarray, v2: np.ndarray, t0: float, 
               central_body: dict, t_span: float = 10.0) -> dict:
    """
    Find ascending/descending node times using events (relative 2-body).
    """
    
    mu = central_body['mu']
    
    h1_hat = np.cross(r1, v1) / np.linalg.norm(np.cross(r1, v1))
    h2_hat = np.cross(r2, v2) / np.linalg.norm(np.cross(r2, v2))
    
    n = np.cross(h1_hat, h2_hat)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        raise ValueError("Planes parallel; no nodes.")
    n_hat = n / n_norm
    
    def relative_ode(t, y):
        r, v = y[:3], y[3:]
        r_norm = np.linalg.norm(r)
        return np.concatenate([v, -mu * r / r_norm**3])
    
    def node_event(t, y):
        #return np.dot(y[:3], n_hat)
        return np.dot(y[:3], h2_hat)
    node_event.terminal = False
    node_event.direction = 0
    
    sol = solve_ivp(relative_ode, [t0, t0 + t_span], np.concatenate([r1, v1]), 
                    events=node_event, dense_output=True, rtol=1e-10, atol=1e-12)
    
    node_times = np.sort(sol.t_events[0])
    if len(node_times) == 0:
        return {'ascending_t': None, 'descending_t': None, 'node_vector': n_hat}
    
    v_rel_nodes = np.array([sol.sol(ti)[3:6] for ti in node_times])
    is_ascending = np.dot(v_rel_nodes, h2_hat) > 0
    
    ascending_t = node_times[is_ascending][0] if np.any(is_ascending) else None
    descending_t = node_times[~is_ascending][0] if np.any(~is_ascending) else None

    r_asc = sol.sol(ascending_t)[:3]
    v_asc = sol.sol(ascending_t)[3:6]

    #plot_orbital_plane(r1, v1, r2, v2)
    #plot_orbital_plane(r_asc, v_asc, r2, v2)
    
    return {
        'ascending_t': ascending_t,
        'descending_t': descending_t,
        'node_vector': n_hat,
    }

def orbital_period(r_vec: np.ndarray, v_vec: np.ndarray, mu: float, axis: int = 0) -> np.ndarray | float:
    """
    Compute orbital period from position and velocity vectors relative to central body.
    Uses Kepler's third law: T = 2π √(a³ / μ), where a is semi-major axis.
    Args:
        r_vec: Position vector [AU], shape (3,) or (N, 3)
        v_vec: Velocity vector [AU/day], matching shape
        mu: Standard gravitational parameter [AU³/day²]
        axis: 0 for single state, 1 for time series
    Returns:
        Period [days], scalar or (N,)
    """
    a = semi_major_axis(r_vec, v_vec, mu, axis=axis)
    
    if np.any(a <= 0):
        raise ValueError("Invalid semi-major axis (a ≤ 0); hyperbolic/parabolic orbit.")
    
    T = 2 * np.pi * np.sqrt(a**3 / mu)  # days
    
    return T

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
    e_data = eccentricity(r_vec, v_vec, mu)
    e_vec = e_data['e_vec']
    e = e_data['e_mag']

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
    e_data = eccentricity(r_vec, v_vec, mu)
    e = e_data['e_mag']
    
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

#
# Inclination to Target Body
#
def inclination_to_target(prop: Propagator, body: dict, target_body: dict) -> None:
    time = prop.t
    r_ssb = prop.r
    v_ssb = prop.v

    r_body = body_position(body, world_time(time))
    v_body = body_velocity(body, world_time(time))
    r_target = body_position(target_body, world_time(time)) - r_body
    v_target = body_velocity(target_body, world_time(time)) - v_body
    r_rel = r_ssb - r_body
    v_rel = v_ssb - v_body
    e_norm = ecliptic_normal(time)
    i_target = inclination(r_target, v_target, ref_normal=e_norm)
    i_orig = inclination(r_rel, v_rel, ref_normal=e_norm)

    #logger.debug(f"e_norm: {e_norm}")
    #logger.debug(f"i_target: {i_target['i_deg_signed']} deg")
    #logger.debug(f"i_orig: {i_orig['i_deg_signed']} deg")
    logger.debug(f"Inclination to target body '{target_body['name']}': {i_orig['i_deg_signed'] - i_target['i_deg_signed']:.4f} deg")

#
# Distance to body
#
def distance_to_body(t: float, y: np.ndarray, body: dict, surface: bool = False) -> float:
    r_ssb = y[:3]
    r_body = body_position(body, world_time(t))
    r_rel = r_ssb - r_body
    distance = np.linalg.norm(r_rel)
    if surface:
        distance -= body['radius']
    distance_km = au_to_km(distance)
    return distance_km

def speed_to_body(t: float, y: np.ndarray, body: dict) -> float:
    v_ssb = y[3:6]
    v_body = body_velocity(body, world_time(t))
    v_rel = v_ssb - v_body
    speed = np.linalg.norm(v_rel)
    speed_km_s = apd_kps(speed)
    return speed_km_s

#
# Moon Orbit Period
#
def period_moon():
    """Orbit period of the Moon around Earth."""
    r_body = body_position(BODIES['moon'], world_time(0))
    v_body = body_velocity(BODIES['moon'], world_time(0))
    y_body = np.concatenate([r_body, v_body, [0.0]])
    state = ssb_to_body_centered(world_time(0), y_body, BODIES['earth'])  # Log position relative to target body
    period_moon = orbital_period(state['r_au'], state['v_aud'], BODIES['earth']['mu'])
    logger.info(f"Moon orbit period: {period_moon:.4f} days ({day_to_sec(period_moon):.2f} sec)")
    return period_moon

#
# Ship Orbit Period
#
def period_ship(prop: Propagator, body: dict):
    """Orbit period of the ship around the given body."""
    state = ssb_to_body_centered(world_time(prop.t), prop.y, body)  # Log position relative to body
    period = orbital_period(state['r_au'], state['v_aud'], body['mu'])
    logger.info(f"Ship orbit period: {period:.4f} days ({day_to_sec(period):.2f} sec) around {body['name']}")
    return period