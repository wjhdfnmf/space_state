# base/star_states.py

import numpy as np
from typing import Tuple
from scipy.integrate import solve_ivp

from .star_logger import logger
from .star_config import CONFIG
from .star_constants import epoch_jd, ts
from .star_propagator import Propagator
from .star_utils import body_position, body_velocity, km_to_au, au_to_km, apd_kps, day_to_sec, apd_to_mps
from .star_orbits import ALL_ORBITS
from guidance.star_tools import time_to_apoapsis, get_apoapsis_state2

from typing import Literal

OrbitName = Literal[
    'leo_iss', 'leo_550', 'geo', 'gto_apo', 'tli_apo',
    'lunar_100km', 'nrho_55000km', 'c3_zero'
]

# 

def get_orbiting_state(BODY, orbit: OrbitName) -> Tuple[np.ndarray, np.ndarray]:
    alt = ALL_ORBITS[orbit].altitude_km

    altitude = km_to_au(alt)
    r0 = np.array([BODY['radius'] + altitude, 0.0, 0.0])
    v_circ = np.sqrt(BODY['mu'] / np.linalg.norm(r0))
    v0 = np.array([0.0, v_circ, 0.0])

    r0_ssb = body_position(BODY, ts.tt_jd(epoch_jd)) + r0
    v0_ssb = body_velocity(BODY, ts.tt_jd(epoch_jd)) + v0

    return r0_ssb, v0_ssb



def get_circular_state_at_apoapsis(BODY, t: float, y0: np.ndarray) -> dict:

    # Relative position and velocity to BODY
    r_rel = y0[:3] - body_position(BODY, ts.tt_jd(epoch_jd + t))
    v_rel = y0[3:6] - body_velocity(BODY, ts.tt_jd(epoch_jd + t))

    # Time to apoapsis (for SSB conversion)
    t_to_apo = time_to_apoapsis(r_rel, v_rel, BODY['mu'])
    if t_to_apo < 0:
        raise ValueError("Already past apoapsis!")
    t_apo = t + t_to_apo

    # State of BODY at apoapsis time
    r_body_apo = body_position(BODY, ts.tt_jd(epoch_jd + t_apo))
    v_body_apo = body_velocity(BODY, ts.tt_jd(epoch_jd + t_apo))

    # Exact apoapsis state relative to BODY
    apo_state = get_apoapsis_state2(r_rel, v_rel, BODY['mu'])

    r_apo_rel = apo_state["r_apo"]
    r_target = np.linalg.norm(r_apo_rel) # Circular orbit radius at apoapsis
    
    # Circular velocity
    v_circ_mag = np.sqrt(BODY['mu'] / r_target)
    h_unit = np.cross(r_apo_rel, apo_state["v_apo"])
    h_unit /= np.linalg.norm(h_unit)
    v_circ_dir = np.cross(h_unit, r_apo_rel)
    v_circ_dir /= np.linalg.norm(v_circ_dir)
    v_circ_rel = v_circ_dir * v_circ_mag

    v_delta = np.linalg.norm(v_circ_rel - apo_state['v_apo'])

    # logger.debug(f"Apoapsis radius (rel): {np.linalg.norm(r_apo_rel):.6f} AU")
    # logger.debug(f"Circular velocity at apoapsis (rel): {v_circ_mag:.6f} AU/day")
    # logger.debug(f"Apoapsis velocity (rel): {np.linalg.norm(apo_state['v_apo']):.6f} AU/day")
    # logger.debug(f"Delta-V to circularize at apoapsis: {v_delta:.6f} AU/day")

    # SSB frame
    r_apo_ssb = r_body_apo + r_apo_rel
    v_apo_ssb = v_body_apo + v_circ_rel

    # km and km/s
    r_apo_rel_km = au_to_km(r_apo_rel)
    v_apo_rel_kms = apd_kps(v_circ_rel)
    t_to_apo_sec = day_to_sec(t_to_apo)

    delta_v_mps = apd_to_mps(v_delta)

    # logger.debug(f"Apoapsis state (SSB frame):")
    # logger.debug(f" r (AU): {r_apo_ssb}, |r| = {np.linalg.norm(r_apo_ssb):.6f} AU")
    # logger.debug(f" v (AU/day): {v_apo_ssb}, |v| = {np.linalg.norm(v_apo_ssb):.6f} AU/day")
    # logger.debug(f" Apoapsis relative state (km, km/s): r={r_apo_rel_km}, v={v_apo_rel_kms}")
    # logger.debug(f" Time to apoapsis (sec): {t_to_apo_sec:.2f}")
    # logger.debug(f" Delta-V to circularize at apoapsis (m/s): {delta_v_mps:.2f}")
    # logger.debug(f" Delta-V to circularize at apoapsis (AU/day): {v_delta:.6f}")

    return {
        'r': r_apo_ssb,
        'v': v_apo_ssb,
        't': t_apo,
        't_to': t_to_apo,
        't_apo': t_apo,
        't_burn': t_apo,
        'delta_v': v_delta,
        'delta_v_mps': delta_v_mps,
        'r_rel_km': r_apo_rel_km,
        'v_rel_kms': v_apo_rel_kms,
        't_to_sec': t_to_apo_sec,
    }


def get_hohmann_state_at_time(
        BODY,
        t0: float,
        y0: np.ndarray,
        t_target: float,
        target_orbit: OrbitName,
        max_step: float = CONFIG.max_step
    ) -> dict:
    """Return a new state along the current orbit that applies the Hohmann transfer at time t"""
    
    # Propagate to target time to get future state
    prop = Propagator()
    prop.result.initialize(y0, t0)
    prop.propagate(t_target, max_step=max_step)
    y_target = prop.result.state

    # Relative position and velocity to BODY
    earth_pos = body_position(BODY, ts.tt_jd(epoch_jd + t_target))
    earth_vel = body_velocity(BODY, ts.tt_jd(epoch_jd + t_target))
    r_rel = y_target[:3] - earth_pos
    v_rel = y_target[3:6] - earth_vel

    # Current and target orbit radii
    r1_au = np.linalg.norm(r_rel)     # Current orbit radius
    r2_au = km_to_au(ALL_ORBITS[target_orbit].altitude_km) + BODY['radius']   # Target orbit radius

    # Compute ΔV to enter Hohmann transfer orbit (prograde)
    a_trans = (r1_au + r2_au) / 2 # Semi-major axis of transfer orbit
    v_current = np.linalg.norm(v_rel)
    v_trans_peri = np.sqrt(BODY['mu'] * (2/r1_au - 1/a_trans))
    dv = v_trans_peri - v_current  # AU/day

    # Apply Δv1 in current velocity direction
    v_dir_rel = v_rel / np.linalg.norm(v_rel)
    v_transfer_rel = v_rel + dv * v_dir_rel
    v_transfer = v_transfer_rel + earth_vel  # AU/day in SSB frame
    y_transfer = np.concatenate([y_target[:3], v_transfer, y_target[6:]])

    # logger.debug(f"Hohmann Transfer Setup at t={t_target} days:")
    # logger.debug(f" r1: {r1_au} AU   r2: {r2_au} AU")
    # logger.debug(f" Δv1: {dv} AU/day")
    # logger.debug(f" v_transfer: {v_transfer} AU/day")
    # logger.debug(f" Transfer semi-major axis: {a_trans} AU")
    # logger.debug(f" Transfer periapsis velocity: {v_trans_peri} AU/day")

    # Return new target state
    return {
        'r_target': y_target[:3],
        'y_target': y_target,
        'v_transfer': v_transfer,
        'y_transfer': y_transfer,
        'r1_au': r1_au,
        'r2_au': r2_au,
        'delta_v': dv,
        'delta_v_mps': apd_to_mps(dv),
        't_burn': t_target,
        't_apo': t_target,
    }


def hohmann_setup(BODY, orbit_p: OrbitName, orbit_a: OrbitName = None, t0: float = 0.0) -> dict:

    periapsis = ALL_ORBITS[orbit_p].altitude_km
    apoapsis = periapsis * 6 if orbit_a is None else ALL_ORBITS[orbit_a].altitude_km
    
    logger.debug(f"Setting up Hohmann transfer orbit around {BODY['name']}")
    if orbit_a is not None:
        logger.debug(f" Target orbit: apoapsis = {apoapsis} km")
    else:
        logger.debug(f" Target orbit: apoapsis = {apoapsis} km")
    logger.debug(f" Initial orbit: periapsis = {periapsis} km")

    # Initial state: 400 km LEO
    r0, v0 = get_orbiting_state(BODY, orbit_p)

    # Reference Earth state
    earth_pos = body_position(BODY, ts.tt_jd(epoch_jd + t0))
    earth_vel = body_velocity(BODY, ts.tt_jd(epoch_jd + t0))

    # Target: GEO at apoapsis
    r1_au = np.linalg.norm(r0 - earth_pos)     # Current orbit radius
    r2_au = km_to_au(apoapsis) + BODY['radius']   # GEO radius
    # Compute Δv1 (prograde)
    a_trans = (r1_au + r2_au) / 2
    v_circ_leo = np.sqrt(BODY['mu'] / r1_au)
    v_trans_peri = np.sqrt(BODY['mu'] * (2/r1_au - 1/a_trans))
    dv1 = v_trans_peri - v_circ_leo  # AU/day

    # Relative state to Earth
    r0_rel = r0 - earth_pos  # AU
    v0_rel = v0 - earth_vel  # AU/day

    # Apply Δv1 in current velocity direction
    v_dir_rel = v0_rel / np.linalg.norm(v0_rel)
    v_transfer_rel = v0_rel + dv1 * v_dir_rel
    v_transfer = v_transfer_rel + earth_vel  # AU/day in SSB frame

    logger.debug(f"Initialized Orbit:")
    logger.debug(f" r0 (AU): {r0}   v0 (AU/day): {v0}")
    logger.debug(f" r1 (AU): {r1_au:.6f}   r2 (AU): {r2_au:.6f}")
    logger.debug(f" Δv1 (AU/day): {dv1:.6f}")
    logger.debug(f" v_transfer (AU/day): {v_transfer}")
    logger.debug(f" Transfer semi-major axis (AU): {a_trans:.6f}")
    logger.debug(f" Transfer periapsis velocity (AU/day): {v_trans_peri:.6f}")
    logger.debug(f" Circular LEO velocity (AU/day): {v_circ_leo:.6f}")



    return {
        'r0': r0,
        'v_transfer': v_transfer,
        'delta_v': dv1,
        'delta_v_mps': apd_to_mps(dv1),

    }



# Escape from Earth (C3 = 0 → parabolic)
def escape_state_earth(BODY: dict, t: float, y0: np.ndarray) -> np.ndarray:
    """Modify velocity at current point to achieve parabolic escape (v_esc)."""
    r_rel = y0[:3] - body_position(BODY, ts.tt_jd(epoch_jd + t))
    v_rel = y0[3:6] - body_velocity(BODY, ts.tt_jd(epoch_jd + t))
    r = np.linalg.norm(r_rel)
    v_esc = np.sqrt(2 * BODY['mu'] / r)
    v_dir = v_rel / np.linalg.norm(v_rel)
    v_new = v_dir * v_esc
    v = v_new + body_velocity(BODY, ts.tt_jd(epoch_jd + t))
    return np.concatenate([y0[:3], v, y0[6:]])
