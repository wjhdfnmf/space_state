# star_states.py

import numpy as np
from typing import Tuple

from .star_logger import logger
from .star_constants import epoch_jd, ts
from .star_utils import body_position, body_velocity, km_to_au

# 

def get_orbiting_state(BODY, altitude_km: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an initial state vector for a circular orbit around a given body at specified altitude (km)"""
    altitude = km_to_au(altitude_km)
    r0 = np.array([BODY['radius'] + altitude, 0.0, 0.0])
    v_circ = np.sqrt(BODY['mu'] / np.linalg.norm(r0))
    v0 = np.array([0.0, v_circ, 0.0])

    r0_ssb = body_position(BODY, ts.tt_jd(epoch_jd)) + r0
    v0_ssb = body_velocity(BODY, ts.tt_jd(epoch_jd)) + v0

    # logger.debug(f"get_orbiting_initial_state: r0_ssb={r0_ssb}, v0_ssb={v0_ssb}")
    # logger.debug(f"get_orbiting_initial_state: altitude={altitude_km} km, v_circ={v_circ} au/day")
    # logger.debug(f"get_orbiting_initial_state: BODY position={body_position(BODY, ts.tt_jd(epoch_jd))}, velocity={body_velocity(BODY, ts.tt_jd(epoch_jd))}")
    # logger.debug(f"get_orbiting_initial_state: r0={r0}, v0={v0}")
    # logger.debug(f"get_orbiting_initial_state: r0 norm={np.linalg.norm(r0)} au, v0 norm={np.linalg.norm(v0)} au/day")
    # logger.debug(f"get_orbiting_initial_state: BODY mu={BODY['mu']}, radius={BODY['radius']} au")

    return r0_ssb, v0_ssb

def hohmann_setup(BODY, altitude_km: float) -> Tuple[np.ndarray, np.ndarray]:
    # Initial state: 400 km LEO
    r0, v0 = get_orbiting_state(BODY, altitude_km)
    t0 = 0.0        # days

    # Reference Earth state
    earth_pos = body_position(BODY, ts.tt_jd(epoch_jd + t0))
    earth_vel = body_velocity(BODY, ts.tt_jd(epoch_jd + t0))

    # Target: GEO at apoapsis
    r1_au = np.linalg.norm(r0 - earth_pos)     # Current orbit radius
    r2_au = km_to_au(altitude_km*6) + BODY['radius']   # GEO radius
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

    return r0, v_transfer