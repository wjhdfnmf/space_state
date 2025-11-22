# star_states.py

import numpy as np

from star_structures import StateVector
from star_constants import epoch_jd, ts
from star_utils import body_position, body_velocity, km_to_au

# 

def get_orbiting_initial_state(BODY, altitude_km: float) -> StateVector:
    """Generate an initial state vector for a circular orbit around a given body at specified altitude (km)"""
    altitude = km_to_au(altitude_km)
    r0 = np.array([BODY['radius'] + altitude, 0.0, 0.0])
    v_circ = np.sqrt(BODY['mu'] / np.linalg.norm(r0))
    v0 = np.array([0.0, v_circ, 0.0])

    r0_ssb = body_position(BODY, ts.tt_jd(epoch_jd)) + r0
    v0_ssb = body_velocity(BODY, ts.tt_jd(epoch_jd)) + v0

    return r0_ssb, v0_ssb