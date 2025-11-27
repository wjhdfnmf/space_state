# base/star_dynamics.py

from typing import Sequence
import numpy as np

from .star_constants import BODIES, ts, epoch_jd

# ------------------- GRAVITATIONAL ACCELERATION -------------------

def n_body_acceleration(t: float, y: np.ndarray, bodies: Sequence[str]) -> np.ndarray:
    """Compute N-body gravitational acceleration in AU/day²."""
    r_sat = y[:3]
    time = ts.tt_jd(epoch_jd + t)
    accel = np.zeros(3)

    for body_name in bodies:
        body = BODIES[body_name]["skyfield"]
        GM = BODIES[body_name]["mu"]
        r = r_sat - body.at(time).position.au
        r_norm = np.linalg.norm(r)
        accel += -GM * r / (r_norm ** 3)

    return accel
