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

# ------------------- THRUST ACCELERATION -------------------

def thrust_acceleration(t: float, y: np.ndarray, ship: object, guidance: object) -> tuple[np.ndarray, float]:
    """Compute thrust acceleration and mass flow rate."""
    if guidance is not None and ship is not None:
        _, thrust_dir, throttle = guidance.update(t, y)
        ship.thrust_direction = thrust_dir
        ship.throttle = throttle
        # Thrust acceleration and mass flow rate
        a_thrust = ship.thrust_vector              # AU/day²
        dm_dt =  - ship.mdot_kgpd                  # kg/day
    else:
        # No guidance, no thrust
        a_thrust = np.array([0.0, 0.0, 0.0])
        dm_dt = 0.0
    
    return a_thrust, dm_dt