# base/star_dynamics.py

from typing import Sequence, Tuple
import numpy as np

from .star_constants import BODIES, ts, epoch_jd, IDX_Q
from .star_attitude import quaternion_to_dcm

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

def thrust_acceleration(t: float, y: np.ndarray, ship: object, guidance: object) -> Tuple[np.ndarray, float]:
    """
    Compute thrust acceleration using ACTUAL spacecraft attitude.

    Args:
        t: Current time (days)
        y: State vector (14 elements)
        ship: Spacecraft object
        guidance: Guidance controller

    Returns:
        a_thrust: Thrust acceleration (AU/day²) in inertial frame
        dm_main: Main engine propellant flow rate (kg/day)
    """
    if guidance is not None and ship is not None:
        # Get desired thrust direction from guidance
        _, thrust_dir_desired, throttle = guidance.update(t, y)

        # Get ACTUAL thrust direction from current quaternion
        q = y[IDX_Q]
        R_body_to_inertial = quaternion_to_dcm(q)
        thrust_dir_actual = R_body_to_inertial @ np.array([0.0, 0.0, 1.0])  # Body +Z axis

        # Set throttle
        ship.throttle = throttle

        # Compute thrust using ACTUAL pointing direction
        if hasattr(ship, 'thrust_vector_from_direction'):
            a_thrust = ship.thrust_vector_from_direction(thrust_dir_actual)
        else:
            # Fallback for ships without the new method
            ship.thrust_direction = thrust_dir_actual
            a_thrust = ship.thrust_vector

        # Main engine mass flow rate
        dm_main = -ship.mdot_kgpd  # kg/day (negative = losing mass)
    else:
        # No guidance, no thrust
        a_thrust = np.array([0.0, 0.0, 0.0])
        dm_main = 0.0

    return a_thrust, dm_main