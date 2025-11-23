# star_dynamics.py

import time
from typing import Callable, Optional, Sequence, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from .star_constants import BODIES, ts, epoch_jd
from .star_structures import PropagationResult
from .star_config import CONFIG
from .star_utils import days_to_seconds
from .star_ship import Spacecraft

from .star_logger import logger

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

# ------------------- PROPAGATION CLASS -------------------
class Propagator:
    def __init__(
        self,
        ship: Spacecraft,
        bodies: Sequence[str],
        guidance: Optional[object] = None,
        result: Optional[PropagationResult] = PropagationResult()
    ):
        self.ship = ship
        self.bodies = bodies
        self.guidance = guidance
        self.result = result

    def dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        r, v, m = y[:3], y[3:6], y[6]

        # Gravitational acceleration
        a_grav = n_body_acceleration(t, y, self.bodies)              # AU/day²

        # Update ship state
        self.ship.update_propellant(m)
        # Update ship thrust direction and throttle from guidance
        if self.guidance:
            vgo, thrust_dir, throttle = self.guidance.update(t, y)  # AU/day, unit vector, 0-1
            self.ship.thrust_direction = thrust_dir
            self.ship.set_throttle(throttle)

        a_thrust = self.ship.get_thrust_vector()                    # AU/day²
        dm_dt = self.ship.total_mdot                                # kg/day

        # Total acceleration
        a = a_grav + a_thrust

        return np.concatenate([v, a, [dm_dt]])

    def propagate(
            self,
            t_end: float,
            events: Optional[Callable] = None,
            max_step: float = CONFIG.max_step,
            rtol: float = CONFIG.rtol,
            atol: float = CONFIG.atol
        ) -> PropagationResult:
        """Propagate the spacecraft state over the given time span in days."""

        logger.debug("Starting propagation from t={:.3f} to t={:.3f} days".format(self.result.time, t_end))

        sol = solve_ivp(
            fun=self.dynamics,
            t_span=[self.result.time, t_end],
            y0=self.result.state,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            events=events,
            dense_output=False,
        )

        self.result.append_raw(sol.t, sol.y)
        self.result.describe_state()