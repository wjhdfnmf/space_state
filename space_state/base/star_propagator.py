# base/star_propagator.py
import time
from typing import Callable, Optional, Sequence, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from .star_dynamics import n_body_acceleration
from .star_structures import PropagationResult, GuidanceCycle, ControllerCycle
from .star_config import CONFIG
from .star_ship import Spacecraft
from .star_logger import logger


# ------------------- PROPAGATOR CLASS -------------------
class Propagator:
    def __init__(
        self,
        ship: Spacecraft = None,
        guidance: object = None,
        result: PropagationResult = None,
        bodies: dict = None,
    ):
        self.ship = ship
        self.guidance = guidance
        self.result = result if result is not None else PropagationResult()
        self.bodies = bodies if bodies is not None else CONFIG.all_bodies
        
    #
    # DYNAMICS FUNCTION
    #

    def dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        r, v, m = y[:3], y[3:6], y[6]
        if self.ship is not None:
            self.ship.update_propellant(m)

        # Gravitational acceleration
        a_grav = n_body_acceleration(t, y, self.bodies)  # AU/day²

        # Update Guidance and thrust
        if self.guidance is not None and self.ship is not None:
            vgo, thrust_dir, throttle = self.guidance.update(t, y) 
            self.ship.thrust_direction = thrust_dir
            self.ship.throttle = throttle

            # Thrust acceleration and mass flow rate
            a_thrust = self.ship.thrust_vector              # AU/day²
            dm_dt =  - self.ship.mdot_kgpd                  # kg/day
        else:
            # No guidance, no thrust
            a_thrust = np.array([0.0, 0.0, 0.0])
            dm_dt = 0.0

        # Total acceleration
        a = a_grav + a_thrust

        return np.concatenate([v, a, [dm_dt]])
    
    #
    # LOG GUIDANCE CYCLES
    #

    def log_guidance(self, sol: object):
        if self.guidance is not None:
        # Log guidance cycles
            for i in range(len(sol.t)):
                t_i = sol.t[i]
                y_i = sol.y[:, i]
                vgo, thrust_dir, throttle = self.guidance.update(t_i, y_i)

                self.result.log_cycle(t_i, ControllerCycle(
                    vgo=vgo,
                    thrust_dir=thrust_dir,
                    throttle=throttle,
                ))
        # No guidance - default to zero thrust
        else:
            for i in range(len(sol.t)):
                t_i = sol.t[i]
                self.result.log_cycle(t_i, ControllerCycle(
                    vgo=0.0,
                    thrust_dir=np.array([0.0, 0.0, 0.0]),
                    throttle=0.0,
                ))

    #
    # PROPAGATION METHOD
    #

    def propagate(
            self,
            t_end: float,
            events: Optional[Callable] = None,
            max_step: float = CONFIG.max_step,
            rtol: float = CONFIG.rtol,
            atol: float = CONFIG.atol
        ) -> PropagationResult:
        """Propagate the spacecraft state over the given time span in days."""

        #logger.debug(f"Propagating from t={self.result.time:.3f} days to t={t_end:.3f} days")

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

        # Append results
        self.result.append_raw(sol.t, sol.y)

        # Log guidance cycles
        self.log_guidance(sol)

        return self.result