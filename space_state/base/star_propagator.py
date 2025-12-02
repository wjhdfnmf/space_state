# base/star_propagator.py
import time
from typing import Callable, Optional, Sequence, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from .star_dynamics import n_body_acceleration, thrust_acceleration
from .star_structures import PropagationResult, GuidanceCycle, ControllerCycle
from .star_utils import log_guidance
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
    
    @property
    def y(self) -> np.ndarray:
        """Current state vector."""
        return self.result.state

    @property
    def t(self) -> float:
        """Current time."""
        return self.result.time
    
    @property
    def r(self) -> np.ndarray:
        """Current position vector."""
        return self.result.state[:3]
        
    @property
    def v(self) -> np.ndarray:
        """Current velocity vector."""
        return self.result.state[3:6]
    
    @property
    def m(self) -> float:
        """Current mass."""
        return self.result.state[6]
        
    #
    # DYNAMICS FUNCTION
    #

    def dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        r, v, m = y[:3], y[3:6], y[6]
        if self.ship is not None:
            self.ship.update_propellant(m)

        # Gravitational acceleration
        a_grav = n_body_acceleration(t, y, self.bodies)  # AU/day²
        a_thrust, dm_dt = thrust_acceleration(t, y, self.ship, self.guidance) # AU/day², kg/day
        a_drag = np.array([0.0, 0.0, 0.0])  # Placeholder for drag acceleration

        # Total acceleration
        a = a_grav + a_thrust + a_drag

        return np.concatenate([v, a, [dm_dt]])

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
        log_guidance(sol, self.guidance, self.result)

        return self.result