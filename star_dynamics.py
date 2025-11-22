# star_dynamics.py

import time
from typing import Callable, Optional, Sequence, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from star_constants import BODIES, ts, epoch_jd
from star_structures import PropagationResult, StateVector
from star_config import DEFAULT_CONFIG
from star_utils import days_to_seconds
from star_ship import Spacecraft

from star_logger import logger

# ------------------- 1. ACCELERATION COMPONENTS -------------------

def n_body_acceleration(
        t: float,            # in days
        y: np.ndarray,       # full state vector [AU, AU/day]
        bodies: list         # list of body names for N-body accel
    ) -> np.ndarray:
    """Compute N-body gravitational acceleration"""

    r = y[:3]
    time = ts.tt_jd(epoch_jd + t)
    accel = np.zeros(3, dtype=np.float64)

    for body_name in bodies:
        body = BODIES[body_name]["skyfield"]
        gm = BODIES[body_name]["mu"]
        radius = BODIES[body_name]["radius"] or 0.0
        if body is None or gm is None:
            logger.error("n_body_accel: body not found - " + body_name)

        r_body = body.at(time).position.au
        dr = r - r_body
        dr_norm = np.linalg.norm(dr)

        if dr_norm > radius:
            accel += -gm * dr / (dr_norm ** 3)
        else:
            logger.warning(f"n_body_accel: ship too close to {body_name} (d={dr_norm:.1f} AU)")

    return accel

def thrust_acceleration(
        throttle: float,     # 0 to 1
        ship: Spacecraft
    ) -> np.ndarray:
    """Compute thrust acceleration vector in AU/day²"""
    if throttle <= 0.0 or ship.thrust == 0.0:
        return np.zeros(3, dtype=np.float64)
    # Convert thrust from N to AU/day²
    thrust_accel = (ship.thrust) / (ship.mass * 1.496e+11) * 86400**2
    return thrust_accel * ship.thrust_direction

# ------------------- 2. EOM FACTORY -------------------

def eom_factory(bodies, ship):

    def eom(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:6]
        m = y[6]

        a = n_body_acceleration(t, y, bodies)

        dm_dt = 0.0  # No thrust for now

        return np.concatenate([v, a, [dm_dt]])

    return eom


# ------------------- 3. PROPAGATION WITH VERBOSE LOGGING -------------------

class Propagator:
    guidance: FiniteBurnZEMZEV
    ship: Spacecraft
    bodies: Sequence[str]
    result: PropagationResult

    def dynamics(self, t, y):
        vgo, thrust_dir, throttle = self.guidance.update(t, y)

        a_grav = n_body_acceleration(t, y, self.bodies)
        

def propagate_orbit(
    state0: StateVector,                            # initial state vector [AU, AU/day]
    t_duration: float,                              # final time in days
    ship: Spacecraft,                               # spacecraft
    bodies: Sequence[str],                          # list of body names for N-body accel
    events: Optional[Sequence[Callable]] = None
) -> PropagationResult:

    rtol = DEFAULT_CONFIG.rtol  
    atol = DEFAULT_CONFIG.atol 
    max_step = DEFAULT_CONFIG.max_step

    logger.info("Starting orbit propagation")
    logger.info(f"  N-body bodies: {bodies or []}")
    logger.debug(f"  Integrator: DOP853, rtol={rtol:.0e}, atol={atol:.0e}, "
                 f"max_step={max_step}")

    y0 = np.hstack([state0.r, state0.v, state0.m]).astype(np.float64)
    t0 = state0.t
    tf = t0 + t_duration
    eom = eom_factory(bodies, ship)

    start_time = time.perf_counter()

    sol = solve_ivp(
        fun=eom,
        t_span=(t0, tf),
        y0=y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=events,
        dense_output=False,
    )

    elapsed = (time.perf_counter() - start_time) / 60.0  # minutes
    n_steps = sol.t.size

    if not sol.success:
        logger.error(f"Integration FAILED: {sol.message}")
        raise RuntimeError(f"Integration failed: {sol.message}")

    logger.info(f"Integration SUCCESS: {n_steps} steps in {elapsed:.2f} min "
                f"({n_steps/elapsed:.1f} steps/s)")

    # Compose result
    result = PropagationResult(
        t=sol.t,
        y=sol.y,
        events=[],
    )

    return result

