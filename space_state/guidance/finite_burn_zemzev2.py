# guidance/finite_burn_zemzev2.py
"""
Finite-Burn ZEM/ZEV Guidance Controller (Version 2)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple

from .star_tools import estimate_burn_duration
from base.star_utils import sec_to_days, day_to_sec, ssb_to_body_centered
from base.star_logger import logger
from base.star_constants import world_time
from base.star_ship import Spacecraft
from base.star_structures import PropagationResult, GuidanceCycle, ControllerCycle

# ————————————————————————
# Powered prediction with mass flow
# ————————————————————————
def _predict_powered(
    t0: float,          # seconds
    y0: np.ndarray,     # [r(3), v(3), m(1)] in km, km/s, kg (body-centered)
    t_end: float,       # seconds
    thrust_dir: np.ndarray,  # unit vector (body-centered frame)
    ship: Spacecraft,
    BODY: dict,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Integrate with constant thrust direction and real mass flow."""

    def eom(t: float, y: np.ndarray) -> np.ndarray:
        r, v, m = y[:3], y[3:6], y[6]

        ship.update_propellant(m)
        ship.thrust_direction = thrust_dir
        ship.throttle = 1.0  # full throttle

        # Gravity
        a_grav = -BODY['mu_km3_s2'] * r / (np.linalg.norm(r) ** 3)  # km/s²
        a_thrust = ship.thrust_vector_km_s2                         # km/s²
        a = a_grav + a_thrust                                       # km/s²

        dm_dt = ship.mdot                                           # kg/s

        return np.concatenate([v, a, [dm_dt]])

    if t_end <= t0:
        logger.debug("Powered prediction: t_end <= t0, returning initial state.")
        return y0[:3], y0[3:6], y0[6]

    sol = solve_ivp(
        fun=eom,
        t_span=(t0, t_end),
        y0=y0,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step = min(10.0, (t_end - t0) / 50)
    )

    if not sol.success:
        logger.warning(f"Powered prediction integration failed: {sol.message}")

    yf = sol.y[:,-1]
    return yf[:3], yf[3:6], yf[6]


# ————————————————————————
# Correct finite-burn ZEM/ZEV VGO and steering
# ————————————————————————
def compute_vgo_and_direction(
    t: float,                    # seconds
    y: np.ndarray,               # [r,v,m] in km, km/s, kg
    t_burn_end: float,           # seconds
    target_r: np.ndarray,        # km
    target_v: np.ndarray,        # km/s
    ship: Spacecraft,
    BODY: dict,
    dir_guess: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Returns (vgo_m_s, unit_thrust_direction)
    Uses optimal λ = T_go / 2 and powered prediction with real mass depletion.
    """

    t_go = t_burn_end - t
    if t_go <= 0:
        return 0.0, dir_guess
    
    if dir_guess is None or np.linalg.norm(dir_guess) < 0.9:
        dir_guess = y[3:6] / (np.linalg.norm(y[3:6]) + 1e-12)

    # 2–3 fixed-point iterations is enough for convergence
    damping = 0.6  # 60% new, 40% old
    for i in range(5):
        pred_r, pred_v, _ = _predict_powered(t, y, t_burn_end, dir_guess, ship, BODY)

        zem = pred_r - target_r     # km
        zev = pred_v - target_v     # km/s

        dv_req = (zem / t_go) + zev  # km/s

        vgo_kms = np.linalg.norm(dv_req)
        new_dir = dv_req / (vgo_kms + 1e-12)

        # Damping to prevent divergence
        dir_guess = damping * new_dir + (1 - damping) * dir_guess
        dir_guess /= np.linalg.norm(dir_guess)

    vgo_mps = vgo_kms * 1000.0  # m/s

    #logger.debug(f"Computed VGO: {vgo_mps:.2f} m/s, Direction: {dir_guess}, t_go: {t_go:.2f} s, ZEM: {np.linalg.norm(zem):.2f} km, ZEV: {np.linalg.norm(zev):.4f} km/s")

    return vgo_mps, dir_guess


# ————————————————————————
# GuidanceController
# ————————————————————————
class FiniteBurnZEMZEV_V2:
    def __init__(
        self,
        target_r: np.ndarray,           # km - body centered
        target_v: np.ndarray,           # km/s - body centered
        delta_v_estimate: float,        # m/s
        init_t_apo: float,              # days
        ship: Spacecraft,
        central_body,
        vgo_threshold: float = 5.0,     # m/s
        cycle_time_sec: float = 1.0,    # seconds
    ):
        self.target_r = np.asarray(target_r)
        self.target_v = np.asarray(target_v)
        self.delta_v_estimate = delta_v_estimate
        self.init_t_apo = init_t_apo
        self.ship = ship
        self.ship_copy = ship.copy()  # For predictions
        self.central_body = central_body
        self.vgo_threshold = vgo_threshold
        self.cycle_time = sec_to_days(cycle_time_sec)

        self.enabled = True
        self.t_burn_end_planned = None  # days - dynamically updated burn end time
        self.last_update = -np.inf
        self.last_vgo = delta_v_estimate
        self.last_dir = np.zeros(3)
        self.last_throttle = 0.0
        self.last_horizon_update = -np.inf

        # Initial burn timing estimate
        burn_time_sec = estimate_burn_duration(
            self.last_vgo,
            self.ship.propellant_mass,
            self.ship.dry_mass,
            self.ship.max_thrust,
            self.ship.max_ve,
            self.ship.max_mdot,
            verbose=False
        )
        half_burn_days = sec_to_days(burn_time_sec / 2.0)
        self.t_burn_end_planned = self.init_t_apo + half_burn_days + sec_to_days(300)  # +5 min margin
        self.t_burn_start_planned = self.init_t_apo - half_burn_days - sec_to_days(300)

        logger.debug("FiniteBurnZEMZEV_V2 Guidance initialized.")
        logger.debug(f"  TARGET POSITION: {np.linalg.norm(self.target_r):.1f} km")
        logger.debug(f"  TARGET RADIUS: {np.linalg.norm(self.target_r):.1f} km")
        logger.debug(f"  TARGET VELOCITY MAGNITUDE: {np.linalg.norm(self.target_v):.3f} km/s")
        #logger.debug(f"  Estimated burn duration: {burn_time_sec:.2f} seconds")
        logger.debug(f"  Apoapsis time: {self.init_t_apo:.6f} days")
        logger.debug(f"  Planned burn start time: {self.t_burn_start_planned:.6f} days")
        logger.debug(f"  Planned burn end time: {self.t_burn_end_planned:.6f} days")
        logger.debug(f"  Delta-V Estimate: {self.delta_v_estimate:.2f} m/s")
    ##------------------------------------------------------------------

    # t: current time in days
    # y: current state in SSB frame [r(3), v(3), m(1)] in au, au/day, kg
    ##------------------------------------------------------------------
    def update(self, t: float, y_ssb: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """ZEM/ZEV Guidance Controller Update. """

        # check active
        if not self.active(t, y_ssb):
            self.last_vgo = 0.0
            self.last_throttle = 0.0
            return 0.0, np.array([0.0, 0.0, 1.0]), 0.0

        # frame transform to body-centered
        state = ssb_to_body_centered(world_time(t), y_ssb, self.central_body)
        self.ship_copy.update_propellant(y_ssb[6], True)  # sync mass for predictions

        # initialize last direction guess
        if np.linalg.norm(self.last_dir) < 0.9:  # Instead of sum() == 0.0
            self.last_dir = state['v_kms'] / (np.linalg.norm(state['v_kms']) + 1e-12)

        # cycle time check
        if t - self.last_update < self.cycle_time:
            direction_ssb = state['R_bc_to_ssb'] @ self.last_dir
            return self.last_vgo, direction_ssb, self.last_throttle
        self.last_update = t

        # time units conversion
        t_sec = day_to_sec(t)
        t_end = day_to_sec(self.t_burn_end_planned)

        # compute optimal direction and exact remaining Δv using current best horizon
        self.last_vgo, self.last_dir = compute_vgo_and_direction(
            t=t_sec,
            y=state['y_km_kms'],
            t_burn_end=t_end,
            target_r=self.target_r,
            target_v=self.target_v,
            ship=self.ship_copy,
            BODY=self.central_body,
            dir_guess=self.last_dir
        )

        self.last_throttle = 1.0 # full throttle

        # Update horizon with exact remaining burn time from rocket equation
        if (t - self.last_horizon_update) > sec_to_days(10.0) or self.last_horizon_update < 0.0:
            self.last_horizon_update = t
            remaining_sec = estimate_burn_duration(
                self.last_vgo,
                self.ship.propellant_mass,
                self.ship.dry_mass,
                self.ship.max_thrust,
                self.ship.max_ve,
                self.ship.max_mdot,
                verbose=False
            )
            if remaining_sec < 1e6:
                self.t_burn_end_planned = t + sec_to_days(remaining_sec)

        # Shutdown
        if self.last_vgo < self.vgo_threshold:
            #logger.info(f"ZEM/ZEV Burn complete. Final VGO = {self.last_vgo:.2f} m/s")
            self.last_vgo = 0.0
            self.last_throttle = 0.0

        direction_ssb = state['R_bc_to_ssb'] @ self.last_dir
        # Return VGO in m/s, direction unit vector, throttle (0–1)
        return self.last_vgo, direction_ssb, self.last_throttle
    
    ##------------------------------------------------------------------
    def active(self, t: float, y: np.ndarray) -> bool:
        """ current time in days - is the controller active """
        if not self.enabled:
            return False

        # Quick VGO estimate if not computed
        if self.last_vgo == 0.0:
            state = ssb_to_body_centered(world_time(t), y, self.central_body)  # Default state
            # Simple distance-based VGO estimate
            dist_km = np.linalg.norm(self.target_r - state['r_km'])
            self.last_vgo = dist_km / 10.0  # Rough m/s (adjust for scale)

        if self.last_vgo < self.vgo_threshold:
            return False

        burn_time_sec = estimate_burn_duration(
            self.last_vgo,
            self.ship.propellant_mass,
            self.ship.dry_mass,
            self.ship.max_thrust,
            self.ship.max_ve,
            self.ship.max_mdot,
            verbose=False
        )

        if burn_time_sec < 5.0:  # Min viable
            return False

        time_to_cutoff_sec = day_to_sec(self.t_burn_end_planned - t)
        margin_sec = max(30.0, burn_time_sec * 0.3)  # Scale margin
        return abs(time_to_cutoff_sec) <= (burn_time_sec / 2.0 + margin_sec)