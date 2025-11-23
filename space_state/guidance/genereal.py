# star_guidance.py
"""
ZEM/ZEV-based finite-burn guidance with real-time VGO and burn duration estimation.

Key Features:
- Dynamic burn duration via Tsiolkovsky + VGO
- Closed-loop steering via ZEM/ZEV
- Auto-shutdown on VGO threshold
- N-body prediction using DOP853
- Modular, readable, and robust
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple

from star_dynamics import n_body_acceleration
from star_utils import sec_to_days, apd_to_mps, days_to_seconds
from star_config import SECONDS_PER_DAY, g0
from star_logger import logger


# ————————————————————————
# Time to Apoapsis Calculation
# From Current State to Apoapsis relative to Central Body
# ————————————————————————
def time_to_apoapsis(r_current, v_current, mu):
    """
    Compute time from current state to apoapsis of current orbit.
    r_current, v_current: AU, AU/day
    mu: AU³/day²
    Returns: days
    """
    r = np.linalg.norm(r_current)
    v = np.linalg.norm(v_current)
    vr = np.dot(r_current, v_current) / r  # radial velocity

    # Specific energy
    epsilon = v**2 / 2 - mu / r

    # Semi-major axis
    a = -mu / (2 * epsilon)

    # Eccentricity
    h = np.cross(r_current, v_current)
    h_mag = np.linalg.norm(h)
    e_vec = np.cross(v_current, h) / mu - r_current / r
    e = np.linalg.norm(e_vec)

    # Mean anomaly at current time
    cos_E = (1 - r / a) / e
    sin_E = (vr * np.sqrt(a * (1 - e**2))) / (e * np.sqrt(mu * a))
    E = np.arctan2(sin_E, cos_E)
    if E < 0:
        E += 2 * np.pi

    M = E - e * np.sin(E)

    # Mean anomaly at apoapsis (E = π)
    M_apo = np.pi - e * np.sin(np.pi)

    # Time to apoapsis
    n = np.sqrt(mu / a**3)  # mean motion
    delta_M = M_apo - M
    if delta_M < 0:
        delta_M += 2 * np.pi
    t_to_apo = delta_M / n

    return t_to_apo


# ————————————————————————
# 1. BURN DURATION ESTIMATION
# ————————————————————————
def estimate_burn_duration(
    dv_req: float,          # in m/s
    thrust_max: float,      # in N
    isp: float,             # in s    
    m0: float,              # in kg
    throttle: float = 1.0,  # 0.0 to 1.0
    dry_mass: float = 0.0,  # in kg
) -> float:
    """
    Estimate finite burn duration in **seconds** using the rocket equation.
    Returns: Burn time in seconds. Returns 0.0 if no burn needed, np.inf if impossible.
    """
    logger.debug("Estimating burn duration.")
    logger.debug(f"Δv required: {dv_req} m/s")

    if dv_req <= 0:
        logger.debug("No Δv required; burn duration = 0.")
        return 0.0

    if m0 <= dry_mass:
        logger.warning("No propellant available (m0 ≤ dry_mass).")
        return 0.0

    ve = isp * g0
    thrust = thrust_max * throttle
    if thrust <= 0:
        logger.error("Thrust ≤ 0; infinite burn time.")
        return np.inf

    mf = m0 * np.exp(-dv_req / ve)
    if mf < dry_mass:
        logger.warning("Insufficient propellant for Δv; infinite burn time.")
        return np.inf

    delta_m = m0 - mf
    if delta_m <= 0:
        logger.debug("Δv already achieved; no burn needed.")
        return 0.0

    mdot = thrust / ve
    burn_time_sec = delta_m / mdot
    return burn_time_sec


# ————————————————————————
# REVISED: Powered prediction with mass flow
# ————————————————————————
def _predict_powered(
    t0: float,          # days
    y0: np.ndarray,     # [r(3), v(3), m(1)] in AU, AU/day, kg
    t_end: float,       # days
    thrust_dir: np.ndarray,  # unit vector (constant during segment)
    spacecraft,
    bodies,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Integrate with constant thrust direction and real mass flow."""
    ve = spacecraft.total_isp * g0
    thrust_mag = spacecraft.total_thrust  # N (assumes throttle=1.0 until shutdown)
    mdot = -thrust_mag / ve                    # kg/s (negative)

    def equations_of_motion(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:6]
        m = y[6]

        a_grav = n_body_acceleration(t, y, bodies)  # AU/day²
        a_thrust = (thrust_mag / m) * thrust_dir * (1 / SECONDS_PER_DAY**2)  # AU/day²
        # Convert N/kg → AU/day²: 1 m/s² = 1 / 149597870700 / (SECONDS_PER_DAY²) AU/day²
        # But we pre-converted in utils or absorb into constant if you add:
        a_thrust_au_per_day2 = a_thrust * (1.0 / 149597870700) * SECONDS_PER_DAY**2

        return np.concatenate([
            v,
            a_grav + a_thrust_au_per_day2,
            [mdot]                               # dm/dt
        ])

    if t_end <= t0:
        return y0[:3], y0[3:6], y0[6]

    sol = solve_ivp(
        fun=equations_of_motion,
        t_span=(t0, t_end),
        y0=y0,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=10 / SECONDS_PER_DAY,
    )
    yf = sol.y[:,-1]
    return yf[:3], yf[3:6], yf[6]


# ————————————————————————
# REVISED: Correct finite-burn ZEM/ZEV VGO and steering
# ————————————————————————
def compute_vgo_and_direction(
    t: float,
    y: np.ndarray,               # [r,v,m]
    t_burn_end: float,
    target_r: np.ndarray,
    target_v: np.ndarray,
    spacecraft,
    bodies,
) -> Tuple[float, np.ndarray]:
    """
    Returns (vgo_m_s, unit_thrust_direction)
    Uses optimal λ = T_go / 2 and powered prediction with real mass depletion.
    """
    T_go_days = t_burn_end - t
    if T_go_days <= 0:
        return 0.0, np.array([0, 0, 1.0])

    T_go_sec = days_to_seconds(T_go_days)

    # First guess direction = last known or ZEM-only
    pred_r, pred_v, _ = _predict_powered(t, y, t_burn_end, np.array([0,0,1.0]), spacecraft, bodies)
    initial_correction = -(pred_r - target_r)  # rough
    dir_guess = initial_correction / (np.linalg.norm(initial_correction) + 1e-12)

    # 1–2 fixed-point iterations is enough for convergence
    for _ in range(2):
        pred_r, pred_v, m_final = _predict_powered(t, y, t_burn_end, dir_guess, spacecraft, bodies)

        zem = pred_r - target_r
        zev = pred_v - target_v

        # Optimal λ for constant thrust + linear gravity = T_go / 2
        lam = T_go_sec / 2.0

        correction = zem + lam * zev * (SECONDS_PER_DAY)  # make units match: zev in AU/day → AU
        dir_guess = -correction / (np.linalg.norm(correction) + 1e-15)

    # Final precise prediction with converged direction
    pred_r, pred_v, m_final = _predict_powered(t, y, t_burn_end, dir_guess, spacecraft, bodies)
    zem = pred_r - target_r
    zev = pred_v - target_v
    lam = T_go_sec / 2.0
    total_correction = zem + lam * zev * SECONDS_PER_DAY
    vgo_au_per_day = np.linalg.norm(total_correction) / T_go_sec
    vgo_m_s = apd_to_mps(vgo_au_per_day)

    return vgo_m_s, dir_guess


# ————————————————————————
# REVISED: GuidanceController (now stable and accurate)
# ————————————————————————
class GuidanceController:
    def __init__(
        self,
        burn: float,                     # in days
        target_r: np.ndarray,
        target_v: np.ndarray,
        spacecraft,
        bodies,
        vgo_threshold: float = 5.0,      # m/s
        cycle_time: float = 5.0,         # seconds
    ):
        self.planned_burn_end = burn
        self.target_r = np.asarray(target_r)
        self.target_v = np.asarray(target_v)
        self.spacecraft = spacecraft
        self.bodies = bodies
        self.vgo_threshold = vgo_threshold
        self.cycle_time = sec_to_days(cycle_time)

        # Start with rough burn time, will be corrected every cycle
        self.t_burn_end_planned = None
        self.last_update = -np.inf

    def update(self, t: float, y: np.ndarray) -> None:
        if t - self.last_update < self.cycle_time:
            return
        self.last_update = t

        # First cycle: initialize horizon with initial guess
        if self.t_burn_end_planned is None:
            self.t_burn_end_planned = self.planned_burn_end

        # Core: compute optimal direction and exact remaining Δv using current best horizon
        vgo, direction = compute_vgo_and_direction(
            t=t,
            y=y,
            t_burn_end=self.t_burn_end_planned,
            target_r=self.target_r,
            target_v=self.target_v,
            spacecraft=self.spacecraft,
            bodies=self.bodies,
        )

        # Update horizon with exact remaining burn time from rocket equation
        remaining_sec = estimate_burn_duration(
            dv_req=vgo,
            thrust_max=self.spacecraft.total_thrust,
            isp=self.spacecraft.total_isp,
            m0=y[6],
            dry_mass=self.spacecraft.dry_mass,
            throttle=1.0,
        )
        self.t_burn_end_planned = t + sec_to_days(remaining_sec)

        # Shutdown
        if vgo < self.vgo_threshold or remaining_sec < 1.0:
            self.spacecraft.set_all_thrusters_throttle(0.0)
            logger.info(f"Burn complete. Final VGO = {vgo:.2f} m/s")
            return

        # Apply
        self.spacecraft.set_thrust_direction(direction)
        # throttle remains 1.0 until shutdown