# test_finite_burn_zemzev.py
import pytest
import numpy as np
from scipy.integrate import solve_ivp

from finite_burn_zemzev import FiniteBurnZEMZEV
from star_dynamics import n_body_acceleration
from star_utils import sec_to_days, au_to_km, body_position, body_velocity, km_to_au, mps2_apd2, convert_gm, calculate_eccentricity, apd_to_mps
from star_constants import BODIES, ts, epoch_jd, c
from star_ship import starship_raptor_vac
from star_config import SECONDS_PER_DAY, DEFAULT_CONFIG
from star_structures import PropagationResult, combine_propagations
from star_states import get_orbiting_initial_state
from star_logger import logger
from star_vis2D import plot_orbital_metrics
from star_guidance import time_to_apoapsis, estimate_burn_duration

ALL_BODIES = DEFAULT_CONFIG.all_bodies
#ALL_BODIES = ['earth']

# ————————————————————————
# 1. FIXTURES
# ————————————————————————
@pytest.fixture
def hohmann_test_data():
    """Setup LEO → GEO transfer with correct target at t_burn_end."""

    # LEO
    r1_km = 6778
    r2_km = 42164
    r1_au = km_to_au(r1_km)
    r2_au = km_to_au(r2_km)
    a_trans = (r1_au + r2_au) / 2

    # Transfer time
    tof_day = np.pi * np.sqrt(a_trans**3 / BODIES['earth']['mu'])
    t_apo = tof_day

    # Initial state: circular LEO at periapsis
    r_peri = np.array([r1_au, 0.0, 0.0])
    v_peri = np.array([0.0, np.sqrt(BODIES['earth']['mu'] * (2/r1_au - 1/a_trans)), 0.0])

    # Earth position
    earth_pos = body_position(BODIES['earth'], ts.tt_jd(epoch_jd))
    earth_vel = body_velocity(BODIES['earth'], ts.tt_jd(epoch_jd))

    r_peri += earth_pos
    v_peri += earth_vel

    # Propagate to apoapsis
    sol = solve_ivp(
        fun=lambda t, y: np.concatenate([y[3:6], n_body_acceleration(t, y, ['earth']), [0.0]]),
        t_span=[0.0, t_apo],
        y0=np.concatenate([r_peri, v_peri, [1e6]]),
        method="DOP853",
        rtol=1e-9, atol=1e-9,
        max_step=60 / SECONDS_PER_DAY,
    )
    r_apo = sol.y[:3, -1]
    v_apo = sol.y[3:6, -1]

    # Target at t_burn_end (centered)
    burn_duration_sec = 190
    t_burn_end = t_apo + sec_to_days(burn_duration_sec / 2)

    # Earth at t_burn_end
    earth_pos_end = body_position(BODIES['earth'], ts.tt_jd(epoch_jd + t_burn_end))

    # Local GEO vector
    r_apo_local = r_apo - earth_pos_end
    r_target_local = r_apo_local / np.linalg.norm(r_apo_local) * r2_au
    r_target = earth_pos_end + r_target_local

    # Circular velocity
    v_circ = np.sqrt(BODIES['earth']['mu'] / r2_au)
    v_dir = np.cross(r_target, [0, 0, 1])
    v_dir = v_dir / np.linalg.norm(v_dir)
    v_target = v_dir * v_circ + body_velocity(BODIES['earth'], ts.tt_jd(epoch_jd + t_burn_end))

    # Spacecraft
    ship = starship_raptor_vac()
    ship.propellant_mass = 1_100_000

    y0 = np.concatenate([r_peri, v_peri, [ship.dry_mass + ship.propellant_mass]])

    logger.debug("Hohmann transfer test data prepared.")
    logger.debug(f" y0 = {y0}  AU, AU/day, kg")
    logger.debug(f" |y0 r| = {np.linalg.norm(y0[:3]):.6f} AU")
    logger.debug(f" |y0 v| = {np.linalg.norm(y0[3:6]):.6f} AU/day")
    logger.debug(f" t0 = {0.0} days")
    logger.debug(f" t_apo = {t_apo:.2f} days")
    logger.debug(f" burn_duration_sec = {burn_duration_sec:.1f} seconds")
    logger.debug(f" t_burn_end = {t_burn_end:.2f} days")
    logger.debug(f" r_target = {r_target} AU")
    logger.debug(f" |r_target| = {np.linalg.norm(r_target):.6f} AU")
    logger.debug(f" v_target = {v_target} AU/day")  
    logger.debug(f" |v_target| = {np.linalg.norm(v_target):.6f} AU/day")
    logger.debug("-----------------------------------------")


    return {
        "y0": y0,
        "t0": 0.0,
        "t_apo": t_apo,
        "t_burn_end": t_burn_end,
        "r_target": r_target,
        "v_target": v_target,
        "ship": ship,
    }

@pytest.fixture
def guidance_new(hohmann_test_data):
    ship = hohmann_test_data["ship"]
    # Ensure full throttle at start
    ship.set_throttle(1.0)

    guidance = FiniteBurnZEMZEV(
        target_r=hohmann_test_data["r_target"],
        target_v=hohmann_test_data["v_target"],
        spacecraft=ship,
        bodies=['earth'],
        vgo_threshold=3.0,        # tighter than default
        cycle_sec=2.0,       # seconds
    )

    logger.debug("FiniteBurnZEMZEV guidance initialized for new interface test.")
    return guidance, hohmann_test_data

def dynamics_with_new_guidance(t, y, ship, bodies, guidance):
    vgo, thrust_dir = guidance.update(t, y)  # ← new interface
    #logger.debug(f"t={t:.2f} days: vgo={vgo:.2f} m/s, thrust_dir={thrust_dir}")
    r, v, m = y[:3], y[3:6], y[6]
    ship.propellant_mass = m - ship.dry_mass

    acc_grav = n_body_acceleration(t, y, bodies)
    acc_thrust = np.zeros(3)
    dm_dt = 0.0

    if ship.propellant_mass > 0 and vgo > guidance.vgo_thr:
        F = ship.total_thrust
        a_thrust_mps2 = (F / m) * thrust_dir                # m/s²
        acc_thrust = mps2_apd2(a_thrust_mps2)                    # AU/day²
        dm_dt = -ship.total_mdot
    
    #logger.debug(f"  acc_grav={acc_grav}, acc_thrust={acc_thrust}, dm_dt={dm_dt}")
    #logger.debug(f"  propellant_mass={ship.propellant_mass:.1f} kg, total_mass={m:.1f} kg")

    return np.concatenate([v, acc_grav + acc_thrust, [dm_dt]])

def test_geo_insertion_new_guidance(guidance_new):
    guidance, data = guidance_new
    ship = data["ship"]

    logger.debug("Starting new ZEM/ZEV guided Hohmann insertion test.")
    logger.debug(f" Thruster total thrust: {ship.total_thrust:.1f} N")
    logger.debug(f" Initial propellant mass: {ship.propellant_mass:,.0f} kg")
    logger.debug(f" Initial total mass: {ship.total_mass:,.0f} kg")
    logger.debug(f" Total mdot: {ship.total_mdot:.3f} kg/s")
    logger.debug(f" a_thrust mps2 at start: {(ship.total_thrust / ship.total_mass):.6f} m/s²")
    logger.debug(f" a_thrust au/day2 at start: {mps2_apd2(ship.total_thrust / ship.total_mass):.6e} AU/day²")
    logger.debug("-----------------------------------------")

    # Start 3 minutes before apogee
    t_burn_start = data["t_apo"] - sec_to_days(180)
    # already at t=0, so propagate coast to t_start first
    sol_coast = solve_ivp(
        lambda t, y: np.concatenate([y[3:6], n_body_acceleration(t, y, ['earth']), [0.0]]),
        (data['t0'], t_burn_start),
        data['y0'],
        method="DOP853",
        rtol=1e-10, atol=1e-12,
    )
    y_start = sol_coast.y[:, -1]

    result_coast = PropagationResult(
        t=sol_coast.t,
        y=sol_coast.y,
        events=[],
        conservation={}
    )

    logger.debug(f" Propagated to burn start at t={t_burn_start:.2f} days.")
    logger.debug(f"  r_start = {y_start[:3]} AU")
    logger.debug(f"  |r_start| = {np.linalg.norm(y_start[:3]):.6f} AU")
    logger.debug(f"  v_start = {y_start[3:6]} AU/day")
    logger.debug(f"  |v_start| = {np.linalg.norm(y_start[3:6]):.6f} AU/day")
    logger.debug(f"  m_start = {y_start[6]:,.1f} kg")
    logger.debug("-----------------------------------------")

    sol = solve_ivp(
        fun=lambda t, y: dynamics_with_new_guidance(t, y, ship, ['earth'], guidance),
        t_span=(t_burn_start, t_burn_start + sec_to_days(900)),  # 15 min window
        y0=y_start,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=10 * sec_to_days(1),
        events=lambda t, y: guidance.vgo_thr - guidance.update(t, y)[0],  # stop on shutdown
    )

    # Final state
    r_f = sol.y[:3, -1]
    v_f = sol.y[3:6, -1]
    m_f = sol.y[6, -1]
    t_f = sol.t[-1]

    result_burn = PropagationResult(
        t=sol.t,
        y=sol.y,
        events=[],
        conservation={}
    )

    logger.debug(f" Propagation complete at t={t_f:.2f} days.")
    logger.debug(f"  r_f = {r_f} AU")
    logger.debug(f"  |r_f| = {np.linalg.norm(r_f):.6f} AU")
    logger.debug(f"  v_f = {v_f} AU/day")
    logger.debug(f"  |v_f| = {np.linalg.norm(v_f):.6f} AU/day")
    logger.debug(f"  m_f = {m_f:,.1f} kg")
    logger.debug(f"  m_i = {data['y0'][6]:,.1f} kg")
    logger.debug("-----------------------------------------")

    earth_pos = body_position(BODIES['earth'], ts.tt_jd(epoch_jd + t_f))
    r_local = r_f - earth_pos
    alt_km = au_to_km(np.linalg.norm(r_local) - BODIES['earth']['radius'])
    v_mag = np.linalg.norm(v_f)
    v_circ = np.sqrt(BODIES['earth']['mu'] / np.linalg.norm(r_local))

    logger.debug("New ZEM/ZEV guided insertion complete.")
    logger.debug(f" Earth position at t_f: {earth_pos} AU")
    logger.debug(f" r_local: {r_local} AU")
    logger.debug(f" r_local norm: {np.linalg.norm(r_local):.6f} AU")
    logger.debug(f" Earth mu: {BODIES['earth']['mu']} AU³/day²")
    logger.debug(f" Final altitude: {alt_km:.2f} km")
    logger.debug(f" Final velocity: {v_f} AU/day")
    logger.debug(f" Final velocity magnitude: {v_mag:.6f} AU/day")
    logger.debug(f" Target circular velocity: {v_circ:.6f} AU/day")
    logger.debug(f" Velocity error: {(v_mag - v_circ)*1000:.1f} m/s")
    logger.debug(f" Burn center offset from apoapsis: {(t_f - data['t_apo'])*86400:+.1f} s")
    logger.debug(f" Propellant used: {data['y0'][6] - m_f:,.0f} kg")
    logger.debug(f" Earth radius: {BODIES['earth']['radius']:.6f} AU")
    logger.debug(f" Computed radius: {np.linalg.norm(r_f):.6f} AU")

    # Continue to propagate coast after burn for visualization
    t_coast_end = t_f + 4   # 4 days later
    t_start = t_f
    y0 = sol.y[:, -1]
    sol_coast2 = solve_ivp(
        lambda t, y: np.concatenate([y[3:6], n_body_acceleration(t, y, ['earth']), [0.0]]),
        (t_start, t_coast_end),
        y0,
        method="DOP853",
        rtol=1e-10, atol=1e-12,
    )
    result_coast_2 = PropagationResult(
        t=sol_coast2.t,
        y=sol_coast2.y,
        events=[],
        conservation={}
    )

    logger.debug(f" Propagated coast to t={t_coast_end:.2f} days for visualization.")
    y_end = sol_coast2.y[:, -1]
    logger.debug(f"  r_end = {y_end[:3]} AU")
    logger.debug(f"  |r_end| = {np.linalg.norm(y_end[:3]):.6f} AU")
    logger.debug(f"  v_end = {y_end[3:6]} AU/day")
    logger.debug(f"  |v_end| = {np.linalg.norm(y_end[3:6]):.6f} AU/day")
    logger.debug(f"  m_end = {y_end[6]:,.1f} kg")
    logger.debug("-----------------------------------------")


    result = combine_propagations([result_coast, result_burn, result_coast_2])

    plot_orbital_metrics(
        result,
        bodies_to_plot=['earth'],
        title="Finite Burn ZEM/ZEV - New Interface Test"
    )

    # ------------------ ASSERTIONS ------------------
    assert 35786 - 30 < alt_km < 35786 + 30, f"Altitude error: {alt_km - 35786:+.1f} km"
    assert abs(v_mag - v_circ) < 0.005, f"Velocity error: {v_mag - v_circ:.3f} km/s"
    assert m_f < data["y0"][6] - 10000, "Insufficient propellant used"
    assert abs(t_f - data["t_apo"]) < sec_to_days(20), "Burn not centered at apogee"
    assert guidance.thrust_dir @ guidance.thrust_dir == pytest.approx(1.0, abs=1e-12)
