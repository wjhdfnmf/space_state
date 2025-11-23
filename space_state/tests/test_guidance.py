# test_star_guidance.py
"""
Unit and integration tests for star_guidance.py
Uses:
- LEO 400 km circular orbit
- Hohmann transfer to GEO
- Starship with 3 Raptor vacuum engines
- N-body (Earth, Moon, Sun)
- ZEM/ZEV guidance with VGO
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from star_logger import logger

from star_guidance import (
    time_to_apoapsis,
    estimate_burn_duration,
    compute_vgo,
    zem_zev_guidance,
    GuidanceController,
)
from star_ship import starship_raptor_vac
from star_dynamics import n_body_acceleration
from star_utils import sec_to_days, km_to_au, au_to_km, apd_to_mps, body_position, body_velocity, calculate_eccentricity
from star_constants import BODIES, ts, epoch_jd
from star_states import get_orbiting_initial_state
from star_config import g0, SECONDS_PER_DAY, DEFAULT_CONFIG

ALL_BODIES = DEFAULT_CONFIG.all_bodies

# ————————————————————————
# INTEGRATION TEST: Hohmann Burn 2 (GEO Insertion)
# ————————————————————————
def hohmann_setup():
    # Initial state: 400 km LEO
    state = get_orbiting_initial_state(BODIES['earth'], 400)
    r0 = state.r    # AU in SSB frame
    v0 = state.v    # AU/day in SSB frame
    t0 = 0.0        # days

    ship = starship_raptor_vac()
    y0 = np.concatenate([r0, v0, [ship.total_mass]])

    # Reference Earth state
    earth_pos = body_position(BODIES['earth'], ts.tt_jd(epoch_jd + t0))
    earth_vel = body_velocity(BODIES['earth'], ts.tt_jd(epoch_jd + t0))

    logger.debug("Hohmann setup:")
    logger.debug(f" Initial state at t0 = {t0} days:")
    logger.debug(f"  r0 = {r0} AU SSB")
    logger.debug(f"  |r0| = {np.linalg.norm(r0):.6f} AU SSB frame")
    logger.debug(f"  Altitude = {au_to_km(np.linalg.norm(r0 - earth_pos) - BODIES['earth']['radius']):.1f} km above earth")
    logger.debug(f"  v0 = : {v0}")
    logger.debug(f"  |v0| = {np.linalg.norm(v0):.6f} AU/day SSB frame")
    logger.debug(f"  MU Earth = {BODIES['earth']['mu']} AU^3/day^2")
    logger.debug(f"  mass = {ship.total_mass:.1f} kg")

    # Target: GEO at apoapsis
    r1_au = np.linalg.norm(r0 - earth_pos)  # Current orbit radius
    r2_au = km_to_au(42164)   # GEO radius
    # Compute Δv1 (prograde)
    a_trans = (r1_au + r2_au) / 2
    v_circ_leo = np.sqrt(BODIES['earth']['mu'] / r1_au)
    v_trans_peri = np.sqrt(BODIES['earth']['mu'] * (2/r1_au - 1/a_trans))
    dv1 = v_trans_peri - v_circ_leo  # AU/day
    t1_burn = t0  # Instantaneous burn at t0

    logger.debug(f" Calculate insertion transfer orbit:  periapsis = {au_to_km(r1_au):.1f} km, apoapsis = {au_to_km(r2_au):.1f} km")
    logger.debug(f"  Target altitude = {au_to_km(r2_au - BODIES['earth']['radius']):.1f} km above earth")
    logger.debug(f"  Transfer semi-major axis: {au_to_km(a_trans):.1f} km")
    logger.debug(f"  v_circ_leo: {apd_to_mps(v_circ_leo):.1f} m/s")
    logger.debug(f"  Δv1: {apd_to_mps(dv1):.1f} m/s")

    # Relative state to Earth
    r0_rel = r0 - earth_pos  # AU
    v0_rel = v0 - earth_vel  # AU/day

    # Apply Δv1 in current velocity direction
    v_dir_rel = v0_rel / np.linalg.norm(v0_rel)
    v_transfer_rel = v0_rel + dv1 * v_dir_rel
    v_transfer = v_transfer_rel + earth_vel  # AU/day in SSB frame
    y1 = np.concatenate([r0, v_transfer, [ship.total_mass]])

    logger.debug(f" After impulsive burn to transfer orbit:")
    logger.debug(f"  v_transfer_rel = {v_transfer_rel} AU/day Earth-relative")
    logger.debug(f"  |v_transfer_rel| = {np.linalg.norm(v_transfer_rel):.6f} AU/day Earth-relative frame")
    logger.debug(f"  v_transfer = {v_transfer} AU/day SSB")
    logger.debug(f"  |v_transfer| = {np.linalg.norm(v_transfer):.6f} AU/day SSB frame")
    
    # Calculate eccentricity of transfer orbit
    e = calculate_eccentricity(r0_rel, v0_rel, BODIES['earth']['mu'])
    e_transfer = calculate_eccentricity(r0_rel, v_transfer_rel, BODIES['earth']['mu'])
    
    logger.debug(f"  eccentricity of current orbit = {e:.6f}")
    logger.debug(f"  eccentricity of transfer orbit = {e_transfer:.6f}")

    # 3. Time to apoapsis
    t_apo = time_to_apoapsis(r0_rel, v_transfer_rel, BODIES['earth']['mu']) + t0

    # 4. Propagate to apoapsis (coast)
    sol = solve_ivp(
        fun=lambda t, y: np.concatenate([y[3:6], n_body_acceleration(t, y[:6], ALL_BODIES), [0.0]]),
        t_span=[0.0, t_apo],
        y0=y1,
        method="DOP853",
        rtol=1e-9, atol=1e-9,
        max_step=60 / SECONDS_PER_DAY,
    )
    r_apo = sol.y[:3, -1]
    v_apo = sol.y[3:6, -1]

    earth_pos_new = body_position(BODIES['earth'], ts.tt_jd(epoch_jd + t_apo))
    earth_vel_new = body_velocity(BODIES['earth'], ts.tt_jd(epoch_jd + t_apo))

    logger.debug(f" Apoapsis state at t_apo = {t_apo:.2f} days:")
    logger.debug(f"  r_apo = {r_apo} AU SSB")
    logger.debug(f"  |r_apo| = {np.linalg.norm(r_apo):.6f} AU SSB frame")
    logger.debug(f"  Altitude = {au_to_km(np.linalg.norm(r_apo - earth_pos_new) - BODIES['earth']['radius']):.1f} km above earth")
    logger.debug(f"  v_apo = {v_apo} AU/day SSB")
    logger.debug(f"  |v_apo| = {np.linalg.norm(v_apo):.6f} AU/day SSB frame")
    logger.debug(f" Position and velocity relative to Earth:")
    logger.debug(f"  r_apo_rel = {r_apo - earth_pos_new} AU")
    logger.debug(f"  v_apo_rel = {v_apo - earth_vel_new} AU/day")

    logger.debug(f" r2au = {r2_au} AU (GEO radius)")

    # Approximate GEO state (circular, equatorial)
    r_apo_rel = r_apo - earth_pos_new
    r_target = earth_pos_new + (r_apo_rel / np.linalg.norm(r_apo_rel)) * r2_au # AU
    v_circ = np.sqrt(BODIES['earth']['mu'] / r2_au) # AU/day
    v_dir = np.cross([0, 0, 1], r_target - earth_pos_new) # AU/day
    v_target = earth_vel_new + v_dir / np.linalg.norm(v_dir) * v_circ # AU/day

    logger.debug("Hohmann setup complete.")
    logger.debug(f"  t_apo: {t_apo:.2f} days")
    logger.debug(f"  r_target (AU): {r_target}")
    logger.debug(f"  v_target (AU/day): {v_target}")
    logger.debug(f" v_circ at GEO: {apd_to_mps(v_circ):.1f} m/s")

    dr2 = au_to_km(np.linalg.norm(r_target - r_apo))  # km
    dv2 = apd_to_mps(np.linalg.norm(v_target - v_apo))  # m/s
    burn_time_sec = estimate_burn_duration(dv2, ship.total_thrust, ship.total_isp, ship.total_mass, 1.0, ship.dry_mass)
    burn_time_days = sec_to_days(burn_time_sec)

    logger.debug("Delta r and v to target at apoapsis:")
    logger.debug(f"  Δr (km): {dr2:.1f} km")
    logger.debug(f"  Δv (m/s): {dv2:.1f} m/s")
    logger.debug(f" Estimated burn duration for circularization burn:")
    logger.debug(f"  burn_time: {burn_time_sec:.1f} seconds")
    logger.debug(f"  burn_time: {burn_time_sec/60:.1f} minutes")
    logger.debug("-----------------------------------------")
    logger.debug("")

    return {
        "y0": y1,
        "t0": t1_burn,
        "t_apo": t_apo,
        "r_target": r_target,
        "v_target": v_target,
        "ship": ship,
        "burn_time": burn_time_days,
    }


def dynamics_with_guidance(t, y, ship, bodies, guidance):
    guidance.update(t, y)
    r, v, m = y[:3], y[3:6], y[6]

    acc_grav = n_body_acceleration(r, t, bodies)
    acc_thrust = np.zeros(3)
    dm_dt = 0.0

    ship.propellant_mass = m - ship.dry_mass
    if ship.active and t < guidance.t_burn_end:
        acc_thrust = ship.get_thrust_vector_au_day2()
        dm_dt = -ship.total_mdot * SECONDS_PER_DAY

    return np.concatenate([v, acc_grav + acc_thrust, [dm_dt]])


def test_hohmann_insertion_guided():
    data = hohmann_setup()

    guidance = GuidanceController(
        t_apo = data["t_apo"],      # in days
        burn = data["burn_time"],   # in days
        target_r=data["r_target"],  # in AU
        target_v=data["v_target"],  # in AU/day
        spacecraft=data["ship"],    # Starship
        bodies=ALL_BODIES,          # list of bodies for N-body accel
        cycle_time=15.0,            # in seconds
        vgo_threshold=3.0,          # in m/s
    )
    logger.debug("Starting guided Hohmann insertion test.")

    # Activate engines
    for thruster in data["ship"].thrusters:
        thruster.set_throttle(1.0)

    logger.debug(f" Thruster total thrust: {data['ship'].total_thrust:.1f} N")

    # Define burn timing
    t_start = data["t_apo"] - (data["burn_time"] / 2 + sec_to_days(50))
    t_end = data["t_apo"] + (data["burn_time"] / 2 + sec_to_days(100))

    # Simulate burn
    sol = solve_ivp(
        fun=lambda t, y: dynamics_with_guidance(t, y, data["ship"], ALL_BODIES, guidance),
        t_span=[t_start, t_end],
        y0=data["y0"],
        method="DOP853",
        rtol=1e-8,
        atol=1e-8,
        max_step=30 / SECONDS_PER_DAY,
    )

    logger.debug("Simulation complete.")

    return

    # Final state
    y_final = sol.y
    r_final = y_final[:3, -1]
    v_final = y_final[3:6, -1]

    logger.debug(f" Final state at t = {sol.t[-1]:.2f} days:")
    logger.debug(f"  r_final = {r_final} AU SSB")
    logger.debug(f"  |r_final| = {np.linalg.norm(r_final):.6f} AU SSB frame")
    logger.debug(f"  v_final = {v_final} AU/day SSB")
    logger.debug(f"  |v_final| = {np.linalg.norm(v_final):.6f} AU/day SSB frame")   

    # Validate
    r_norm_km = np.linalg.norm(r_final) * 1.495978707e8 / 1000
    altitude_final = r_norm_km - BODIES['earth']['radius']
    v_norm = np.linalg.norm(v_final) * 1.495978707e8 / SECONDS_PER_DAY / 1000  # km/s
    v_circ_target = np.sqrt(BODIES['earth']['mu'] / r_norm_km)

    assert 35700 < altitude_final < 35850  # GEO ±50 km
    assert abs(v_norm - v_circ_target) < 0.05  # <50 m/s
    assert guidance.t_burn_end <= sol.t[-1]  # burn ended

if __name__ == "__main__":
    test_hohmann_insertion_guided()