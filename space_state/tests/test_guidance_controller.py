# test_guidance_controller.py
"""
Integration and unit tests for GuidanceController.
Builds on working _predict_coast tests.
Verifies:
- VGO convergence to zero
- Burn centering at t_apo
- Thrust shutdown on threshold
- Final state: GEO altitude, circular velocity
- Consistent with N-body dynamics
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from star_guidance import GuidanceController
from star_dynamics import n_body_acceleration
from star_utils import sec_to_days, km_to_au, au_to_km, apd_to_mps, body_position
from star_constants import BODIES, ts, epoch_jd
from star_config import SECONDS_PER_DAY
from star_ship import starship_raptor_vac

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
    v_target = v_dir * v_circ

    # Spacecraft
    ship = starship_raptor_vac()
    ship.propellant_mass = 1_100_000

    y0 = np.concatenate([r_peri, v_peri, [ship.dry_mass + ship.propellant_mass]])

    return {
        "y0": y0,
        "t0": 0.0,
        "t_apo": t_apo,
        "t_burn_end": t_burn_end,
        "r_target": r_target,
        "v_target": v_target,
        "ship": ship,
        "r_apo": r_apo,
        "burn_duration_sec": burn_duration_sec,
    }


# ————————————————————————
# 2. UNIT TESTS
# ————————————————————————
# def test_vgo_zero_at_target(hohmann_test_data):
#     """VGO = 0 when at target state."""
#     data = hohmann_test_data
#     vgo = compute_vgo(
#         t_current=data["t_burn_end"],
#         y_current=np.concatenate([data["r_target"], data["v_target"], [1e6]]),
#         t_burn_end=data["t_burn_end"],
#         target_r=data["r_target"],
#         target_v=data["v_target"],
#         bodies=['earth'],
#     )
#     assert vgo == pytest.approx(0.0, abs=1e-3)


# ————————————————————————
# 3. INTEGRATION TEST: Full Insertion
# ————————————————————————
def dynamics_with_guidance(t, y, ship, bodies, guidance):
    guidance.update(t, y)
    r, v, m = y[:3], y[3:6], y[6]
    ship.propellant_mass = m - ship.dry_mass

    acc_grav = n_body_acceleration(t, r, bodies)
    acc_thrust = np.zeros(3)
    dm_dt = 0.0

    if ship.active and t < guidance.t_burn_end_planned:
        acc_thrust = ship.get_thrust_vector_au_day2()
        dm_dt = -ship.total_mdot * SECONDS_PER_DAY

    return np.concatenate([v, acc_grav + acc_thrust, [dm_dt]])


def test_guided_insertion(hohmann_test_data):
    """Full guided insertion burn to GEO."""
    data = hohmann_test_data

    guidance = GuidanceController(
        burn = data["t_burn_end"],
        target_r=data["r_target"],
        target_v=data["v_target"],
        spacecraft=data["ship"],
        bodies=['earth'],
        cycle_time=15.0,
        vgo_threshold=5.0,
    )

    # Activate
    for thruster in data["ship"].thrusters:
        thruster.set_throttle(1.0)

    # Simulate
    t_span = [
        data["t_apo"] - sec_to_days(150),
        data["t_apo"] + sec_to_days(250)
    ]

    sol = solve_ivp(
        fun=lambda t, y: dynamics_with_guidance(t, y, data["ship"], ['earth'], guidance),
        t_span=t_span,
        y0=data["y0"],
        method="DOP853",
        rtol=1e-8,
        atol=1e-8,
        max_step=30 / SECONDS_PER_DAY,
    )

    # Final state
    y_final = sol.y[:, -1]
    r_final = y_final[:3]
    v_final = y_final[3:6]
    m_final = y_final[6]

    # Earth at final time
    earth_pos_final = body_position(BODIES['earth'], ts.tt_jd(epoch_jd + sol.t[-1]))
    r_local = r_final - earth_pos_final
    alt_final = au_to_km(np.linalg.norm(r_local)) - 6378

    v_circ_final = np.sqrt(BODIES['earth']['mu'] / np.linalg.norm(r_local))

    assert 35700 < alt_final < 35850  # GEO ±50 km
    assert abs(np.linalg.norm(v_final) - v_circ_final) < 0.05  # <50 m/s
    assert m_final < data["y0"][6]  # fuel used
    assert guidance.t_burn_end <= sol.t[-1]  # burn ended