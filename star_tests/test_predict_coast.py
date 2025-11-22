# test_predict_coast.py
"""
Unit and integration tests for _predict_coast.
Ensures:
- Correct coast prediction
- Handles t_end <= t_start
- Uses N-body acceleration
- Returns r, v in AU, AU/day
- Consistent with solve_ivp
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from star_guidance import _predict_coast
from star_dynamics import n_body_acceleration
from star_utils import sec_to_days, au3d2_km3s2, km_to_au, au_to_km, apd_kps
from star_constants import BODIES
from star_config import SECONDS_PER_DAY, DEFAULT_CONFIG

# ————————————————————————
# 1. UNIT TESTS
# ————————————————————————
def test_predict_coast_no_time():
    """t_end <= t_start → return current state."""
    t_start = 100.0
    t_end = 99.0
    y_rv = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1000.0])  # r, v, m
    r_pred, v_pred = _predict_coast(t_start, y_rv, t_end, [])
    assert np.allclose(r_pred, y_rv[:3])
    assert np.allclose(v_pred, y_rv[3:6])


def test_predict_coast_short_coast():
    """Short coast with Earth only."""
    t_start = 0.0
    t_end = sec_to_days(60)  # 60 seconds
    r_au = km_to_au(6778)
    r0 = np.array([r_au, 0.0, 0.0])  # LEO

    # CORRECT: mu in AU³/day², r in AU
    v_circ_au_day = np.sqrt(BODIES['earth']['mu'] / r_au)
    v0 = np.array([0.0, v_circ_au_day, 0.0])  # AU/day
    y_rv = np.concatenate([r0, v0, [1e6]])

    # Earth only
    r_pred, v_pred = _predict_coast(t_start, y_rv, t_end, ['earth'])

    # Expected: ~460 km in 60 s
    v_circ_km_s = apd_kps(v_circ_au_day)
    expected_dist_km = v_circ_km_s * 60
    assert au_to_km(np.linalg.norm(r_pred - r0)) == pytest.approx(expected_dist_km, abs=10)


# ————————————————————————
# 2. INTEGRATION TEST: Full N-body
# ————————————————————————
@pytest.fixture
def leo_state():
    """LEO state at t=0."""
    r_km = 6778
    r_au = km_to_au(r_km)
    v_circ_kms = np.sqrt(au3d2_km3s2(BODIES['earth']['mu']) / r_km)
    v_circ_au_day = v_circ_kms / km_to_au(1) * SECONDS_PER_DAY
    r0 = np.array([r_au, 0.0, 0.0])
    v0 = np.array([0.0, v_circ_au_day, 0.0])
    return np.concatenate([r0, v0, [1e6]]), r_au, v_circ_au_day


def test_predict_coast_nbody(leo_state):
    """Predict 1 hour coast with full N-body."""
    y_rv, r_au, v_au_day = leo_state
    t_start = 0.0
    t_end = sec_to_days(3600)  # 1 hour

    r_pred, v_pred = _predict_coast(t_start, y_rv, t_end, DEFAULT_CONFIG.all_bodies)

    # Distance traveled ~ v * t
    expected_dist_km = (v_au_day * 1.495978707e8 / SECONDS_PER_DAY) * 3600
    actual_dist_km = au_to_km(np.linalg.norm(r_pred - y_rv[:3]))
    assert actual_dist_km == pytest.approx(expected_dist_km, rel=0.01)

    # Velocity magnitude conserved (circular)
    assert np.linalg.norm(v_pred) == pytest.approx(v_au_day, rel=1e-6)


# ————————————————————————
# 3. REGRESSION: Compare with solve_ivp
# ————————————————————————
def test_predict_coast_vs_solve_ivp(leo_state):
    """_predict_coast must match solve_ivp."""
    y_rv, _, _ = leo_state
    t_start = 0.0
    t_end = sec_to_days(1800)  # 30 min

    # Manual solve_ivp
    sol = solve_ivp(
        fun=lambda t, y: np.concatenate([y[3:6], n_body_acceleration(t, y, ['earth']), [0.0]]),
        t_span=[t_start, t_end],
        y0=y_rv,
        method="DOP853",
        rtol=1e-9, atol=1e-9,
        max_step=60 / SECONDS_PER_DAY,
    )
    r_manual, v_manual = sol.y[:3, -1], sol.y[3:6, -1]

    # _predict_coast
    r_pred, v_pred = _predict_coast(t_start, y_rv, t_end, ['earth'])

    assert np.allclose(r_pred, r_manual, rtol=1e-9, atol=1e-12)
    assert np.allclose(v_pred, v_manual, rtol=1e-9, atol=1e-12)


# ————————————————————————
# Run with: pytest test_predict_coast.py -v
# ————————————————————————