"""
Tests for Proximity Operations and Rendezvous Guidance
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from guidance.proximity_ops import (
    ApproachMode, RelativeState, ManeuverPlan,
    ProximityGuidanceConfig, ProximityGuidance,
    inertial_to_lvlh, lvlh_to_inertial, get_relative_state,
    cw_state_transition, cw_delta_v_to_target,
    compute_vbar_approach, compute_rbar_approach,
    compute_station_keeping, compute_hohmann_transfer_relative,
    predict_closest_approach, compute_safe_trajectory,
)
from base.star_constants import BODIES, AU_TO_KM, DAY_TO_SEC


# ISS orbital parameters for testing
ISS_ALTITUDE_KM = 420.0
EARTH_RADIUS_KM = 6371.0
MU_EARTH = BODIES['earth']['mu_km3_s2']
ISS_ORBIT_RADIUS = EARTH_RADIUS_KM + ISS_ALTITUDE_KM
ISS_MEAN_MOTION = np.sqrt(MU_EARTH / ISS_ORBIT_RADIUS**3)  # rad/s


class TestFrameTransforms:
    """Test LVLH frame transformations."""

    def test_inertial_to_lvlh_origin(self):
        """Chaser at same position as target gives zero LVLH."""
        r_target = np.array([1.0, 0.0, 0.0])  # AU
        v_target = np.array([0.0, 0.01, 0.0])  # AU/day

        r_chaser = r_target.copy()
        v_chaser = v_target.copy()

        r_lvlh, v_lvlh = inertial_to_lvlh(r_target, v_target, r_chaser, v_chaser)

        assert np.allclose(r_lvlh, [0, 0, 0], atol=1e-10)
        assert np.allclose(v_lvlh, [0, 0, 0], atol=1e-10)

    def test_inertial_to_lvlh_radial_offset(self):
        """Radial offset maps to x-axis in LVLH."""
        r_target = np.array([1.0, 0.0, 0.0])  # AU
        v_target = np.array([0.0, 0.01, 0.0])  # AU/day

        # Chaser 1 km above target (radial direction)
        offset_au = 1.0 / AU_TO_KM  # 1 km in AU
        r_chaser = r_target + np.array([offset_au, 0.0, 0.0])
        v_chaser = v_target.copy()

        r_lvlh, v_lvlh = inertial_to_lvlh(r_target, v_target, r_chaser, v_chaser)

        # Should have ~1 km in x (radial)
        assert abs(r_lvlh[0] - 1.0) < 0.01  # Within 10 m
        assert abs(r_lvlh[1]) < 0.01
        assert abs(r_lvlh[2]) < 0.01

    def test_inertial_to_lvlh_along_track_offset(self):
        """Along-track offset maps to y-axis in LVLH."""
        r_target = np.array([1.0, 0.0, 0.0])  # AU
        v_target = np.array([0.0, 0.01, 0.0])  # AU/day (velocity along y)

        # Chaser 1 km ahead of target (velocity direction)
        offset_au = 1.0 / AU_TO_KM
        r_chaser = r_target + np.array([0.0, offset_au, 0.0])
        v_chaser = v_target.copy()

        r_lvlh, v_lvlh = inertial_to_lvlh(r_target, v_target, r_chaser, v_chaser)

        # Should have ~1 km in y (along-track)
        assert abs(r_lvlh[0]) < 0.01
        assert abs(r_lvlh[1] - 1.0) < 0.01
        assert abs(r_lvlh[2]) < 0.01

    def test_round_trip_transform(self):
        """LVLH -> inertial -> LVLH preserves state."""
        r_target = np.array([1.0, 0.0, 0.0])
        v_target = np.array([0.0, 0.01, 0.0])

        r_lvlh_orig = np.array([0.5, -1.0, 0.2])  # km
        v_lvlh_orig = np.array([0.1, -0.05, 0.02])  # m/s

        # LVLH to inertial
        r_chaser, v_chaser = lvlh_to_inertial(r_target, v_target, r_lvlh_orig, v_lvlh_orig)

        # Back to LVLH
        r_lvlh_back, v_lvlh_back = inertial_to_lvlh(r_target, v_target, r_chaser, v_chaser)

        assert np.allclose(r_lvlh_orig, r_lvlh_back, atol=1e-6)
        assert np.allclose(v_lvlh_orig, v_lvlh_back, atol=1e-6)


class TestRelativeState:
    """Test RelativeState class."""

    def test_relative_state_properties(self):
        """RelativeState has correct properties."""
        r = np.array([1.0, -2.0, 0.5])
        v = np.array([0.1, -0.2, 0.05])

        state = RelativeState(
            r_lvlh=r,
            v_lvlh=v,
            range_km=np.linalg.norm(r),
            range_rate_mps=-0.1,
            t=0.0,
        )

        assert state.x == 1.0
        assert state.y == -2.0
        assert state.z == 0.5
        assert abs(state.range_km - np.sqrt(1 + 4 + 0.25)) < 0.01


class TestCWDynamics:
    """Test Clohessy-Wiltshire equations."""

    def test_cw_stationary_stays_at_origin(self):
        """Zero state at origin stays at origin."""
        r0 = np.array([0.0, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 0.0])

        r, v = cw_state_transition(r0, v0, ISS_MEAN_MOTION, 100.0)

        assert np.allclose(r, [0, 0, 0], atol=1e-10)
        assert np.allclose(v, [0, 0, 0], atol=1e-10)

    def test_cw_radial_offset_oscillates(self):
        """Radial offset with zero velocity oscillates."""
        r0 = np.array([1.0, 0.0, 0.0])  # 1 km above
        v0 = np.array([0.0, 0.0, 0.0])

        # Half period
        T_half = np.pi / ISS_MEAN_MOTION

        r, v = cw_state_transition(r0, v0, ISS_MEAN_MOTION, T_half)

        # Should oscillate (x changes sign or returns)
        # Due to CW dynamics, x will drift along with y
        assert np.linalg.norm(r) > 0  # Should have moved

    def test_cw_z_motion_decoupled(self):
        """Z (cross-track) motion is simple harmonic."""
        r0 = np.array([0.0, 0.0, 1.0])  # 1 km cross-track
        v0 = np.array([0.0, 0.0, 0.0])

        # Full period
        T = 2 * np.pi / ISS_MEAN_MOTION

        r, v = cw_state_transition(r0, v0, ISS_MEAN_MOTION, T)

        # Should return to initial position (simple harmonic)
        assert abs(r[2] - 1.0) < 0.01  # Within 10 m

    def test_cw_drift_in_y(self):
        """Lower altitude leads to along-track drift."""
        # Below target (negative x) should drift ahead (positive y)
        r0 = np.array([-0.1, 0.0, 0.0])  # 100 m below
        v0 = np.array([0.0, 0.0, 0.0])

        # One orbit
        T = 2 * np.pi / ISS_MEAN_MOTION

        r, v = cw_state_transition(r0, v0, ISS_MEAN_MOTION, T)

        # Should have drifted ahead (positive y)
        assert r[1] > 0


class TestDeltaVCalculation:
    """Test delta-V computation for CW transfers."""

    def test_cw_delta_v_to_target_zero_initial_velocity(self):
        """Compute delta-V from stationary to reach target."""
        r0 = np.array([0.0, -1.0, 0.0])  # 1 km behind
        v0 = np.array([0.0, 0.0, 0.0])
        rf = np.array([0.0, -0.2, 0.0])  # Target: 200 m behind

        dt = 600.0  # 10 minutes

        dv = cw_delta_v_to_target(r0, v0, rf, ISS_MEAN_MOTION, dt)

        # Should have positive y component to move forward
        assert dv[1] > 0
        # Delta-V should be reasonable (< 1 m/s for 800 m transfer)
        assert np.linalg.norm(dv) < 2.0

    def test_cw_delta_v_reaches_target(self):
        """Apply delta-V and verify arrival at target."""
        r0 = np.array([0.0, -1.0, 0.0])
        v0 = np.array([0.0, 0.0, 0.0])
        rf = np.array([0.0, -0.2, 0.0])
        dt = 600.0

        dv = cw_delta_v_to_target(r0, v0, rf, ISS_MEAN_MOTION, dt)

        # Apply delta-V and propagate
        v0_new = v0 + dv
        r_final, v_final = cw_state_transition(r0, v0_new, ISS_MEAN_MOTION, dt)

        # Should arrive close to target
        assert np.linalg.norm(r_final - rf) < 0.05  # Within 50 m


class TestApproachStrategies:
    """Test approach maneuver computation."""

    def test_vbar_approach_plan(self):
        """V-bar approach creates valid maneuver plan."""
        rel_state = RelativeState(
            r_lvlh=np.array([0.0, -1.0, 0.0]),  # 1 km behind on V-bar
            v_lvlh=np.array([0.0, 0.0, 0.0]),
            range_km=1.0,
            range_rate_mps=0.0,
            t=0.0,
        )

        plan = compute_vbar_approach(rel_state, ISS_MEAN_MOTION)

        assert isinstance(plan, ManeuverPlan)
        assert plan.duration_s > 0
        assert np.linalg.norm(plan.delta_v_lvlh) > 0
        assert "V-bar" in plan.description

    def test_rbar_approach_plan(self):
        """R-bar approach creates valid maneuver plan."""
        rel_state = RelativeState(
            r_lvlh=np.array([-1.0, 0.0, 0.0]),  # 1 km below on R-bar
            v_lvlh=np.array([0.0, 0.0, 0.0]),
            range_km=1.0,
            range_rate_mps=0.0,
            t=0.0,
        )

        plan = compute_rbar_approach(rel_state, ISS_MEAN_MOTION)

        assert isinstance(plan, ManeuverPlan)
        assert np.linalg.norm(plan.delta_v_lvlh) > 0
        assert "R-bar" in plan.description

    def test_station_keeping_nulls_velocity(self):
        """Station-keeping counters relative velocity."""
        rel_state = RelativeState(
            r_lvlh=np.array([0.0, -0.2, 0.0]),  # 200 m standoff
            v_lvlh=np.array([0.1, -0.05, 0.0]),  # Drifting
            range_km=0.2,
            range_rate_mps=0.05,
            t=0.0,
        )

        plan = compute_station_keeping(rel_state, ISS_MEAN_MOTION)

        # Should counter the velocity
        assert np.linalg.norm(plan.delta_v_lvlh) > 0

    def test_hohmann_transfer_two_burns(self):
        """Hohmann transfer generates two maneuvers."""
        rel_state = RelativeState(
            r_lvlh=np.array([0.0, -1.0, 0.0]),
            v_lvlh=np.array([0.0, 0.0, 0.0]),
            range_km=1.0,
            range_rate_mps=0.0,
            t=0.0,
        )

        plans = compute_hohmann_transfer_relative(rel_state, ISS_MEAN_MOTION, 0.1)

        assert len(plans) == 2
        assert plans[0].t_execute < plans[1].t_execute


class TestProximityGuidance:
    """Test ProximityGuidance controller."""

    def test_guidance_initialization(self):
        """ProximityGuidance initializes correctly."""
        guidance = ProximityGuidance(
            target_body='earth',
            target_altitude_km=420.0,
        )

        assert guidance.mean_motion > 0
        assert guidance.orbital_period_s > 0
        # ISS period ~92 minutes = 5500 seconds
        assert 5000 < guidance.orbital_period_s < 6000

    def test_guidance_config(self):
        """Custom config is applied."""
        config = ProximityGuidanceConfig(
            mode=ApproachMode.R_BAR,
            target_standoff_km=0.5,
        )

        guidance = ProximityGuidance(
            target_body='earth',
            target_altitude_km=420.0,
            config=config,
        )

        assert guidance.config.mode == ApproachMode.R_BAR
        assert guidance.config.target_standoff_km == 0.5

    def test_guidance_status(self):
        """Status returns valid info."""
        guidance = ProximityGuidance()

        status = guidance.get_status()
        assert status['initialized'] == False

        # After update
        r_target = np.array([1.0, 0.0, 0.0])
        v_target = np.array([0.0, 0.01, 0.0])
        r_chaser = r_target + np.array([0.001, 0.0, 0.0])  # Slightly offset
        v_chaser = v_target.copy()

        guidance.update(r_target, v_target, r_chaser, v_chaser, t=0.0)

        status = guidance.get_status()
        assert status['initialized'] == True
        assert 'range_km' in status


class TestTrajectoryPrediction:
    """Test trajectory prediction utilities."""

    def test_predict_closest_approach(self):
        """Closest approach prediction works."""
        rel_state = RelativeState(
            r_lvlh=np.array([0.0, -1.0, 0.0]),  # 1 km behind
            v_lvlh=np.array([0.0, 0.1, 0.0]),   # Approaching at 0.1 m/s
            range_km=1.0,
            range_rate_mps=-0.1,
            t=0.0,
        )

        t_closest, range_closest, pos_closest = predict_closest_approach(
            rel_state, ISS_MEAN_MOTION
        )

        # Should predict closer approach than initial
        assert range_closest < 1.0

    def test_safe_trajectory_check(self):
        """Safe trajectory check identifies violations."""
        # Safe trajectory - passing at 1 km
        rel_state_safe = RelativeState(
            r_lvlh=np.array([1.0, 0.0, 0.0]),
            v_lvlh=np.array([0.0, 0.1, 0.0]),
            range_km=1.0,
            range_rate_mps=0.0,
            t=0.0,
        )

        assert compute_safe_trajectory(rel_state_safe, ISS_MEAN_MOTION, 0.2) == True

        # Unsafe - on collision course
        rel_state_unsafe = RelativeState(
            r_lvlh=np.array([0.0, -0.1, 0.0]),  # Very close
            v_lvlh=np.array([0.0, 0.5, 0.0]),   # Approaching fast
            range_km=0.1,
            range_rate_mps=-0.5,
            t=0.0,
        )

        # This should be unsafe (range already < 0.2 km)
        assert compute_safe_trajectory(rel_state_unsafe, ISS_MEAN_MOTION, 0.2) == False


class TestApproachModes:
    """Test approach mode enum."""

    def test_all_modes_exist(self):
        """All approach modes are defined."""
        assert ApproachMode.V_BAR.value == "v_bar"
        assert ApproachMode.R_BAR.value == "r_bar"
        assert ApproachMode.STATION_KEEP.value == "hold"
        assert ApproachMode.INTERCEPT.value == "intercept"
        assert ApproachMode.COELLIPTIC.value == "coelliptic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
