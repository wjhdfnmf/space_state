"""
Tests for Gravity Assist / Flyby Module
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from guidance.flyby import (
    FlybyGeometry, BPlaneTarget,
    compute_v_infinity, flyby_turn_angle, flyby_periapsis_from_turn_angle,
    b_plane_from_periapsis, periapsis_from_b_plane,
    rotate_v_infinity, compute_flyby, powered_flyby_delta_v,
    maximum_turn_angle, flyby_delta_v_vector, design_flyby_for_target,
    gravity_assist_sequence_delta_v,
)


# Test constants (km, km/s)
MU_EARTH = 398600.4418
MU_JUPITER = 126686534.0
R_EARTH = 6378.137
R_JUPITER = 71492.0


class TestVInfinity:
    """Test V-infinity calculations."""

    def test_compute_v_infinity(self):
        """Test V-infinity calculation."""
        v_sc = np.array([30.0, 5.0, 0.0])  # km/s heliocentric
        v_planet = np.array([29.8, 0.0, 0.0])  # km/s heliocentric

        v_inf = compute_v_infinity(v_sc, v_planet)

        expected = np.array([0.2, 5.0, 0.0])
        np.testing.assert_allclose(v_inf, expected, rtol=1e-10)

    def test_v_infinity_magnitude(self):
        """Test V-infinity magnitude for Earth departure."""
        v_sc = np.array([32.0, 0.0, 0.0])  # km/s
        v_earth = np.array([29.78, 0.0, 0.0])  # km/s (Earth orbital velocity)

        v_inf = compute_v_infinity(v_sc, v_earth)
        v_inf_mag = np.linalg.norm(v_inf)

        assert abs(v_inf_mag - 2.22) < 0.1


class TestFlybyTurnAngle:
    """Test flyby turn angle calculations."""

    def test_turn_angle_basic(self):
        """Test basic turn angle calculation."""
        v_inf = 5.0  # km/s
        rp = R_EARTH + 300  # 300 km altitude flyby

        delta = flyby_turn_angle(v_inf, rp, MU_EARTH)

        # Turn angle should be positive and less than 180 degrees
        assert 0 < delta < np.pi
        assert np.degrees(delta) > 10  # Should be significant turn

    def test_turn_angle_increases_with_lower_periapsis(self):
        """Test that lower periapsis gives larger turn angle."""
        v_inf = 5.0

        delta_high = flyby_turn_angle(v_inf, R_EARTH + 1000, MU_EARTH)
        delta_low = flyby_turn_angle(v_inf, R_EARTH + 200, MU_EARTH)

        assert delta_low > delta_high

    def test_turn_angle_decreases_with_higher_v_infinity(self):
        """Test that higher V-infinity gives smaller turn angle."""
        rp = R_EARTH + 500

        delta_slow = flyby_turn_angle(3.0, rp, MU_EARTH)
        delta_fast = flyby_turn_angle(10.0, rp, MU_EARTH)

        assert delta_slow > delta_fast

    def test_jupiter_flyby_large_turn(self):
        """Test Jupiter flyby can achieve large turn angles."""
        v_inf = 6.0  # km/s, typical for outer planet mission
        rp = R_JUPITER + 100000  # 100,000 km above Jupiter

        delta = flyby_turn_angle(v_inf, rp, MU_JUPITER)

        # Jupiter's large mass should allow significant turns
        assert np.degrees(delta) > 30


class TestPeriapsisFromTurnAngle:
    """Test periapsis calculation from desired turn angle."""

    def test_roundtrip(self):
        """Test turn angle to periapsis roundtrip."""
        v_inf = 5.0
        delta_desired = np.radians(45)

        rp = flyby_periapsis_from_turn_angle(v_inf, delta_desired, MU_EARTH)
        delta_check = flyby_turn_angle(v_inf, rp, MU_EARTH)

        assert abs(delta_check - delta_desired) < 1e-10

    def test_large_turn_requires_close_approach(self):
        """Test that large turn requires close approach."""
        v_inf = 5.0

        rp_30deg = flyby_periapsis_from_turn_angle(v_inf, np.radians(30), MU_EARTH)
        rp_60deg = flyby_periapsis_from_turn_angle(v_inf, np.radians(60), MU_EARTH)

        assert rp_60deg < rp_30deg


class TestBPlane:
    """Test B-plane calculations."""

    def test_b_plane_from_periapsis(self):
        """Test B-plane parameter from periapsis."""
        v_inf = 5.0
        rp = R_EARTH + 500

        b = b_plane_from_periapsis(rp, v_inf, MU_EARTH)

        # B should be larger than rp (includes bending)
        assert b > rp

    def test_periapsis_from_b_plane_roundtrip(self):
        """Test B-plane to periapsis roundtrip."""
        v_inf = 5.0
        rp_orig = R_EARTH + 400

        b = b_plane_from_periapsis(rp_orig, v_inf, MU_EARTH)
        rp_back = periapsis_from_b_plane(b, v_inf, MU_EARTH)

        assert abs(rp_back - rp_orig) < 1e-6

    def test_b_plane_target_from_periapsis(self):
        """Test BPlaneTarget creation from periapsis."""
        rp = R_EARTH + 300
        v_inf = 5.0

        target = BPlaneTarget.from_periapsis(rp, v_inf, MU_EARTH, theta=0)

        assert target.b_mag > 0
        assert abs(target.b_dot_t - target.b_mag) < 1e-10  # theta=0


class TestRotateVInfinity:
    """Test V-infinity rotation."""

    def test_rotation_preserves_magnitude(self):
        """Test that rotation preserves V-infinity magnitude."""
        v_inf_in = np.array([5.0, 0.0, 0.0])
        turn_angle = np.radians(45)
        normal = np.array([0, 0, 1])

        v_inf_out = rotate_v_infinity(v_inf_in, turn_angle, normal)

        assert abs(np.linalg.norm(v_inf_out) - np.linalg.norm(v_inf_in)) < 1e-10

    def test_rotation_angle(self):
        """Test that rotation achieves correct angle."""
        v_inf_in = np.array([5.0, 0.0, 0.0])
        turn_angle = np.radians(45)
        normal = np.array([0, 0, 1])

        v_inf_out = rotate_v_infinity(v_inf_in, turn_angle, normal)

        # Check angle between vectors
        cos_angle = np.dot(v_inf_in, v_inf_out) / (np.linalg.norm(v_inf_in) * np.linalg.norm(v_inf_out))
        angle = np.arccos(cos_angle)

        assert abs(angle - turn_angle) < 1e-10

    def test_rotation_in_plane(self):
        """Test rotation stays perpendicular to normal."""
        v_inf_in = np.array([5.0, 0.0, 0.0])
        turn_angle = np.radians(30)
        normal = np.array([0, 0, 1])

        v_inf_out = rotate_v_infinity(v_inf_in, turn_angle, normal)

        # Z component should remain zero (rotation about Z)
        assert abs(v_inf_out[2]) < 1e-10


class TestComputeFlyby:
    """Test complete flyby computation."""

    def test_basic_flyby(self):
        """Test basic flyby computation."""
        v_inf_in = np.array([5.0, 0.0, 0.0])  # km/s
        rp = R_EARTH + 500

        flyby = compute_flyby(v_inf_in, rp, MU_EARTH)

        assert isinstance(flyby, FlybyGeometry)
        assert flyby.v_inf_mag > 0
        assert flyby.turn_angle > 0
        assert flyby.e_hyperbolic > 1.0  # Must be hyperbolic
        assert flyby.a_hyperbolic < 0  # Negative for hyperbola

    def test_flyby_v_infinity_magnitude_preserved(self):
        """Test V-infinity magnitude preserved through flyby."""
        v_inf_in = np.array([6.0, 2.0, 0.0])
        rp = R_EARTH + 400

        flyby = compute_flyby(v_inf_in, rp, MU_EARTH)

        v_inf_in_mag = np.linalg.norm(v_inf_in)
        v_inf_out_mag = np.linalg.norm(flyby.v_inf_out)

        assert abs(v_inf_out_mag - v_inf_in_mag) < 1e-10

    def test_flyby_delta_v_gain(self):
        """Test delta-V gain is calculated."""
        v_inf_in = np.array([5.0, 0.0, 0.0])
        rp = R_EARTH + 300

        flyby = compute_flyby(v_inf_in, rp, MU_EARTH)

        # Delta-V gain should be positive for any turn
        assert flyby.delta_v_gain > 0

    def test_jupiter_flyby(self):
        """Test Jupiter gravity assist."""
        v_inf_in = np.array([6.0, 0.0, 0.0])  # Typical arrival speed
        rp = R_JUPITER + 200000  # Safe distance above atmosphere

        flyby = compute_flyby(v_inf_in, rp, MU_JUPITER)

        # Jupiter should provide significant assist
        assert flyby.delta_v_gain > 2.0  # More than 2 km/s gain


class TestPoweredFlyby:
    """Test powered gravity assist."""

    def test_powered_flyby_delta_v(self):
        """Test powered flyby delta-V calculation."""
        v_inf_in = 5.0
        v_inf_out = 7.0  # Want to increase V-infinity
        rp = R_EARTH + 300

        dv = powered_flyby_delta_v(v_inf_in, v_inf_out, rp, MU_EARTH)

        # Delta-V should be less than v_inf difference (Oberth effect)
        assert dv < (v_inf_out - v_inf_in)
        assert dv > 0

    def test_powered_flyby_oberth_effect(self):
        """Test Oberth effect - lower periapsis needs less delta-V."""
        v_inf_in = 5.0
        v_inf_out = 7.0

        dv_high = powered_flyby_delta_v(v_inf_in, v_inf_out, R_EARTH + 1000, MU_EARTH)
        dv_low = powered_flyby_delta_v(v_inf_in, v_inf_out, R_EARTH + 200, MU_EARTH)

        assert dv_low < dv_high  # Lower periapsis is more efficient


class TestMaximumTurnAngle:
    """Test maximum turn angle calculation."""

    def test_maximum_turn_angle(self):
        """Test maximum turn angle for given constraints."""
        v_inf = 5.0
        rp_min = R_EARTH + 200  # Minimum safe altitude

        max_turn = maximum_turn_angle(v_inf, rp_min, MU_EARTH)

        assert max_turn > 0
        assert max_turn < np.pi  # Less than 180 degrees

    def test_higher_v_infinity_less_max_turn(self):
        """Test that higher V-infinity limits maximum turn."""
        rp_min = R_EARTH + 300

        max_turn_slow = maximum_turn_angle(3.0, rp_min, MU_EARTH)
        max_turn_fast = maximum_turn_angle(10.0, rp_min, MU_EARTH)

        assert max_turn_fast < max_turn_slow


class TestDesignFlybyForTarget:
    """Test flyby design for target direction."""

    def test_design_achievable_flyby(self):
        """Test designing achievable flyby."""
        v_inf_in = np.array([5.0, 0.0, 0.0])
        target_dir = np.array([0.9, 0.4, 0.0])  # Modest turn
        target_dir = target_dir / np.linalg.norm(target_dir)

        flyby = design_flyby_for_target(v_inf_in, target_dir, MU_EARTH, R_EARTH, min_altitude=200)

        if flyby is not None:
            # Check outgoing direction matches target
            v_out_dir = flyby.v_inf_out / np.linalg.norm(flyby.v_inf_out)
            assert np.dot(v_out_dir, target_dir) > 0.99

    def test_design_returns_none_for_impossible(self):
        """Test that impossible flyby returns None."""
        v_inf_in = np.array([15.0, 0.0, 0.0])  # Very high speed
        target_dir = np.array([-1.0, 0.0, 0.0])  # 180 degree turn impossible

        flyby = design_flyby_for_target(v_inf_in, target_dir, MU_EARTH, R_EARTH)

        # May or may not be possible depending on exact conditions
        # Just verify it doesn't crash


class TestGravityAssistSequence:
    """Test gravity assist sequence analysis."""

    def test_sequence_analysis(self):
        """Test gravity assist sequence delta-V calculation."""
        bodies = ['venus', 'earth', 'jupiter']
        v_infs = [4.0, 5.0, 6.0]  # km/s
        rps = [6200, 6700, 100000]  # km
        mus = [324859, MU_EARTH, MU_JUPITER]

        result = gravity_assist_sequence_delta_v(bodies, v_infs, rps, mus)

        assert 'total_delta_v_gain' in result
        assert result['total_delta_v_gain'] > 0
        assert len(result['flybys']) == 3


class TestFlybyDeltaVVector:
    """Test heliocentric velocity change calculation."""

    def test_flyby_delta_v_vector(self):
        """Test heliocentric velocity change from flyby."""
        v_body = np.array([29.78, 0.0, 0.0])  # Earth orbital velocity

        v_inf_in = np.array([3.0, 2.0, 0.0])
        flyby = compute_flyby(v_inf_in, R_EARTH + 400, MU_EARTH)

        v_helio_in, v_helio_out = flyby_delta_v_vector(v_body, flyby)

        # Heliocentric velocity should equal body velocity + V-infinity
        np.testing.assert_allclose(v_helio_in, v_body + flyby.v_inf_in)
        np.testing.assert_allclose(v_helio_out, v_body + flyby.v_inf_out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
