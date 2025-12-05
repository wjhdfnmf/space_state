"""
Tests for Patched Conics Module
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from guidance.patched_conics import (
    ReferenceFrame, TrajectorySegment, PatchedConicTrajectory,
    sphere_of_influence_radius, get_soi_radius, SOI_RADII_AU,
    heliocentric_to_planetocentric, planetocentric_to_heliocentric,
    check_soi_entry, compute_c3, v_infinity_from_c3,
    departure_v_infinity, arrival_conditions,
    interplanetary_transfer_type, estimate_transfer_delta_v,
)


class TestSOIRadius:
    """Test sphere of influence radius calculations."""

    def test_soi_formula(self):
        """Test SOI radius formula."""
        a = 1.0  # AU
        m_planet = 1.0
        m_sun = 1000.0

        r_soi = sphere_of_influence_radius(a, m_planet, m_sun)

        # Should be a * (m_planet/m_sun)^0.4
        expected = a * (m_planet / m_sun) ** 0.4
        assert abs(r_soi - expected) < 1e-10

    def test_earth_soi_reasonable(self):
        """Test Earth SOI is reasonable."""
        r_soi = SOI_RADII_AU.get('earth', 0)

        # Earth SOI is about 929,000 km = 0.0062 AU
        assert 0.005 < r_soi < 0.008

    def test_jupiter_soi_large(self):
        """Test Jupiter has largest SOI."""
        r_jupiter = SOI_RADII_AU.get('jupiter', 0)
        r_earth = SOI_RADII_AU.get('earth', 0)

        assert r_jupiter > r_earth * 10  # Much larger

    def test_moon_soi_small(self):
        """Test Moon has small SOI."""
        r_moon = SOI_RADII_AU.get('moon', 0)
        r_earth = SOI_RADII_AU.get('earth', 0)

        assert r_moon < r_earth / 10  # Much smaller


class TestFrameTransformations:
    """Test coordinate frame transformations."""

    def test_helio_to_planetocentric(self):
        """Test heliocentric to planetocentric transformation."""
        r_helio = np.array([1.0, 0.0, 0.0])  # At 1 AU
        v_helio = np.array([0.0, 30.0, 0.0])  # km/s

        r_planet = np.array([1.0, 0.01, 0.0])  # Planet slightly offset
        v_planet = np.array([0.0, 29.78, 0.0])  # km/s

        r_pc, v_pc = heliocentric_to_planetocentric(r_helio, v_helio, r_planet, v_planet)

        # Position should be relative
        expected_r = r_helio - r_planet
        np.testing.assert_allclose(r_pc, expected_r)

        # Velocity should be relative
        expected_v = v_helio - v_planet
        np.testing.assert_allclose(v_pc, expected_v)

    def test_planetocentric_to_helio(self):
        """Test planetocentric to heliocentric transformation."""
        r_pc = np.array([0.01, 0.0, 0.0])
        v_pc = np.array([5.0, 0.0, 0.0])

        r_planet = np.array([1.0, 0.0, 0.0])
        v_planet = np.array([0.0, 29.78, 0.0])

        r_helio, v_helio = planetocentric_to_heliocentric(r_pc, v_pc, r_planet, v_planet)

        expected_r = r_pc + r_planet
        expected_v = v_pc + v_planet
        np.testing.assert_allclose(r_helio, expected_r)
        np.testing.assert_allclose(v_helio, expected_v)

    def test_roundtrip_transformation(self):
        """Test roundtrip transformation."""
        r_helio = np.array([1.1, 0.2, 0.05])
        v_helio = np.array([2.0, 28.0, 1.0])

        r_planet = np.array([1.0, 0.0, 0.0])
        v_planet = np.array([0.0, 29.78, 0.0])

        # To planetocentric
        r_pc, v_pc = heliocentric_to_planetocentric(r_helio, v_helio, r_planet, v_planet)

        # Back to heliocentric
        r_back, v_back = planetocentric_to_heliocentric(r_pc, v_pc, r_planet, v_planet)

        np.testing.assert_allclose(r_back, r_helio)
        np.testing.assert_allclose(v_back, v_helio)


class TestSOIEntry:
    """Test SOI entry detection."""

    def test_inside_soi(self):
        """Test detection when inside SOI."""
        r_helio = np.array([1.001, 0.0, 0.0])
        r_planet = np.array([1.0, 0.0, 0.0])

        inside = check_soi_entry(r_helio, r_planet, 'earth')

        # 0.001 AU = ~150,000 km, well inside Earth SOI (~930,000 km)
        assert inside

    def test_outside_soi(self):
        """Test detection when outside SOI."""
        r_helio = np.array([1.1, 0.0, 0.0])
        r_planet = np.array([1.0, 0.0, 0.0])

        inside = check_soi_entry(r_helio, r_planet, 'earth')

        # 0.1 AU = ~15 million km, outside Earth SOI
        assert not inside

    def test_jupiter_large_soi(self):
        """Test Jupiter's large SOI."""
        r_helio = np.array([5.5, 0.0, 0.0])  # Near Jupiter
        r_planet = np.array([5.2, 0.0, 0.0])  # Jupiter at ~5.2 AU

        inside = check_soi_entry(r_helio, r_planet, 'jupiter')

        # 0.3 AU = ~45 million km, inside Jupiter SOI (~48 million km)
        assert inside


class TestC3:
    """Test characteristic energy calculations."""

    def test_compute_c3(self):
        """Test C3 calculation."""
        v_inf = 3.0  # km/s

        c3 = compute_c3(v_inf)

        assert c3 == 9.0  # v_inf²

    def test_v_infinity_from_c3(self):
        """Test V-infinity from C3."""
        c3 = 16.0

        v_inf = v_infinity_from_c3(c3)

        assert v_inf == 4.0

    def test_c3_roundtrip(self):
        """Test C3 roundtrip."""
        v_inf_orig = 5.5

        c3 = compute_c3(v_inf_orig)
        v_inf_back = v_infinity_from_c3(c3)

        assert abs(v_inf_back - v_inf_orig) < 1e-10

    def test_negative_c3_raises(self):
        """Test negative C3 raises error."""
        with pytest.raises(ValueError):
            v_infinity_from_c3(-1.0)


class TestDepartureArrival:
    """Test departure and arrival calculations."""

    def test_departure_v_infinity(self):
        """Test departure V-infinity calculation."""
        v_helio_sc = np.array([32.0, 1.0, 0.0])  # km/s
        v_helio_planet = np.array([29.78, 0.0, 0.0])  # km/s

        v_inf, v_inf_mag, c3 = departure_v_infinity(v_helio_sc, v_helio_planet)

        expected_v_inf = v_helio_sc - v_helio_planet
        np.testing.assert_allclose(v_inf, expected_v_inf)
        assert abs(v_inf_mag - np.linalg.norm(expected_v_inf)) < 1e-10
        assert abs(c3 - v_inf_mag**2) < 1e-10

    def test_arrival_conditions(self):
        """Test arrival conditions calculation."""
        v_helio_sc = np.array([20.0, 5.0, 0.0])  # km/s
        v_helio_planet = np.array([24.1, 0.0, 0.0])  # Mars velocity

        mu_mars = 42828.0  # km³/s²

        result = arrival_conditions(v_helio_sc, v_helio_planet, mu_mars)

        assert 'v_infinity' in result
        assert 'c3' in result
        assert result['v_infinity'] > 0
        assert result['c3'] > 0


class TestTransferTypes:
    """Test transfer classification."""

    def test_low_energy_transfer(self):
        """Test low-energy transfer classification."""
        result = interplanetary_transfer_type(5.0, 5.0)
        assert "Low-energy" in result

    def test_medium_energy_transfer(self):
        """Test medium-energy transfer classification."""
        result = interplanetary_transfer_type(20.0, 15.0)
        assert "Medium-energy" in result

    def test_high_energy_transfer(self):
        """Test high-energy transfer classification."""
        result = interplanetary_transfer_type(50.0, 40.0)
        assert "High-energy" in result


class TestEstimateTransferDeltaV:
    """Test transfer delta-V estimation."""

    def test_earth_mars_transfer(self):
        """Test Earth to Mars transfer estimate."""
        result = estimate_transfer_delta_v('earth', 'mars')

        assert 'v_inf_departure_km_s' in result
        assert 'v_inf_arrival_km_s' in result
        assert 'transfer_time_days' in result

        # Earth-Mars Hohmann takes about 259 days
        assert 200 < result['transfer_time_days'] < 300

        # Departure C3 should be reasonable
        assert result['c3_departure'] < 20  # km²/s²

    def test_earth_venus_transfer(self):
        """Test Earth to Venus transfer."""
        result = estimate_transfer_delta_v('earth', 'venus')

        # Venus is inner planet, transfer time shorter
        assert result['transfer_time_days'] < 200

    def test_earth_jupiter_transfer(self):
        """Test Earth to Jupiter transfer."""
        result = estimate_transfer_delta_v('earth', 'jupiter')

        # Jupiter is far, long transfer
        assert result['transfer_time_days'] > 500

    def test_unknown_body_raises(self):
        """Test unknown body raises error."""
        with pytest.raises(ValueError):
            estimate_transfer_delta_v('earth', 'unknown_planet')


class TestReferenceFrame:
    """Test reference frame enum."""

    def test_heliocentric_exists(self):
        """Test heliocentric frame exists."""
        assert ReferenceFrame.HELIOCENTRIC is not None

    def test_planetocentric_exists(self):
        """Test planetocentric frame exists."""
        assert ReferenceFrame.PLANETOCENTRIC is not None


class TestTrajectorySegment:
    """Test trajectory segment dataclass."""

    def test_segment_creation(self):
        """Test trajectory segment creation."""
        segment = TrajectorySegment(
            frame=ReferenceFrame.HELIOCENTRIC,
            central_body='sun',
            r_initial=np.array([1.0, 0.0, 0.0]),
            v_initial=np.array([0.0, 0.0172, 0.0]),  # AU/day
            t_start=0.0,
            t_end=100.0,
            mu=0.000296  # AU³/day² (Sun)
        )

        assert segment.frame == ReferenceFrame.HELIOCENTRIC
        assert segment.central_body == 'sun'
        assert segment.t_end > segment.t_start


class TestPatchedConicTrajectory:
    """Test patched conic trajectory class."""

    def test_trajectory_creation(self):
        """Test trajectory with multiple segments."""
        seg1 = TrajectorySegment(
            frame=ReferenceFrame.PLANETOCENTRIC,
            central_body='earth',
            r_initial=np.array([6678.0, 0.0, 0.0]),
            v_initial=np.array([0.0, 10.0, 0.0]),
            t_start=0.0,
            t_end=0.1,
            mu=398600.0
        )

        seg2 = TrajectorySegment(
            frame=ReferenceFrame.HELIOCENTRIC,
            central_body='sun',
            r_initial=np.array([1.0, 0.0, 0.0]),
            v_initial=np.array([0.0, 0.0172, 0.0]),
            t_start=0.1,
            t_end=200.0,
            mu=0.000296
        )

        trajectory = PatchedConicTrajectory(
            segments=[seg1, seg2],
            soi_crossings=[{'time': 0.1, 'body': 'earth', 'type': 'exit'}],
            total_time=200.0
        )

        assert len(trajectory.segments) == 2
        assert len(trajectory.soi_crossings) == 1

    def test_trajectory_summary(self):
        """Test trajectory summary generation."""
        seg1 = TrajectorySegment(
            frame=ReferenceFrame.HELIOCENTRIC,
            central_body='sun',
            r_initial=np.array([1.0, 0.0, 0.0]),
            v_initial=np.array([0.0, 0.02, 0.0]),
            t_start=0.0,
            t_end=100.0,
            mu=0.000296
        )

        trajectory = PatchedConicTrajectory(
            segments=[seg1],
            soi_crossings=[],
            total_time=100.0
        )

        summary = trajectory.summary()

        assert 'HELIOCENTRIC' in summary
        assert 'sun' in summary
        assert '100' in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
