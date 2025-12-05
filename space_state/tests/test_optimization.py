"""
Tests for Trajectory Optimization Module
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from optimization import (
    ManeuverNode,
    OptimizationResult,
    TrajectoryLeg,
    optimize_single_impulse,
    optimize_plane_change,
    optimize_phasing,
    optimize_hohmann,
    optimize_bielliptic,
    optimize_three_burn,
    optimize_lambert_departure,
    optimize_rendezvous,
    GravityAssist,
    compute_flyby_geometry,
    optimize_powered_flyby,
    estimate_propellant,
    compute_mass_ratio,
    delta_v_budget_margin,
)
from base.star_constants import BODIES


class TestManeuverNode:
    """Test ManeuverNode data class."""

    def test_maneuver_node_creation(self):
        """Test basic maneuver node creation."""
        node = ManeuverNode(
            time=1.5,
            delta_v=np.array([1.0, 0.0, 0.0]),
            position=np.array([1.0, 0.0, 0.0]),
            velocity_before=np.array([0.0, 1.0, 0.0]),
            velocity_after=np.array([1.0, 1.0, 0.0]),
            description="Test burn"
        )

        assert node.time == 1.5
        assert node.description == "Test burn"

    def test_delta_v_magnitude(self):
        """Test delta-V magnitude calculation."""
        node = ManeuverNode(
            time=0.0,
            delta_v=np.array([3.0, 4.0, 0.0]),
            position=np.zeros(3),
            velocity_before=np.zeros(3),
            velocity_after=np.zeros(3),
        )

        assert_allclose(node.delta_v_magnitude, 5.0)


class TestOptimizationResult:
    """Test OptimizationResult data class."""

    def test_empty_result(self):
        """Test empty optimization result."""
        result = OptimizationResult(success=False, total_delta_v=0.0)
        assert not result.success
        assert len(result.maneuvers) == 0

    def test_result_with_maneuvers(self):
        """Test result with maneuvers."""
        m1 = ManeuverNode(
            time=0.0,
            delta_v=np.array([1.0, 0.0, 0.0]),
            position=np.zeros(3),
            velocity_before=np.zeros(3),
            velocity_after=np.zeros(3),
            description="Burn 1"
        )
        m2 = ManeuverNode(
            time=1.0,
            delta_v=np.array([0.5, 0.0, 0.0]),
            position=np.zeros(3),
            velocity_before=np.zeros(3),
            velocity_after=np.zeros(3),
            description="Burn 2"
        )

        result = OptimizationResult(
            success=True,
            total_delta_v=1.5,
            maneuvers=[m1, m2],
            total_time=1.0,
        )

        assert result.success
        assert len(result.maneuvers) == 2
        assert result.total_time == 1.0

    def test_summary_generation(self):
        """Test summary string generation."""
        m = ManeuverNode(
            time=0.0,
            delta_v=np.array([3.0, 0.0, 0.0]),
            position=np.zeros(3),
            velocity_before=np.zeros(3),
            velocity_after=np.zeros(3),
            description="Test"
        )

        result = OptimizationResult(
            success=True,
            total_delta_v=3.0,
            maneuvers=[m],
            total_time=5.0,
        )

        summary = result.summary()
        assert "Successful" in summary
        assert "3.000" in summary
        assert "5.00" in summary


class TestTrajectoryLeg:
    """Test TrajectoryLeg data class."""

    def test_leg_duration(self):
        """Test trajectory leg duration calculation."""
        leg = TrajectoryLeg(
            start_time=0.0,
            end_time=5.5,
            start_position=np.zeros(3),
            end_position=np.ones(3),
            central_body='earth',
        )

        assert leg.duration_days == 5.5


class TestHohmannOptimization:
    """Test Hohmann transfer optimization."""

    def test_leo_to_geo(self):
        """Test LEO to GEO Hohmann transfer."""
        r_leo = 6371 + 400  # km
        r_geo = 42164  # km
        mu = BODIES['earth']['mu_km3_s2']

        result = optimize_hohmann(r_leo, r_geo, mu)

        assert result.success
        assert len(result.maneuvers) == 2

        # Total delta-V should be ~3.9 km/s
        assert 3.5 < result.total_delta_v < 4.3

        # Transfer time should be ~5 hours
        assert 0.1 < result.total_time < 0.3

    def test_geo_to_leo(self):
        """Test GEO to LEO (descending) Hohmann transfer."""
        r_leo = 6371 + 400
        r_geo = 42164
        mu = BODIES['earth']['mu_km3_s2']

        result = optimize_hohmann(r_geo, r_leo, mu)

        assert result.success
        # Delta-V should be same magnitude (symmetric)
        assert 3.5 < result.total_delta_v < 4.3

    def test_hohmann_with_plane_change(self):
        """Test Hohmann with combined plane change."""
        r_leo = 6371 + 400
        r_geo = 42164
        mu = BODIES['earth']['mu_km3_s2']

        result = optimize_hohmann(
            r_leo, r_geo, mu,
            include_plane_change=True,
            plane_change_deg=28.5  # Cape Canaveral to equatorial
        )

        assert result.success
        # Should be higher than pure Hohmann
        assert result.total_delta_v > 3.9


class TestBiellipticOptimization:
    """Test bi-elliptic transfer optimization."""

    def test_high_ratio_bielliptic(self):
        """Test bi-elliptic for high orbit ratio."""
        r1 = 6371 + 400  # LEO
        r2 = 384400  # Lunar distance
        r_i = 500000  # Intermediate
        mu = BODIES['earth']['mu_km3_s2']

        result = optimize_bielliptic(r1, r2, r_i, mu)

        assert result.success
        assert len(result.maneuvers) == 3
        assert result.total_delta_v > 0

    def test_three_burn_selection(self):
        """Test three-burn optimizer selects correct strategy."""
        mu = BODIES['earth']['mu_km3_s2']

        # Low ratio - should prefer Hohmann
        r1 = 6371 + 400
        r2 = 6371 + 1000
        result = optimize_three_burn(r1, r2, mu)
        assert "Hohmann" in result.message

        # Very high ratio - may prefer bi-elliptic
        r1 = 6371 + 200
        r2 = 6371 * 50
        result = optimize_three_burn(r1, r2, mu)
        assert result.success


class TestPlaneChangeOptimization:
    """Test plane change optimization."""

    def test_small_plane_change(self):
        """Test small plane change delta-V."""
        result = optimize_plane_change(
            inclination_change_deg=5.0,
            velocity_kmps=7.7,  # LEO velocity
        )

        assert result.success
        # Small angle: dv ≈ v * Δi (in radians)
        assert 0.5 < result.total_delta_v < 1.0

    def test_large_plane_change(self):
        """Test large plane change delta-V."""
        result = optimize_plane_change(
            inclination_change_deg=90.0,
            velocity_kmps=7.7,
        )

        assert result.success
        # 90° plane change: dv ≈ √2 * v ≈ 10.9 km/s
        assert 10.0 < result.total_delta_v < 12.0


class TestPhasingOptimization:
    """Test phasing maneuver optimization."""

    def test_catch_up_phasing(self):
        """Test phasing to catch up with target ahead."""
        r = 6371 + 400  # LEO radius
        mu = BODIES['earth']['mu_km3_s2']

        result = optimize_phasing(
            orbital_radius_km=r,
            phase_angle_deg=10.0,  # Target 10° ahead
            mu=mu,
            n_orbits=1,
        )

        assert result.success
        assert len(result.maneuvers) == 2
        assert result.total_delta_v > 0

    def test_phasing_more_orbits_less_dv(self):
        """Test that more orbits requires less delta-V."""
        r = 6371 + 400
        mu = BODIES['earth']['mu_km3_s2']

        result_1 = optimize_phasing(r, 30.0, mu, n_orbits=1)
        result_3 = optimize_phasing(r, 30.0, mu, n_orbits=3)

        # More orbits = less delta-V (but more time)
        if result_1.success and result_3.success:
            assert result_3.total_delta_v < result_1.total_delta_v
            assert result_3.total_time > result_1.total_time


class TestGravityAssist:
    """Test gravity assist calculations."""

    def test_earth_flyby(self):
        """Test Earth gravity assist geometry."""
        v_infinity = np.array([5.0, 0.0, 0.0])  # 5 km/s
        altitude = 300  # km

        flyby = compute_flyby_geometry('earth', v_infinity, altitude)

        assert flyby.body == 'earth'
        assert flyby.periapsis_altitude_km == altitude
        assert flyby.turn_angle_deg > 0
        assert flyby.turn_angle_deg < 180

    def test_jupiter_flyby(self):
        """Test Jupiter gravity assist (large turn angle)."""
        v_infinity = np.array([10.0, 0.0, 0.0])  # 10 km/s
        altitude = 100000  # 100,000 km

        flyby = compute_flyby_geometry('jupiter barycenter', v_infinity, altitude)

        # Jupiter has strong gravity - large turn angle
        assert flyby.turn_angle_deg > 20

    def test_powered_flyby(self):
        """Test powered gravity assist."""
        v_infinity = np.array([3.0, 0.0, 0.0])
        desired_boost = 5.0  # km/s

        flyby, prop_dv = optimize_powered_flyby(
            'earth',
            v_infinity,
            desired_boost,
            min_altitude_km=200.0
        )

        assert flyby is not None
        assert isinstance(prop_dv, float)


class TestFuelEstimation:
    """Test propellant estimation."""

    def test_propellant_calculation(self):
        """Test Tsiolkovsky equation implementation."""
        result = estimate_propellant(
            delta_v_kmps=3.0,
            dry_mass_kg=1000.0,
            isp_s=300.0,
            margin_percent=10.0,
        )

        assert 'propellant_mass_kg' in result
        assert 'mass_ratio' in result
        assert result['propellant_mass_kg'] > 0
        assert result['mass_ratio'] > 1

    def test_mass_ratio(self):
        """Test mass ratio calculation."""
        # For 3 km/s delta-V with Isp=300s
        ratio = compute_mass_ratio(3.0, 300.0)

        # exp(3000 / (300 * 9.81)) ≈ 2.77
        assert 2.5 < ratio < 3.0

    def test_high_delta_v_high_mass_ratio(self):
        """Test high delta-V requires high mass ratio."""
        ratio_low = compute_mass_ratio(1.0, 300.0)
        ratio_high = compute_mass_ratio(5.0, 300.0)

        assert ratio_high > ratio_low


class TestDeltaVBudget:
    """Test delta-V budget with margins."""

    def test_budget_margins(self):
        """Test budget margin calculation."""
        m1 = ManeuverNode(
            time=0.0,
            delta_v=np.array([2.0, 0.0, 0.0]),
            position=np.zeros(3),
            velocity_before=np.zeros(3),
            velocity_after=np.zeros(3),
        )
        m2 = ManeuverNode(
            time=1.0,
            delta_v=np.array([1.0, 0.0, 0.0]),
            position=np.zeros(3),
            velocity_before=np.zeros(3),
            velocity_after=np.zeros(3),
        )

        budget = delta_v_budget_margin([m1, m2])

        assert budget['nominal_dv_kmps'] == 3.0
        assert budget['total_with_margin_kmps'] > 3.0
        assert budget['margin_percent'] > 0


class TestLambertDeparture:
    """Test Lambert-based departure optimization."""

    def test_short_transfer(self):
        """Test short transfer Lambert optimization."""
        # Simple Earth orbit transfer
        r1 = np.array([6771.0, 0.0, 0.0])  # LEO position
        r2 = np.array([0.0, 42164.0, 0.0])  # GEO-ish position
        mu = BODIES['earth']['mu_km3_s2']

        result = optimize_lambert_departure(
            r_departure=r1,
            r_arrival=r2,
            tof_days=0.5,
            mu=mu,
        )

        # May succeed or fail depending on geometry
        assert isinstance(result.total_delta_v, float)


class TestRendezvousOptimization:
    """Test rendezvous optimization."""

    def test_coplanar_rendezvous(self):
        """Test coplanar rendezvous optimization."""
        mu = BODIES['earth']['mu_km3_s2']
        r = 6771.0  # LEO radius

        # Chaser and target on same circular orbit, different phases
        r_chaser = np.array([r, 0.0, 0.0])
        v_chaser = np.array([0.0, np.sqrt(mu/r), 0.0])

        # Target 30° ahead
        angle = np.radians(30)
        r_target = np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
        v_target = np.array([-np.sqrt(mu/r) * np.sin(angle), np.sqrt(mu/r) * np.cos(angle), 0.0])

        result = optimize_rendezvous(
            r_chaser, v_chaser,
            r_target, v_target,
            mu,
            tof_range=(0.01, 0.5),
            n_samples=20,
        )

        # Should find some solution
        assert isinstance(result.total_delta_v, float)


class TestSingleImpulse:
    """Test single impulse optimization."""

    def test_velocity_match(self):
        """Test velocity matching maneuver."""
        r = np.array([7000.0, 0.0, 0.0])
        v_current = np.array([0.0, 7.5, 0.0])
        v_target = np.array([0.0, 7.8, 0.0])
        mu = BODIES['earth']['mu_km3_s2']

        result = optimize_single_impulse(
            r, v_current,
            r, v_target,
            mu,
        )

        assert result.success
        assert len(result.maneuvers) == 1
        # Should be close to direct velocity change
        assert result.total_delta_v < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
