# tests/test_station_keeping.py
"""
Tests for Station Keeping Module

Tests GEO station keeping, Lagrange point calculations,
and low-thrust propulsion budgets.
"""

import pytest
import numpy as np

import sys
sys.path.insert(0, '/home/boss/data/space_state')

from space_state.guidance.station_keeping import (
    StationKeepingMode, GEOStationKeepingBudget, LagrangePoint, LowThrustBudget,
    geo_inclination_drift, geo_north_south_delta_v, geo_east_west_delta_v,
    geo_eccentricity_delta_v, estimate_geo_station_keeping_budget,
    compute_lagrange_points, sun_earth_lagrange_points, earth_moon_lagrange_points,
    lagrange_station_keeping_budget, low_thrust_delta_v, estimate_low_thrust_budget,
    hohmann_correction, plane_change_delta_v, combined_maneuver_delta_v,
    MU_EARTH, MU_SUN, MU_MOON, R_GEO,
)


class TestGEOStationKeeping:
    """Test GEO station keeping calculations."""

    def test_geo_inclination_drift_upward(self):
        """Test inclination drift toward equilibrium from below."""
        initial = 0.0
        final = geo_inclination_drift(initial, years=1.0)
        # Should drift toward ~15 degrees
        assert final > initial
        assert final < 1.0  # Not too much drift in one year

    def test_geo_inclination_drift_downward(self):
        """Test inclination drift toward equilibrium from above."""
        initial = 20.0
        final = geo_inclination_drift(initial, years=1.0)
        # Should drift toward ~15 degrees
        assert final < initial
        assert final > 14.0  # Shouldn't go below equilibrium

    def test_geo_inclination_drift_at_equilibrium(self):
        """Test inclination at equilibrium stays stable."""
        equilibrium = 14.67
        final = geo_inclination_drift(equilibrium, years=1.0)
        # Should stay close to equilibrium
        assert abs(final - equilibrium) < 0.1

    def test_geo_north_south_delta_v(self):
        """Test N-S delta-V calculation."""
        # Correcting 0.1 degree inclination
        dv = geo_north_south_delta_v(0.1)
        # Should be a few m/s
        assert 1 < dv < 10

    def test_geo_north_south_delta_v_annual(self):
        """Test annual N-S delta-V is in expected range."""
        # Annual drift is ~0.85 degrees
        dv_annual = geo_north_south_delta_v(0.85)
        # Typical annual N-S budget is ~45-50 m/s
        assert 40 < dv_annual < 55

    def test_geo_east_west_delta_v(self):
        """Test E-W delta-V calculation."""
        drift_rate = 0.01  # degrees per day
        dv = geo_east_west_delta_v(drift_rate, correction_period=14)
        assert dv > 0
        assert dv < 10  # Should be modest

    def test_geo_eccentricity_delta_v(self):
        """Test eccentricity correction delta-V."""
        dv = geo_eccentricity_delta_v(0.001, target_eccentricity=0.0001)
        # v_geo is ~3075 m/s, delta_e is 0.0009
        assert 2 < dv < 4

    def test_estimate_geo_station_keeping_budget(self):
        """Test full GEO station keeping budget."""
        budget = estimate_geo_station_keeping_budget()

        assert isinstance(budget, GEOStationKeepingBudget)
        assert budget.north_south > 0
        assert budget.east_west > 0
        assert budget.eccentricity > 0
        assert budget.total == budget.north_south + budget.east_west + budget.eccentricity
        assert budget.maneuvers_per_year > 0

    def test_geo_budget_typical_values(self):
        """Test GEO budget matches typical mission values."""
        budget = estimate_geo_station_keeping_budget()

        # Typical GEO station keeping is 45-55 m/s/year for N-S
        assert 40 < budget.north_south < 55
        # E-W is typically 2-6 m/s/year
        assert 2 < budget.east_west < 10
        # Eccentricity control typically 1-3 m/s/year
        assert 1 < budget.eccentricity < 5
        # Total should be reasonable (50-60 m/s/year for fully controlled GEO)
        assert 45 < budget.total < 70


class TestLagrangePoints:
    """Test Lagrange point calculations."""

    def test_compute_lagrange_points_basic(self):
        """Test basic Lagrange point calculation."""
        # Use simple test masses
        m1, m2 = 1e6, 1e3
        distance = 1000.0

        points = compute_lagrange_points(m1, m2, distance)

        assert 'L1' in points
        assert 'L2' in points
        assert 'L3' in points
        assert 'L4' in points
        assert 'L5' in points

    def test_lagrange_point_stability(self):
        """Test L4/L5 are stable, L1/L2/L3 are unstable."""
        m1, m2 = 1e6, 1e3
        distance = 1000.0
        points = compute_lagrange_points(m1, m2, distance)

        assert points['L1'].stable is False
        assert points['L2'].stable is False
        assert points['L3'].stable is False
        assert points['L4'].stable is True
        assert points['L5'].stable is True

    def test_lagrange_l4_l5_equilateral(self):
        """Test L4 and L5 form equilateral triangle."""
        m1, m2 = 1e6, 1e3
        distance = 1000.0
        points = compute_lagrange_points(m1, m2, distance)

        # L4 and L5 should be symmetric about x-axis
        assert abs(points['L4'].y + points['L5'].y) < 1e-10
        assert abs(points['L4'].x - points['L5'].x) < 1e-10

        # y should be sqrt(3)/2 ≈ 0.866
        assert abs(abs(points['L4'].y) - np.sqrt(3)/2) < 0.01

    def test_sun_earth_lagrange_points(self):
        """Test Sun-Earth Lagrange points."""
        points = sun_earth_lagrange_points()

        # L1 is about 1.5 million km from Earth
        assert 1.4e6 < points['L1'].distance_km < 1.6e6

        # L2 is also about 1.5 million km from Earth
        assert 1.4e6 < points['L2'].distance_km < 1.6e6

    def test_earth_moon_lagrange_points(self):
        """Test Earth-Moon Lagrange points."""
        points = earth_moon_lagrange_points()

        # Should have all 5 points
        assert len(points) == 5

        # L1 should be between Earth and Moon
        assert points['L1'].x < 1.0  # Closer to Earth
        assert points['L1'].x > 0.0

    def test_lagrange_station_keeping_budget_sun_earth(self):
        """Test station keeping budgets for Sun-Earth points."""
        l1_budget = lagrange_station_keeping_budget('L1', 'sun-earth')
        l2_budget = lagrange_station_keeping_budget('L2', 'sun-earth')
        l4_budget = lagrange_station_keeping_budget('L4', 'sun-earth')

        # L1/L2 require active station keeping (~5 m/s/year)
        assert 1 < l1_budget < 20
        assert 1 < l2_budget < 20

        # L4/L5 are stable (minimal)
        assert l4_budget == 0

    def test_lagrange_station_keeping_budget_earth_moon(self):
        """Test station keeping budgets for Earth-Moon points."""
        l1_budget = lagrange_station_keeping_budget('L1', 'earth-moon')
        l2_budget = lagrange_station_keeping_budget('L2', 'earth-moon')

        # Earth-Moon L1/L2 require more delta-V (~100 m/s/year)
        assert 50 < l1_budget < 200
        assert 50 < l2_budget < 200


class TestLowThrustPropulsion:
    """Test low-thrust propulsion calculations."""

    def test_low_thrust_delta_v_basic(self):
        """Test basic low-thrust delta-V calculation."""
        thrust = 0.1  # 100 mN
        mass = 1000.0  # 1000 kg
        burn_time = 3600.0  # 1 hour

        dv = low_thrust_delta_v(thrust, mass, burn_time)

        # acceleration = 0.1/1000 = 0.0001 m/s²
        # dv = 0.0001 * 3600 = 0.36 m/s
        assert abs(dv - 0.36) < 0.01

    def test_low_thrust_delta_v_scaling(self):
        """Test delta-V scales correctly with parameters."""
        dv1 = low_thrust_delta_v(0.1, 1000, 3600)
        dv2 = low_thrust_delta_v(0.2, 1000, 3600)  # Double thrust

        assert abs(dv2 / dv1 - 2.0) < 1e-10

    def test_estimate_low_thrust_budget(self):
        """Test low-thrust propulsion budget estimation."""
        budget = estimate_low_thrust_budget(
            thrust_n=0.1,
            isp_s=3000,
            spacecraft_mass=1000,
            delta_v_requirement=50  # 50 m/s/year
        )

        assert isinstance(budget, LowThrustBudget)
        assert budget.thrust_n == 0.1
        assert budget.isp_s == 3000
        assert budget.delta_v_per_year == 50
        assert budget.propellant_per_year > 0

    def test_low_thrust_propellant_mass(self):
        """Test propellant mass calculation is reasonable."""
        budget = estimate_low_thrust_budget(
            thrust_n=0.1,
            isp_s=3000,
            spacecraft_mass=1000,
            delta_v_requirement=50
        )

        # With Isp=3000s, v_exhaust ≈ 29430 m/s
        # For 50 m/s, mass ratio ≈ exp(50/29430) ≈ 1.0017
        # Propellant ≈ 1000 * 0.0017 ≈ 1.7 kg
        assert 1 < budget.propellant_per_year < 5


class TestOrbitManeuvers:
    """Test orbital maneuver calculations."""

    def test_hohmann_correction_raises(self):
        """Test Hohmann transfer for orbit raising."""
        r1 = 42164.0  # GEO
        r2 = 42200.0  # Slightly higher

        dv1, dv2 = hohmann_correction(r1, r2, MU_EARTH)

        assert dv1 > 0
        assert dv2 > 0
        # For small altitude change, delta-V should be small
        assert dv1 < 0.1  # km/s
        assert dv2 < 0.1

    def test_hohmann_correction_lowers(self):
        """Test Hohmann transfer for orbit lowering."""
        r1 = 42200.0
        r2 = 42164.0  # Lower orbit

        dv1, dv2 = hohmann_correction(r1, r2, MU_EARTH)

        assert dv1 > 0
        assert dv2 > 0

    def test_plane_change_delta_v_small(self):
        """Test plane change for small angle."""
        velocity = 3.075  # km/s (GEO velocity)
        angle = 1.0  # 1 degree

        dv = plane_change_delta_v(velocity, angle)

        # For small angles: dv ≈ v * angle_rad
        expected = velocity * np.radians(angle)
        assert abs(dv - expected) < 0.001

    def test_plane_change_delta_v_large(self):
        """Test plane change for larger angle."""
        velocity = 3.075
        angle = 28.5  # ISS inclination change

        dv = plane_change_delta_v(velocity, angle)

        # Should be significant
        assert 1.0 < dv < 2.0

    def test_combined_maneuver_more_efficient(self):
        """Test combined maneuver is more efficient than separate."""
        velocity = 7.8  # km/s (LEO velocity)
        altitude_change = 100  # km
        plane_change = 5.0  # degrees

        dv_combined = combined_maneuver_delta_v(
            velocity, altitude_change, plane_change, MU_EARTH
        )

        # Separate maneuvers
        dv_plane = plane_change_delta_v(velocity, plane_change)

        # Combined should be less than or equal to sum
        assert dv_combined <= dv_plane + 0.1


class TestStationKeepingMode:
    """Test station keeping mode enum."""

    def test_modes_exist(self):
        """Test all expected modes exist."""
        assert StationKeepingMode.NORTH_SOUTH
        assert StationKeepingMode.EAST_WEST
        assert StationKeepingMode.ECCENTRICITY
        assert StationKeepingMode.COMBINED

    def test_modes_distinct(self):
        """Test modes are distinct."""
        modes = [
            StationKeepingMode.NORTH_SOUTH,
            StationKeepingMode.EAST_WEST,
            StationKeepingMode.ECCENTRICITY,
            StationKeepingMode.COMBINED,
        ]
        assert len(set(modes)) == 4


class TestDataclassRepresentations:
    """Test dataclass string representations."""

    def test_geo_budget_repr(self):
        """Test GEO budget string representation."""
        budget = GEOStationKeepingBudget(
            north_south=50.0,
            east_west=4.0,
            eccentricity=2.0,
            total=56.0,
            maneuvers_per_year=24
        )

        repr_str = repr(budget)
        assert "50.0" in repr_str
        assert "4.0" in repr_str
        assert "56.0" in repr_str

    def test_lagrange_point_fields(self):
        """Test LagrangePoint has expected fields."""
        point = LagrangePoint(
            name='L1',
            x=0.99,
            y=0.0,
            stable=False,
            distance_km=1.5e6
        )

        assert point.name == 'L1'
        assert point.x == 0.99
        assert point.y == 0.0
        assert point.stable is False
        assert point.distance_km == 1.5e6

    def test_low_thrust_budget_fields(self):
        """Test LowThrustBudget has expected fields."""
        budget = LowThrustBudget(
            thrust_n=0.1,
            isp_s=3000,
            delta_v_per_year=50.0,
            propellant_per_year=1.7
        )

        assert budget.thrust_n == 0.1
        assert budget.isp_s == 3000
        assert budget.delta_v_per_year == 50.0
        assert budget.propellant_per_year == 1.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
