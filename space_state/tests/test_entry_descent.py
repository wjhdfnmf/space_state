# tests/test_entry_descent.py
"""
Tests for Entry, Descent, and Landing Module
"""

import pytest
import numpy as np

import sys
sys.path.insert(0, '/home/boss/data/space_state')

from space_state.guidance.entry_descent import (
    EntryType, AtmosphereModel, EntryConditions, EntryResult,
    ATMOSPHERES, atmospheric_density, ballistic_coefficient,
    entry_corridor_bounds, peak_deceleration_estimate, peak_heating_rate,
    total_heat_load, simulate_entry, skip_entry_analysis,
    parachute_deployment_conditions, mars_edl_profile, earth_reentry_corridor,
    R_EARTH, R_MARS, G0,
)


class TestAtmosphereModel:
    """Test atmosphere models."""

    def test_earth_atmosphere_exists(self):
        """Test Earth atmosphere model exists."""
        assert 'earth' in ATMOSPHERES
        earth = ATMOSPHERES['earth']
        assert earth.name == 'Earth'
        assert earth.surface_density > 1.0
        assert earth.scale_height > 5.0

    def test_mars_atmosphere_exists(self):
        """Test Mars atmosphere model exists."""
        assert 'mars' in ATMOSPHERES
        mars = ATMOSPHERES['mars']
        assert mars.name == 'Mars'
        assert mars.surface_density < 0.1  # Thin atmosphere
        assert mars.entry_altitude == 125.0

    def test_venus_atmosphere_exists(self):
        """Test Venus atmosphere model exists."""
        assert 'venus' in ATMOSPHERES
        venus = ATMOSPHERES['venus']
        assert venus.surface_density > 50  # Very thick
        assert venus.surface_pressure > 9e6

    def test_titan_atmosphere_exists(self):
        """Test Titan atmosphere model exists."""
        assert 'titan' in ATMOSPHERES
        titan = ATMOSPHERES['titan']
        assert titan.entry_altitude > 1000  # High atmosphere


class TestAtmosphericDensity:
    """Test atmospheric density calculations."""

    def test_density_at_surface(self):
        """Test density at surface matches model."""
        earth = ATMOSPHERES['earth']
        rho = atmospheric_density(0.0, earth)
        assert abs(rho - earth.surface_density) < 0.01

    def test_density_decreases_with_altitude(self):
        """Test density decreases with altitude."""
        earth = ATMOSPHERES['earth']
        rho_0 = atmospheric_density(0, earth)
        rho_10 = atmospheric_density(10, earth)
        rho_50 = atmospheric_density(50, earth)

        assert rho_0 > rho_10 > rho_50

    def test_density_scale_height(self):
        """Test exponential decrease follows scale height."""
        earth = ATMOSPHERES['earth']
        H = earth.scale_height

        rho_0 = atmospheric_density(0, earth)
        rho_H = atmospheric_density(H, earth)

        # At one scale height, density should be 1/e
        assert abs(rho_H / rho_0 - 1/np.e) < 0.01


class TestBallisticCoefficient:
    """Test ballistic coefficient calculations."""

    def test_ballistic_coefficient_basic(self):
        """Test basic ballistic coefficient calculation."""
        mass = 1000  # kg
        cd = 1.5
        area = 10  # m²

        beta = ballistic_coefficient(mass, cd, area)

        assert beta == mass / (cd * area)
        assert abs(beta - 66.67) < 0.1

    def test_ballistic_coefficient_scaling(self):
        """Test β scales correctly."""
        beta1 = ballistic_coefficient(1000, 1.5, 10)
        beta2 = ballistic_coefficient(2000, 1.5, 10)

        # Double mass = double β
        assert abs(beta2 / beta1 - 2.0) < 0.01


class TestEntryCorridor:
    """Test entry corridor calculations."""

    def test_entry_corridor_earth_leo(self):
        """Test entry corridor for LEO reentry."""
        steep, shallow = entry_corridor_bounds(8.0, 'earth', max_decel_g=10.0)

        # Both should be negative (descending)
        assert steep < 0
        assert shallow < 0
        # Steep limit should be steeper (more negative)
        assert steep < shallow

    def test_entry_corridor_lunar_return(self):
        """Test entry corridor for lunar return."""
        steep, shallow = entry_corridor_bounds(11.0, 'earth', max_decel_g=10.0)

        # Lunar return has tighter corridor
        assert steep < 0
        assert shallow < 0

    def test_entry_corridor_mars(self):
        """Test entry corridor for Mars entry."""
        steep, shallow = entry_corridor_bounds(5.5, 'mars', max_decel_g=15.0)

        # Mars has steep entries
        assert steep < -10  # Steeper than -10 degrees


class TestPeakDeceleration:
    """Test peak deceleration estimates."""

    def test_peak_decel_earth_leo(self):
        """Test peak deceleration for LEO reentry."""
        g_peak = peak_deceleration_estimate(
            velocity_km_s=7.8,
            flight_path_angle_deg=-5.0,
            ballistic_coef=100,
            body='earth'
        )

        # Typical LEO reentry: 3-8 g's
        assert 1 < g_peak < 15

    def test_peak_decel_increases_with_steeper_angle(self):
        """Test steeper entry = higher peak g."""
        g_shallow = peak_deceleration_estimate(7.8, -3.0, 100, 'earth')
        g_steep = peak_deceleration_estimate(7.8, -7.0, 100, 'earth')

        assert g_steep > g_shallow

    def test_peak_decel_increases_with_velocity(self):
        """Test higher velocity = higher peak g."""
        g_low_v = peak_deceleration_estimate(7.0, -5.0, 100, 'earth')
        g_high_v = peak_deceleration_estimate(11.0, -5.0, 100, 'earth')

        assert g_high_v > g_low_v


class TestHeatRate:
    """Test heat rate calculations."""

    def test_peak_heat_rate_positive(self):
        """Test heat rate is positive."""
        q = peak_heating_rate(7.8, 1.0, 60.0, 'earth')
        assert q > 0

    def test_heat_rate_increases_with_velocity(self):
        """Test heat rate scales with velocity cubed."""
        q_low = peak_heating_rate(5.0, 1.0, 60.0, 'earth')
        q_high = peak_heating_rate(10.0, 1.0, 60.0, 'earth')

        # q ∝ V³, so q_high/q_low ≈ (10/5)³ = 8
        assert q_high / q_low > 5

    def test_heat_rate_decreases_with_larger_nose(self):
        """Test larger nose radius = lower heat rate."""
        q_small = peak_heating_rate(7.8, 0.5, 60.0, 'earth')
        q_large = peak_heating_rate(7.8, 2.0, 60.0, 'earth')

        assert q_small > q_large


class TestTotalHeatLoad:
    """Test total heat load calculations."""

    def test_heat_load_positive(self):
        """Test heat load is positive."""
        Q = total_heat_load(7.8, 100, 1.0, 'earth')
        assert Q > 0

    def test_heat_load_increases_with_velocity(self):
        """Test heat load scales with velocity."""
        Q_low = total_heat_load(7.0, 100, 1.0, 'earth')
        Q_high = total_heat_load(11.0, 100, 1.0, 'earth')

        assert Q_high > Q_low


class TestEntrySimulation:
    """Test entry trajectory simulation."""

    def test_simulate_entry_completes(self):
        """Test entry simulation runs to completion."""
        entry = EntryConditions(
            velocity=7.8,
            flight_path_angle=-5.0,
            altitude=120.0,
            latitude=0.0,
            longitude=0.0,
            azimuth=90.0
        )

        result = simulate_entry(
            entry=entry,
            ballistic_coef=100,
            nose_radius_m=1.0,
            body='earth',
            dt=1.0
        )

        assert isinstance(result, EntryResult)
        assert result.peak_deceleration_g > 0
        assert result.peak_heat_rate_w_cm2 > 0
        assert len(result.trajectory) > 0

    def test_simulate_entry_trajectory_descends(self):
        """Test trajectory altitude decreases."""
        entry = EntryConditions(
            velocity=7.8,
            flight_path_angle=-5.0,
            altitude=120.0,
            latitude=0.0,
            longitude=0.0,
            azimuth=90.0
        )

        result = simulate_entry(entry, 100, 1.0, 'earth', dt=1.0)

        # First altitude should be higher than last
        assert result.trajectory[0].altitude > result.trajectory[-1].altitude

    def test_simulate_entry_velocity_decreases(self):
        """Test velocity decreases during entry."""
        entry = EntryConditions(
            velocity=7.8,
            flight_path_angle=-5.0,
            altitude=120.0,
            latitude=0.0,
            longitude=0.0,
            azimuth=90.0
        )

        result = simulate_entry(entry, 100, 1.0, 'earth', dt=1.0)

        # Velocity should decrease
        assert result.trajectory[0].velocity > result.trajectory[-1].velocity


class TestSkipEntry:
    """Test skip entry analysis."""

    def test_skip_entry_analysis(self):
        """Test skip entry returns valid results."""
        result = skip_entry_analysis(
            entry_velocity_km_s=11.0,
            entry_fpa_deg=-4.0,
            ballistic_coef=100,
            lift_to_drag=0.3,
            body='earth'
        )

        assert 'entry_velocity_km_s' in result
        assert 'skip_out_velocity_km_s' in result
        assert 'velocity_loss_km_s' in result
        assert result['skip_out_velocity_km_s'] < result['entry_velocity_km_s']


class TestParachuteDeployment:
    """Test parachute deployment conditions."""

    def test_parachute_conditions_earth(self):
        """Test parachute deployment conditions for Earth."""
        cond = parachute_deployment_conditions(
            mach_target=2.0,
            altitude_km=10.0,
            body='earth'
        )

        assert cond['mach_number'] == 2.0
        assert cond['altitude_km'] == 10.0
        assert cond['velocity_m_s'] > 0
        assert cond['dynamic_pressure_pa'] > 0

    def test_parachute_conditions_mars(self):
        """Test parachute deployment for Mars."""
        cond = parachute_deployment_conditions(2.0, 10.0, 'mars')

        # Mars has thinner atmosphere
        assert cond['density_kg_m3'] < 0.01


class TestMarsEDL:
    """Test Mars EDL profile generation."""

    def test_mars_edl_profile(self):
        """Test Mars EDL profile generation."""
        profile = mars_edl_profile(
            entry_velocity_km_s=5.5,
            entry_fpa_deg=-12.0,
            vehicle_mass_kg=900.0,
            heatshield_diameter_m=4.5
        )

        assert 'entry_conditions' in profile
        assert 'ballistic_coefficient_kg_m2' in profile
        assert 'entry_result' in profile
        assert 'parachute_deployment' in profile
        assert 'timeline' in profile

    def test_mars_edl_reasonable_values(self):
        """Test Mars EDL produces reasonable values."""
        profile = mars_edl_profile()

        result = profile['entry_result']
        # Peak deceleration should be reasonable
        assert 1 < result.peak_deceleration_g < 30


class TestEarthReentryCorridor:
    """Test Earth reentry corridor calculations."""

    def test_capsule_corridor(self):
        """Test capsule entry corridor."""
        corridor = earth_reentry_corridor(11.0, 'capsule')

        assert corridor['vehicle_type'] == 'capsule'
        assert corridor['lift_to_drag'] > 0
        assert corridor['steep_limit_deg'] < corridor['shallow_limit_deg']
        assert corridor['corridor_width_deg'] > 0

    def test_ballistic_corridor(self):
        """Test ballistic entry corridor."""
        corridor = earth_reentry_corridor(8.0, 'ballistic')

        assert corridor['lift_to_drag'] == 0
        assert corridor['corridor_width_deg'] > 0

    def test_shuttle_corridor(self):
        """Test Shuttle-like entry corridor."""
        corridor = earth_reentry_corridor(7.8, 'shuttle')

        assert corridor['lift_to_drag'] > 1.0


class TestEntryType:
    """Test EntryType enum."""

    def test_entry_types_exist(self):
        """Test all entry types exist."""
        assert EntryType.BALLISTIC
        assert EntryType.LIFTING
        assert EntryType.SKIP
        assert EntryType.AEROBRAKING


class TestConstants:
    """Test module constants."""

    def test_earth_radius(self):
        """Test Earth radius is correct."""
        assert 6370 < R_EARTH < 6380

    def test_mars_radius(self):
        """Test Mars radius is correct."""
        assert 3380 < R_MARS < 3400

    def test_g0(self):
        """Test standard gravity."""
        assert abs(G0 - 9.80665) < 0.001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
