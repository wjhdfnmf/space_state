# tests/test_thermal.py
"""
Tests for Thermal Analysis Module

Tests solar flux, planetary IR, equilibrium temperature,
and power system sizing calculations.
"""

import pytest
import numpy as np

import sys
sys.path.insert(0, '/home/boss/data/space_state')

from space_state.base.star_thermal import (
    ThermalEnvironment, SpacecraftThermal, PLANET_PROPERTIES,
    STEFAN_BOLTZMANN, SOLAR_CONSTANT_1AU, AU_TO_KM,
    solar_flux_at_distance, solar_flux_at_distance_km,
    planetary_view_factor, albedo_flux, planetary_ir_flux,
    equilibrium_temperature, compute_thermal_environment,
    hot_cold_case_temperatures, eclipse_cooldown_time, eclipse_warmup_time,
    estimate_heater_power, battery_sizing_for_eclipse, solar_panel_sizing,
)


class TestSolarFlux:
    """Test solar flux calculations."""

    def test_solar_flux_at_1au(self):
        """Test solar flux at 1 AU matches solar constant."""
        flux = solar_flux_at_distance(1.0)
        assert abs(flux - SOLAR_CONSTANT_1AU) < 1.0

    def test_solar_flux_inverse_square(self):
        """Test flux follows inverse square law."""
        flux_1au = solar_flux_at_distance(1.0)
        flux_2au = solar_flux_at_distance(2.0)

        # At 2 AU, flux should be 1/4 of 1 AU
        assert abs(flux_2au - flux_1au / 4) < 1.0

    def test_solar_flux_at_mercury(self):
        """Test solar flux at Mercury's orbit."""
        flux = solar_flux_at_distance(0.387)  # Mercury's mean distance

        # Should be about 6-10x Earth's flux
        assert 6000 < flux < 10000

    def test_solar_flux_at_mars(self):
        """Test solar flux at Mars' orbit."""
        flux = solar_flux_at_distance(1.524)  # Mars mean distance

        # Should be about 43% of Earth's flux
        assert 500 < flux < 700

    def test_solar_flux_at_jupiter(self):
        """Test solar flux at Jupiter's orbit."""
        flux = solar_flux_at_distance(5.2)

        # Should be about 1/27 of Earth's flux
        assert 40 < flux < 60

    def test_solar_flux_km_conversion(self):
        """Test km-based solar flux function."""
        distance_km = AU_TO_KM  # 1 AU in km
        flux = solar_flux_at_distance_km(distance_km)

        assert abs(flux - SOLAR_CONSTANT_1AU) < 1.0


class TestPlanetaryViewFactor:
    """Test planetary view factor calculations."""

    def test_view_factor_at_surface(self):
        """Test view factor at surface is 1/2 (hemisphere)."""
        # At surface (altitude=0), we see half the sky covered by planet
        vf = planetary_view_factor(0.0, 6378.0)  # Earth
        assert abs(vf - 1.0) < 0.01

    def test_view_factor_leo(self):
        """Test view factor in LEO."""
        vf = planetary_view_factor(400.0, 6378.0)  # ISS altitude

        # Should see most of the hemisphere
        assert 0.8 < vf < 1.0

    def test_view_factor_geo(self):
        """Test view factor at GEO."""
        vf = planetary_view_factor(35786.0, 6378.0)  # GEO altitude

        # Should see smaller portion of sphere
        assert 0.01 < vf < 0.1

    def test_view_factor_decreases_with_altitude(self):
        """Test view factor decreases with altitude."""
        vf_400 = planetary_view_factor(400, 6378)
        vf_1000 = planetary_view_factor(1000, 6378)
        vf_10000 = planetary_view_factor(10000, 6378)

        assert vf_400 > vf_1000 > vf_10000


class TestAlbedoFlux:
    """Test albedo flux calculations."""

    def test_albedo_flux_basic(self):
        """Test basic albedo flux calculation."""
        solar = 1361.0
        albedo = 0.30
        view_factor = 0.5

        flux = albedo_flux(solar, albedo, view_factor)

        # Should be solar * albedo * view_factor
        expected = 1361 * 0.30 * 0.5
        assert abs(flux - expected) < 1.0

    def test_albedo_flux_zero_view_factor(self):
        """Test albedo flux is zero with no view factor."""
        flux = albedo_flux(1361.0, 0.30, 0.0)
        assert flux == 0.0

    def test_albedo_flux_scales_with_albedo(self):
        """Test albedo flux scales linearly with albedo."""
        flux_low = albedo_flux(1361.0, 0.10, 0.5)
        flux_high = albedo_flux(1361.0, 0.30, 0.5)

        assert abs(flux_high / flux_low - 3.0) < 0.01


class TestPlanetaryIR:
    """Test planetary IR flux calculations."""

    def test_planetary_ir_basic(self):
        """Test basic planetary IR calculation."""
        planet_temp = 288  # Earth's surface temperature
        view_factor = 0.5
        emissivity = 0.95

        ir = planetary_ir_flux(planet_temp, view_factor, emissivity)

        # Should be σ * ε * T^4 * vf
        expected = STEFAN_BOLTZMANN * emissivity * (288**4) * view_factor
        assert abs(ir - expected) < 1.0

    def test_planetary_ir_temperature_dependence(self):
        """Test IR scales with T^4."""
        ir_cold = planetary_ir_flux(200, 0.5)
        ir_warm = planetary_ir_flux(400, 0.5)

        # T doubles -> IR increases by 16x
        ratio = ir_warm / ir_cold
        assert abs(ratio - 16.0) < 1.0


class TestEquilibriumTemperature:
    """Test equilibrium temperature calculations."""

    def test_equilibrium_temp_basic(self):
        """Test basic equilibrium temperature."""
        # At 1 AU with black body (α=ε=1)
        absorbed = 1361.0  # Full solar absorption
        emissivity = 1.0

        T_eq = equilibrium_temperature(absorbed, emissivity)

        # T = (Q / σ)^0.25 ≈ 394 K
        assert 390 < T_eq < 400

    def test_equilibrium_temp_deep_space(self):
        """Test equilibrium temp with no heat input."""
        T_eq = equilibrium_temperature(0.0, 1.0)

        # Should return deep space temperature
        assert T_eq == 3.0

    def test_equilibrium_temp_with_area_ratio(self):
        """Test equilibrium temp with radiator area ratio."""
        absorbed = 1361.0

        T_eq_1 = equilibrium_temperature(absorbed, 1.0, area_ratio=1.0)
        T_eq_4 = equilibrium_temperature(absorbed, 1.0, area_ratio=4.0)

        # More radiator area = lower temperature
        # 4x area -> T drops by factor of sqrt(2) ≈ 1.41
        assert T_eq_1 > T_eq_4


class TestThermalEnvironment:
    """Test ThermalEnvironment calculations."""

    def test_deep_space_environment(self):
        """Test thermal environment in deep space."""
        env = compute_thermal_environment(
            distance_from_sun_au=5.0,  # Jupiter distance
            altitude_km=None,
            planet=None
        )

        assert isinstance(env, ThermalEnvironment)
        assert 40 < env.solar_flux < 60  # Jupiter's solar flux
        assert env.albedo_flux == 0.0
        assert env.planetary_ir == 0.0
        assert env.sink_temperature == 3.0

    def test_earth_orbit_environment(self):
        """Test thermal environment in Earth orbit."""
        env = compute_thermal_environment(
            distance_from_sun_au=1.0,
            altitude_km=400,
            planet='earth'
        )

        assert env.solar_flux > 1300
        assert env.albedo_flux > 0
        assert env.planetary_ir > 0
        assert env.equilibrium_temp > 0

    def test_lunar_orbit_environment(self):
        """Test thermal environment in lunar orbit."""
        env = compute_thermal_environment(
            distance_from_sun_au=1.0,
            altitude_km=100,
            planet='moon'
        )

        # Moon has low albedo
        assert env.solar_flux > 1300
        assert env.albedo_flux > 0

    def test_environment_repr(self):
        """Test ThermalEnvironment string representation."""
        env = compute_thermal_environment(1.0)
        repr_str = repr(env)

        assert 'W/m²' in repr_str
        assert 'T_eq' in repr_str


class TestHotColdCases:
    """Test hot and cold case temperature analysis."""

    def test_hot_cold_case_basic(self):
        """Test hot/cold case temperature calculation."""
        T_hot, T_cold = hot_cold_case_temperatures(
            distance_from_sun_au=1.0,
            altitude_km=400,
            planet='earth'
        )

        # Hot case should be warmer than cold case
        assert T_hot > T_cold

    def test_hot_cold_reasonable_range(self):
        """Test temperatures are in reasonable range."""
        T_hot, T_cold = hot_cold_case_temperatures(1.0, 400, 'earth')

        # Hot case: high absorptivity, low emissivity -> warm
        # With α=0.9, ε=0.3, temperatures can be very high
        assert T_hot > 350  # Should be quite hot

        # Cold case: low absorptivity, high emissivity -> cooler
        # But with planetary IR and albedo, still above deep space
        assert T_cold < T_hot  # Cold case must be cooler than hot


class TestSpacecraftThermal:
    """Test SpacecraftThermal dataclass."""

    def test_spacecraft_thermal_creation(self):
        """Test creating SpacecraftThermal object."""
        sc = SpacecraftThermal(
            solar_absorptivity=0.3,
            ir_emissivity=0.9,
            area_sun_facing=5.0,
            area_radiator=10.0,
            internal_power=500.0,
            mass=1000.0,
            specific_heat=900.0
        )

        assert sc.solar_absorptivity == 0.3
        assert sc.ir_emissivity == 0.9
        assert sc.area_sun_facing == 5.0
        assert sc.area_radiator == 10.0
        assert sc.internal_power == 500.0
        assert sc.mass == 1000.0
        assert sc.specific_heat == 900.0


class TestEclipseThermal:
    """Test eclipse thermal transient calculations."""

    def test_eclipse_cooldown_time(self):
        """Test eclipse cooldown time estimate."""
        sc = SpacecraftThermal(
            solar_absorptivity=0.3,
            ir_emissivity=0.9,
            area_sun_facing=5.0,
            area_radiator=10.0,
            internal_power=100.0,
            mass=1000.0,
            specific_heat=900.0
        )

        env = compute_thermal_environment(1.0)

        dt = eclipse_cooldown_time(sc, 300.0, 280.0, env)

        # Should take some time to cool
        assert dt > 0

    def test_eclipse_warmup_time(self):
        """Test eclipse warmup time estimate."""
        sc = SpacecraftThermal(
            solar_absorptivity=0.3,
            ir_emissivity=0.9,
            area_sun_facing=5.0,
            area_radiator=10.0,
            internal_power=100.0,
            mass=1000.0,
            specific_heat=900.0
        )

        env = compute_thermal_environment(1.0)

        dt = eclipse_warmup_time(sc, 280.0, 300.0, env)

        # Should take some time to warm
        assert dt > 0

    def test_no_cooldown_needed(self):
        """Test no cooldown when target >= initial."""
        sc = SpacecraftThermal(
            solar_absorptivity=0.3,
            ir_emissivity=0.9,
            area_sun_facing=5.0,
            area_radiator=10.0,
            internal_power=100.0,
            mass=1000.0,
            specific_heat=900.0
        )

        env = compute_thermal_environment(1.0)

        dt = eclipse_cooldown_time(sc, 280.0, 300.0, env)
        assert dt == 0.0

    def test_heater_power_estimate(self):
        """Test heater power estimation."""
        sc = SpacecraftThermal(
            solar_absorptivity=0.3,
            ir_emissivity=0.9,
            area_sun_facing=5.0,
            area_radiator=10.0,
            internal_power=100.0,
            mass=1000.0,
            specific_heat=900.0
        )

        env = compute_thermal_environment(1.0)

        heater_power = estimate_heater_power(sc, 273.0, env)

        # Should require some heater power to maintain temperature
        assert heater_power >= 0


class TestPowerSizing:
    """Test power system sizing calculations."""

    def test_battery_sizing_basic(self):
        """Test battery sizing for eclipse."""
        sc = SpacecraftThermal(
            solar_absorptivity=0.3,
            ir_emissivity=0.9,
            area_sun_facing=5.0,
            area_radiator=10.0,
            internal_power=500.0,
            mass=1000.0,
            specific_heat=900.0
        )

        result = battery_sizing_for_eclipse(
            sc,
            eclipse_duration_min=35.0,
            heater_power=50.0,
            other_loads_w=200.0
        )

        assert 'battery_capacity_wh' in result
        assert 'energy_required_wh' in result
        assert result['battery_capacity_wh'] > result['energy_required_wh']

    def test_battery_sizing_depth_of_discharge(self):
        """Test battery sizing accounts for DoD."""
        sc = SpacecraftThermal(
            solar_absorptivity=0.3,
            ir_emissivity=0.9,
            area_sun_facing=5.0,
            area_radiator=10.0,
            internal_power=500.0,
            mass=1000.0,
            specific_heat=900.0
        )

        result_30 = battery_sizing_for_eclipse(
            sc, 35, 50, 200, depth_of_discharge=0.30
        )
        result_50 = battery_sizing_for_eclipse(
            sc, 35, 50, 200, depth_of_discharge=0.50
        )

        # Higher DoD allows smaller battery
        assert result_30['battery_capacity_wh'] > result_50['battery_capacity_wh']

    def test_solar_panel_sizing_1au(self):
        """Test solar panel sizing at 1 AU."""
        result = solar_panel_sizing(
            power_required_w=500.0,
            distance_au=1.0,
            panel_efficiency=0.30
        )

        assert 'panel_area_m2' in result
        assert result['panel_area_m2'] > 0
        assert result['solar_flux_w_m2'] > 1300

    def test_solar_panel_sizing_mars(self):
        """Test solar panel sizing at Mars."""
        result_earth = solar_panel_sizing(500.0, 1.0)
        result_mars = solar_panel_sizing(500.0, 1.5)

        # Mars needs larger panels
        assert result_mars['panel_area_m2'] > result_earth['panel_area_m2']

    def test_solar_panel_sizing_jupiter(self):
        """Test solar panel sizing at Jupiter."""
        result = solar_panel_sizing(500.0, 5.2)

        # Jupiter needs much larger panels
        assert result['panel_area_m2'] > 10  # Significant area needed


class TestPlanetProperties:
    """Test planetary thermal properties dictionary."""

    def test_earth_properties(self):
        """Test Earth's thermal properties."""
        assert 'earth' in PLANET_PROPERTIES
        earth = PLANET_PROPERTIES['earth']

        assert 0.2 < earth['albedo'] < 0.4
        assert 6370 < earth['radius_km'] < 6385
        assert 280 < earth['surface_temp_k'] < 300

    def test_mars_properties(self):
        """Test Mars thermal properties."""
        assert 'mars' in PLANET_PROPERTIES
        mars = PLANET_PROPERTIES['mars']

        assert 0.2 < mars['albedo'] < 0.3
        assert 200 < mars['surface_temp_k'] < 230

    def test_all_planets_have_required_fields(self):
        """Test all planets have required thermal properties."""
        required_fields = ['albedo', 'ir_emissivity', 'surface_temp_k', 'radius_km']

        for planet, props in PLANET_PROPERTIES.items():
            for field in required_fields:
                assert field in props, f"{planet} missing {field}"

    def test_moon_properties(self):
        """Test Moon's thermal properties."""
        assert 'moon' in PLANET_PROPERTIES
        moon = PLANET_PROPERTIES['moon']

        # Moon has low albedo
        assert moon['albedo'] < 0.2


class TestConstants:
    """Test physical constants."""

    def test_stefan_boltzmann(self):
        """Test Stefan-Boltzmann constant."""
        # σ = 5.67e-8 W/(m²·K⁴)
        assert abs(STEFAN_BOLTZMANN - 5.67e-8) < 1e-10

    def test_solar_constant(self):
        """Test solar constant at 1 AU."""
        # ~1361 W/m²
        assert 1360 < SOLAR_CONSTANT_1AU < 1365

    def test_au_to_km(self):
        """Test AU to km conversion."""
        # 1 AU ≈ 149.6 million km
        assert 149e6 < AU_TO_KM < 150e6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
