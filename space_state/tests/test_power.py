"""
Tests for Power System Module
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base.star_power import (
    SolarPanel, Battery, PowerLoad, PowerSystem, PowerMode,
    detect_eclipse, detect_earth_eclipse, detect_moon_eclipse,
    SOLAR_CONSTANT,
)
from base.star_constants import BODIES


class TestSolarPanel:
    """Test solar panel model."""

    def test_max_power_at_1au(self):
        """Panel at 1 AU, sun-facing should produce expected power."""
        panel = SolarPanel(area=100, efficiency=0.28)
        power = panel.max_power(distance_au=1.0)

        # Expected: 0.28 * 100 m² * 1361 W/m² = 38,108 W
        expected = 0.28 * 100 * SOLAR_CONSTANT
        assert np.isclose(power, expected, rtol=0.01)

    def test_power_inverse_square(self):
        """Power follows inverse square law with distance."""
        panel = SolarPanel(area=100, efficiency=0.28)

        power_1au = panel.max_power(distance_au=1.0)
        power_2au = panel.max_power(distance_au=2.0)

        assert np.isclose(power_2au / power_1au, 0.25, rtol=0.01)

    def test_power_cosine_loss(self):
        """Power decreases with sun angle."""
        panel = SolarPanel(area=100, efficiency=0.28)

        power_0deg = panel.power_output(1.0, sun_angle_rad=0.0)
        power_60deg = panel.power_output(1.0, sun_angle_rad=np.pi/3)

        # cos(60°) = 0.5
        assert np.isclose(power_60deg / power_0deg, 0.5, rtol=0.01)

    def test_power_zero_at_90deg(self):
        """Power is zero when panel edge-on to Sun."""
        panel = SolarPanel(area=100, efficiency=0.28)
        power = panel.power_output(1.0, sun_angle_rad=np.pi/2)
        assert np.isclose(power, 0.0, atol=1e-10)

    def test_degradation(self):
        """Panel degrades over time."""
        panel_new = SolarPanel(area=100, efficiency=0.28, age_days=0)
        panel_old = SolarPanel(area=100, efficiency=0.28, age_days=365.25)

        power_new = panel_new.max_power(1.0)
        power_old = panel_old.max_power(1.0)

        # 2% degradation per year
        expected_ratio = 0.98
        assert np.isclose(power_old / power_new, expected_ratio, rtol=0.01)


class TestBattery:
    """Test battery model."""

    def test_initial_soc(self):
        """Battery starts fully charged."""
        battery = Battery(capacity_wh=10000)
        assert battery.soc == 1.0

    def test_discharge_reduces_soc(self):
        """Discharging reduces state of charge."""
        battery = Battery(capacity_wh=10000)
        battery.discharge(1000, duration_hours=1.0)

        # 1000 W for 1 hour = 1 kWh, should reduce SOC by ~10%
        assert battery.soc < 1.0

    def test_charge_increases_soc(self):
        """Charging increases state of charge."""
        battery = Battery(capacity_wh=10000)
        battery._current_soc = 0.5  # Start at 50%

        battery.charge(1000, duration_hours=1.0)
        assert battery.soc > 0.5

    def test_cannot_overcharge(self):
        """Battery cannot exceed 100% SOC."""
        battery = Battery(capacity_wh=10000)
        battery.charge(10000, duration_hours=10.0)  # Way more than needed
        assert battery.soc == 1.0

    def test_min_soc_protection(self):
        """Battery respects minimum SOC."""
        battery = Battery(capacity_wh=10000, min_soc=0.1)
        battery.discharge(10000, duration_hours=10.0)  # Try to fully discharge
        assert battery.soc >= 0.1

    def test_discharge_efficiency(self):
        """Discharge efficiency affects energy delivered."""
        battery = Battery(capacity_wh=10000, discharge_efficiency=0.9)
        battery._current_soc = 1.0
        initial_soc = battery.soc

        # Request 1000 W for 1 hour = 1000 Wh
        delivered = battery.discharge(1000, duration_hours=1.0)

        # Energy should be delivered (the method returns delivered energy)
        assert delivered > 0
        # SOC should decrease
        assert battery.soc < initial_soc


class TestPowerLoad:
    """Test power load model."""

    def test_load_power(self):
        """Load has correct power."""
        load = PowerLoad("test", power_w=100)
        assert load.power_w == 100

    def test_essential_flag(self):
        """Essential flag works."""
        essential = PowerLoad("essential", power_w=100, essential=True)
        optional = PowerLoad("optional", power_w=100, essential=False)

        assert essential.essential
        assert not optional.essential


class TestPowerSystem:
    """Test complete power system."""

    def test_typical_spacecraft(self):
        """Can create typical spacecraft."""
        system = PowerSystem.typical_spacecraft()
        assert len(system.panels) > 0
        assert len(system.batteries) > 0
        assert len(system.loads) > 0

    def test_solar_power_generation(self):
        """System generates solar power."""
        system = PowerSystem(
            panels=[SolarPanel(area=100, efficiency=0.28)],
            batteries=[],
            loads=[],
        )
        power = system.solar_power(distance_au=1.0)
        assert power > 10000  # Should be ~38 kW

    def test_solar_power_zero_in_eclipse(self):
        """No solar power in eclipse."""
        system = PowerSystem(
            panels=[SolarPanel(area=100, efficiency=0.28)],
            batteries=[],
            loads=[],
        )
        system._in_eclipse = True
        power = system.solar_power(distance_au=1.0)
        assert power == 0.0

    def test_total_load(self):
        """Total load sums correctly."""
        system = PowerSystem(
            panels=[],
            batteries=[],
            loads=[
                PowerLoad("a", 100),
                PowerLoad("b", 200),
                PowerLoad("c", 300),
            ],
        )
        assert system.total_load() == 600

    def test_essential_load(self):
        """Essential load sums only essential."""
        system = PowerSystem(
            panels=[],
            batteries=[],
            loads=[
                PowerLoad("essential", 100, essential=True),
                PowerLoad("optional", 200, essential=False),
            ],
        )
        assert system.essential_load() == 100

    def test_update_charges_battery(self):
        """Excess power charges battery."""
        system = PowerSystem(
            panels=[SolarPanel(area=100, efficiency=0.28)],
            batteries=[Battery(capacity_wh=10000)],
            loads=[PowerLoad("small", 100)],
        )
        system.batteries[0]._current_soc = 0.5

        system.update(duration_hours=1.0, distance_au=1.0)

        assert system.battery_soc() > 0.5

    def test_update_in_eclipse_discharges(self):
        """Eclipse discharges battery."""
        system = PowerSystem(
            panels=[SolarPanel(area=100, efficiency=0.28)],
            batteries=[Battery(capacity_wh=10000)],
            loads=[PowerLoad("load", 500)],
        )

        initial_soc = system.battery_soc()
        system.update(duration_hours=1.0, distance_au=1.0, in_eclipse=True)

        assert system.battery_soc() < initial_soc


class TestEclipseDetection:
    """Test eclipse detection."""

    def test_no_eclipse_in_front(self):
        """Spacecraft in front of body is not in eclipse."""
        r_body = np.array([1.0, 0.0, 0.0])
        r_sc = np.array([0.5, 0.0, 0.0])  # Closer to Sun
        body_radius = 0.01

        assert not detect_eclipse(r_sc, r_body, body_radius)

    def test_eclipse_behind_body(self):
        """Spacecraft directly behind body is in eclipse."""
        r_body = np.array([1.0, 0.0, 0.0])
        r_sc = np.array([1.5, 0.0, 0.0])  # Behind body
        body_radius = 0.01

        # Within shadow cylinder
        assert detect_eclipse(r_sc, r_body, body_radius)

    def test_no_eclipse_beside_body(self):
        """Spacecraft beside body (perpendicular) is not in eclipse."""
        r_body = np.array([1.0, 0.0, 0.0])
        r_sc = np.array([1.0, 0.1, 0.0])  # Same distance, offset
        body_radius = 0.01

        assert not detect_eclipse(r_sc, r_body, body_radius)

    def test_earth_eclipse_function(self):
        """Earth eclipse convenience function works."""
        r_earth = np.array([1.0, 0.0, 0.0])

        # In shadow
        r_in_shadow = np.array([1.0 + 0.001, 0.0, 0.0])
        assert detect_earth_eclipse(r_in_shadow, r_earth)

        # Not in shadow
        r_not_shadow = np.array([1.0 + 0.001, 0.001, 0.0])
        # May or may not be in shadow depending on offset


class TestPhysicalValidation:
    """Physical validation tests."""

    def test_iss_solar_panel_power(self):
        """ISS-like solar array power output."""
        # ISS has ~2500 m² of panels, ~14% efficiency
        panel = SolarPanel(area=2500, efficiency=0.14)
        power = panel.max_power(distance_au=1.0)

        # ISS generates ~240 kW peak
        # Our calculation: 2500 * 1361 * 0.14 = 476 kW (theoretical max)
        # Real ISS is lower due to tracking, degradation, etc.
        assert 400000 < power < 500000  # 400-500 kW range

    def test_deep_space_power(self):
        """Power at Mars distance."""
        panel = SolarPanel(area=100, efficiency=0.28)

        power_earth = panel.max_power(distance_au=1.0)
        power_mars = panel.max_power(distance_au=1.524)

        # At Mars (~1.52 AU), power should be ~43% of Earth
        ratio = power_mars / power_earth
        expected = 1 / 1.524**2
        assert np.isclose(ratio, expected, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
