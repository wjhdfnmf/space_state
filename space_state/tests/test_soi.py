"""
Tests for Sphere of Influence and Integration Events
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base.star_soi import (
    OrbitalRegime, SOIInfo, SOI_RADII,
    determine_soi, get_soi_info, get_body_position, get_body_velocity,
    create_soi_events, create_periapsis_event, create_apoapsis_event,
    create_atmosphere_entry_event, create_altitude_event, create_impact_event,
)
from base.star_environment import (
    MultiBodyAtmosphere, compute_drag_acceleration, ATMOSPHERE_PARAMS,
)
from base.star_constants import BODIES


class TestSOIRadii:
    """Test SOI radius values."""

    def test_earth_soi_reasonable(self):
        """Earth SOI should be ~924,000 km (~0.00617 AU)."""
        r_soi = SOI_RADII['earth']
        r_soi_km = r_soi * 1.496e8
        assert 800000 < r_soi_km < 1000000

    def test_moon_soi_reasonable(self):
        """Moon SOI should be ~65,000 km."""
        r_soi = SOI_RADII['moon']
        r_soi_km = r_soi * 1.496e8
        assert 50000 < r_soi_km < 80000

    def test_jupiter_soi_large(self):
        """Jupiter should have much larger SOI than Earth."""
        # Jupiter and Saturn both have massive SOIs
        # Saturn's SOI can be larger than Jupiter's due to larger orbital distance
        assert SOI_RADII['jupiter barycenter'] > SOI_RADII['earth'] * 10
        assert SOI_RADII['saturn barycenter'] > SOI_RADII['earth'] * 10


class TestDetermineSOI:
    """Test SOI determination."""

    def test_heliocentric_far_from_planets(self):
        """Spacecraft far from planets is heliocentric."""
        r_sc = np.array([0.5, 0.5, 0.0])  # Somewhere in inner solar system
        body, regime, dist = determine_soi(r_sc, time=0.0)

        # Should be heliocentric (not in any planet's SOI)
        assert regime == OrbitalRegime.HELIOCENTRIC

    def test_earth_leo(self):
        """Spacecraft in LEO should be in Earth SOI."""
        # Get Earth's position at t=0
        r_earth = get_body_position('earth', 0.0)

        # Place spacecraft 400 km above Earth
        alt_au = 400 / 1.496e8
        r_sc = r_earth + np.array([alt_au, 0, 0])

        body, regime, dist = determine_soi(r_sc, time=0.0)

        assert body == 'earth'
        assert regime == OrbitalRegime.EARTH_ORBIT

    def test_lunar_orbit(self):
        """Spacecraft near Moon should be in lunar SOI."""
        r_moon = get_body_position('moon', 0.0)

        # Place spacecraft 1000 km from Moon
        alt_au = 1000 / 1.496e8
        r_sc = r_moon + np.array([alt_au, 0, 0])

        body, regime, dist = determine_soi(r_sc, time=0.0)

        assert body == 'moon'
        assert regime == OrbitalRegime.LUNAR_ORBIT


class TestSOIInfo:
    """Test detailed SOI information."""

    def test_soi_info_structure(self):
        """SOI info returns all expected fields."""
        r_earth = get_body_position('earth', 0.0)
        v_earth = get_body_velocity('earth', 0.0)

        alt_au = 400 / 1.496e8
        r_sc = r_earth + np.array([alt_au, 0, 0])
        v_sc = v_earth + np.array([0, 0.0005, 0])  # Some orbital velocity

        info = get_soi_info(r_sc, v_sc, time=0.0)

        assert isinstance(info, SOIInfo)
        assert hasattr(info, 'body')
        assert hasattr(info, 'regime')
        assert hasattr(info, 'altitude_km')
        assert hasattr(info, 'in_atmosphere')

    def test_altitude_calculation(self):
        """Altitude should be reasonable for LEO."""
        r_earth = get_body_position('earth', 0.0)
        v_earth = get_body_velocity('earth', 0.0)

        # Earth radius + 400 km altitude
        earth_radius_km = 6371.0
        target_alt_km = 400
        total_dist_km = earth_radius_km + target_alt_km
        total_dist_au = total_dist_km / 1.496e8

        # Place spacecraft at Earth position + radial offset
        r_earth_hat = r_earth / np.linalg.norm(r_earth)
        r_sc = r_earth + total_dist_au * r_earth_hat
        v_sc = v_earth

        info = get_soi_info(r_sc, v_sc, time=0.0)

        # Altitude should be close to 400 km
        assert abs(info.altitude_km - target_alt_km) < 50  # Within 50 km (numerical precision)


class TestMultiBodyAtmosphere:
    """Test multi-body atmosphere model."""

    def test_earth_surface_density(self):
        """Earth surface density should be ~1.225 kg/m³."""
        atm = MultiBodyAtmosphere()
        rho = atm.density('earth', 0.0)
        assert np.isclose(rho, 1.225, rtol=0.01)

    def test_earth_density_decreases_with_altitude(self):
        """Earth density should decrease with altitude."""
        atm = MultiBodyAtmosphere()

        rho_0 = atm.density('earth', 0.0)
        rho_100 = atm.density('earth', 100.0)
        rho_300 = atm.density('earth', 300.0)

        assert rho_100 < rho_0
        assert rho_300 < rho_100

    def test_mars_lower_density(self):
        """Mars surface density should be much lower than Earth."""
        atm = MultiBodyAtmosphere()

        rho_earth = atm.density('earth', 0.0)
        rho_mars = atm.density('mars barycenter', 0.0)

        assert rho_mars < rho_earth * 0.05  # Mars < 5% of Earth

    def test_venus_higher_density(self):
        """Venus surface density should be much higher than Earth."""
        atm = MultiBodyAtmosphere()

        rho_earth = atm.density('earth', 0.0)
        rho_venus = atm.density('venus barycenter', 0.0)

        assert rho_venus > rho_earth * 10  # Venus > 10x Earth

    def test_no_atmosphere_above_max(self):
        """No density above maximum altitude."""
        atm = MultiBodyAtmosphere()

        # Earth max is 1000 km
        rho = atm.density('earth', 1001.0)
        assert rho == 0.0

    def test_unknown_body_zero_density(self):
        """Unknown bodies return zero density."""
        atm = MultiBodyAtmosphere()
        rho = atm.density('unknown_body', 100.0)
        assert rho == 0.0


class TestDragAcceleration:
    """Test drag acceleration calculation."""

    def test_drag_in_leo(self):
        """Drag should be non-zero in LEO."""
        r_earth = get_body_position('earth', 0.0)
        v_earth = get_body_velocity('earth', 0.0)

        # Place spacecraft at Earth position + (Earth radius + 300 km) radially
        earth_radius_km = 6371.0
        alt_km = 300
        total_dist_au = (earth_radius_km + alt_km) / 1.496e8
        r_earth_hat = r_earth / np.linalg.norm(r_earth)
        r_sc = r_earth + total_dist_au * r_earth_hat

        # LEO velocity is ~7.7 km/s, which is ~0.000445 AU/day
        # Add orbital velocity perpendicular to radial
        v_orbital = np.array([0, 0.000445, 0])
        v_sc = v_earth + v_orbital

        atm = MultiBodyAtmosphere(area=10.0, drag_coefficient=2.2)
        a_drag = atm.acceleration(r_sc, v_sc, mass=1000,
                                  body_name='earth', r_body=r_earth, v_body=v_earth)

        assert np.linalg.norm(a_drag) > 0

    def test_drag_opposes_velocity(self):
        """Drag should oppose velocity direction."""
        r_earth = get_body_position('earth', 0.0)
        v_earth = get_body_velocity('earth', 0.0)

        # Place at correct altitude
        earth_radius_km = 6371.0
        alt_km = 200
        total_dist_au = (earth_radius_km + alt_km) / 1.496e8
        r_earth_hat = r_earth / np.linalg.norm(r_earth)
        r_sc = r_earth + total_dist_au * r_earth_hat

        # Velocity in +y direction relative to Earth
        v_orbital = np.array([0, 0.0005, 0])
        v_sc = v_earth + v_orbital

        atm = MultiBodyAtmosphere(area=10.0)
        a_drag = atm.acceleration(r_sc, v_sc, mass=1000,
                                  body_name='earth', r_body=r_earth, v_body=v_earth)

        # Drag should oppose the relative velocity direction
        if np.linalg.norm(a_drag) > 0:
            # Dot product of drag and relative velocity should be negative
            v_rel = v_sc - v_earth
            assert np.dot(a_drag, v_rel) < 0

    def test_no_drag_in_deep_space(self):
        """No drag far from planets."""
        r_sc = np.array([2.0, 0.0, 0.0])  # 2 AU from Sun
        v_sc = np.array([0.0, 0.01, 0.0])

        a_drag = compute_drag_acceleration(r_sc, v_sc, mass=1000, time=0.0)

        assert np.allclose(a_drag, [0, 0, 0])


class TestIntegrationEvents:
    """Test integration event functions."""

    def test_soi_event_creation(self):
        """Can create SOI crossing event."""
        event = create_soi_events('earth', direction=0)
        assert callable(event)
        assert hasattr(event, 'terminal')
        assert hasattr(event, 'direction')

    def test_periapsis_event_creation(self):
        """Can create periapsis event."""
        event = create_periapsis_event('earth')
        assert callable(event)

    def test_apoapsis_event_creation(self):
        """Can create apoapsis event."""
        event = create_apoapsis_event('earth')
        assert callable(event)

    def test_atmosphere_entry_event_creation(self):
        """Can create atmosphere entry event."""
        event = create_atmosphere_entry_event('earth')
        assert callable(event)

    def test_altitude_event_creation(self):
        """Can create altitude crossing event."""
        event = create_altitude_event('earth', altitude_km=100.0)
        assert callable(event)

    def test_impact_event_terminal(self):
        """Impact event should be terminal."""
        event = create_impact_event('earth')
        assert event.terminal == True

    def test_periapsis_event_evaluates(self):
        """Periapsis event function evaluates correctly."""
        event = create_periapsis_event('earth')

        # Create a simple state
        r_earth = get_body_position('earth', 0.0)
        alt_au = 400 / 1.496e8
        r_sc = r_earth + np.array([alt_au, 0, 0])
        v_sc = np.array([0, 0.0005, 0])  # Some velocity

        state = np.zeros(14)
        state[:3] = r_sc
        state[3:6] = v_sc

        # Event function should return a number
        result = event(0.0, state)
        assert isinstance(result, (int, float, np.floating))


class TestHeatFlux:
    """Test heat flux estimation."""

    def test_heat_flux_increases_with_velocity(self):
        """Heat flux should increase with velocity (v³)."""
        atm = MultiBodyAtmosphere()

        q_slow = atm.heat_flux(80.0, 5000.0, 'earth')  # 5 km/s
        q_fast = atm.heat_flux(80.0, 7000.0, 'earth')  # 7 km/s

        # Should scale roughly as v³
        assert q_fast > q_slow * 1.5

    def test_heat_flux_decreases_with_altitude(self):
        """Heat flux should decrease with altitude."""
        atm = MultiBodyAtmosphere()

        q_low = atm.heat_flux(80.0, 7000.0, 'earth')   # Peak heating altitude
        q_high = atm.heat_flux(300.0, 7000.0, 'earth')  # Higher altitude

        # Higher altitude should have less heat flux
        assert q_high < q_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
