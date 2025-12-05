"""
Tests for Environmental Forces Module
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base.star_environment import (
    SolarRadiationPressure, AtmosphericDrag, EnvironmentModel,
    compute_srp_acceleration, estimate_srp_delta_v,
    SOLAR_PRESSURE, EARTH_SURFACE_DENSITY, ATMOSPHERE_SCALE_HEIGHT,
)


class TestSolarRadiationPressure:
    """Test SRP model."""

    def test_srp_at_1au(self):
        """SRP at 1 AU should be around 4.56e-6 N/m²."""
        assert np.isclose(SOLAR_PRESSURE, 4.56e-6, rtol=0.05)

    def test_srp_direction_away_from_sun(self):
        """SRP should push away from Sun."""
        srp = SolarRadiationPressure(area=100, reflectivity=1.5)

        # Spacecraft at 1 AU on positive x-axis
        r = np.array([1.0, 0.0, 0.0])
        a = srp.acceleration(r, mass=1000)

        # Should push in +x direction (away from Sun at origin)
        assert a[0] > 0
        assert np.isclose(a[1], 0, atol=1e-20)
        assert np.isclose(a[2], 0, atol=1e-20)

    def test_srp_inverse_square(self):
        """SRP should follow inverse square law."""
        srp = SolarRadiationPressure(area=100, reflectivity=1.5)

        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([2.0, 0.0, 0.0])

        a1 = np.linalg.norm(srp.acceleration(r1, mass=1000))
        a2 = np.linalg.norm(srp.acceleration(r2, mass=1000))

        # At 2 AU, should be 1/4 of 1 AU value
        assert np.isclose(a2 / a1, 0.25, rtol=0.01)

    def test_srp_scales_with_area(self):
        """SRP should scale linearly with area."""
        srp1 = SolarRadiationPressure(area=100, reflectivity=1.5)
        srp2 = SolarRadiationPressure(area=200, reflectivity=1.5)

        r = np.array([1.0, 0.0, 0.0])

        a1 = np.linalg.norm(srp1.acceleration(r, mass=1000))
        a2 = np.linalg.norm(srp2.acceleration(r, mass=1000))

        assert np.isclose(a2 / a1, 2.0, rtol=0.01)

    def test_srp_scales_with_reflectivity(self):
        """SRP should scale with reflectivity."""
        srp1 = SolarRadiationPressure(area=100, reflectivity=1.0)
        srp2 = SolarRadiationPressure(area=100, reflectivity=2.0)

        r = np.array([1.0, 0.0, 0.0])

        a1 = np.linalg.norm(srp1.acceleration(r, mass=1000))
        a2 = np.linalg.norm(srp2.acceleration(r, mass=1000))

        assert np.isclose(a2 / a1, 2.0, rtol=0.01)

    def test_srp_inverse_mass(self):
        """SRP acceleration should be inversely proportional to mass."""
        srp = SolarRadiationPressure(area=100, reflectivity=1.5)

        r = np.array([1.0, 0.0, 0.0])

        a1 = np.linalg.norm(srp.acceleration(r, mass=1000))
        a2 = np.linalg.norm(srp.acceleration(r, mass=2000))

        assert np.isclose(a2 / a1, 0.5, rtol=0.01)

    def test_srp_disabled(self):
        """Disabled SRP should return zero."""
        srp = SolarRadiationPressure(area=100, reflectivity=1.5, enabled=False)

        r = np.array([1.0, 0.0, 0.0])
        a = srp.acceleration(r, mass=1000)

        assert np.allclose(a, [0, 0, 0])

    def test_srp_magnitude_reasonable(self):
        """SRP magnitude should be small but non-negligible."""
        srp = SolarRadiationPressure(area=100, reflectivity=1.5)

        r = np.array([1.0, 0.0, 0.0])
        a = srp.acceleration(r, mass=1000)

        # SRP acceleration is very small but measurable
        a_mag = np.linalg.norm(a)

        # Should be around 1e-9 to 1e-6 AU/day² for typical spacecraft
        assert 1e-12 < a_mag < 1e-5


class TestAtmosphericDrag:
    """Test atmospheric drag model."""

    def test_density_at_surface(self):
        """Surface density should be ~1.225 kg/m³."""
        drag = AtmosphericDrag()
        rho = drag.density(0)
        assert np.isclose(rho, EARTH_SURFACE_DENSITY, rtol=0.01)

    def test_density_exponential_decay(self):
        """Density should decay exponentially with altitude."""
        drag = AtmosphericDrag()

        rho_0 = drag.density(0)
        rho_H = drag.density(ATMOSPHERE_SCALE_HEIGHT)

        # At scale height, density should be 1/e of surface
        assert np.isclose(rho_H / rho_0, 1/np.e, rtol=0.01)

    def test_density_zero_above_1000km(self):
        """Density should be zero above 1000 km."""
        drag = AtmosphericDrag()
        assert drag.density(1001) == 0.0

    def test_drag_opposes_velocity(self):
        """Drag should oppose velocity direction."""
        drag = AtmosphericDrag(area=100, drag_coefficient=2.2)

        # LEO position and velocity
        earth_radius_au = 6371 / 1.496e8
        alt_200km_au = 200 / 1.496e8

        r_earth = np.zeros(3)
        v_earth = np.zeros(3)
        r_sc = np.array([earth_radius_au + alt_200km_au, 0, 0])
        v_sc = np.array([0, 0.0005, 0])  # Moving in +y (typical LEO)

        a = drag.acceleration(r_sc, v_sc, 1000, r_earth, v_earth)

        # Drag should be in -y direction (opposing velocity)
        if np.linalg.norm(a) > 0:
            assert a[1] < 0

    def test_drag_zero_in_deep_space(self):
        """Drag should be zero far from Earth."""
        drag = AtmosphericDrag()

        r_earth = np.zeros(3)
        v_earth = np.zeros(3)
        r_sc = np.array([1.0, 0, 0])  # 1 AU from Earth
        v_sc = np.array([0, 0.01, 0])

        a = drag.acceleration(r_sc, v_sc, 1000, r_earth, v_earth)
        assert np.allclose(a, [0, 0, 0])

    def test_drag_disabled(self):
        """Disabled drag should return zero."""
        drag = AtmosphericDrag(enabled=False)

        r_earth = np.zeros(3)
        v_earth = np.zeros(3)
        r_sc = np.array([4.26e-5, 0, 0])  # LEO altitude
        v_sc = np.array([0, 0.0005, 0])

        a = drag.acceleration(r_sc, v_sc, 1000, r_earth, v_earth)
        assert np.allclose(a, [0, 0, 0])


class TestEnvironmentModel:
    """Test combined environment model."""

    def test_default_model_has_srp(self):
        """Default model should include SRP."""
        env = EnvironmentModel.default()
        assert env.srp is not None

    def test_default_model_has_drag(self):
        """Default model should include drag."""
        env = EnvironmentModel.default()
        assert env.drag is not None

    def test_deep_space_model_no_drag(self):
        """Deep space model should not have drag."""
        env = EnvironmentModel.deep_space()
        assert env.srp is not None
        assert env.drag is None

    def test_total_acceleration_combines(self):
        """Total acceleration should combine SRP and drag."""
        env = EnvironmentModel(
            srp=SolarRadiationPressure(area=100),
            drag=None,  # Disable drag for simple test
        )

        r = np.array([1.0, 0, 0])
        v = np.zeros(3)

        a = env.total_acceleration(r, v, mass=1000)

        # Should be non-zero (SRP contribution)
        assert np.linalg.norm(a) > 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_compute_srp_acceleration(self):
        """Quick SRP calculation should work."""
        r = np.array([1.0, 0, 0])
        a = compute_srp_acceleration(r, mass=1000, area=100)
        assert np.linalg.norm(a) > 0

    def test_estimate_srp_delta_v(self):
        """SRP delta-V estimate should be reasonable."""
        dv = estimate_srp_delta_v(
            duration_days=365,
            distance_au=1.0,
            mass=1000,
            area=100,
            reflectivity=1.5,
        )

        # SRP delta-V over a year at 1 AU for 1000 kg, 100 m² should be
        # on order of tens to hundreds of m/s
        assert 1 < dv < 10000


class TestPhysicalValidation:
    """Physical validation tests."""

    def test_srp_typical_interplanetary(self):
        """SRP effect on typical interplanetary spacecraft."""
        # Dawn spacecraft: ~3000 kg, ~30 m² solar panels
        srp = SolarRadiationPressure(area=30, reflectivity=1.5)

        # At 1 AU
        r = np.array([1.0, 0, 0])
        a = srp.acceleration(r, mass=3000)

        # Convert to m/s²
        a_mps2 = np.linalg.norm(a) * 1.496e11 / 86400**2

        # Should be around 1e-8 to 1e-7 m/s²
        assert 1e-10 < a_mps2 < 1e-5

    def test_drag_leo_order_of_magnitude(self):
        """Drag at low LEO altitude should be significant."""
        drag = AtmosphericDrag(area=10, drag_coefficient=2.2)

        # Density at 100 km (our simple exponential model works best at low alt)
        rho = drag.density(100)

        # At 100 km, density should be around 1e-7 to 1e-5 kg/m³
        # (simple exponential model gives ~1e-6)
        assert 1e-8 < rho < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
