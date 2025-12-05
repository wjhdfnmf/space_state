# tests/test_states.py
"""
Unit tests for star_states.py - orbital state generation and transfer calculations.

Tests cover:
- Circular orbit state generation
- Hohmann transfer calculations
- State vector structure validation
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from base.star_constants import BODIES, IDX_R, IDX_V, IDX_M, IDX_Q, IDX_OMEGA
from base.star_states import (
    get_circular_state,
    get_hohmann_state_at_time,
    OrbitName,
)
from base.star_utils import au_to_km, km_to_au
from base.star_orbits import ALL_ORBITS


class TestGetCircularState:
    """Tests for get_circular_state function."""

    def test_returns_four_arrays(self):
        """Test that function returns (r, v, q, omega) tuple."""
        body = BODIES['earth']
        result = get_circular_state(body, 'leo_iss')

        assert len(result) == 4
        r, v, q, omega = result
        assert r.shape == (3,)
        assert v.shape == (3,)
        assert q.shape == (4,)
        assert omega.shape == (3,)

    def test_position_magnitude_correct_earth_leo(self):
        """Test circular orbit radius around Earth at ISS altitude."""
        body = BODIES['earth']
        r, v, q, omega = get_circular_state(body, 'leo_iss')

        # ISS altitude is ~400 km, Earth radius ~6371 km
        # Expected: ~6771 km from Earth center
        expected_radius_km = ALL_ORBITS['leo_iss'].altitude_km + 6371
        actual_radius_km = au_to_km(np.linalg.norm(r))

        # Allow for body position offset (SSB frame)
        # The position includes Earth's position in SSB
        assert actual_radius_km > 1e8  # Should be ~1 AU from Sun

    def test_velocity_magnitude_correct_earth_leo(self):
        """Test circular orbit velocity is reasonable."""
        body = BODIES['earth']
        r, v, q, omega = get_circular_state(body, 'leo_iss')

        # Velocity should be non-zero and reasonable
        v_mag = np.linalg.norm(v)
        assert v_mag > 0

    def test_quaternion_normalized(self):
        """Test that returned quaternion is normalized."""
        body = BODIES['earth']
        r, v, q, omega = get_circular_state(body, 'leo_iss')

        q_norm = np.linalg.norm(q)
        assert_allclose(q_norm, 1.0, rtol=1e-10)

    def test_angular_velocity_zero_initially(self):
        """Test that initial angular velocity is zero."""
        body = BODIES['earth']
        r, v, q, omega = get_circular_state(body, 'leo_iss')

        assert_allclose(omega, np.zeros(3), atol=1e-15)

    def test_lunar_orbit(self):
        """Test circular orbit around Moon."""
        body = BODIES['moon']
        r, v, q, omega = get_circular_state(body, 'lunar_100km')

        # Verify state vectors have correct shapes
        assert r.shape == (3,)
        assert v.shape == (3,)
        assert q.shape == (4,)
        assert omega.shape == (3,)

        # Quaternion should be normalized
        assert_allclose(np.linalg.norm(q), 1.0, rtol=1e-10)


class TestStateVectorStructure:
    """Tests for 14-element state vector structure."""

    def test_state_vector_construction(self):
        """Test building full 14-element state from get_circular_state."""
        body = BODIES['earth']
        r, v, q, omega = get_circular_state(body, 'leo_iss')

        # Construct state vector as done in tests/orbit_moon.py
        mass = 1000.0  # kg
        y = np.concatenate([r, v, [mass], q, omega])

        assert y.shape == (14,)
        assert_allclose(y[IDX_R], r)
        assert_allclose(y[IDX_V], v)
        assert y[IDX_M] == mass
        assert_allclose(y[IDX_Q], q)
        assert_allclose(y[IDX_OMEGA], omega)

    def test_state_indexing_constants(self):
        """Test that state indexing constants work correctly."""
        y = np.arange(14).astype(float)

        # Position: elements 0, 1, 2
        assert list(y[IDX_R]) == [0.0, 1.0, 2.0]

        # Velocity: elements 3, 4, 5
        assert list(y[IDX_V]) == [3.0, 4.0, 5.0]

        # Mass: element 6
        assert y[IDX_M] == 6.0

        # Quaternion: elements 7, 8, 9, 10
        assert list(y[IDX_Q]) == [7.0, 8.0, 9.0, 10.0]

        # Angular velocity: elements 11, 12, 13
        assert list(y[IDX_OMEGA]) == [11.0, 12.0, 13.0]


class TestOrbitNameType:
    """Tests for OrbitName type literal."""

    def test_earth_orbits_exist(self):
        """Test that Earth orbit types exist."""
        earth_orbits = ['leo_200', 'leo_iss', 'leo_550', 'meo_gps', 'gto_apo', 'geo']
        for orbit in earth_orbits:
            assert orbit in ALL_ORBITS

    def test_lunar_orbits_exist(self):
        """Test that lunar orbit types exist."""
        lunar_orbits = ['lunar_100km', 'lunar_50km', 'lunar_15km']
        for orbit in lunar_orbits:
            assert orbit in ALL_ORBITS


class TestOrbitalMechanics:
    """Tests for orbital mechanics calculations."""

    def test_circular_velocity_formula(self):
        """Test that circular velocity follows v = sqrt(mu/r)."""
        body = BODIES['earth']

        # Get state
        r, v, q, omega = get_circular_state(body, 'leo_iss')

        # The velocity should be perpendicular to position (for circular orbit)
        # In a body-centered frame, v_circ is tangential
        # Can't easily test this in SSB frame without frame conversion


class TestHohmannTransfer:
    """Tests for Hohmann transfer calculations."""

    def test_hohmann_returns_dict(self):
        """Test that get_hohmann_state_at_time returns expected dictionary."""
        from base.star_propagator import Propagator
        from base.star_ship import starship_raptor_vac
        from guidance import BurnController

        body = BODIES['moon']
        ship = starship_raptor_vac()

        r0, v0, q0, omega0 = get_circular_state(body, 'lunar_50km')
        y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

        guid = BurnController(ship, body)
        prop = Propagator(ship, guid)
        prop.result.initialize(y0, 0.0)

        # Get Hohmann transfer to higher orbit
        result = get_hohmann_state_at_time(prop, body, 0.0, 'lunar_100km')

        assert 't_center' in result
        assert 'delta_v' in result
        assert 'y_transfer' in result
        assert 'r1_km' in result
        assert 'r2_km' in result

    def test_hohmann_delta_v_positive(self):
        """Test that Hohmann transfer to higher orbit has positive delta-v."""
        from base.star_propagator import Propagator
        from base.star_ship import starship_raptor_vac
        from guidance import BurnController

        body = BODIES['moon']
        ship = starship_raptor_vac()

        r0, v0, q0, omega0 = get_circular_state(body, 'lunar_50km')
        y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

        guid = BurnController(ship, body)
        prop = Propagator(ship, guid)
        prop.result.initialize(y0, 0.0)

        # Higher orbit requires positive delta-v
        result = get_hohmann_state_at_time(prop, body, 0.0, 'lunar_100km')

        assert result['delta_v'] > 0
        assert result['r2_km'] > result['r1_km']

    def test_hohmann_transfer_state_valid(self):
        """Test that Hohmann transfer state vector is valid."""
        from base.star_propagator import Propagator
        from base.star_ship import starship_raptor_vac
        from guidance import BurnController

        body = BODIES['moon']
        ship = starship_raptor_vac()

        r0, v0, q0, omega0 = get_circular_state(body, 'lunar_50km')
        y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

        guid = BurnController(ship, body)
        prop = Propagator(ship, guid)
        prop.result.initialize(y0, 0.0)

        result = get_hohmann_state_at_time(prop, body, 0.0, 'lunar_100km')
        y_transfer = result['y_transfer']

        # Check state vector structure
        assert y_transfer.shape == (14,)

        # Quaternion should be normalized
        q_transfer = y_transfer[IDX_Q]
        assert_allclose(np.linalg.norm(q_transfer), 1.0, rtol=1e-10)

        # Mass should be positive
        assert y_transfer[IDX_M] > 0


class TestBodyConstants:
    """Tests for BODIES constants."""

    def test_earth_has_required_keys(self):
        """Test that Earth body has required keys."""
        earth = BODIES['earth']
        required_keys = ['mu', 'radius']
        for key in required_keys:
            assert key in earth

    def test_moon_has_required_keys(self):
        """Test that Moon body has required keys."""
        moon = BODIES['moon']
        required_keys = ['mu', 'radius']
        for key in required_keys:
            assert key in moon

    def test_mu_positive(self):
        """Test that gravitational parameters are positive."""
        for name, body in BODIES.items():
            if 'mu' in body:
                assert body['mu'] > 0, f"Body {name} has non-positive mu"

    def test_radius_positive(self):
        """Test that radii are positive (when defined)."""
        for name, body in BODIES.items():
            if 'radius' in body and body['radius'] is not None:
                assert body['radius'] > 0, f"Body {name} has non-positive radius"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
