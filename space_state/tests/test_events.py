"""
Tests for Orbital Event Functions

Tests the event factory functions for use with scipy.integrate.solve_ivp.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base.star_events import (
    periapsis_event, apoapsis_event,
    altitude_event, distance_event,
    soi_entry_event, soi_exit_event,
    ascending_node_event, descending_node_event,
    throttle_events,
)
from base.star_constants import BODIES


class TestPeriapsisEvent:
    """Test periapsis event detection."""

    def test_periapsis_event_creation(self):
        """Test periapsis event factory creates callable."""
        event = periapsis_event(BODIES['moon'])
        assert callable(event)
        assert hasattr(event, 'terminal')
        assert hasattr(event, 'direction')
        assert hasattr(event, 'name')

    def test_periapsis_event_attributes(self):
        """Test periapsis event has correct attributes."""
        event = periapsis_event(BODIES['moon'], terminal=True)
        assert event.terminal is True
        assert event.direction == -1  # Crossing from positive to negative
        assert 'Periapsis' in event.name

    def test_periapsis_event_non_terminal(self):
        """Test non-terminal periapsis event."""
        event = periapsis_event(BODIES['moon'], terminal=False)
        assert event.terminal is False


class TestApoapsisEvent:
    """Test apoapsis event detection."""

    def test_apoapsis_event_creation(self):
        """Test apoapsis event factory creates callable."""
        event = apoapsis_event(BODIES['earth'])
        assert callable(event)
        assert hasattr(event, 'terminal')
        assert hasattr(event, 'direction')

    def test_apoapsis_event_direction(self):
        """Apoapsis should trigger on rising zero crossing."""
        event = apoapsis_event(BODIES['earth'])
        assert event.direction == +1  # Crossing from negative to positive


class TestAltitudeEvent:
    """Test altitude crossing events."""

    def test_altitude_event_creation(self):
        """Test altitude event factory."""
        event = altitude_event(BODIES['moon'], altitude_km=100.0)
        assert callable(event)
        assert 'Altitude' in event.name
        assert '100' in event.name

    def test_altitude_event_ascending(self):
        """Test ascending altitude crossing."""
        event = altitude_event(BODIES['moon'], altitude_km=100.0, direction=+1)
        assert event.direction == +1

    def test_altitude_event_descending(self):
        """Test descending altitude crossing."""
        event = altitude_event(BODIES['moon'], altitude_km=100.0, direction=-1)
        assert event.direction == -1

    def test_altitude_event_either_direction(self):
        """Test either direction altitude crossing."""
        event = altitude_event(BODIES['moon'], altitude_km=100.0, direction=0)
        assert event.direction == 0


class TestDistanceEvent:
    """Test distance events."""

    def test_distance_event_from_center(self):
        """Test distance from body center."""
        event = distance_event(BODIES['earth'], distance_km=10000.0, surface=False)
        assert callable(event)
        assert 'Distance' in event.name

    def test_distance_event_from_surface(self):
        """Test distance from body surface (altitude)."""
        event = distance_event(BODIES['moon'], distance_km=100.0, surface=True)
        assert callable(event)


class TestSOIEvents:
    """Test sphere of influence crossing events."""

    def test_soi_entry_event_creation(self):
        """Test SOI entry event factory."""
        event = soi_entry_event(BODIES['moon'])
        assert callable(event)
        assert event.direction == -1  # Distance decreasing (entering)
        assert 'Entry' in event.name

    def test_soi_exit_event_creation(self):
        """Test SOI exit event factory."""
        event = soi_exit_event(BODIES['moon'])
        assert callable(event)
        assert event.direction == +1  # Distance increasing (exiting)
        assert 'Exit' in event.name

    def test_soi_events_complementary(self):
        """SOI entry and exit should have opposite directions."""
        entry = soi_entry_event(BODIES['moon'])
        exit_ev = soi_exit_event(BODIES['moon'])
        assert entry.direction == -exit_ev.direction


class TestNodeEvents:
    """Test ascending/descending node events."""

    def test_ascending_node_creation(self):
        """Test ascending node event factory."""
        event = ascending_node_event(BODIES['earth'])
        assert callable(event)
        assert event.direction == +1  # Crossing from below (ascending)
        assert 'Ascending' in event.name

    def test_descending_node_creation(self):
        """Test descending node event factory."""
        event = descending_node_event(BODIES['earth'])
        assert callable(event)
        assert event.direction == -1  # Crossing from above (descending)
        assert 'Descending' in event.name

    def test_node_events_non_terminal_default(self):
        """Node events should default to non-terminal."""
        asc = ascending_node_event(BODIES['moon'])
        desc = descending_node_event(BODIES['moon'])
        assert asc.terminal is False
        assert desc.terminal is False

    def test_node_events_custom_reference_plane(self):
        """Test node events with custom reference plane."""
        # Use ecliptic plane (tilted from equator)
        ecliptic_normal = np.array([0, -0.3978, 0.9175])  # Approx
        event = ascending_node_event(BODIES['earth'], reference_plane=ecliptic_normal)
        assert callable(event)


class TestThrottleEvents:
    """Test propulsion-related events."""

    def test_throttle_events_with_none_guidance(self):
        """Test throttle events when guidance is None."""
        # Mock ship object
        class MockShip:
            dry_mass = 5000

        events = throttle_events(MockShip(), None)

        # Should return list of events
        assert isinstance(events, list)
        assert len(events) == 3  # engine_on, engine_off, propellant_depletion

    def test_throttle_events_attributes(self):
        """Test throttle event attributes."""
        class MockShip:
            dry_mass = 5000

        events = throttle_events(MockShip(), None)

        # All events should be terminal
        for event in events:
            assert event.terminal is True
            assert hasattr(event, 'name')

    def test_throttle_events_names(self):
        """Test throttle event names."""
        class MockShip:
            dry_mass = 5000

        events = throttle_events(MockShip(), None)

        names = [e.name for e in events]
        assert 'Engine Ignition' in names
        assert 'Engine Cutoff' in names
        assert 'Propellant Depletion' in names


class TestEventCallability:
    """Test that events can be called with proper arguments."""

    def test_periapsis_event_callable(self):
        """Test periapsis event can be called with t and y."""
        event = periapsis_event(BODIES['moon'])

        # Mock state: position and velocity relative to Moon
        # y = [rx, ry, rz, vx, vy, vz, mass, q0, q1, q2, q3, wx, wy, wz]
        t = 0.0
        y = np.array([
            0.003, 0.0, 0.0,     # Position (AU) - roughly at Moon distance
            0.0, 0.001, 0.0,     # Velocity (AU/day)
            10000.0,             # Mass (kg)
            1.0, 0.0, 0.0, 0.0,  # Quaternion
            0.0, 0.0, 0.0        # Angular velocity
        ])

        # Should return a float (the event function value)
        result = event(t, y)
        assert isinstance(result, (int, float))

    def test_altitude_event_callable(self):
        """Test altitude event can be called with t and y."""
        event = altitude_event(BODIES['moon'], altitude_km=100.0)

        t = 0.0
        y = np.array([
            0.003, 0.0, 0.0,
            0.0, 0.001, 0.0,
            10000.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])

        result = event(t, y)
        assert isinstance(result, (int, float))


class TestEventSignConvention:
    """Test that event functions follow scipy sign convention."""

    def test_event_returns_numeric(self):
        """Event functions should return numeric values."""
        event = periapsis_event(BODIES['earth'])

        t = 0.0
        y = np.array([
            1.0, 0.0, 0.0,       # 1 AU from origin
            0.0, 0.017, 0.0,     # ~Earth orbital velocity
            10000.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])

        result = event(t, y)
        assert np.isfinite(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
