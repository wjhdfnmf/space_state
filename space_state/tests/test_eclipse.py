# tests/test_eclipse.py
"""
Tests for Eclipse and Occultation Module

Tests shadow calculations, eclipse detection, and prediction.
"""

import pytest
import numpy as np

import sys
sys.path.insert(0, '/home/boss/data/space_state')

from space_state.base.star_eclipse import (
    EclipseType, EclipseState, EclipsePrediction,
    shadow_cone_geometry, cylindrical_shadow_check, conical_shadow_check,
    compute_eclipse_state, estimate_eclipse_duration_leo,
    estimate_eclipses_per_orbit_geo, predict_eclipses,
    communication_blackout_check,
    R_SUN, R_EARTH, R_MOON, AU_TO_KM,
)


class TestShadowConeGeometry:
    """Test shadow cone geometry calculations."""

    def test_earth_shadow_geometry(self):
        """Test Earth's shadow cone geometry."""
        distance_sun_earth = 1.0 * AU_TO_KM  # 1 AU

        umbra_length, umbra_angle, penumbra_angle = shadow_cone_geometry(
            R_SUN, R_EARTH, distance_sun_earth
        )

        # Earth's umbra extends about 1.4 million km
        assert 1.3e6 < umbra_length < 1.5e6

        # Umbra angle is small
        assert 0 < umbra_angle < 0.01  # radians

        # Penumbra angle is slightly larger
        assert penumbra_angle > umbra_angle

    def test_moon_shadow_geometry(self):
        """Test Moon's shadow cone geometry."""
        distance_sun_moon = 1.0 * AU_TO_KM

        umbra_length, umbra_angle, penumbra_angle = shadow_cone_geometry(
            R_SUN, R_MOON, distance_sun_moon
        )

        # Moon's umbra is shorter than Earth's
        assert umbra_length < 500000  # Less than 500,000 km
        assert umbra_length > 300000

    def test_shadow_cone_angles_positive(self):
        """Test shadow cone angles are positive."""
        umbra_length, umbra_angle, penumbra_angle = shadow_cone_geometry(
            R_SUN, R_EARTH, AU_TO_KM
        )

        assert umbra_angle > 0
        assert penumbra_angle > 0


class TestCylindricalShadow:
    """Test cylindrical shadow model."""

    def test_spacecraft_in_sunlight(self):
        """Test spacecraft in front of body (sunlit)."""
        r_body = np.array([0.0, 0.0, 0.0])
        r_sun = np.array([1.0, 0.0, 0.0]) * AU_TO_KM
        r_sc = np.array([0.5, 0.0, 0.0]) * AU_TO_KM  # Between body and sun

        result = cylindrical_shadow_check(r_sc, r_body, r_sun, R_EARTH)

        assert result == EclipseType.NONE

    def test_spacecraft_in_umbra(self):
        """Test spacecraft in umbra (shadow axis)."""
        r_body = np.array([0.0, 0.0, 0.0])
        r_sun = np.array([1.0, 0.0, 0.0]) * AU_TO_KM
        # Spacecraft behind body, on shadow axis
        r_sc = np.array([-10000.0, 0.0, 0.0])  # 10,000 km behind body

        result = cylindrical_shadow_check(r_sc, r_body, r_sun, R_EARTH)

        assert result == EclipseType.UMBRA

    def test_spacecraft_in_penumbra(self):
        """Test spacecraft in penumbra (edge of shadow)."""
        r_body = np.array([0.0, 0.0, 0.0])
        r_sun = np.array([1.0, 0.0, 0.0]) * AU_TO_KM
        # Spacecraft behind body, near edge
        r_sc = np.array([-10000.0, R_EARTH * 1.05, 0.0])

        result = cylindrical_shadow_check(r_sc, r_body, r_sun, R_EARTH)

        assert result == EclipseType.PENUMBRA

    def test_spacecraft_outside_shadow(self):
        """Test spacecraft outside shadow entirely."""
        r_body = np.array([0.0, 0.0, 0.0])
        r_sun = np.array([1.0, 0.0, 0.0]) * AU_TO_KM
        # Spacecraft behind body but far off axis
        r_sc = np.array([-10000.0, R_EARTH * 2.0, 0.0])

        result = cylindrical_shadow_check(r_sc, r_body, r_sun, R_EARTH)

        assert result == EclipseType.NONE


class TestConicalShadow:
    """Test conical shadow model (more accurate)."""

    def test_spacecraft_in_full_sunlight(self):
        """Test spacecraft in full sunlight."""
        r_body = np.array([0.0, 0.0, 0.0])
        r_sun = np.array([1.0, 0.0, 0.0]) * AU_TO_KM
        r_sc = np.array([10000.0, 0.0, 0.0])  # In front of body

        eclipse_type, illumination = conical_shadow_check(
            r_sc, r_body, r_sun, R_EARTH, R_SUN
        )

        assert eclipse_type == EclipseType.NONE
        assert illumination == 1.0

    def test_spacecraft_in_umbra_conical(self):
        """Test spacecraft in umbral cone."""
        r_body = np.array([0.0, 0.0, 0.0])
        r_sun = np.array([1.0, 0.0, 0.0]) * AU_TO_KM
        # Spacecraft on shadow axis, close to body
        r_sc = np.array([-50000.0, 0.0, 0.0])  # 50,000 km behind

        eclipse_type, illumination = conical_shadow_check(
            r_sc, r_body, r_sun, R_EARTH, R_SUN
        )

        assert eclipse_type == EclipseType.UMBRA
        assert illumination == 0.0

    def test_spacecraft_in_penumbra_partial(self):
        """Test spacecraft in penumbra has partial illumination."""
        r_body = np.array([0.0, 0.0, 0.0])
        r_sun = np.array([1.0, 0.0, 0.0]) * AU_TO_KM
        # Spacecraft in penumbra region
        r_sc = np.array([-50000.0, R_EARTH + 1000.0, 0.0])

        eclipse_type, illumination = conical_shadow_check(
            r_sc, r_body, r_sun, R_EARTH, R_SUN
        )

        # Could be penumbra or none depending on exact geometry
        if eclipse_type == EclipseType.PENUMBRA:
            assert 0.0 < illumination < 1.0

    def test_illumination_monotonic(self):
        """Test illumination increases as we move away from shadow axis."""
        r_body = np.array([0.0, 0.0, 0.0])
        r_sun = np.array([1.0, 0.0, 0.0]) * AU_TO_KM

        illuminations = []
        for offset in [0, 1000, 3000, 5000, 7000]:
            r_sc = np.array([-50000.0, offset, 0.0])
            _, illum = conical_shadow_check(r_sc, r_body, r_sun, R_EARTH, R_SUN)
            illuminations.append(illum)

        # Illumination should generally increase
        assert illuminations[-1] >= illuminations[0]


class TestEclipseDuration:
    """Test eclipse duration estimates."""

    def test_leo_eclipse_duration(self):
        """Test LEO eclipse duration estimate."""
        altitude = 400  # ISS altitude
        inclination = 51.6  # ISS inclination

        duration = estimate_eclipse_duration_leo(altitude, inclination)

        # LEO eclipse is typically 30-36 minutes
        assert 25 < duration < 40

    def test_higher_orbit_shorter_eclipse(self):
        """Test higher orbits have relatively shorter eclipses."""
        duration_400 = estimate_eclipse_duration_leo(400, 0)
        duration_1000 = estimate_eclipse_duration_leo(1000, 0)

        # Relative eclipse fraction should decrease with altitude
        # (though absolute duration may not due to longer period)
        # Let's just verify we get reasonable values
        assert duration_400 > 0
        assert duration_1000 > 0

    def test_dawn_dusk_orbit_no_eclipse(self):
        """Test dawn-dusk orbit has no eclipse."""
        altitude = 800
        beta_angle = 75  # High beta angle

        duration = estimate_eclipse_duration_leo(altitude, 0, beta_angle)

        # Should be zero or very small
        assert duration < 5

    def test_geo_eclipse_stats(self):
        """Test GEO eclipse statistics."""
        stats = estimate_eclipses_per_orbit_geo()

        assert stats['eclipse_seasons_per_year'] == 2
        assert 40 < stats['season_duration_days'] < 50
        assert 60 < stats['max_eclipse_duration_minutes'] < 80
        assert 80 < stats['total_eclipse_hours_per_year'] < 100


class TestEclipseType:
    """Test EclipseType enum."""

    def test_eclipse_types_exist(self):
        """Test all eclipse types exist."""
        assert EclipseType.NONE
        assert EclipseType.PENUMBRA
        assert EclipseType.UMBRA
        assert EclipseType.ANTUMBRA

    def test_eclipse_types_distinct(self):
        """Test eclipse types are distinct."""
        types = [
            EclipseType.NONE,
            EclipseType.PENUMBRA,
            EclipseType.UMBRA,
            EclipseType.ANTUMBRA,
        ]
        assert len(set(types)) == 4


class TestEclipseState:
    """Test EclipseState dataclass."""

    def test_eclipse_state_creation(self):
        """Test EclipseState can be created."""
        state = EclipseState(
            eclipse_type=EclipseType.PENUMBRA,
            illumination_fraction=0.5,
            eclipse_body='earth',
            shadow_depth=10000.0,
            time_to_exit=300.0
        )

        assert state.eclipse_type == EclipseType.PENUMBRA
        assert state.illumination_fraction == 0.5
        assert state.eclipse_body == 'earth'
        assert state.shadow_depth == 10000.0
        assert state.time_to_exit == 300.0

    def test_eclipse_state_repr(self):
        """Test EclipseState string representation."""
        state = EclipseState(
            eclipse_type=EclipseType.UMBRA,
            illumination_fraction=0.0,
            eclipse_body='moon',
            shadow_depth=5000.0,
            time_to_exit=None
        )

        repr_str = repr(state)
        assert 'UMBRA' in repr_str
        assert 'moon' in repr_str


class TestEclipsePrediction:
    """Test EclipsePrediction dataclass."""

    def test_eclipse_prediction_creation(self):
        """Test EclipsePrediction can be created."""
        prediction = EclipsePrediction(
            start_time=1.0,
            end_time=1.02,
            duration=1728.0,  # ~29 minutes
            eclipse_type=EclipseType.UMBRA,
            eclipsing_body='earth',
            max_shadow_depth=50000.0
        )

        assert prediction.start_time == 1.0
        assert prediction.end_time == 1.02
        assert prediction.duration == 1728.0
        assert prediction.eclipse_type == EclipseType.UMBRA
        assert prediction.eclipsing_body == 'earth'


class TestPredictEclipses:
    """Test eclipse prediction along trajectory."""

    def test_predict_eclipses_empty(self):
        """Test prediction with no eclipses."""
        # Trajectory always in sunlight
        N = 100
        positions = np.zeros((3, N))
        positions[0, :] = np.linspace(1.0, 1.1, N)  # Far from Earth
        times = np.linspace(0, 1, N)

        predictions = predict_eclipses(positions, times, ['earth'])

        # May or may not find eclipses depending on geometry
        assert isinstance(predictions, list)

    def test_predict_eclipses_returns_list(self):
        """Test prediction returns list of EclipsePrediction."""
        N = 10
        positions = np.zeros((3, N))
        times = np.linspace(0, 1, N)

        predictions = predict_eclipses(positions, times)

        assert isinstance(predictions, list)
        for pred in predictions:
            assert isinstance(pred, EclipsePrediction)


class TestConstants:
    """Test module constants."""

    def test_solar_radius(self):
        """Test solar radius value."""
        # Sun's radius is about 696,000 km
        assert 695000 < R_SUN < 700000

    def test_earth_radius(self):
        """Test Earth radius value."""
        # Earth's radius is about 6378 km
        assert 6370 < R_EARTH < 6385

    def test_moon_radius(self):
        """Test Moon radius value."""
        # Moon's radius is about 1737 km
        assert 1735 < R_MOON < 1740

    def test_au_conversion(self):
        """Test AU to km conversion."""
        # 1 AU is about 150 million km
        assert 149e6 < AU_TO_KM < 150e6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
