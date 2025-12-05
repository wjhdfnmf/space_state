"""
Tests for Orbital Elements Module
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base.star_elements import (
    OrbitalElements,
    cartesian_to_keplerian, keplerian_to_cartesian,
    true_to_eccentric_anomaly, eccentric_to_true_anomaly,
    mean_to_eccentric_anomaly, eccentric_to_mean_anomaly,
    true_to_mean_anomaly, mean_to_true_anomaly,
    propagate_kepler, time_since_periapsis, time_to_true_anomaly,
    vis_viva, circular_velocity, escape_velocity, orbital_period,
    flight_path_angle, radius_at_true_anomaly,
    # Plane change functions
    plane_change_delta_v, combined_plane_change_delta_v,
    optimal_plane_change_location, plane_change_direction,
    compute_inclination_from_state,
    # Transfer orbit functions
    hohmann_transfer, bielliptic_transfer,
)


# Test constants (using Earth-like values in km, km/s)
MU_EARTH = 398600.4418  # km³/s²
R_EARTH = 6378.137      # km


class TestCartesianToKeplerian:
    """Test Cartesian to Keplerian conversion."""

    def test_circular_equatorial_orbit(self):
        """Test circular equatorial orbit."""
        r = 7000.0  # km
        v = np.sqrt(MU_EARTH / r)

        r_vec = np.array([r, 0, 0])
        v_vec = np.array([0, v, 0])

        elements = cartesian_to_keplerian(r_vec, v_vec, MU_EARTH)

        assert abs(elements.a - r) < 1.0  # Semi-major axis
        assert elements.e < 0.001  # Nearly circular
        assert abs(elements.i) < 0.001  # Equatorial

    def test_elliptical_orbit(self):
        """Test elliptical orbit with known elements."""
        # Create orbit with a=10000km, e=0.2
        a = 10000.0
        e = 0.2

        # At periapsis
        rp = a * (1 - e)
        vp = np.sqrt(MU_EARTH * (2/rp - 1/a))

        r_vec = np.array([rp, 0, 0])
        v_vec = np.array([0, vp, 0])

        elements = cartesian_to_keplerian(r_vec, v_vec, MU_EARTH)

        assert abs(elements.a - a) / a < 0.001
        assert abs(elements.e - e) < 0.001
        assert abs(elements.nu) < 0.001  # At periapsis, nu = 0

    def test_inclined_orbit(self):
        """Test inclined orbit."""
        r = 7000.0
        v = np.sqrt(MU_EARTH / r)
        inc = np.radians(45)

        r_vec = np.array([r, 0, 0])
        v_vec = np.array([0, v * np.cos(inc), v * np.sin(inc)])

        elements = cartesian_to_keplerian(r_vec, v_vec, MU_EARTH)

        assert abs(np.degrees(elements.i) - 45) < 1.0

    def test_hyperbolic_orbit(self):
        """Test hyperbolic escape trajectory."""
        r = 7000.0
        v_escape = np.sqrt(2 * MU_EARTH / r)
        v = v_escape * 1.2  # 20% above escape

        r_vec = np.array([r, 0, 0])
        v_vec = np.array([0, v, 0])

        elements = cartesian_to_keplerian(r_vec, v_vec, MU_EARTH)

        assert elements.e > 1.0  # Hyperbolic
        assert elements.a < 0  # Negative semi-major axis


class TestKeplerianToCartesian:
    """Test Keplerian to Cartesian conversion."""

    def test_roundtrip_circular(self):
        """Test roundtrip conversion for circular orbit."""
        r_orig = np.array([7000.0, 0, 0])
        v_orig = np.array([0, np.sqrt(MU_EARTH / 7000), 0])

        elements = cartesian_to_keplerian(r_orig, v_orig, MU_EARTH)
        r_new, v_new = keplerian_to_cartesian(elements, MU_EARTH)

        np.testing.assert_allclose(r_new, r_orig, rtol=1e-6)
        np.testing.assert_allclose(v_new, v_orig, rtol=1e-6)

    def test_roundtrip_elliptical(self):
        """Test roundtrip for elliptical orbit at arbitrary true anomaly."""
        a = 12000.0
        e = 0.3
        i = np.radians(28.5)
        raan = np.radians(45)
        omega = np.radians(30)
        nu = np.radians(60)

        elements = OrbitalElements(a=a, e=e, i=i, raan=raan, omega=omega, nu=nu, mu=MU_EARTH)
        r, v = keplerian_to_cartesian(elements, MU_EARTH)

        elements_back = cartesian_to_keplerian(r, v, MU_EARTH)

        assert abs(elements_back.a - a) / a < 1e-6
        assert abs(elements_back.e - e) < 1e-6
        assert abs(elements_back.i - i) < 1e-6

    def test_velocity_magnitude_vis_viva(self):
        """Test velocity magnitude matches vis-viva."""
        a = 10000.0
        e = 0.25
        nu = np.radians(90)

        elements = OrbitalElements(a=a, e=e, i=0, raan=0, omega=0, nu=nu, mu=MU_EARTH)
        r, v = keplerian_to_cartesian(elements, MU_EARTH)

        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        v_expected = vis_viva(r_mag, a, MU_EARTH)

        assert abs(v_mag - v_expected) / v_expected < 1e-6


class TestAnomalyConversions:
    """Test anomaly conversions."""

    def test_true_eccentric_roundtrip(self):
        """Test true to eccentric anomaly roundtrip."""
        e = 0.3
        for nu in np.linspace(0, 2*np.pi, 12):
            E = true_to_eccentric_anomaly(nu, e)
            nu_back = eccentric_to_true_anomaly(E, e)
            # Handle wraparound
            diff = abs(nu - nu_back)
            if diff > np.pi:
                diff = 2*np.pi - diff
            assert diff < 1e-10

    def test_mean_eccentric_roundtrip(self):
        """Test mean to eccentric anomaly roundtrip."""
        e = 0.4
        for M in np.linspace(0, 2*np.pi, 12):
            E = mean_to_eccentric_anomaly(M, e)
            M_back = eccentric_to_mean_anomaly(E, e)
            # Handle wraparound
            M_back = M_back % (2*np.pi)
            diff = abs(M - M_back)
            if diff > np.pi:
                diff = 2*np.pi - diff
            assert diff < 1e-10

    def test_keplers_equation(self):
        """Test Kepler's equation: M = E - e*sin(E)."""
        e = 0.5
        M = np.radians(45)
        E = mean_to_eccentric_anomaly(M, e)

        M_check = E - e * np.sin(E)
        assert abs(M - M_check) < 1e-12

    def test_hyperbolic_anomaly(self):
        """Test hyperbolic anomaly conversions."""
        e = 1.5  # Hyperbolic
        nu = np.radians(30)

        H = true_to_eccentric_anomaly(nu, e)
        nu_back = eccentric_to_true_anomaly(H, e)

        assert abs(nu - nu_back) < 1e-10


class TestKeplerPropagation:
    """Test Kepler propagation."""

    def test_one_period(self):
        """Test propagation through one orbital period."""
        a = 8000.0
        T = orbital_period(a, MU_EARTH)

        elements = OrbitalElements(a=a, e=0.1, i=0, raan=0, omega=0, nu=0, mu=MU_EARTH)
        elements_after = propagate_kepler(elements, T, MU_EARTH)

        # Should return to same true anomaly (within 2π)
        nu_diff = abs(elements_after.nu - elements.nu) % (2*np.pi)
        if nu_diff > np.pi:
            nu_diff = 2*np.pi - nu_diff
        assert nu_diff < 1e-6

    def test_half_period(self):
        """Test propagation through half period."""
        a = 8000.0
        e = 0.2
        T = orbital_period(a, MU_EARTH)

        elements = OrbitalElements(a=a, e=e, i=0, raan=0, omega=0, nu=0, mu=MU_EARTH)
        elements_half = propagate_kepler(elements, T/2, MU_EARTH)

        # Should be at apoapsis (nu = π)
        assert abs(elements_half.nu - np.pi) < 0.01

    def test_time_since_periapsis_zero(self):
        """Test time since periapsis at periapsis is zero."""
        elements = OrbitalElements(a=8000, e=0.1, i=0, raan=0, omega=0, nu=0, mu=MU_EARTH)
        t = time_since_periapsis(elements, MU_EARTH)
        assert abs(t) < 1e-10


class TestOrbitalVelocities:
    """Test orbital velocity calculations."""

    def test_circular_velocity(self):
        """Test circular velocity at LEO."""
        r = R_EARTH + 400  # 400 km altitude
        v_circ = circular_velocity(r, MU_EARTH)

        # LEO velocity should be ~7.7 km/s
        assert 7.5 < v_circ < 8.0

    def test_escape_velocity(self):
        """Test escape velocity relationship."""
        r = 7000.0
        v_circ = circular_velocity(r, MU_EARTH)
        v_esc = escape_velocity(r, MU_EARTH)

        # v_escape = sqrt(2) * v_circular
        assert abs(v_esc / v_circ - np.sqrt(2)) < 1e-10

    def test_vis_viva_at_periapsis(self):
        """Test vis-viva at periapsis."""
        a = 10000.0
        e = 0.3
        rp = a * (1 - e)

        v = vis_viva(rp, a, MU_EARTH)

        # Check with specific energy
        energy = -MU_EARTH / (2 * a)
        v_check = np.sqrt(2 * (energy + MU_EARTH / rp))

        assert abs(v - v_check) < 1e-10


class TestOrbitalPeriod:
    """Test orbital period calculations."""

    def test_leo_period(self):
        """Test LEO orbital period."""
        a = R_EARTH + 400  # 400 km altitude
        T = orbital_period(a, MU_EARTH)

        # LEO period should be ~90 minutes = 5400 seconds
        assert 5000 < T < 6000

    def test_geo_period(self):
        """Test GEO orbital period (should be ~sidereal day)."""
        # GEO altitude is about 35,786 km
        a = R_EARTH + 35786
        T = orbital_period(a, MU_EARTH)

        # Should be close to 86164 seconds (sidereal day = 23h 56m 4s)
        assert abs(T - 86164) < 100


class TestFlightPathAngle:
    """Test flight path angle calculations."""

    def test_circular_zero_fpa(self):
        """Test circular orbit has zero flight path angle."""
        elements = OrbitalElements(a=8000, e=0.0001, i=0, raan=0, omega=0, nu=np.pi/4, mu=MU_EARTH)
        gamma = flight_path_angle(elements)
        assert abs(gamma) < 0.001

    def test_periapsis_zero_fpa(self):
        """Test flight path angle is zero at periapsis."""
        elements = OrbitalElements(a=8000, e=0.3, i=0, raan=0, omega=0, nu=0, mu=MU_EARTH)
        gamma = flight_path_angle(elements)
        assert abs(gamma) < 1e-10

    def test_apoapsis_zero_fpa(self):
        """Test flight path angle is zero at apoapsis."""
        elements = OrbitalElements(a=8000, e=0.3, i=0, raan=0, omega=0, nu=np.pi, mu=MU_EARTH)
        gamma = flight_path_angle(elements)
        assert abs(gamma) < 1e-10


class TestRadiusAtTrueAnomaly:
    """Test radius calculation at true anomaly."""

    def test_periapsis_radius(self):
        """Test radius at periapsis."""
        a = 10000.0
        e = 0.25
        rp = radius_at_true_anomaly(a, e, 0)
        assert abs(rp - a * (1 - e)) < 1e-10

    def test_apoapsis_radius(self):
        """Test radius at apoapsis."""
        a = 10000.0
        e = 0.25
        ra = radius_at_true_anomaly(a, e, np.pi)
        assert abs(ra - a * (1 + e)) < 1e-10

    def test_semi_latus_rectum(self):
        """Test radius at 90 degrees equals semi-latus rectum."""
        a = 10000.0
        e = 0.25
        p = a * (1 - e**2)
        r90 = radius_at_true_anomaly(a, e, np.pi/2)
        assert abs(r90 - p) < 1e-10


class TestOrbitalElementsDataclass:
    """Test OrbitalElements dataclass methods."""

    def test_period_property(self):
        """Test period property."""
        a = 8000.0
        elements = OrbitalElements(a=a, e=0.1, i=0, raan=0, omega=0, nu=0, mu=MU_EARTH)
        T_expected = orbital_period(a, MU_EARTH)
        assert abs(elements.period - T_expected) < 1e-10

    def test_periapsis_property(self):
        """Test periapsis property."""
        a = 10000.0
        e = 0.3
        elements = OrbitalElements(a=a, e=e, i=0, raan=0, omega=0, nu=0, mu=MU_EARTH)
        assert abs(elements.periapsis - a * (1 - e)) < 1e-10

    def test_apoapsis_property(self):
        """Test apoapsis property."""
        a = 10000.0
        e = 0.3
        elements = OrbitalElements(a=a, e=e, i=0, raan=0, omega=0, nu=0, mu=MU_EARTH)
        assert abs(elements.apoapsis - a * (1 + e)) < 1e-10

    def test_hyperbolic_no_apoapsis(self):
        """Test hyperbolic orbit has no apoapsis."""
        elements = OrbitalElements(a=-10000, e=1.5, i=0, raan=0, omega=0, nu=0, mu=MU_EARTH)
        assert elements.apoapsis is None

    def test_to_dict(self):
        """Test dictionary conversion."""
        elements = OrbitalElements(a=8000, e=0.1, i=np.radians(28.5), raan=0, omega=0, nu=0, mu=MU_EARTH)
        d = elements.to_dict()
        assert 'a' in d
        assert 'e' in d
        assert abs(d['i'] - 28.5) < 0.01  # Should be in degrees


class TestPlaneChangeManeuvers:
    """Test plane change maneuver calculations."""

    def test_plane_change_delta_v_zero(self):
        """Zero inclination change requires zero delta-V."""
        v = 7.5  # km/s
        delta_i = 0.0
        dv = plane_change_delta_v(v, delta_i)
        assert abs(dv) < 1e-10

    def test_plane_change_delta_v_formula(self):
        """Test plane change formula: Δv = 2v sin(Δi/2)."""
        v = 7.5  # km/s
        delta_i = np.radians(10)  # 10° change
        dv = plane_change_delta_v(v, delta_i)
        expected = 2 * v * np.sin(delta_i / 2)
        assert abs(dv - expected) < 1e-10

    def test_plane_change_delta_v_symmetric(self):
        """Plane change delta-V is same magnitude for +/- inclination change."""
        v = 7.5
        delta_i = np.radians(15)
        dv_pos = plane_change_delta_v(v, delta_i)
        dv_neg = plane_change_delta_v(v, -delta_i)
        assert abs(dv_pos - dv_neg) < 1e-10

    def test_combined_plane_change_reduces_to_simple(self):
        """Combined plane change with same velocity reduces to simple formula."""
        v = 7.5
        delta_i = np.radians(10)
        # When v1 = v2 = v, combined formula should give same result
        dv_combined = combined_plane_change_delta_v(v, v, delta_i)
        dv_simple = plane_change_delta_v(v, delta_i)
        assert abs(dv_combined - dv_simple) < 1e-10

    def test_combined_plane_change_law_of_cosines(self):
        """Test combined plane change follows law of cosines."""
        v1 = 7.5
        v2 = 8.0
        delta_i = np.radians(20)
        dv = combined_plane_change_delta_v(v1, v2, delta_i)
        expected = np.sqrt(v1**2 + v2**2 - 2*v1*v2*np.cos(delta_i))
        assert abs(dv - expected) < 1e-10


class TestOptimalPlaneChangeLocation:
    """Test optimal plane change location finding."""

    def test_ascending_descending_separated_180(self):
        """Ascending and descending nodes are 180° apart in true anomaly."""
        elements = OrbitalElements(a=8000, e=0.1, i=np.radians(30),
                                   raan=np.radians(45), omega=np.radians(60), nu=0)
        result = optimal_plane_change_location(elements)

        nu_asc = result['ascending_node_nu']
        nu_desc = result['descending_node_nu']

        # Difference should be π (with 2π periodicity)
        diff = abs(nu_asc - nu_desc)
        if diff > np.pi:
            diff = 2*np.pi - diff
        assert abs(diff - np.pi) < 0.01

    def test_optimal_location_returns_valid_type(self):
        """Optimal type should be ascending or descending."""
        elements = OrbitalElements(a=10000, e=0.2, i=np.radians(45),
                                   raan=0, omega=0, nu=0)
        result = optimal_plane_change_location(elements)
        assert result['optimal_type'] in ['ascending', 'descending']


class TestInclinationFromState:
    """Test inclination computation from state vectors."""

    def test_equatorial_orbit_zero_inclination(self):
        """Equatorial orbit should have zero inclination."""
        r = np.array([7000, 0, 0])  # In equatorial plane
        v = np.array([0, 7.5, 0])   # Velocity in equatorial plane
        inc = compute_inclination_from_state(r, v)
        assert abs(inc) < 0.001

    def test_polar_orbit_90_inclination(self):
        """Polar orbit should have 90° inclination."""
        r = np.array([7000, 0, 0])
        v = np.array([0, 0, 7.5])  # Velocity perpendicular to equator
        inc = compute_inclination_from_state(r, v)
        assert abs(np.degrees(inc) - 90) < 0.1

    def test_inclined_orbit(self):
        """Test orbit with known inclination."""
        r = np.array([7000, 0, 0])
        inc_target = np.radians(45)
        v = np.array([0, 7.5 * np.cos(inc_target), 7.5 * np.sin(inc_target)])
        inc = compute_inclination_from_state(r, v)
        assert abs(np.degrees(inc) - 45) < 1.0


class TestHohmannTransfer:
    """Test Hohmann transfer calculations."""

    def test_hohmann_transfer_basic(self):
        """Test basic Hohmann transfer properties."""
        r1 = 6678.0   # LEO radius (300 km altitude)
        r2 = 42164.0  # GEO radius
        result = hohmann_transfer(r1, r2, MU_EARTH)

        # Semi-major axis of transfer orbit
        a_expected = (r1 + r2) / 2
        assert abs(result['a_transfer'] - a_expected) < 1.0

        # Total delta-V should be positive
        assert result['dv_total'] > 0
        assert result['dv1'] > 0
        assert result['dv2'] > 0

    def test_hohmann_leo_to_geo(self):
        """Test LEO to GEO transfer delta-V is approximately correct."""
        r1 = R_EARTH + 200   # 200 km LEO
        r2 = R_EARTH + 35786  # GEO
        result = hohmann_transfer(r1, r2, MU_EARTH)

        # LEO to GEO should require ~3.9 km/s total
        assert 3.5 < result['dv_total'] < 4.5

    def test_hohmann_transfer_time(self):
        """Test Hohmann transfer time is half period."""
        r1 = 7000.0
        r2 = 14000.0
        result = hohmann_transfer(r1, r2, MU_EARTH)

        # Transfer time should be π√(a³/μ)
        a = result['a_transfer']
        expected_time = np.pi * np.sqrt(a**3 / MU_EARTH)
        assert abs(result['transfer_time'] - expected_time) < 1.0


class TestBiellipticTransfer:
    """Test bi-elliptic transfer calculations."""

    def test_bielliptic_basic(self):
        """Test basic bi-elliptic transfer properties."""
        r1 = 7000.0
        r2 = 105000.0  # r2/r1 = 15 > 11.94, bielliptic can be better
        rb = 200000.0  # Intermediate apoapsis

        result = bielliptic_transfer(r1, r2, rb, MU_EARTH)

        # Should have three delta-V maneuvers
        assert result['dv1'] > 0
        assert result['dv2'] > 0
        assert result['dv3'] > 0
        assert abs(result['dv_total'] - (result['dv1'] + result['dv2'] + result['dv3'])) < 1e-10

    def test_bielliptic_transfer_ellipses(self):
        """Test bi-elliptic uses correct transfer ellipse parameters."""
        r1 = 7000.0
        r2 = 50000.0
        rb = 100000.0

        result = bielliptic_transfer(r1, r2, rb, MU_EARTH)

        # First ellipse: periapsis at r1, apoapsis at rb
        assert abs(result['a1'] - (r1 + rb) / 2) < 1.0

        # Second ellipse: periapsis at r2, apoapsis at rb
        assert abs(result['a2'] - (rb + r2) / 2) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
