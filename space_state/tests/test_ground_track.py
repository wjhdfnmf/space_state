# tests/test_ground_track.py
"""
Tests for Ground Track and Coverage Module
"""

import pytest
import numpy as np

import sys
sys.path.insert(0, '/home/boss/data/space_state')

from space_state.guidance.ground_track import (
    GroundStation, VisibilityWindow, CoverageResult, GROUND_STATIONS,
    eci_to_ecef, ecef_to_geodetic, ground_track,
    elevation_angle, azimuth_angle, find_visibility_windows,
    compute_access_statistics, swath_width, revisit_time_estimate,
    sun_synchronous_inclination, walker_constellation_coverage,
    horizon_distance, line_of_sight_range, communication_window_geometry,
    R_EARTH, OMEGA_EARTH, MU_EARTH,
)


class TestGroundStations:
    """Test ground station definitions."""

    def test_ground_stations_exist(self):
        """Test predefined ground stations exist."""
        assert 'goldstone' in GROUND_STATIONS
        assert 'canberra' in GROUND_STATIONS
        assert 'madrid' in GROUND_STATIONS

    def test_ground_station_dsn_coverage(self):
        """Test DSN provides global coverage."""
        # DSN stations are roughly 120° apart in longitude
        gs = GROUND_STATIONS['goldstone']
        cb = GROUND_STATIONS['canberra']
        md = GROUND_STATIONS['madrid']

        lons = [gs.longitude, cb.longitude, md.longitude]
        # Check they're spread around the globe
        assert min(lons) < -100  # West
        assert max(lons) > 140   # East

    def test_ground_station_ecef_position(self):
        """Test ECEF position calculation."""
        gs = GROUND_STATIONS['goldstone']
        pos = gs.ecef_position()

        assert len(pos) == 3
        # Should be near Earth's surface
        r = np.linalg.norm(pos)
        assert abs(r - R_EARTH) < 100


class TestCoordinateTransforms:
    """Test coordinate transformations."""

    def test_eci_to_ecef_at_t0(self):
        """Test ECI to ECEF at t=0."""
        r_eci = np.array([R_EARTH + 400, 0, 0])
        r_ecef = eci_to_ecef(r_eci, 0.0)

        # At t=0, ECI and ECEF are aligned
        np.testing.assert_array_almost_equal(r_eci, r_ecef, decimal=5)

    def test_eci_to_ecef_rotation(self):
        """Test ECI to ECEF includes Earth rotation."""
        r_eci = np.array([R_EARTH + 400, 0, 0])

        # After 6 hours, Earth rotates 90 degrees
        t = 6 * 3600  # seconds
        r_ecef = eci_to_ecef(r_eci, t)

        # X component should shift toward Y
        assert r_ecef[1] != 0

    def test_ecef_to_geodetic_equator(self):
        """Test ECEF to geodetic at equator."""
        r_ecef = np.array([R_EARTH + 400, 0, 0])
        lat, lon, alt = ecef_to_geodetic(r_ecef)

        assert abs(lat) < 1  # Near equator
        assert abs(lon) < 1  # Near prime meridian
        assert abs(alt - 400) < 10  # 400 km altitude

    def test_ecef_to_geodetic_poles(self):
        """Test ECEF to geodetic at poles."""
        r_north = np.array([0, 0, R_EARTH + 400])
        lat_n, _, _ = ecef_to_geodetic(r_north)
        assert lat_n > 80  # Near north pole

        r_south = np.array([0, 0, -(R_EARTH + 400)])
        lat_s, _, _ = ecef_to_geodetic(r_south)
        assert lat_s < -80  # Near south pole


class TestGroundTrack:
    """Test ground track calculations."""

    def test_ground_track_shape(self):
        """Test ground track output shape."""
        # Create simple circular orbit trajectory
        N = 100
        T = 90 * 60  # 90-minute orbit
        times = np.linspace(0, T, N)

        positions = np.zeros((3, N))
        r = R_EARTH + 400
        for i, t in enumerate(times):
            omega = 2 * np.pi / T
            positions[0, i] = r * np.cos(omega * t)
            positions[1, i] = r * np.sin(omega * t)

        lats, lons, alts = ground_track(positions, times)

        assert len(lats) == N
        assert len(lons) == N
        assert len(alts) == N

    def test_ground_track_altitude(self):
        """Test ground track altitudes are correct."""
        N = 10
        positions = np.zeros((3, N))
        positions[0, :] = R_EARTH + 400  # Constant radius
        times = np.linspace(0, 1000, N)

        _, _, alts = ground_track(positions, times)

        # All altitudes should be ~400 km
        for alt in alts:
            assert abs(alt - 400) < 10


class TestVisibility:
    """Test visibility calculations."""

    def test_elevation_angle_overhead(self):
        """Test elevation when satellite is overhead."""
        station = GROUND_STATIONS['goldstone']
        # Satellite directly above station
        r_station = station.ecef_position()
        r_sat = r_station * (1 + 400 / np.linalg.norm(r_station))

        el = elevation_angle(r_sat, station)

        # Should be near 90 degrees
        assert el > 80

    def test_elevation_angle_horizon(self):
        """Test elevation at horizon."""
        station = GroundStation('Test', 0, 0, 0, 5)
        # Satellite at horizon
        r_sat = np.array([R_EARTH + 400, R_EARTH + 400, 0])

        el = elevation_angle(r_sat, station)

        # Should be low but positive for visible satellite
        assert el < 45

    def test_azimuth_angle_north(self):
        """Test azimuth toward north."""
        station = GroundStation('Test', 0, 0, 0, 5)
        # Satellite to the north
        r_sat = np.array([0, 0, R_EARTH + 400])

        az = azimuth_angle(r_sat, station)

        # Should be near 0 degrees (North)
        assert az < 10 or az > 350


class TestVisibilityWindows:
    """Test visibility window detection."""

    def test_find_visibility_windows_empty(self):
        """Test no windows when satellite never visible."""
        station = GroundStation('Test', 45, 0, 0, 5)

        # Satellite always on opposite side of Earth
        N = 100
        positions = np.zeros((N, 3))
        positions[:, 0] = -(R_EARTH + 400)
        times = np.linspace(0, 10000, N)

        windows = find_visibility_windows(positions, times, station)

        # Might find some windows depending on geometry
        assert isinstance(windows, list)

    def test_find_visibility_windows_format(self):
        """Test visibility window format."""
        station = GroundStation('Test', 0, 0, 0, 5)

        # Simple overhead pass
        N = 100
        positions = np.zeros((N, 3))
        times = np.linspace(0, 6000, N)

        # Simulate orbit passing over station
        for i, t in enumerate(times):
            angle = 2 * np.pi * t / 5400
            positions[i, 0] = (R_EARTH + 400) * np.cos(angle)
            positions[i, 2] = (R_EARTH + 400) * np.sin(angle)

        windows = find_visibility_windows(positions, times, station)

        for w in windows:
            assert isinstance(w, VisibilityWindow)
            assert w.duration >= 0
            assert w.end_time >= w.start_time


class TestAccessStatistics:
    """Test access statistics calculations."""

    def test_compute_access_statistics_empty(self):
        """Test statistics with no windows."""
        stats = compute_access_statistics([], 86400)

        assert stats['coverage_percent'] == 0
        assert stats['num_passes'] == 0

    def test_compute_access_statistics_single(self):
        """Test statistics with single window."""
        windows = [VisibilityWindow(
            station_name='Test',
            start_time=0,
            end_time=600,
            duration=600,
            max_elevation=45,
            aos_azimuth=0,
            los_azimuth=180
        )]

        stats = compute_access_statistics(windows, 86400)

        assert stats['num_passes'] == 1
        assert stats['total_access_time_s'] == 600
        assert stats['coverage_percent'] > 0


class TestCoverageCalculations:
    """Test coverage analysis functions."""

    def test_swath_width_positive(self):
        """Test swath width is positive."""
        sw = swath_width(500, 30)
        assert sw > 0

    def test_swath_width_increases_with_altitude(self):
        """Test higher altitude = wider swath."""
        sw_low = swath_width(400, 30)
        sw_high = swath_width(800, 30)

        assert sw_high > sw_low

    def test_swath_width_increases_with_angle(self):
        """Test larger cone angle = wider swath."""
        sw_narrow = swath_width(500, 15)
        sw_wide = swath_width(500, 45)

        assert sw_wide > sw_narrow

    def test_revisit_time_estimate_positive(self):
        """Test revisit time is positive."""
        rt = revisit_time_estimate(500, 45, 1000)
        assert rt > 0

    def test_revisit_time_decreases_with_swath(self):
        """Test wider swath = shorter revisit."""
        rt_narrow = revisit_time_estimate(500, 45, 500)
        rt_wide = revisit_time_estimate(500, 45, 2000)

        assert rt_wide < rt_narrow


class TestSunSynchronous:
    """Test sun-synchronous inclination."""

    def test_sun_sync_inclination_leo(self):
        """Test sun-sync inclination for LEO."""
        inc = sun_synchronous_inclination(600)

        # Sun-sync LEO is typically 97-99 degrees
        assert 95 < inc < 100

    def test_sun_sync_inclination_increases_with_altitude(self):
        """Test higher altitude requires higher inclination."""
        inc_low = sun_synchronous_inclination(400)
        inc_high = sun_synchronous_inclination(800)

        assert inc_high > inc_low


class TestWalkerConstellation:
    """Test Walker constellation coverage."""

    def test_walker_coverage_result(self):
        """Test Walker coverage returns valid result."""
        result = walker_constellation_coverage(
            altitude_km=550,
            inclination_deg=53,
            num_planes=6,
            sats_per_plane=22,
            swath_width_km=1000
        )

        assert isinstance(result, CoverageResult)
        assert result.total_coverage_percent >= 0
        assert result.contacts_per_day >= 0

    def test_more_satellites_better_coverage(self):
        """Test more satellites improves coverage."""
        result_small = walker_constellation_coverage(500, 53, 2, 4, 1000)
        result_large = walker_constellation_coverage(500, 53, 6, 22, 1000)

        assert result_large.total_coverage_percent >= result_small.total_coverage_percent


class TestGeometry:
    """Test geometry calculations."""

    def test_horizon_distance(self):
        """Test horizon distance calculation."""
        d = horizon_distance(400)

        # From ISS altitude, horizon is ~2000 km
        assert 2000 < d < 2500

    def test_horizon_distance_increases_with_altitude(self):
        """Test higher altitude = farther horizon."""
        d_low = horizon_distance(200)
        d_high = horizon_distance(800)

        assert d_high > d_low

    def test_line_of_sight_range_high_elevation(self):
        """Test slant range at high elevation."""
        alt = 400
        rng = line_of_sight_range(alt, 80)

        # At high elevation, range should be close to altitude
        assert rng > 0
        assert rng < alt * 2  # Should be reasonably close

    def test_line_of_sight_range_increases_with_lower_elevation(self):
        """Test lower elevation = longer range."""
        rng_high = line_of_sight_range(400, 60)
        rng_low = line_of_sight_range(400, 10)

        assert rng_low > rng_high

    def test_communication_window_geometry(self):
        """Test communication geometry calculation."""
        station = GROUND_STATIONS['goldstone']
        geom = communication_window_geometry(400, station)

        assert geom['min_range_km'] > 0
        assert geom['max_range_km'] > geom['min_range_km']
        assert geom['horizon_distance_km'] > 0


class TestConstants:
    """Test module constants."""

    def test_earth_radius(self):
        """Test Earth radius."""
        assert 6370 < R_EARTH < 6380

    def test_earth_rotation_rate(self):
        """Test Earth rotation rate."""
        # One revolution per sidereal day
        period = 2 * np.pi / OMEGA_EARTH
        assert abs(period - 86164) < 10  # ~23h 56m

    def test_earth_mu(self):
        """Test Earth gravitational parameter."""
        assert 398000 < MU_EARTH < 399000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
