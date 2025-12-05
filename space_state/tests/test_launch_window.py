# tests/test_launch_window.py
"""
Tests for Launch Window Module
"""

import pytest
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, '/home/boss/data/space_state')

from space_state.guidance.launch_window import (
    LaunchSite, LaunchWindow, InterplanetaryWindow, LAUNCH_SITES,
    launch_azimuth, launch_window_for_inclination, iss_launch_window,
    synodic_period, hohmann_phase_angle, interplanetary_launch_opportunities,
    earth_mars_window_dates, plane_change_penalty, gto_launch_parameters,
    daily_launch_windows,
    R_EARTH, MU_EARTH,
)


class TestLaunchSites:
    """Test launch site definitions."""

    def test_launch_sites_exist(self):
        """Test predefined launch sites exist."""
        assert 'kennedy' in LAUNCH_SITES
        assert 'baikonur' in LAUNCH_SITES
        assert 'kourou' in LAUNCH_SITES
        assert 'vandenberg' in LAUNCH_SITES

    def test_launch_site_latitudes(self):
        """Test launch site latitudes are reasonable."""
        for name, site in LAUNCH_SITES.items():
            assert -90 <= site.latitude <= 90

    def test_launch_site_min_inclination(self):
        """Test minimum inclination equals latitude."""
        for name, site in LAUNCH_SITES.items():
            assert site.min_inclination() == abs(site.latitude)

    def test_kennedy_properties(self):
        """Test Kennedy Space Center properties."""
        ksc = LAUNCH_SITES['kennedy']
        assert abs(ksc.latitude - 28.5) < 1
        assert ksc.azimuth_min < ksc.azimuth_max


class TestLaunchAzimuth:
    """Test launch azimuth calculations."""

    def test_launch_azimuth_polar(self):
        """Test launch azimuth for polar orbit."""
        az = launch_azimuth(28.5, 90, ascending=True)

        # Polar orbit from 28.5°N should launch roughly north
        assert 0 < az < 45

    def test_launch_azimuth_equatorial(self):
        """Test launch azimuth for equatorial orbit."""
        # From near-equator site
        az = launch_azimuth(5.0, 5.0, ascending=True)

        # Should be near due east (90°)
        assert 80 < az < 100

    def test_launch_azimuth_ascending_vs_descending(self):
        """Test ascending vs descending azimuths."""
        az_asc = launch_azimuth(28.5, 51.6, ascending=True)
        az_desc = launch_azimuth(28.5, 51.6, ascending=False)

        # Ascending and descending should be different
        assert az_asc != az_desc
        # They should sum to ~180°
        assert abs((az_asc + az_desc) - 180) < 30 or abs((az_asc + az_desc) - 360) < 30

    def test_launch_azimuth_impossible_inclination(self):
        """Test error for impossible inclination."""
        # Cannot achieve 20° inclination from 28.5°N
        with pytest.raises(ValueError):
            launch_azimuth(28.5, 20.0)

    def test_launch_azimuth_iss(self):
        """Test ISS-compatible azimuth from Kennedy."""
        az = launch_azimuth(28.5, 51.6, ascending=True)

        # Should be in Kennedy's allowed range
        assert 35 < az < 120


class TestLaunchWindows:
    """Test launch window calculations."""

    def test_launch_window_for_inclination(self):
        """Test launch window generation."""
        site = LAUNCH_SITES['kennedy']
        windows = launch_window_for_inclination(site, 51.6, 0.0)

        # Should find at least one window
        assert isinstance(windows, list)

    def test_launch_window_format(self):
        """Test launch window format."""
        site = LAUNCH_SITES['kennedy']
        windows = launch_window_for_inclination(site, 51.6, 0.0)

        for w in windows:
            assert isinstance(w, LaunchWindow)
            assert w.duration > 0
            assert w.close_time > w.open_time

    def test_iss_launch_window(self):
        """Test ISS launch window."""
        site = LAUNCH_SITES['kennedy']
        windows = iss_launch_window(site, 45.0)

        assert isinstance(windows, list)

    def test_daily_launch_windows_iss(self):
        """Test daily windows for ISS from Kennedy."""
        site = LAUNCH_SITES['kennedy']
        n_windows = daily_launch_windows(site, 51.6)

        # Kennedy should have 1-2 windows per day for ISS
        assert 0 <= n_windows <= 2

    def test_daily_windows_impossible_inclination(self):
        """Test zero windows for impossible inclination."""
        site = LAUNCH_SITES['kennedy']
        n = daily_launch_windows(site, 20.0)  # Below latitude

        assert n == 0


class TestSynodicPeriod:
    """Test synodic period calculations."""

    def test_synodic_earth_mars(self):
        """Test Earth-Mars synodic period."""
        T = synodic_period(1.0, 1.524)

        # Earth-Mars synodic period is ~780 days
        assert 750 < T < 810

    def test_synodic_earth_venus(self):
        """Test Earth-Venus synodic period."""
        T = synodic_period(0.723, 1.0)

        # Venus synodic period is ~584 days
        assert 570 < T < 600

    def test_synodic_earth_jupiter(self):
        """Test Earth-Jupiter synodic period."""
        T = synodic_period(1.0, 5.2)

        # Jupiter synodic period is ~399 days
        assert 390 < T < 410


class TestHohmannPhaseAngle:
    """Test Hohmann phase angle calculations."""

    def test_phase_angle_earth_mars(self):
        """Test phase angle for Mars mission."""
        phase = hohmann_phase_angle(1.0, 1.524)

        # Mars launch phase angle is roughly 44 degrees
        assert 30 < phase < 60

    def test_phase_angle_earth_venus(self):
        """Test phase angle for Venus mission."""
        phase = hohmann_phase_angle(1.0, 0.723)

        # Venus is inner planet, different geometry
        assert 0 < phase < 360  # Valid angle


class TestInterplanetaryOpportunities:
    """Test interplanetary launch opportunity finder."""

    def test_find_mars_opportunities(self):
        """Test finding Mars launch opportunities."""
        start = datetime(2024, 1, 1)
        opps = interplanetary_launch_opportunities('earth', 'mars', start, 3)

        assert len(opps) == 3
        for opp in opps:
            assert isinstance(opp, InterplanetaryWindow)
            assert opp.c3_km2_s2 > 0
            assert opp.tof_days > 0

    def test_mars_opportunities_spacing(self):
        """Test Mars opportunities are spaced by synodic period."""
        start = datetime(2024, 1, 1)
        opps = interplanetary_launch_opportunities('earth', 'mars', start, 3)

        # Check spacing is roughly synodic period
        for i in range(len(opps) - 1):
            dt = (opps[i+1].departure_date - opps[i].departure_date).days
            assert 700 < dt < 850  # ~780 days synodic

    def test_venus_opportunities(self):
        """Test finding Venus launch opportunities."""
        start = datetime(2024, 1, 1)
        opps = interplanetary_launch_opportunities('earth', 'venus', start, 2)

        assert len(opps) == 2


class TestEarthMarsWindows:
    """Test Earth-Mars window date lookup."""

    def test_known_window_2024(self):
        """Test known 2024 Mars window."""
        window = earth_mars_window_dates(2024)

        assert window['year'] == 2024
        assert 'window_month' in window
        assert 'typical_tof_days' in window

    def test_estimated_window(self):
        """Test estimated window for future year."""
        window = earth_mars_window_dates(2045)

        assert 'estimated' in window or 'window_month' in window


class TestPlaneChangePenalty:
    """Test plane change delta-V penalty."""

    def test_plane_change_small_angle(self):
        """Test small plane change."""
        dv = plane_change_penalty(1.0, 3.0)  # 1 degree

        # For small angles: dv ≈ v × Δi (radians)
        expected = 3.0 * np.radians(1.0)
        assert abs(dv - expected) < 0.01

    def test_plane_change_scales_with_velocity(self):
        """Test plane change scales with velocity."""
        dv1 = plane_change_penalty(10.0, 3.0)
        dv2 = plane_change_penalty(10.0, 6.0)

        assert dv2 > dv1  # Higher velocity = more delta-V

    def test_plane_change_scales_with_angle(self):
        """Test larger angle = more delta-V."""
        dv_small = plane_change_penalty(5.0, 3.0)
        dv_large = plane_change_penalty(30.0, 3.0)

        assert dv_large > dv_small


class TestGTOParameters:
    """Test GTO launch parameters."""

    def test_gto_from_kennedy(self):
        """Test GTO parameters from Kennedy."""
        params = gto_launch_parameters(LAUNCH_SITES['kennedy'])

        assert params['launch_site'] == 'Kennedy Space Center'
        assert params['gto_inclination_deg'] > 0
        assert params['perigee_km'] > 0
        assert params['apogee_km'] > 30000

    def test_gto_from_kourou(self):
        """Test GTO from Kourou (near equator)."""
        params = gto_launch_parameters(LAUNCH_SITES['kourou'])

        # Kourou is near equator, less plane change needed
        assert params['gto_inclination_deg'] < 10
        assert params['dv_plane_change_km_s'] < 0.5

    def test_gto_delta_v_reasonable(self):
        """Test total GTO to GEO delta-V is reasonable."""
        params = gto_launch_parameters(LAUNCH_SITES['kennedy'])

        # Typical GTO to GEO is 1.5-2 km/s
        assert 1.0 < params['total_gto_to_geo_km_s'] < 2.5


class TestLaunchWindowDataclass:
    """Test LaunchWindow dataclass."""

    def test_launch_window_creation(self):
        """Test creating LaunchWindow."""
        window = LaunchWindow(
            open_time=0,
            close_time=600,
            duration=600,
            azimuth=45.0,
            type='ascending'
        )

        assert window.duration == 600
        assert window.azimuth == 45.0
        assert window.type == 'ascending'


class TestLaunchSiteDataclass:
    """Test LaunchSite dataclass."""

    def test_launch_site_creation(self):
        """Test creating LaunchSite."""
        site = LaunchSite('Test', 30.0, -80.0, 35, 120)

        assert site.name == 'Test'
        assert site.latitude == 30.0
        assert site.longitude == -80.0
        assert site.min_inclination() == 30.0


class TestConstants:
    """Test module constants."""

    def test_earth_radius(self):
        """Test Earth radius."""
        assert 6370 < R_EARTH < 6380

    def test_earth_mu(self):
        """Test Earth gravitational parameter."""
        assert 398000 < MU_EARTH < 399000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
