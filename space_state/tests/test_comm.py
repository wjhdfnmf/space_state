"""
Tests for Radio Communications Model
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base.star_comm import (
    Band, BAND_FREQUENCIES,
    Antenna, Transmitter, Receiver, GroundStation,
    DSN_STATIONS, NEAR_EARTH_STATIONS,
    free_space_path_loss, atmospheric_loss,
    compute_link_budget, max_data_rate,
    earth_rotation_angle, station_position_inertial,
    compute_elevation, is_visible,
    find_contact_windows, ContactWindow,
    CommSystem, estimate_data_volume,
)
from base.star_constants import AU_TO_KM
from base.star_soi import get_body_position


class TestBandDefinitions:
    """Test frequency band definitions."""

    def test_all_bands_have_frequencies(self):
        """All bands have defined frequencies."""
        for band in Band:
            assert band in BAND_FREQUENCIES
            assert BAND_FREQUENCIES[band] > 0

    def test_frequency_order(self):
        """Bands are in correct frequency order."""
        assert BAND_FREQUENCIES[Band.VHF] < BAND_FREQUENCIES[Band.UHF]
        assert BAND_FREQUENCIES[Band.UHF] < BAND_FREQUENCIES[Band.S_BAND]
        assert BAND_FREQUENCIES[Band.S_BAND] < BAND_FREQUENCIES[Band.X_BAND]
        assert BAND_FREQUENCIES[Band.X_BAND] < BAND_FREQUENCIES[Band.KA_BAND]


class TestAntenna:
    """Test antenna model."""

    def test_antenna_wavelength(self):
        """Wavelength calculation is correct."""
        antenna = Antenna("Test", diameter_m=1.0, frequency_hz=3e8)  # 1 meter wavelength
        assert abs(antenna.wavelength_m - 1.0) < 0.01

    def test_antenna_gain_increases_with_diameter(self):
        """Larger dishes have higher gain."""
        small = Antenna("Small", diameter_m=1.0, frequency_hz=8.4e9)
        large = Antenna("Large", diameter_m=10.0, frequency_hz=8.4e9)

        assert large.gain_db > small.gain_db

    def test_antenna_gain_increases_with_frequency(self):
        """Higher frequency gives higher gain for same dish."""
        low_freq = Antenna("LowF", diameter_m=5.0, frequency_hz=2e9)
        high_freq = Antenna("HighF", diameter_m=5.0, frequency_hz=8e9)

        assert high_freq.gain_db > low_freq.gain_db

    def test_dsn_70m_gain(self):
        """70m DSN dish has ~70+ dBi gain at X-band."""
        dsn70 = Antenna("DSS-14", diameter_m=70.0, frequency_hz=8.4e9)
        assert dsn70.gain_db > 70

    def test_beamwidth_decreases_with_frequency(self):
        """Higher frequency means narrower beam."""
        low = Antenna("Low", diameter_m=5.0, frequency_hz=2e9)
        high = Antenna("High", diameter_m=5.0, frequency_hz=8e9)

        assert high.beamwidth_deg < low.beamwidth_deg


class TestTransmitterReceiver:
    """Test transmitter and receiver models."""

    def test_eirp_calculation(self):
        """EIRP = power + gain - losses."""
        antenna = Antenna("Test", diameter_m=1.0, frequency_hz=8.4e9)
        tx = Transmitter(power_w=100, antenna=antenna, line_loss_db=1.0)

        # EIRP = 10*log10(100) + gain - 1 = 20 + gain - 1
        expected = 20 + antenna.gain_db - 1
        assert abs(tx.eirp_dbw - expected) < 0.01

    def test_g_over_t(self):
        """G/T calculation is correct."""
        antenna = Antenna("Test", diameter_m=10.0, frequency_hz=8.4e9)
        rx = Receiver(antenna=antenna, noise_temp_k=100.0, line_loss_db=0.5)

        # G/T = gain - 10*log10(100) - 0.5 = gain - 20 - 0.5
        expected = antenna.gain_db - 20 - 0.5
        assert abs(rx.g_over_t_db - expected) < 0.01


class TestGroundStations:
    """Test ground station database."""

    def test_dsn_stations_exist(self):
        """All three DSN stations are defined."""
        assert 'goldstone' in DSN_STATIONS
        assert 'madrid' in DSN_STATIONS
        assert 'canberra' in DSN_STATIONS

    def test_dsn_coverage(self):
        """DSN stations provide ~120° longitude separation."""
        gs = DSN_STATIONS['goldstone']
        md = DSN_STATIONS['madrid']
        cb = DSN_STATIONS['canberra']

        # Check approximate longitudes for global coverage
        assert -130 < gs.longitude_deg < -100  # California
        assert -10 < md.longitude_deg < 0      # Spain
        assert 140 < cb.longitude_deg < 160    # Australia

    def test_ground_station_ecef(self):
        """Ground station ECEF position is reasonable."""
        gs = DSN_STATIONS['goldstone']
        ecef = gs.ecef_position()

        # Should be on Earth's surface (~6371 km radius)
        radius = np.linalg.norm(ecef)
        assert 6350 < radius < 6400


class TestFreeSpacePathLoss:
    """Test FSPL calculations."""

    def test_fspl_increases_with_distance(self):
        """Path loss increases with distance."""
        freq = 8.4e9

        loss_1km = free_space_path_loss(1, freq)
        loss_10km = free_space_path_loss(10, freq)
        loss_100km = free_space_path_loss(100, freq)

        assert loss_10km > loss_1km
        assert loss_100km > loss_10km

    def test_fspl_increases_with_frequency(self):
        """Path loss increases with frequency."""
        dist = 1000  # km

        loss_s = free_space_path_loss(dist, 2e9)
        loss_x = free_space_path_loss(dist, 8e9)
        loss_ka = free_space_path_loss(dist, 32e9)

        assert loss_x > loss_s
        assert loss_ka > loss_x

    def test_fspl_distance_squared(self):
        """FSPL scales with distance squared (6 dB per doubling)."""
        freq = 8.4e9

        loss_1 = free_space_path_loss(1000, freq)
        loss_2 = free_space_path_loss(2000, freq)

        # Doubling distance adds ~6 dB
        assert abs((loss_2 - loss_1) - 6.02) < 0.1

    def test_fspl_leo_realistic(self):
        """LEO path loss is realistic (~170 dB at X-band)."""
        loss = free_space_path_loss(2000, 8.4e9)  # 2000 km slant range
        assert 165 < loss < 180


class TestAtmosphericLoss:
    """Test atmospheric attenuation."""

    def test_atm_loss_increases_at_low_elevation(self):
        """Lower elevation means more atmosphere."""
        loss_90 = atmospheric_loss(90, 8.4e9)  # Zenith
        loss_10 = atmospheric_loss(10, 8.4e9)  # Near horizon

        assert loss_10 > loss_90

    def test_atm_loss_increases_with_frequency(self):
        """Higher frequency has more attenuation."""
        loss_s = atmospheric_loss(45, 2e9)
        loss_ka = atmospheric_loss(45, 32e9)

        assert loss_ka > loss_s


class TestLinkBudget:
    """Test complete link budget calculation."""

    def test_link_budget_contains_all_terms(self):
        """Link budget includes all expected terms."""
        antenna = Antenna("Test", diameter_m=5.0, frequency_hz=8.4e9)
        tx = Transmitter(power_w=50, antenna=antenna)
        rx = Receiver(antenna=antenna)

        budget = compute_link_budget(tx, rx, distance_km=1000)

        assert 'eirp_dbw' in budget
        assert 'fspl_db' in budget
        assert 'received_power_dbw' in budget
        assert 'cn0_db_hz' in budget
        assert 'eb_n0' in budget

    def test_link_budget_power_balance(self):
        """Received power = EIRP - losses + Rx gain."""
        antenna = Antenna("Test", diameter_m=5.0, frequency_hz=8.4e9)
        tx = Transmitter(power_w=50, antenna=antenna)
        rx = Receiver(antenna=antenna, line_loss_db=0.5)

        budget = compute_link_budget(tx, rx, distance_km=1000, elevation_deg=90)

        expected_pr = (budget['eirp_dbw'] - budget['total_loss_db'] +
                      budget['rx_gain_db'] - rx.line_loss_db)

        assert abs(budget['received_power_dbw'] - expected_pr) < 0.1


class TestMaxDataRate:
    """Test data rate calculations."""

    def test_max_rate_decreases_with_distance(self):
        """Data rate decreases with distance."""
        antenna = Antenna("Test", diameter_m=10.0, frequency_hz=8.4e9)
        tx = Transmitter(power_w=20, antenna=antenna)
        rx = Receiver(antenna=antenna)

        rate_near = max_data_rate(tx, rx, 1000)
        rate_far = max_data_rate(tx, rx, 10000)

        assert rate_far < rate_near

    def test_leo_data_rate_reasonable(self):
        """LEO comm can achieve >1 Mbps."""
        # Small LEO spacecraft to ground
        sc_antenna = Antenna("LEO", diameter_m=0.3, frequency_hz=8.4e9)
        gs_antenna = Antenna("GS", diameter_m=11.0, frequency_hz=8.4e9)

        tx = Transmitter(power_w=10, antenna=sc_antenna)
        rx = Receiver(antenna=gs_antenna, noise_temp_k=100)

        rate = max_data_rate(tx, rx, 2000)  # 2000 km slant range

        assert rate > 1e6  # > 1 Mbps


class TestVisibility:
    """Test visibility and elevation calculations."""

    def test_earth_rotation(self):
        """Earth rotation angle increases with time."""
        angle_0 = earth_rotation_angle(0.0)
        angle_1 = earth_rotation_angle(1.0)  # 1 day later

        # Should rotate ~360 degrees in 1 day
        delta = angle_1 - angle_0
        assert abs(delta - 2*np.pi) < 0.1

    def test_elevation_directly_overhead(self):
        """Spacecraft directly above station has ~90° elevation."""
        station = GroundStation(
            name="Test",
            latitude_deg=0.0,
            longitude_deg=0.0,
        )

        # Place spacecraft directly above on equator at t=0
        # The spacecraft needs to be above the station, not just offset from Earth center
        # Station at lat=0, lon=0 is in +x direction from Earth center (at t=0)
        # We place spacecraft at 400 km altitude above the surface
        r_earth = get_body_position('earth', 0.0)
        earth_radius = 6371  # km
        altitude = 400  # km
        offset = (earth_radius + altitude) / AU_TO_KM
        sc_pos = r_earth + np.array([offset, 0, 0])

        elev = compute_elevation(station, sc_pos, 0.0)

        # Should be high elevation (may not be exactly 90° due to Earth rotation)
        assert elev > 45

    def test_visibility_check(self):
        """Visibility function works correctly."""
        station = GroundStation(
            name="Test",
            latitude_deg=35.0,
            longitude_deg=-117.0,
            min_elevation_deg=10.0,
        )

        # Get station position and place spacecraft directly above it
        from base.star_comm import station_position_inertial
        station_pos = station_position_inertial(station, 0.0)
        r_earth = get_body_position('earth', 0.0)

        # Direction from Earth center to station
        zenith_dir = station_pos - r_earth
        zenith_hat = zenith_dir / np.linalg.norm(zenith_dir)

        # Place spacecraft 1000 km above station (in zenith direction)
        offset = 1000 / AU_TO_KM
        sc_high = station_pos + zenith_hat * offset

        # Should be visible (directly overhead)
        assert is_visible(station, sc_high, 0.0)


class TestContactWindows:
    """Test contact window finding."""

    def test_empty_trajectory_no_contacts(self):
        """Empty trajectory gives no contacts."""
        contacts = find_contact_windows(
            np.array([]).reshape(0, 3),
            np.array([]),
        )
        assert len(contacts) == 0

    def test_contact_window_structure(self):
        """ContactWindow has correct structure."""
        window = ContactWindow(
            station_name="Test",
            start_time=0.0,
            end_time=0.1,
            max_elevation_deg=45.0,
            min_range_km=1000.0,
        )

        assert window.station_name == "Test"
        assert window.duration_minutes > 0


class TestCommSystem:
    """Test spacecraft comm system."""

    def test_comm_system_rate_limits(self):
        """Comm system respects rate limits."""
        antenna = Antenna("Test", diameter_m=1.0, frequency_hz=8.4e9)
        tx = Transmitter(power_w=100, antenna=antenna)
        rx = Receiver(antenna=antenna)

        comm = CommSystem(
            transmitter=tx,
            receiver=rx,
            max_data_rate_bps=1e6,
            min_data_rate_bps=100,
        )

        station = DSN_STATIONS['goldstone']

        # Very close - should hit max
        rate_close = comm.achievable_rate(station, 100)
        assert rate_close <= 1e6

        # Very far - should hit min
        rate_far = comm.achievable_rate(station, 1e12)
        assert rate_far >= 100


class TestDataVolumeEstimation:
    """Test data volume estimation."""

    def test_estimate_returns_valid_structure(self):
        """Data volume estimate has correct structure."""
        contacts = [
            ContactWindow("Goldstone DSN", 0.0, 0.01, 45.0, 1000.0),
        ]

        antenna = Antenna("Test", diameter_m=1.0, frequency_hz=8.4e9)
        comm = CommSystem(
            transmitter=Transmitter(power_w=20, antenna=antenna),
            receiver=Receiver(antenna=antenna),
        )

        result = estimate_data_volume(contacts, comm, DSN_STATIONS)

        assert 'total_bits' in result
        assert 'total_mbytes' in result
        assert 'num_contacts' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
