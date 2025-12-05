# base/star_comm.py
"""
Radio Communications Model

Provides link budget calculations for spacecraft communications:
- Free-space path loss
- Antenna gain modeling
- Data rate vs distance
- Ground station visibility
- Contact window scheduling

All calculations are optional and do not affect core propagation.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .star_constants import BODIES, AU_TO_KM, DAY_TO_SEC
from .star_soi import get_body_position
from .star_logger import logger


# Speed of light (m/s)
C = 299792458.0

# Boltzmann constant (J/K)
K_BOLTZMANN = 1.380649e-23


class Band(Enum):
    """Communication frequency bands."""
    S_BAND = "s_band"      # 2-4 GHz
    X_BAND = "x_band"      # 8-12 GHz
    KA_BAND = "ka_band"    # 26-40 GHz
    UHF = "uhf"            # 300-3000 MHz
    VHF = "vhf"            # 30-300 MHz


# Standard frequency bands (Hz)
BAND_FREQUENCIES = {
    Band.S_BAND: 2.2e9,    # 2.2 GHz
    Band.X_BAND: 8.4e9,    # 8.4 GHz
    Band.KA_BAND: 32e9,    # 32 GHz
    Band.UHF: 400e6,       # 400 MHz
    Band.VHF: 150e6,       # 150 MHz
}


@dataclass
class Antenna:
    """Antenna model."""
    name: str
    diameter_m: float = 1.0           # Dish diameter (m)
    efficiency: float = 0.55          # Aperture efficiency
    frequency_hz: float = 8.4e9       # Operating frequency
    pointing_loss_db: float = 0.5     # Pointing loss (dB)

    @property
    def wavelength_m(self) -> float:
        """Signal wavelength (m)."""
        return C / self.frequency_hz

    @property
    def gain_db(self) -> float:
        """Antenna gain (dBi)."""
        # G = η * (π * D / λ)²
        gain_linear = self.efficiency * (np.pi * self.diameter_m / self.wavelength_m) ** 2
        return 10 * np.log10(gain_linear)

    @property
    def beamwidth_deg(self) -> float:
        """Half-power beamwidth (degrees)."""
        # θ ≈ 70 * λ / D for parabolic dish
        return 70 * self.wavelength_m / self.diameter_m


@dataclass
class Transmitter:
    """Transmitter model."""
    power_w: float = 20.0             # Transmit power (W)
    antenna: Antenna = None           # Transmit antenna
    line_loss_db: float = 1.0         # Cable/waveguide losses

    @property
    def eirp_dbw(self) -> float:
        """Effective Isotropic Radiated Power (dBW)."""
        power_dbw = 10 * np.log10(self.power_w)
        gain = self.antenna.gain_db if self.antenna else 0
        return power_dbw + gain - self.line_loss_db


@dataclass
class Receiver:
    """Receiver model."""
    antenna: Antenna = None
    noise_temp_k: float = 150.0       # System noise temperature (K)
    line_loss_db: float = 0.5         # Receive line losses
    implementation_loss_db: float = 2.0  # Demodulator losses

    @property
    def g_over_t_db(self) -> float:
        """Figure of merit G/T (dB/K)."""
        gain = self.antenna.gain_db if self.antenna else 0
        noise_temp_db = 10 * np.log10(self.noise_temp_k)
        return gain - noise_temp_db - self.line_loss_db


@dataclass
class GroundStation:
    """Ground station definition."""
    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    antenna: Antenna = None
    receiver: Receiver = None
    transmitter: Transmitter = None
    min_elevation_deg: float = 10.0   # Minimum elevation for contact

    def ecef_position(self) -> np.ndarray:
        """Position in Earth-Centered Earth-Fixed frame (km)."""
        lat = np.radians(self.latitude_deg)
        lon = np.radians(self.longitude_deg)

        # WGS84 ellipsoid
        a = 6378.137  # Equatorial radius (km)
        f = 1 / 298.257223563
        e2 = 2*f - f**2

        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

        x = (N + self.altitude_m/1000) * np.cos(lat) * np.cos(lon)
        y = (N + self.altitude_m/1000) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - e2) + self.altitude_m/1000) * np.sin(lat)

        return np.array([x, y, z])


# ------------------- DSN GROUND STATIONS -------------------

# Deep Space Network stations
DSN_STATIONS = {
    'goldstone': GroundStation(
        name="Goldstone DSN",
        latitude_deg=35.4267,
        longitude_deg=-116.89,
        altitude_m=1031,
        antenna=Antenna("DSS-14", diameter_m=70.0, frequency_hz=8.4e9),
        min_elevation_deg=6.0,
    ),
    'madrid': GroundStation(
        name="Madrid DSN",
        latitude_deg=40.4314,
        longitude_deg=-4.2481,
        altitude_m=834,
        antenna=Antenna("DSS-63", diameter_m=70.0, frequency_hz=8.4e9),
        min_elevation_deg=6.0,
    ),
    'canberra': GroundStation(
        name="Canberra DSN",
        latitude_deg=-35.4014,
        longitude_deg=148.9817,
        altitude_m=689,
        antenna=Antenna("DSS-43", diameter_m=70.0, frequency_hz=8.4e9),
        min_elevation_deg=6.0,
    ),
}

# Near-Earth stations
NEAR_EARTH_STATIONS = {
    'wallops': GroundStation(
        name="Wallops Flight Facility",
        latitude_deg=37.9402,
        longitude_deg=-75.4664,
        altitude_m=12,
        antenna=Antenna("WFF-S", diameter_m=11.0, frequency_hz=2.2e9),
        min_elevation_deg=5.0,
    ),
    'svalbard': GroundStation(
        name="Svalbard Ground Station",
        latitude_deg=78.2306,
        longitude_deg=15.3897,
        altitude_m=458,
        antenna=Antenna("SvalSat", diameter_m=13.0, frequency_hz=8.4e9),
        min_elevation_deg=3.0,
    ),
}


# ------------------- LINK BUDGET CALCULATIONS -------------------

def free_space_path_loss(distance_km: float, frequency_hz: float) -> float:
    """
    Calculate free-space path loss.

    FSPL = (4πR/λ)² = (4πRf/c)²

    Args:
        distance_km: Distance (km)
        frequency_hz: Frequency (Hz)

    Returns:
        Path loss (dB)
    """
    distance_m = distance_km * 1000
    wavelength_m = C / frequency_hz

    loss_linear = (4 * np.pi * distance_m / wavelength_m) ** 2
    return 10 * np.log10(loss_linear)


def atmospheric_loss(elevation_deg: float, frequency_hz: float) -> float:
    """
    Estimate atmospheric attenuation.

    Simplified model for clear sky conditions.

    Args:
        elevation_deg: Elevation angle (degrees)
        frequency_hz: Frequency (Hz)

    Returns:
        Atmospheric loss (dB)
    """
    if elevation_deg < 5:
        elevation_deg = 5  # Avoid singularity

    # Zenith attenuation (approximate)
    freq_ghz = frequency_hz / 1e9
    if freq_ghz < 10:
        zenith_loss = 0.04  # S/X band
    elif freq_ghz < 20:
        zenith_loss = 0.1   # Ku band
    else:
        zenith_loss = 0.3   # Ka band

    # Scale by air mass (secant of zenith angle)
    zenith_deg = 90 - elevation_deg
    air_mass = 1 / np.cos(np.radians(zenith_deg))

    return zenith_loss * air_mass


def compute_link_budget(
    tx: Transmitter,
    rx: Receiver,
    distance_km: float,
    elevation_deg: float = 90.0,
) -> Dict[str, float]:
    """
    Compute full link budget.

    Args:
        tx: Transmitter
        rx: Receiver
        distance_km: Link distance (km)
        elevation_deg: Elevation angle for atmospheric loss

    Returns:
        Dictionary with all link budget parameters
    """
    freq_hz = tx.antenna.frequency_hz if tx.antenna else 8.4e9

    # Losses
    fspl = free_space_path_loss(distance_km, freq_hz)
    atm_loss = atmospheric_loss(elevation_deg, freq_hz)
    pointing_loss = (tx.antenna.pointing_loss_db if tx.antenna else 0) + \
                   (rx.antenna.pointing_loss_db if rx.antenna else 0)

    total_loss = fspl + atm_loss + pointing_loss

    # Received power
    eirp = tx.eirp_dbw
    rx_gain = rx.antenna.gain_db if rx.antenna else 0
    pr_dbw = eirp - total_loss + rx_gain - rx.line_loss_db

    # Noise power density
    n0_dbw_hz = 10 * np.log10(K_BOLTZMANN * rx.noise_temp_k)

    # C/N0 (carrier to noise density ratio)
    cn0_db_hz = pr_dbw - n0_dbw_hz

    # Eb/N0 for various data rates
    data_rates = {
        '1_kbps': 1e3,
        '10_kbps': 1e4,
        '100_kbps': 1e5,
        '1_mbps': 1e6,
        '10_mbps': 1e7,
    }

    eb_n0 = {}
    for name, rate in data_rates.items():
        eb_n0[name] = cn0_db_hz - 10 * np.log10(rate) - rx.implementation_loss_db

    return {
        'distance_km': distance_km,
        'frequency_hz': freq_hz,
        'eirp_dbw': eirp,
        'fspl_db': fspl,
        'atmospheric_loss_db': atm_loss,
        'pointing_loss_db': pointing_loss,
        'total_loss_db': total_loss,
        'rx_gain_db': rx_gain,
        'received_power_dbw': pr_dbw,
        'noise_density_dbw_hz': n0_dbw_hz,
        'cn0_db_hz': cn0_db_hz,
        'eb_n0': eb_n0,
    }


def max_data_rate(
    tx: Transmitter,
    rx: Receiver,
    distance_km: float,
    required_eb_n0_db: float = 3.0,
    elevation_deg: float = 90.0,
) -> float:
    """
    Calculate maximum achievable data rate.

    Args:
        tx: Transmitter
        rx: Receiver
        distance_km: Link distance
        required_eb_n0_db: Required Eb/N0 for acceptable BER
        elevation_deg: Elevation angle

    Returns:
        Maximum data rate (bps)
    """
    budget = compute_link_budget(tx, rx, distance_km, elevation_deg)

    cn0 = budget['cn0_db_hz']
    available_margin = cn0 - required_eb_n0_db - rx.implementation_loss_db

    # R = 10^(margin/10)
    max_rate = 10 ** (available_margin / 10)

    return max_rate


# ------------------- VISIBILITY CALCULATIONS -------------------

def earth_rotation_angle(t_days: float) -> float:
    """
    Calculate Earth rotation angle.

    Args:
        t_days: Time from epoch (days)

    Returns:
        Rotation angle (radians)
    """
    # Earth rotates ~360.985647 degrees per solar day
    omega_earth = 2 * np.pi / 0.99726966  # rad/day (sidereal)
    return omega_earth * t_days


def station_position_inertial(station: GroundStation, t_days: float) -> np.ndarray:
    """
    Get ground station position in inertial frame.

    Args:
        station: Ground station
        t_days: Time from epoch (days)

    Returns:
        Position in inertial frame (AU)
    """
    # ECEF position
    ecef_km = station.ecef_position()

    # Rotate by Earth rotation angle
    theta = earth_rotation_angle(t_days)

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])

    eci_km = R @ ecef_km

    # Add Earth position
    r_earth = get_body_position('earth', t_days)

    # Convert to AU and add Earth position
    return r_earth + eci_km / AU_TO_KM


def compute_elevation(
    station: GroundStation,
    spacecraft_pos_au: np.ndarray,
    t_days: float,
) -> float:
    """
    Compute elevation angle from ground station to spacecraft.

    Args:
        station: Ground station
        spacecraft_pos_au: Spacecraft position (AU, inertial)
        t_days: Current time (days)

    Returns:
        Elevation angle (degrees), negative if below horizon
    """
    # Station position in inertial frame
    station_pos = station_position_inertial(station, t_days)

    # Vector from station to spacecraft
    r_rel = spacecraft_pos_au - station_pos
    range_km = np.linalg.norm(r_rel) * AU_TO_KM

    if range_km < 1e-6:
        return 90.0

    r_hat = r_rel / np.linalg.norm(r_rel)

    # Local zenith direction (from Earth center through station)
    r_earth = get_body_position('earth', t_days)
    zenith = station_pos - r_earth
    zenith_hat = zenith / np.linalg.norm(zenith)

    # Elevation = 90° - angle from zenith
    cos_zenith_angle = np.dot(r_hat, zenith_hat)
    zenith_angle_deg = np.degrees(np.arccos(np.clip(cos_zenith_angle, -1, 1)))

    elevation_deg = 90.0 - zenith_angle_deg

    return elevation_deg


def is_visible(
    station: GroundStation,
    spacecraft_pos_au: np.ndarray,
    t_days: float,
) -> bool:
    """Check if spacecraft is visible from ground station."""
    elevation = compute_elevation(station, spacecraft_pos_au, t_days)
    return elevation >= station.min_elevation_deg


@dataclass
class ContactWindow:
    """A communication contact window."""
    station_name: str
    start_time: float      # days
    end_time: float        # days
    max_elevation_deg: float
    min_range_km: float

    @property
    def duration_minutes(self) -> float:
        """Contact duration in minutes."""
        return (self.end_time - self.start_time) * 24 * 60


def find_contact_windows(
    spacecraft_positions: np.ndarray,  # (N, 3) positions in AU
    times: np.ndarray,                  # (N,) times in days
    stations: List[GroundStation] = None,
    min_duration_minutes: float = 1.0,
) -> List[ContactWindow]:
    """
    Find all contact windows with ground stations.

    Args:
        spacecraft_positions: Array of spacecraft positions (AU)
        times: Array of times (days)
        stations: List of ground stations (default: DSN)
        min_duration_minutes: Minimum contact duration

    Returns:
        List of ContactWindow objects
    """
    if stations is None:
        stations = list(DSN_STATIONS.values())

    contacts = []

    for station in stations:
        in_contact = False
        contact_start = None
        max_elev = 0
        min_range = float('inf')

        for i, (pos, t) in enumerate(zip(spacecraft_positions, times)):
            elevation = compute_elevation(station, pos, t)

            if elevation >= station.min_elevation_deg:
                # Calculate range
                station_pos = station_position_inertial(station, t)
                range_km = np.linalg.norm(pos - station_pos) * AU_TO_KM

                if not in_contact:
                    # Start of contact
                    in_contact = True
                    contact_start = t
                    max_elev = elevation
                    min_range = range_km
                else:
                    # Update max elevation and min range
                    max_elev = max(max_elev, elevation)
                    min_range = min(min_range, range_km)
            else:
                if in_contact:
                    # End of contact
                    duration = (t - contact_start) * 24 * 60
                    if duration >= min_duration_minutes:
                        contacts.append(ContactWindow(
                            station_name=station.name,
                            start_time=contact_start,
                            end_time=t,
                            max_elevation_deg=max_elev,
                            min_range_km=min_range,
                        ))
                    in_contact = False

        # Handle contact at end of data
        if in_contact:
            duration = (times[-1] - contact_start) * 24 * 60
            if duration >= min_duration_minutes:
                contacts.append(ContactWindow(
                    station_name=station.name,
                    start_time=contact_start,
                    end_time=times[-1],
                    max_elevation_deg=max_elev,
                    min_range_km=min_range,
                ))

    # Sort by start time
    contacts.sort(key=lambda c: c.start_time)

    return contacts


# ------------------- COMMUNICATION SYSTEM -------------------

@dataclass
class CommSystem:
    """Spacecraft communication system."""
    transmitter: Transmitter
    receiver: Receiver
    max_data_rate_bps: float = 1e6    # 1 Mbps max
    min_data_rate_bps: float = 10.0   # 10 bps minimum

    def achievable_rate(
        self,
        ground_station: GroundStation,
        distance_km: float,
        elevation_deg: float = 90.0,
    ) -> float:
        """
        Calculate achievable data rate to ground station.

        Args:
            ground_station: Target ground station
            distance_km: Distance (km)
            elevation_deg: Elevation angle

        Returns:
            Achievable data rate (bps)
        """
        # Use ground station receiver
        rx = ground_station.receiver or Receiver(
            antenna=ground_station.antenna,
            noise_temp_k=25.0,  # DSN typical
        )

        rate = max_data_rate(self.transmitter, rx, distance_km,
                            required_eb_n0_db=3.0, elevation_deg=elevation_deg)

        # Clamp to system limits
        return np.clip(rate, self.min_data_rate_bps, self.max_data_rate_bps)


def estimate_data_volume(
    contacts: List[ContactWindow],
    comm_system: CommSystem,
    ground_stations: Dict[str, GroundStation],
) -> Dict[str, Any]:
    """
    Estimate total data volume that can be transmitted.

    Args:
        contacts: List of contact windows
        comm_system: Spacecraft comm system
        ground_stations: Dictionary of ground stations

    Returns:
        Data volume statistics
    """
    total_bits = 0
    total_duration = 0

    details = []

    for contact in contacts:
        # Find ground station
        station = None
        for gs in ground_stations.values():
            if gs.name == contact.station_name:
                station = gs
                break

        if station is None:
            continue

        # Average rate over contact
        rate = comm_system.achievable_rate(
            station,
            contact.min_range_km,
            contact.max_elevation_deg
        )

        duration_s = contact.duration_minutes * 60
        bits = rate * duration_s

        total_bits += bits
        total_duration += duration_s

        details.append({
            'station': contact.station_name,
            'start': contact.start_time,
            'duration_min': contact.duration_minutes,
            'rate_bps': rate,
            'data_bits': bits,
        })

    return {
        'total_bits': total_bits,
        'total_mbytes': total_bits / 8 / 1e6,
        'total_contact_minutes': total_duration / 60,
        'num_contacts': len(contacts),
        'details': details,
    }
