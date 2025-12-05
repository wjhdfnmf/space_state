# guidance/launch_window.py
"""
Launch Window Analysis Module

Provides calculations for:
- Launch window determination
- Launch azimuth for target orbit
- Plane change avoidance
- Interplanetary launch windows
- Synodic period calculations

Key Concepts:
- Launch Window: Time when orbital plane passes over launch site
- Launch Azimuth: Direction of launch (from North)
- Dog-leg Maneuver: Plane change during ascent (costly)

Applications:
- LEO mission planning
- ISS rendezvous
- GTO launches
- Interplanetary departure
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from base.star_logger import logger


# Constants
R_EARTH = 6378.137  # km
MU_EARTH = 398600.4418  # km³/s²
OMEGA_EARTH = 7.292115e-5  # rad/s
MU_SUN = 1.32712440041e11  # km³/s²
AU_TO_KM = 149597870.7


@dataclass
class LaunchSite:
    """Launch site parameters."""
    name: str
    latitude: float     # degrees
    longitude: float    # degrees
    azimuth_min: float  # degrees (minimum safe azimuth)
    azimuth_max: float  # degrees (maximum safe azimuth)

    def min_inclination(self) -> float:
        """Minimum achievable inclination (equals latitude)."""
        return abs(self.latitude)


# Common launch sites
LAUNCH_SITES = {
    'kennedy': LaunchSite('Kennedy Space Center', 28.524, -80.651, 35, 120),
    'cape_canaveral': LaunchSite('Cape Canaveral', 28.396, -80.605, 35, 120),
    'vandenberg': LaunchSite('Vandenberg SFB', 34.632, -120.611, 147, 201),
    'baikonur': LaunchSite('Baikonur', 45.965, 63.305, 35, 99),
    'kourou': LaunchSite('Kourou', 5.232, -52.769, -15, 90),
    'tanegashima': LaunchSite('Tanegashima', 30.400, 130.975, 90, 180),
    'jiuquan': LaunchSite('Jiuquan', 40.958, 100.291, 57, 70),
    'xichang': LaunchSite('Xichang', 28.246, 102.027, 97, 104),
    'satish_dhawan': LaunchSite('Satish Dhawan', 13.720, 80.230, 102, 140),
    'plesetsk': LaunchSite('Plesetsk', 62.927, 40.577, 0, 180),
    'wallops': LaunchSite('Wallops', 37.940, -75.466, 90, 180),
}


@dataclass
class LaunchWindow:
    """Launch window definition."""
    open_time: float       # seconds from reference epoch
    close_time: float      # seconds from reference epoch
    duration: float        # seconds
    azimuth: float         # degrees (launch azimuth)
    type: str              # 'ascending' or 'descending'


@dataclass
class InterplanetaryWindow:
    """Interplanetary launch window."""
    departure_date: datetime
    arrival_date: datetime
    c3_km2_s2: float       # Departure C3
    dv_departure: float    # km/s (Earth departure delta-V)
    dv_arrival: float      # km/s (arrival delta-V for orbit insertion)
    tof_days: float        # Time of flight
    phase_angle: float     # degrees


def launch_azimuth(
    site_latitude: float,
    target_inclination: float,
    ascending: bool = True
) -> float:
    """
    Calculate launch azimuth for target orbital inclination.

    Parameters
    ----------
    site_latitude : float
        Launch site latitude (degrees)
    target_inclination : float
        Target orbital inclination (degrees)
    ascending : bool
        True for ascending node passage, False for descending

    Returns
    -------
    float
        Launch azimuth (degrees from North)

    Notes
    -----
    Only valid if inclination >= |latitude|
    """
    lat = np.radians(abs(site_latitude))
    inc = np.radians(target_inclination)

    if target_inclination < abs(site_latitude):
        raise ValueError(
            f"Inclination {target_inclination}° cannot be achieved from "
            f"latitude {site_latitude}° without dog-leg maneuver"
        )

    # sin(azimuth) = cos(inclination) / cos(latitude)
    sin_az = np.cos(inc) / np.cos(lat)
    sin_az = np.clip(sin_az, -1, 1)

    if ascending:
        azimuth = np.degrees(np.arcsin(sin_az))
    else:
        azimuth = 180 - np.degrees(np.arcsin(sin_az))

    # Ensure azimuth is in [0, 360)
    if azimuth < 0:
        azimuth += 360

    return azimuth


def launch_window_for_inclination(
    site: LaunchSite,
    target_inclination: float,
    raan: float,
    epoch: float = 0.0
) -> List[LaunchWindow]:
    """
    Calculate launch windows for target orbital plane.

    Parameters
    ----------
    site : LaunchSite
        Launch site
    target_inclination : float
        Target inclination (degrees)
    raan : float
        Right ascension of ascending node (degrees)
    epoch : float
        Reference epoch (seconds since J2000)

    Returns
    -------
    list
        List of LaunchWindow objects for next 24 hours
    """
    if target_inclination < site.min_inclination():
        return []  # Cannot achieve this inclination

    windows = []

    # Calculate launch azimuths
    az_asc = launch_azimuth(site.latitude, target_inclination, ascending=True)
    az_desc = launch_azimuth(site.latitude, target_inclination, ascending=False)

    # Earth rotation period
    T_sidereal = 86164.0905  # seconds

    # Calculate when launch site rotates under orbital plane
    # This depends on RAAN and site longitude

    site_lon = np.radians(site.longitude)
    raan_rad = np.radians(raan)

    # GMST at epoch (simplified)
    gmst_epoch = (epoch / 86400) * 2 * np.pi % (2 * np.pi)

    # Angular difference to RAAN
    delta_lon_asc = raan_rad - site_lon - gmst_epoch
    delta_lon_desc = delta_lon_asc + np.pi

    # Normalize to [0, 2π)
    delta_lon_asc = delta_lon_asc % (2 * np.pi)
    delta_lon_desc = delta_lon_desc % (2 * np.pi)

    # Time until site rotates to plane
    t_asc = delta_lon_asc / OMEGA_EARTH
    t_desc = delta_lon_desc / OMEGA_EARTH

    # Create windows (one ascending, one descending per day)
    window_duration = 600  # 10-minute window (simplified)

    for day in range(1):  # Next 24 hours
        offset = day * T_sidereal

        # Ascending window
        if site.azimuth_min <= az_asc <= site.azimuth_max:
            windows.append(LaunchWindow(
                open_time=epoch + t_asc + offset - window_duration / 2,
                close_time=epoch + t_asc + offset + window_duration / 2,
                duration=window_duration,
                azimuth=az_asc,
                type='ascending'
            ))

        # Descending window
        if site.azimuth_min <= az_desc <= site.azimuth_max:
            windows.append(LaunchWindow(
                open_time=epoch + t_desc + offset - window_duration / 2,
                close_time=epoch + t_desc + offset + window_duration / 2,
                duration=window_duration,
                azimuth=az_desc,
                type='descending'
            ))

    return sorted(windows, key=lambda w: w.open_time)


def iss_launch_window(
    site: LaunchSite,
    iss_raan: float,
    epoch: float = 0.0
) -> List[LaunchWindow]:
    """
    Calculate ISS rendezvous launch windows.

    ISS orbital parameters:
    - Inclination: 51.6°
    - Altitude: ~420 km

    Parameters
    ----------
    site : LaunchSite
        Launch site
    iss_raan : float
        Current ISS RAAN (degrees)
    epoch : float
        Reference epoch

    Returns
    -------
    list
        Launch windows for ISS rendezvous
    """
    return launch_window_for_inclination(site, 51.6, iss_raan, epoch)


def synodic_period(a1_au: float, a2_au: float) -> float:
    """
    Calculate synodic period between two planets.

    Parameters
    ----------
    a1_au : float
        Semi-major axis of inner planet (AU)
    a2_au : float
        Semi-major axis of outer planet (AU)

    Returns
    -------
    float
        Synodic period (days)
    """
    # Orbital periods (years)
    T1 = a1_au ** 1.5
    T2 = a2_au ** 1.5

    # Synodic period
    T_syn = abs(1 / (1/T1 - 1/T2))

    return T_syn * 365.25  # Convert to days


def hohmann_phase_angle(a1_au: float, a2_au: float) -> float:
    """
    Calculate required phase angle for Hohmann transfer.

    Parameters
    ----------
    a1_au : float
        Departure orbit semi-major axis (AU)
    a2_au : float
        Arrival orbit semi-major axis (AU)

    Returns
    -------
    float
        Phase angle (degrees)
    """
    # Transfer semi-major axis
    a_transfer = (a1_au + a2_au) / 2

    # Transfer time (half period)
    T_transfer = np.pi * np.sqrt(a_transfer**3 / (MU_SUN / AU_TO_KM**3))

    # Target planet moves during transfer
    if a2_au > a1_au:  # Outer planet
        n2 = np.sqrt(MU_SUN / AU_TO_KM**3 / a2_au**3)
        angle_traveled = n2 * T_transfer
        phase = 180 - np.degrees(angle_traveled)
    else:  # Inner planet
        n2 = np.sqrt(MU_SUN / AU_TO_KM**3 / a2_au**3)
        angle_traveled = n2 * T_transfer
        phase = 180 + np.degrees(angle_traveled)

    return phase % 360


def interplanetary_launch_opportunities(
    departure_body: str,
    arrival_body: str,
    start_date: datetime,
    num_opportunities: int = 3
) -> List[InterplanetaryWindow]:
    """
    Find interplanetary launch opportunities.

    Parameters
    ----------
    departure_body : str
        Departure body name ('earth')
    arrival_body : str
        Arrival body name ('mars', 'venus', etc.)
    start_date : datetime
        Start date for search
    num_opportunities : int
        Number of opportunities to find

    Returns
    -------
    list
        List of InterplanetaryWindow objects
    """
    # Planetary semi-major axes (AU)
    sma = {
        'mercury': 0.387,
        'venus': 0.723,
        'earth': 1.000,
        'mars': 1.524,
        'jupiter': 5.203,
        'saturn': 9.537,
        'uranus': 19.19,
        'neptune': 30.07,
    }

    a1 = sma.get(departure_body, 1.0)
    a2 = sma.get(arrival_body, 1.524)

    # Synodic period
    T_syn = synodic_period(a1, a2)

    # Phase angle for Hohmann transfer
    phase = hohmann_phase_angle(a1, a2)

    # Transfer time
    a_transfer = (a1 + a2) / 2
    tof = np.pi * np.sqrt(a_transfer**3) * 365.25  # days (Hohmann)

    # C3 estimate (Hohmann)
    v_earth = np.sqrt(MU_SUN / AU_TO_KM / a1)  # km/s
    v_transfer = np.sqrt(MU_SUN / AU_TO_KM * (2/a1 - 1/a_transfer))  # km/s
    v_inf = abs(v_transfer - v_earth)
    c3 = v_inf ** 2

    # Departure delta-V (from LEO)
    v_leo = np.sqrt(MU_EARTH / (R_EARTH + 200))  # km/s
    v_hyp = np.sqrt(v_inf**2 + 2*MU_EARTH/(R_EARTH + 200))
    dv_dep = v_hyp - v_leo

    # Arrival delta-V (for orbit insertion)
    v_target = np.sqrt(MU_SUN / AU_TO_KM / a2)
    v_arr_helio = np.sqrt(MU_SUN / AU_TO_KM * (2/a2 - 1/a_transfer))
    v_inf_arr = abs(v_arr_helio - v_target)

    opportunities = []
    current_date = start_date

    for i in range(num_opportunities):
        departure = current_date + timedelta(days=i * T_syn)
        arrival = departure + timedelta(days=tof)

        opportunities.append(InterplanetaryWindow(
            departure_date=departure,
            arrival_date=arrival,
            c3_km2_s2=c3,
            dv_departure=dv_dep,
            dv_arrival=v_inf_arr,
            tof_days=tof,
            phase_angle=phase
        ))

    return opportunities


def earth_mars_window_dates(year: int) -> Dict[str, Any]:
    """
    Get approximate Earth-Mars launch window for a given year.

    Windows occur roughly every 26 months.

    Parameters
    ----------
    year : int
        Year (2020-2040)

    Returns
    -------
    dict
        Window parameters
    """
    # Known Mars windows (approximate)
    known_windows = {
        2020: {'month': 7, 'tof': 203},   # Mars 2020
        2022: {'month': 9, 'tof': 260},
        2024: {'month': 10, 'tof': 280},
        2026: {'month': 11, 'tof': 240},
        2028: {'month': 12, 'tof': 220},
        2031: {'month': 2, 'tof': 200},
        2033: {'month': 4, 'tof': 190},
        2035: {'month': 5, 'tof': 210},
        2037: {'month': 7, 'tof': 230},
        2039: {'month': 8, 'tof': 250},
    }

    if year in known_windows:
        w = known_windows[year]
        return {
            'year': year,
            'window_month': w['month'],
            'typical_tof_days': w['tof'],
            'c3_range': (8, 15),  # km²/s² typical range
            'window_duration_days': 30,
        }
    else:
        # Estimate based on synodic period
        ref_year = 2020
        ref_month = 7
        years_diff = year - ref_year
        synodic_years = 2.135  # ~26 months

        periods = round(years_diff / synodic_years)
        estimated_year = ref_year + periods * synodic_years
        month_offset = (years_diff - periods * synodic_years) * 12

        return {
            'year': year,
            'window_month': int((ref_month + month_offset - 1) % 12 + 1),
            'typical_tof_days': 220,
            'c3_range': (8, 20),
            'window_duration_days': 30,
            'estimated': True
        }


def plane_change_penalty(
    inclination_difference: float,
    velocity_km_s: float
) -> float:
    """
    Calculate delta-V penalty for plane change during launch.

    Parameters
    ----------
    inclination_difference : float
        Required plane change (degrees)
    velocity_km_s : float
        Velocity at maneuver point (km/s)

    Returns
    -------
    float
        Delta-V penalty (km/s)
    """
    delta_i = np.radians(inclination_difference)
    return 2 * velocity_km_s * np.sin(delta_i / 2)


def gto_launch_parameters(
    site: LaunchSite,
    target_longitude: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate GTO launch parameters.

    Parameters
    ----------
    site : LaunchSite
        Launch site
    target_longitude : float, optional
        Target GEO longitude (degrees)

    Returns
    -------
    dict
        GTO launch parameters
    """
    # GTO parameters
    perigee = 200  # km
    apogee = 35786  # km (GEO altitude)
    a = (R_EARTH + perigee + R_EARTH + apogee) / 2

    # Minimum inclination from site
    min_inc = site.min_inclination()

    # Launch azimuth for minimum inclination
    az = launch_azimuth(site.latitude, min_inc)

    # Plane change at apogee to reach 0° inclination
    v_apogee = np.sqrt(MU_EARTH * (2/(R_EARTH + apogee) - 1/a))  # km/s
    dv_plane_change = plane_change_penalty(min_inc, v_apogee)

    # Circularization at GEO
    v_geo = np.sqrt(MU_EARTH / (R_EARTH + apogee))
    dv_circ = v_geo - v_apogee

    return {
        'launch_site': site.name,
        'azimuth_deg': az,
        'gto_inclination_deg': min_inc,
        'perigee_km': perigee,
        'apogee_km': apogee,
        'dv_plane_change_km_s': dv_plane_change,
        'dv_circularization_km_s': dv_circ,
        'total_gto_to_geo_km_s': np.sqrt(dv_plane_change**2 + dv_circ**2),
    }


def daily_launch_windows(
    site: LaunchSite,
    target_inclination: float
) -> int:
    """
    Determine number of launch windows per day.

    Parameters
    ----------
    site : LaunchSite
        Launch site
    target_inclination : float
        Target orbital inclination (degrees)

    Returns
    -------
    int
        Number of windows per day (0, 1, or 2)
    """
    if target_inclination < site.min_inclination():
        return 0  # Cannot achieve inclination

    # Calculate launch azimuths
    az_asc = launch_azimuth(site.latitude, target_inclination, ascending=True)
    az_desc = launch_azimuth(site.latitude, target_inclination, ascending=False)

    windows = 0
    if site.azimuth_min <= az_asc <= site.azimuth_max:
        windows += 1
    if site.azimuth_min <= az_desc <= site.azimuth_max:
        windows += 1

    return windows
