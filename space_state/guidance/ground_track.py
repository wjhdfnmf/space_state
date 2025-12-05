# guidance/ground_track.py
"""
Ground Track and Coverage Analysis Module

Provides calculations for:
- Ground track projection
- Ground station visibility windows
- Coverage analysis for constellations
- Access time calculations
- Communication link geometry

Coordinate Systems:
- ECI (Earth-Centered Inertial)
- ECEF (Earth-Centered Earth-Fixed)
- Geodetic (latitude, longitude, altitude)

Applications:
- Ground station contact scheduling
- Constellation coverage analysis
- Observation planning
- Communication link availability
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum, auto

from base.star_logger import logger


# Earth parameters
R_EARTH = 6378.137  # km
OMEGA_EARTH = 7.292115e-5  # rad/s (Earth rotation rate)
MU_EARTH = 398600.4418  # km³/s²
J2 = 1.08263e-3  # Earth J2 coefficient


@dataclass
class GroundStation:
    """Ground station location and parameters."""
    name: str
    latitude: float        # degrees
    longitude: float       # degrees
    altitude: float        # km above sea level
    min_elevation: float   # degrees (minimum elevation for contact)

    def ecef_position(self) -> np.ndarray:
        """Get ECEF position vector (km)."""
        lat = np.radians(self.latitude)
        lon = np.radians(self.longitude)
        r = R_EARTH + self.altitude

        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)

        return np.array([x, y, z])


@dataclass
class VisibilityWindow:
    """Visibility window between spacecraft and ground station."""
    station_name: str
    start_time: float      # seconds from epoch
    end_time: float        # seconds from epoch
    duration: float        # seconds
    max_elevation: float   # degrees
    aos_azimuth: float     # degrees (Acquisition of Signal)
    los_azimuth: float     # degrees (Loss of Signal)


@dataclass
class CoverageResult:
    """Coverage analysis results."""
    total_coverage_percent: float
    gap_max_minutes: float
    gap_mean_minutes: float
    revisit_time_max_minutes: float
    revisit_time_mean_minutes: float
    contacts_per_day: float


# Pre-defined ground stations
GROUND_STATIONS = {
    'goldstone': GroundStation('Goldstone', 35.426, -116.890, 1.0, 5.0),
    'canberra': GroundStation('Canberra', -35.401, 148.981, 0.7, 5.0),
    'madrid': GroundStation('Madrid', 40.431, -4.248, 0.8, 5.0),
    'cape_canaveral': GroundStation('Cape Canaveral', 28.396, -80.605, 0.0, 5.0),
    'kourou': GroundStation('Kourou', 5.232, -52.769, 0.0, 5.0),
    'wallops': GroundStation('Wallops', 37.940, -75.466, 0.0, 5.0),
    'baikonur': GroundStation('Baikonur', 45.965, 63.305, 0.1, 5.0),
    'svalbard': GroundStation('Svalbard', 78.229, 15.407, 0.5, 5.0),
    'mcmurdo': GroundStation('McMurdo', -77.846, 166.668, 0.2, 5.0),
}


def eci_to_ecef(r_eci: np.ndarray, t: float) -> np.ndarray:
    """
    Convert ECI position to ECEF.

    Parameters
    ----------
    r_eci : np.ndarray
        Position in ECI (km)
    t : float
        Time since epoch (seconds)

    Returns
    -------
    np.ndarray
        Position in ECEF (km)
    """
    theta = OMEGA_EARTH * t  # Earth rotation angle

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Rotation matrix ECI -> ECEF
    R = np.array([
        [cos_t, sin_t, 0],
        [-sin_t, cos_t, 0],
        [0, 0, 1]
    ])

    return R @ r_eci


def ecef_to_geodetic(r_ecef: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert ECEF position to geodetic coordinates.

    Parameters
    ----------
    r_ecef : np.ndarray
        Position in ECEF (km)

    Returns
    -------
    lat, lon, alt : tuple
        Latitude (deg), Longitude (deg), Altitude (km)
    """
    x, y, z = r_ecef

    # Longitude
    lon = np.degrees(np.arctan2(y, x))

    # Latitude (simplified spherical)
    r_xy = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, r_xy))

    # Altitude
    r = np.linalg.norm(r_ecef)
    alt = r - R_EARTH

    return lat, lon, alt


def ground_track(
    r_eci_trajectory: np.ndarray,
    times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ground track from ECI trajectory.

    Parameters
    ----------
    r_eci_trajectory : np.ndarray
        ECI positions (km), shape (3, N) or (N, 3)
    times : np.ndarray
        Times since epoch (seconds), shape (N,)

    Returns
    -------
    latitudes, longitudes, altitudes : tuple
        Ground track coordinates (degrees, degrees, km)
    """
    # Ensure correct shape
    if r_eci_trajectory.shape[0] == 3:
        positions = r_eci_trajectory.T
    else:
        positions = r_eci_trajectory

    N = len(times)
    lats = np.zeros(N)
    lons = np.zeros(N)
    alts = np.zeros(N)

    for i in range(N):
        r_ecef = eci_to_ecef(positions[i], times[i])
        lat, lon, alt = ecef_to_geodetic(r_ecef)
        lats[i] = lat
        lons[i] = lon
        alts[i] = alt

    return lats, lons, alts


def elevation_angle(
    r_sat_ecef: np.ndarray,
    station: GroundStation
) -> float:
    """
    Calculate elevation angle from ground station to satellite.

    Parameters
    ----------
    r_sat_ecef : np.ndarray
        Satellite position in ECEF (km)
    station : GroundStation
        Ground station

    Returns
    -------
    float
        Elevation angle (degrees)
    """
    r_station = station.ecef_position()

    # Vector from station to satellite
    r_rel = r_sat_ecef - r_station
    r_rel_norm = np.linalg.norm(r_rel)

    if r_rel_norm < 1e-6:
        return 90.0

    # Local vertical at station (radial direction)
    r_station_unit = r_station / np.linalg.norm(r_station)

    # Elevation angle
    sin_el = np.dot(r_rel, r_station_unit) / r_rel_norm
    el = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))

    return el


def azimuth_angle(
    r_sat_ecef: np.ndarray,
    station: GroundStation
) -> float:
    """
    Calculate azimuth angle from ground station to satellite.

    Parameters
    ----------
    r_sat_ecef : np.ndarray
        Satellite position in ECEF (km)
    station : GroundStation
        Ground station

    Returns
    -------
    float
        Azimuth angle (degrees, from North)
    """
    r_station = station.ecef_position()

    # Vector from station to satellite
    r_rel = r_sat_ecef - r_station

    # Local coordinate system at station
    lat = np.radians(station.latitude)
    lon = np.radians(station.longitude)

    # East unit vector
    e_east = np.array([-np.sin(lon), np.cos(lon), 0])

    # North unit vector
    e_north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
        np.cos(lat)
    ])

    # Project onto local horizontal
    east_comp = np.dot(r_rel, e_east)
    north_comp = np.dot(r_rel, e_north)

    # Azimuth from North
    az = np.degrees(np.arctan2(east_comp, north_comp))
    if az < 0:
        az += 360

    return az


def find_visibility_windows(
    r_eci_trajectory: np.ndarray,
    times: np.ndarray,
    station: GroundStation,
    min_elevation: Optional[float] = None
) -> List[VisibilityWindow]:
    """
    Find visibility windows for a ground station.

    Parameters
    ----------
    r_eci_trajectory : np.ndarray
        ECI positions (km), shape (3, N) or (N, 3)
    times : np.ndarray
        Times since epoch (seconds)
    station : GroundStation
        Ground station
    min_elevation : float, optional
        Minimum elevation angle. Default uses station's value.

    Returns
    -------
    list
        List of VisibilityWindow objects
    """
    if min_elevation is None:
        min_elevation = station.min_elevation

    # Ensure correct shape
    if r_eci_trajectory.shape[0] == 3:
        positions = r_eci_trajectory.T
    else:
        positions = r_eci_trajectory

    windows = []
    in_view = False
    window_start = None
    max_el = 0.0
    aos_az = 0.0

    for i, t in enumerate(times):
        r_ecef = eci_to_ecef(positions[i], t)
        el = elevation_angle(r_ecef, station)
        az = azimuth_angle(r_ecef, station)

        if el >= min_elevation:
            if not in_view:
                # Start of visibility window
                in_view = True
                window_start = t
                max_el = el
                aos_az = az
            else:
                max_el = max(max_el, el)
        else:
            if in_view:
                # End of visibility window
                in_view = False
                windows.append(VisibilityWindow(
                    station_name=station.name,
                    start_time=window_start,
                    end_time=t,
                    duration=t - window_start,
                    max_elevation=max_el,
                    aos_azimuth=aos_az,
                    los_azimuth=az
                ))

    # Handle case where visibility continues to end
    if in_view:
        windows.append(VisibilityWindow(
            station_name=station.name,
            start_time=window_start,
            end_time=times[-1],
            duration=times[-1] - window_start,
            max_elevation=max_el,
            aos_azimuth=aos_az,
            los_azimuth=azimuth_angle(eci_to_ecef(positions[-1], times[-1]), station)
        ))

    return windows


def compute_access_statistics(
    windows: List[VisibilityWindow],
    total_time: float
) -> Dict[str, float]:
    """
    Compute access statistics from visibility windows.

    Parameters
    ----------
    windows : list
        List of VisibilityWindow objects
    total_time : float
        Total time span analyzed (seconds)

    Returns
    -------
    dict
        Access statistics
    """
    if not windows:
        return {
            'total_access_time_s': 0.0,
            'coverage_percent': 0.0,
            'num_passes': 0,
            'mean_pass_duration_s': 0.0,
            'max_pass_duration_s': 0.0,
            'mean_gap_s': 0.0,
            'max_gap_s': total_time,
        }

    total_access = sum(w.duration for w in windows)
    durations = [w.duration for w in windows]

    # Calculate gaps
    gaps = []
    for i in range(len(windows) - 1):
        gap = windows[i + 1].start_time - windows[i].end_time
        gaps.append(gap)

    return {
        'total_access_time_s': total_access,
        'coverage_percent': 100 * total_access / total_time,
        'num_passes': len(windows),
        'mean_pass_duration_s': np.mean(durations),
        'max_pass_duration_s': max(durations),
        'mean_gap_s': np.mean(gaps) if gaps else 0.0,
        'max_gap_s': max(gaps) if gaps else 0.0,
    }


def swath_width(
    altitude_km: float,
    half_angle_deg: float
) -> float:
    """
    Calculate instrument swath width.

    Parameters
    ----------
    altitude_km : float
        Orbital altitude (km)
    half_angle_deg : float
        Instrument half-cone angle (degrees)

    Returns
    -------
    float
        Swath width on ground (km)
    """
    h = altitude_km
    theta = np.radians(half_angle_deg)

    # Spherical Earth geometry
    # sin(rho) = R / (R + h)
    rho = np.arcsin(R_EARTH / (R_EARTH + h))

    # Central angle of swath
    eta = np.pi / 2 - theta - rho
    if eta < 0:
        # Instrument sees beyond horizon
        eta = 0

    # Swath half-width
    lambda_half = np.arccos(np.sin(theta + rho) * np.sin(rho) +
                            np.cos(theta + rho) * np.cos(rho))

    # Ground distance
    swath = 2 * R_EARTH * lambda_half

    return swath


def revisit_time_estimate(
    altitude_km: float,
    inclination_deg: float,
    swath_width_km: float
) -> float:
    """
    Estimate revisit time for a single satellite.

    Parameters
    ----------
    altitude_km : float
        Orbital altitude (km)
    inclination_deg : float
        Inclination (degrees)
    swath_width_km : float
        Ground swath width (km)

    Returns
    -------
    float
        Approximate revisit time at equator (hours)
    """
    # Orbital period
    a = R_EARTH + altitude_km
    T = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)  # seconds

    # Ground track spacing
    # Earth rotates during one orbit
    ground_track_spacing = 2 * np.pi * R_EARTH * np.cos(0) * (T / 86400)

    # Revolutions needed to cover equator
    if swath_width_km > 0:
        revolutions_to_cover = ground_track_spacing / swath_width_km
    else:
        revolutions_to_cover = float('inf')

    # Revisit time
    revisit_hours = revolutions_to_cover * T / 3600

    return revisit_hours


def sun_synchronous_inclination(altitude_km: float) -> float:
    """
    Calculate sun-synchronous inclination.

    Parameters
    ----------
    altitude_km : float
        Orbital altitude (km)

    Returns
    -------
    float
        Required inclination (degrees)
    """
    a = R_EARTH + altitude_km
    n = np.sqrt(MU_EARTH / a**3)  # Mean motion (rad/s)

    # Nodal precession rate for sun-sync: -2π / year
    omega_sun = 2 * np.pi / (365.25 * 86400)  # rad/s

    # J2 effect: dΩ/dt = -1.5 * n * J2 * (R/a)² * cos(i)
    # Solve for cos(i)
    cos_i = omega_sun / (-1.5 * n * J2 * (R_EARTH / a)**2)

    if abs(cos_i) > 1:
        return float('nan')  # Not possible

    return np.degrees(np.arccos(cos_i))


def walker_constellation_coverage(
    altitude_km: float,
    inclination_deg: float,
    num_planes: int,
    sats_per_plane: int,
    swath_width_km: float
) -> CoverageResult:
    """
    Estimate Walker constellation coverage characteristics.

    Parameters
    ----------
    altitude_km : float
        Orbital altitude (km)
    inclination_deg : float
        Inclination (degrees)
    num_planes : int
        Number of orbital planes
    sats_per_plane : int
        Satellites per plane
    swath_width_km : float
        Instrument swath width (km)

    Returns
    -------
    CoverageResult
        Coverage analysis results
    """
    total_sats = num_planes * sats_per_plane

    # Orbital period
    a = R_EARTH + altitude_km
    T = 2 * np.pi * np.sqrt(a**3 / MU_EARTH) / 60  # minutes

    # Ground track spacing (simplified)
    circumference_equator = 2 * np.pi * R_EARTH

    # Coverage band per satellite
    coverage_band = swath_width_km * np.sin(np.radians(inclination_deg))

    # Total coverage (simplified)
    total_coverage = min(100, 100 * total_sats * coverage_band / circumference_equator)

    # Gap estimates (very approximate)
    single_revisit = revisit_time_estimate(altitude_km, inclination_deg, swath_width_km)
    constellation_revisit = single_revisit / total_sats

    # Contacts per day (approximate)
    contacts = 86400 / (T * 60) * (swath_width_km / circumference_equator) * total_sats

    return CoverageResult(
        total_coverage_percent=total_coverage,
        gap_max_minutes=constellation_revisit * 60,
        gap_mean_minutes=constellation_revisit * 60 * 0.5,
        revisit_time_max_minutes=constellation_revisit * 60,
        revisit_time_mean_minutes=constellation_revisit * 60 * 0.7,
        contacts_per_day=contacts
    )


def horizon_distance(altitude_km: float) -> float:
    """
    Calculate distance to horizon from given altitude.

    Parameters
    ----------
    altitude_km : float
        Altitude above surface (km)

    Returns
    -------
    float
        Distance to horizon (km)
    """
    h = altitude_km
    return np.sqrt(2 * R_EARTH * h + h**2)


def line_of_sight_range(
    altitude_km: float,
    elevation_deg: float
) -> float:
    """
    Calculate slant range for given elevation angle.

    Parameters
    ----------
    altitude_km : float
        Satellite altitude (km)
    elevation_deg : float
        Elevation angle from ground (degrees)

    Returns
    -------
    float
        Slant range (km)
    """
    el = np.radians(elevation_deg)
    r = R_EARTH + altitude_km

    # Using law of sines in the Earth-satellite-station triangle
    rho = np.arcsin(R_EARTH / r * np.cos(el))
    lambda_angle = np.pi / 2 - el - rho

    # Slant range
    D = R_EARTH * np.sin(lambda_angle) / np.cos(el)

    return D


def communication_window_geometry(
    altitude_km: float,
    station: GroundStation
) -> Dict[str, float]:
    """
    Calculate communication geometry parameters.

    Parameters
    ----------
    altitude_km : float
        Satellite altitude (km)
    station : GroundStation
        Ground station

    Returns
    -------
    dict
        Communication geometry parameters
    """
    min_el = station.min_elevation

    # Maximum range (at minimum elevation)
    max_range = line_of_sight_range(altitude_km, min_el)

    # Minimum range (at zenith)
    min_range = altitude_km

    # Maximum view time estimate
    v_ground = np.sqrt(MU_EARTH / (R_EARTH + altitude_km))  # km/s
    arc_length = 2 * horizon_distance(altitude_km)
    max_pass_time = arc_length / v_ground

    return {
        'min_range_km': min_range,
        'max_range_km': max_range,
        'horizon_distance_km': horizon_distance(altitude_km),
        'max_pass_duration_s': max_pass_time,
        'min_elevation_deg': min_el,
    }
