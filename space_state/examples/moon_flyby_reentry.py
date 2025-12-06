#!/usr/bin/env python3
"""
Moon Flyby and Earth Reentry Mission

This mission demonstrates a complete circumlunar free-return trajectory including:
1. Trans-Lunar Injection (TLI) from Low Earth Orbit
2. Lunar flyby with gravity assist
3. Free-return trajectory back to Earth
4. Atmospheric entry with thermal and attitude analysis

=============================================================================
SCIENTIFIC BACKGROUND AND NASA REFERENCES
=============================================================================

Apollo Free-Return Trajectory (NASA Historical Reference):
- Apollo 8, 10, 11, 12, 13 used free-return trajectories
- Transfer time to Moon: ~3 days (72 hours)
- Total round trip: ~6 days
- Entry velocity: 11.0-11.1 km/s (lunar return)
- Entry interface: 121 km (400,000 ft)

Apollo Entry Parameters:
- Entry Flight Path Angle: -6.5° to -7.5° (nominal -6.5°)
- Entry Corridor: ±1° (very narrow for lunar return)
- Peak deceleration: 6-7 g's
- Peak heating rate: ~1,000-1,500 W/cm²
- Skip-out boundary: ~-5.3°
- Undershoot boundary: ~-7.5° (g-limit)

Key Delta-V Budget (Apollo):
- TLI: 3,100-3,200 m/s
- Midcourse corrections: 10-50 m/s
- Entry: 0 m/s (ballistic)
- Total: ~3,150-3,250 m/s

Apollo Command Module:
- Entry mass: ~5,500 kg
- Diameter: 3.9 m
- Nose radius: ~4.7 m
- Ballistic coefficient: ~350-400 kg/m²
- L/D ratio: ~0.3 (lift-to-drag)

Free-Return Trajectory Geometry:
- Achieves ~180° transfer angle
- Moon flyby altitude: 100-300 km typical
- Uses lunar gravity to return spacecraft to Earth
- Entry corridor targeting done at TLI

Reference:
- NASA SP-368: Apollo by the Numbers
- NASA TM X-58040: Apollo Experience Report - Entry
- https://www.nasa.gov/mission_pages/apollo/missions/

Author: Space State Development Team
Date: December 2025
"""

import numpy as np
import sys
from datetime import datetime, timedelta
import logging
import os

# Reduce logging before imports
os.environ['LOG_LEVEL'] = 'WARNING'
logging.getLogger().setLevel(logging.WARNING)

# Ensure correct path
sys.path.insert(0, '/home/boss/data/space_state/space_state')

# Core imports
from base import (
    BODIES, CONFIG, Propagator, starship_raptor_vac,
    get_circular_state,
    vis_viva, circular_velocity, orbital_period, escape_velocity,
    # Mission logging imports
    FlightDataRecorder, MissionController, DisplayMode,
    create_fdr, generate_mission_plots,
)
from base.star_constants import AU_TO_KM, DAY_TO_SEC, KM_TO_AU, epoch_jd, ts
from base.star_utils import (
    sec_to_days, day_to_sec, min_to_days,
    km_to_au, au_to_km, body_position, body_velocity,
    ssb_to_body_centered,
)
from guidance.star_tools import distance_to_body
from base.star_lambert import (
    solve_lambert, lambert, LambertSolution, TransferType,
    compute_c3, compute_delta_v_escape,
)
from base.star_elements import (
    OrbitalElements, cartesian_to_keplerian, keplerian_to_cartesian,
    vis_viva, circular_velocity, escape_velocity, orbital_period,
    hohmann_transfer,
)
from base.star_events import (
    periapsis_event, apoapsis_event, altitude_event,
    soi_entry_event, soi_exit_event,
)
from base.star_ship import Spacecraft, Thruster

# Flyby module
from guidance.flyby import (
    compute_flyby, compute_v_infinity, flyby_turn_angle,
    b_plane_from_periapsis, FlybyGeometry,
)

# Entry descent module
from guidance.entry_descent import (
    ATMOSPHERES, EntryConditions, EntryResult,
    simulate_entry, ballistic_coefficient,
    peak_deceleration_estimate, peak_heating_rate,
    earth_reentry_corridor, parachute_deployment_conditions,
)

# Thermal module
from base.star_thermal import (
    compute_thermal_environment, ThermalEnvironment,
    solar_flux_at_distance, equilibrium_temperature,
    STEFAN_BOLTZMANN, SpacecraftThermal,
)

# Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Mission Configuration
# =============================================================================

class MoonFlybyConfig:
    """Moon flyby and reentry mission configuration."""

    # Mission name
    NAME = "Circumlunar Free-Return and Earth Reentry"

    # Departure orbit (LEO)
    LEO_ALTITUDE_KM = 185       # Apollo parking orbit altitude
    DEPARTURE_BODY = 'earth'

    # Target body
    TARGET_BODY = 'moon'

    # Flyby parameters
    FLYBY_ALTITUDE_KM = 200     # Minimum lunar flyby altitude
    FREE_RETURN = True          # Use free-return trajectory

    # Entry parameters (Apollo-like)
    ENTRY_INTERFACE_KM = 121    # Standard entry interface
    NOMINAL_ENTRY_FPA_DEG = -6.5  # Nominal entry flight path angle
    ENTRY_CORRIDOR_DEG = 1.0    # Entry corridor half-width

    # Vehicle parameters (Apollo CM-like)
    VEHICLE_MASS_KG = 5500      # Entry mass
    VEHICLE_DIAMETER_M = 3.9    # Heat shield diameter
    NOSE_RADIUS_M = 4.7         # Nose radius for heating
    DRAG_COEFFICIENT = 1.3      # Hypersonic Cd
    LIFT_TO_DRAG = 0.3          # L/D ratio

    # Simulation parameters
    MAX_MISSION_DURATION_DAYS = 8.0

    # Visualization flags
    SHOW_TRAJECTORY_3D = True
    SHOW_THERMAL_PLOTS = True
    SHOW_ATTITUDE_PLOTS = True
    SAVE_PLOTS = True
    INTERACTIVE_MODE = 'static'  # 'static', 'interactive', 'dashboard', 'full'

    # Mission logging
    ENABLE_FDR = True
    FDR_INTERVAL = 0.001  # Record every 0.001 days (~86 sec)
    DISPLAY_MODE = DisplayMode.COMPACT


# Physical constants
R_EARTH_KM = 6378.137
R_MOON_KM = 1737.4
MU_EARTH_KM = 398600.4418      # km³/s²
MU_MOON_KM = 4902.8            # km³/s²


# =============================================================================
# Utility Functions
# =============================================================================

def print_header(text: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_subheader(text: str) -> None:
    """Print formatted subsection header."""
    print(f"\n--- {text} ---")


def print_metric(name: str, value, unit: str = "", fmt: str = ".2f") -> None:
    """Print formatted metric."""
    if isinstance(value, (int, float)):
        print(f"  {name:<30}: {value:{fmt}} {unit}")
    else:
        print(f"  {name:<30}: {value} {unit}")


# =============================================================================
# Free-Return Trajectory Design
# =============================================================================

def compute_free_return_parameters(
    t_departure: float,
    flyby_altitude_km: float = 200.0,
) -> dict:
    """
    Compute free-return trajectory parameters.

    A free-return trajectory uses the Moon's gravity to return the
    spacecraft to Earth without propulsive maneuvers.

    Parameters
    ----------
    t_departure : float
        Departure time (days from epoch)
    flyby_altitude_km : float
        Lunar flyby altitude (km)

    Returns
    -------
    dict
        Free-return trajectory parameters
    """
    earth = BODIES['earth']
    moon = BODIES['moon']

    # Get Earth and Moon positions at departure
    time_dep = ts.tt_jd(epoch_jd + t_departure)
    r_earth = body_position(earth, time_dep)
    v_earth = body_velocity(earth, time_dep)

    # Estimate Moon position at arrival (~3 days later)
    t_arrival_moon = t_departure + 3.0  # 3-day transfer
    time_arr = ts.tt_jd(epoch_jd + t_arrival_moon)
    r_moon_arrival = body_position(moon, time_arr)
    v_moon_arrival = body_velocity(moon, time_arr)

    # Earth-Moon distance
    em_distance_au = np.linalg.norm(r_moon_arrival - r_earth)
    em_distance_km = au_to_km(em_distance_au)

    # For free-return, we need to achieve a specific C3 and trajectory
    # Apollo used ~10.9 km²/s² C3 for lunar missions

    # Compute transfer using Lambert solver
    tof_days = 3.0  # 3-day transfer to Moon

    # Position vectors (Earth-centered for TLI)
    r1_ssb = r_earth  # Start at Earth
    r2_ssb = r_moon_arrival  # End at Moon

    # Solve Lambert problem for transfer
    solution = solve_lambert(
        r1_ssb, r2_ssb, tof_days,
        BODIES['sun']['mu'],
        transfer_type=TransferType.SHORT_WAY
    )

    if solution is None or not solution.converged:
        print("Warning: Lambert solver did not converge, using Hohmann estimate")
        # Fallback to Hohmann-like estimate
        v_tli_estimate = 3.1  # km/s TLI delta-V
        c3_estimate = 10.9   # km²/s²
    else:
        # Compute required departure velocity
        v_depart_helio = solution.v1

        # Convert to Earth-centered
        v_earth_helio = v_earth
        v_depart_earth = v_depart_helio - v_earth_helio

        # V-infinity at Earth departure
        v_inf_depart = v_depart_earth * AU_TO_KM / DAY_TO_SEC  # km/s

        # C3 = v_inf²
        c3_estimate = np.dot(v_inf_depart, v_inf_depart)
        v_tli_estimate = None  # Will compute from C3

    # Flyby parameters
    rp_flyby = R_MOON_KM + flyby_altitude_km

    # Return trajectory parameters
    # For free-return, spacecraft returns to Earth entry interface
    # Entry velocity ~11 km/s from lunar return

    return {
        'departure_time': t_departure,
        'arrival_moon_time': t_arrival_moon,
        'tof_to_moon_days': tof_days,
        'earth_moon_distance_km': em_distance_km,
        'c3_km2s2': c3_estimate,
        'flyby_periapsis_km': rp_flyby,
        'flyby_altitude_km': flyby_altitude_km,
        'entry_velocity_km_s': 11.0,  # Apollo lunar return
        'lambert_solution': solution,
    }


def compute_tli_delta_v(c3_km2s2: float, parking_alt_km: float = 185.0) -> float:
    """
    Compute TLI delta-V to achieve required C3.

    Parameters
    ----------
    c3_km2s2 : float
        Characteristic energy (km²/s²)
    parking_alt_km : float
        Parking orbit altitude (km)

    Returns
    -------
    float
        TLI delta-V (km/s)
    """
    r_parking = R_EARTH_KM + parking_alt_km

    # Circular velocity in parking orbit
    v_circ = np.sqrt(MU_EARTH_KM / r_parking)

    # Hyperbolic departure velocity (vis-viva with negative C3 term)
    # v² = C3 + 2μ/r
    v_depart = np.sqrt(c3_km2s2 + 2 * MU_EARTH_KM / r_parking)

    return v_depart - v_circ


# =============================================================================
# Mission Phases
# =============================================================================

def phase1_mission_design() -> dict:
    """Phase 1: Mission design and trajectory planning."""
    print_header("PHASE 1: MISSION DESIGN")

    # Select departure window
    t_departure = 0.5  # Half day from epoch

    # Compute free-return parameters
    fr_params = compute_free_return_parameters(
        t_departure,
        MoonFlybyConfig.FLYBY_ALTITUDE_KM
    )

    print_subheader("Free-Return Trajectory Design")
    print_metric("Earth-Moon Distance", fr_params['earth_moon_distance_km'], "km", ".0f")
    print_metric("Transfer Time to Moon", fr_params['tof_to_moon_days'], "days")
    print_metric("Departure C3", fr_params['c3_km2s2'], "km²/s²")
    print_metric("Moon Flyby Altitude", fr_params['flyby_altitude_km'], "km", ".0f")
    print_metric("Entry Velocity", fr_params['entry_velocity_km_s'], "km/s")

    # Compute TLI delta-V
    tli_dv = compute_tli_delta_v(fr_params['c3_km2s2'], MoonFlybyConfig.LEO_ALTITUDE_KM)
    print_metric("Required TLI ΔV", tli_dv * 1000, "m/s", ".1f")

    # Apollo reference comparison
    print_subheader("Apollo Reference Comparison")
    print_metric("Apollo TLI ΔV", 3120, "m/s (typical)", ".0f")
    print_metric("Apollo C3", 10.9, "km²/s² (typical)")
    print_metric("Apollo Entry Velocity", 11.0, "km/s")

    # Entry corridor analysis
    corridor = earth_reentry_corridor(
        fr_params['entry_velocity_km_s'],
        vehicle_type='capsule'
    )

    print_subheader("Entry Corridor Analysis")
    print_metric("Steep Limit (g-limit)", corridor['steep_limit_deg'], "°")
    print_metric("Shallow Limit (skip-out)", corridor['shallow_limit_deg'], "°")
    print_metric("Corridor Width", corridor['corridor_width_deg'], "°")
    print_metric("Nominal Entry FPA", corridor['nominal_entry_deg'], "°")
    print_metric("Vehicle L/D", corridor['lift_to_drag'])

    fr_params['tli_delta_v_km_s'] = tli_dv
    fr_params['entry_corridor'] = corridor

    return fr_params


def phase2_leo_setup(mission: dict) -> Propagator:
    """Phase 2: Set up spacecraft in LEO parking orbit."""
    print_header("PHASE 2: LEO PARKING ORBIT")

    earth = BODIES['earth']

    # Create spacecraft (Apollo CM-like)
    rcs_thruster = Thruster(thrust_max=50000, isp=300)  # 50 kN RCS
    ship = Spacecraft(
        name="Apollo-like Capsule",
        dry_mass=MoonFlybyConfig.VEHICLE_MASS_KG * 0.9,  # 90% dry
        propellant_mass=MoonFlybyConfig.VEHICLE_MASS_KG * 0.1,  # 10% prop for corrections
        thrusters=[rcs_thruster]
    )

    # Initial position in LEO
    r_leo_km = R_EARTH_KM + MoonFlybyConfig.LEO_ALTITUDE_KM
    r_leo_au = km_to_au(r_leo_km)

    # Circular velocity
    v_leo_kms = np.sqrt(MU_EARTH_KM / r_leo_km)
    v_leo_aud = v_leo_kms * DAY_TO_SEC / AU_TO_KM

    # Earth position at departure
    t_dep = mission['departure_time']
    time_dep = ts.tt_jd(epoch_jd + t_dep)
    r_earth = body_position(earth, time_dep)
    v_earth = body_velocity(earth, time_dep)

    # Position in ecliptic plane
    r_rel = np.array([r_leo_au, 0, 0])
    v_rel = np.array([0, v_leo_aud, 0])

    r0 = r_earth + r_rel
    v0 = v_earth + v_rel

    # Default attitude state (identity quaternion, no angular velocity)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion [w, x, y, z]
    omega0 = np.zeros(3)  # No initial rotation (rad/s)

    # Create state vector
    y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

    # Create propagator (no guidance for coasting)
    prop = Propagator(ship=ship, bodies=["sun", "earth", "moon"])
    prop.result.initialize(y0, t_dep)

    print_subheader("Parking Orbit Parameters")
    print_metric("Orbit Altitude", MoonFlybyConfig.LEO_ALTITUDE_KM, "km", ".0f")
    print_metric("Orbital Velocity", v_leo_kms, "km/s")
    print_metric("Orbital Period", orbital_period(r_leo_km, MU_EARTH_KM) / 60, "min")
    print_metric("Spacecraft Mass", ship.mass, "kg", ".0f")

    return prop


def phase3_tli_burn(prop: Propagator, mission: dict) -> Propagator:
    """Phase 3: Trans-Lunar Injection burn."""
    print_header("PHASE 3: TRANS-LUNAR INJECTION")

    earth = BODIES['earth']
    moon = BODIES['moon']

    # TLI delta-V required
    tli_dv_kms = mission['tli_delta_v_km_s']
    tli_dv_aud = tli_dv_kms * DAY_TO_SEC / AU_TO_KM

    print_subheader("TLI Burn Parameters")
    print_metric("Required ΔV", tli_dv_kms * 1000, "m/s", ".1f")

    # Current velocity direction (prograde burn)
    v_current = prop.v
    v_dir = v_current / np.linalg.norm(v_current)

    # Apply TLI burn (impulsive for simplicity)
    v_new = prop.v + tli_dv_aud * v_dir

    # Update state
    new_state = prop.y.copy()
    new_state[3:6] = v_new
    t_current = prop.t

    # Create new propagator with post-TLI state
    new_prop = Propagator(ship=prop.ship, bodies=["sun", "earth", "moon"])
    new_prop.result.initialize(new_state, t_current)
    prop = new_prop

    # Calculate achieved C3
    time_now = ts.tt_jd(epoch_jd + prop.t)
    r_earth = body_position(earth, time_now)
    v_earth = body_velocity(earth, time_now)
    v_rel = (prop.v - v_earth) * AU_TO_KM / DAY_TO_SEC  # km/s
    c3_achieved = np.dot(v_rel, v_rel)

    print_metric("Achieved C3", c3_achieved, "km²/s²")
    print_metric("Burn Complete at t=", prop.t, "days")

    # Estimate burn duration (for reference)
    thrust = prop.ship.max_thrust
    isp = prop.ship.max_isp
    m0 = prop.ship.mass
    if isp > 0:
        mdot = thrust / (isp * 9.80665)
        dv_mps = tli_dv_kms * 1000
        t_burn = m0 / mdot * (1 - np.exp(-dv_mps / (isp * 9.80665)))
    else:
        t_burn = 0  # Impulsive approximation

    print_metric("Estimated Burn Duration", t_burn, "seconds", ".0f")

    mission['c3_achieved'] = c3_achieved

    return prop


def phase4_lunar_transit(prop: Propagator, mission: dict) -> Propagator:
    """Phase 4: Coast to Moon."""
    print_header("PHASE 4: TRANS-LUNAR COAST")

    earth = BODIES['earth']
    moon = BODIES['moon']

    # Propagate to near Moon
    t_arrival = mission['arrival_moon_time']

    print_subheader("Transit Progress")
    print(f"  {'Day':<8} {'Dist Earth':<15} {'Dist Moon':<15} {'Velocity':<12}")
    print(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*12}")

    # Checkpoint propagation with high density (max_step=0.01 days = ~15 min)
    n_checkpoints = 20  # Increased for better progress visibility
    checkpoints = np.linspace(prop.t, t_arrival - 0.1, n_checkpoints)

    for checkpoint in checkpoints:
        prop.propagate(checkpoint, max_step=0.01)  # Fine resolution

        dist_earth = distance_to_body(prop.t, prop.y, earth, surface=True)
        dist_moon = distance_to_body(prop.t, prop.y, moon, surface=True)
        v_mag = np.linalg.norm(prop.v) * AU_TO_KM / DAY_TO_SEC

        print(f"  {prop.t:<8.2f} {dist_earth:>12,.0f} km  {dist_moon:>12,.0f} km  {v_mag:>9.2f} km/s")

    return prop


def phase5_lunar_flyby(prop: Propagator, mission: dict) -> tuple:
    """Phase 5: Lunar flyby."""
    print_header("PHASE 5: LUNAR FLYBY")

    moon = BODIES['moon']
    earth = BODIES['earth']

    # Find closest approach to Moon
    print_subheader("Finding Lunar Closest Approach")

    t_start = prop.t
    min_dist = float('inf')
    t_periapsis = prop.t

    # Search for periapsis with fine resolution (~15 min steps)
    step = 0.01  # ~15 minutes
    while prop.t < t_start + 1.0:  # Search for 1 day
        prop.propagate(prop.t + step, max_step=0.005)  # Fine resolution for flyby
        dist = distance_to_body(prop.t, prop.y, moon, surface=True)

        if dist < min_dist:
            min_dist = dist
            t_periapsis = prop.t
        elif dist > min_dist * 1.5 and min_dist < 100000:
            break

    print_metric("Periapsis Time", t_periapsis, "days")
    print_metric("Flyby Altitude", min_dist, "km", ".0f")

    # Get state at periapsis
    time_peri = ts.tt_jd(epoch_jd + t_periapsis)
    r_moon = body_position(moon, time_peri)
    v_moon = body_velocity(moon, time_peri)

    # Relative velocity at flyby
    v_rel = prop.v - v_moon
    v_inf = np.linalg.norm(v_rel) * AU_TO_KM / DAY_TO_SEC

    print_metric("V-infinity at Moon", v_inf, "km/s")

    # Calculate turn angle
    rp_flyby = R_MOON_KM + min_dist
    try:
        turn_angle = flyby_turn_angle(v_inf, rp_flyby, MU_MOON_KM)
        print_metric("Turn Angle", np.degrees(turn_angle), "°")
    except ValueError:
        turn_angle = 0
        print("  Warning: Could not compute turn angle")

    # Store flyby data
    flyby_data = {
        'periapsis_time': t_periapsis,
        'altitude_km': min_dist,
        'v_infinity_km_s': v_inf,
        'turn_angle_deg': np.degrees(turn_angle) if turn_angle else 0,
    }

    return prop, flyby_data


def phase6_earth_return(prop: Propagator, mission: dict) -> tuple:
    """Phase 6: Return trajectory to Earth."""
    print_header("PHASE 6: EARTH RETURN COAST")

    earth = BODIES['earth']
    moon = BODIES['moon']

    # Propagate until Earth approach
    # Free-return should bring us back in ~3 more days
    t_return = prop.t + 3.5

    print_subheader("Return Transit Progress")
    print(f"  {'Day':<8} {'Dist Earth':<15} {'Dist Moon':<15} {'Velocity':<12}")
    print(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*12}")

    # High density propagation (max_step=0.01 days = ~15 min)
    n_checkpoints = 20  # Increased for better progress visibility
    checkpoints = np.linspace(prop.t, t_return, n_checkpoints)

    trajectory_data = []

    for checkpoint in checkpoints:
        prop.propagate(checkpoint, max_step=0.01)  # Fine resolution

        dist_earth = distance_to_body(prop.t, prop.y, earth, surface=True)
        dist_moon = distance_to_body(prop.t, prop.y, moon, surface=True)
        v_mag = np.linalg.norm(prop.v) * AU_TO_KM / DAY_TO_SEC

        print(f"  {prop.t:<8.2f} {dist_earth:>12,.0f} km  {dist_moon:>12,.0f} km  {v_mag:>9.2f} km/s")

        trajectory_data.append({
            'time': prop.t,
            'r': prop.r.copy(),
            'v': prop.v.copy(),
            'dist_earth': dist_earth,
            'velocity': v_mag,
        })

    # Find entry interface
    print_subheader("Approaching Entry Interface")

    # Propagate until entry interface (~121 km altitude)
    entry_interface_found = False
    while prop.t < t_return + 2.0:
        prop.propagate(prop.t + 0.001, max_step=0.001)  # Fine steps
        dist_earth = distance_to_body(prop.t, prop.y, earth, surface=True)

        if dist_earth < MoonFlybyConfig.ENTRY_INTERFACE_KM:
            entry_interface_found = True
            break

    if entry_interface_found:
        print_metric("Entry Interface Time", prop.t, "days")
        print_metric("Entry Altitude", dist_earth, "km", ".0f")

        # Calculate entry velocity and FPA
        time_entry = ts.tt_jd(epoch_jd + prop.t)
        r_earth = body_position(earth, time_entry)
        v_earth = body_velocity(earth, time_entry)

        r_rel = (prop.r - r_earth) * AU_TO_KM  # km
        v_rel = (prop.v - v_earth) * AU_TO_KM / DAY_TO_SEC  # km/s

        v_entry = np.linalg.norm(v_rel)
        print_metric("Entry Velocity", v_entry, "km/s")

        # Flight path angle
        r_unit = r_rel / np.linalg.norm(r_rel)
        v_unit = v_rel / np.linalg.norm(v_rel)
        fpa = np.arcsin(np.dot(r_unit, v_unit))
        fpa_deg = np.degrees(fpa)
        print_metric("Entry Flight Path Angle", fpa_deg, "°")

        entry_state = {
            'time': prop.t,
            'altitude_km': dist_earth,
            'velocity_km_s': v_entry,
            'fpa_deg': fpa_deg,
            'r_rel_km': r_rel,
            'v_rel_km_s': v_rel,
        }
    else:
        print("  Warning: Entry interface not reached")
        entry_state = None

    return prop, trajectory_data, entry_state


def phase7_atmospheric_entry(entry_state: dict, mission: dict) -> EntryResult:
    """Phase 7: Atmospheric entry simulation."""
    print_header("PHASE 7: ATMOSPHERIC ENTRY")

    if entry_state is None:
        print("  Error: No entry state available")
        return None

    # Entry conditions
    entry = EntryConditions(
        velocity=entry_state['velocity_km_s'],
        flight_path_angle=entry_state['fpa_deg'],
        altitude=entry_state['altitude_km'],
        latitude=0.0,
        longitude=0.0,
        azimuth=90.0
    )

    # Vehicle parameters (Apollo CM-like)
    area = np.pi * (MoonFlybyConfig.VEHICLE_DIAMETER_M / 2)**2
    beta = ballistic_coefficient(
        MoonFlybyConfig.VEHICLE_MASS_KG,
        MoonFlybyConfig.DRAG_COEFFICIENT,
        area
    )

    print_subheader("Entry Parameters")
    print_metric("Entry Velocity", entry.velocity, "km/s")
    print_metric("Entry FPA", entry.flight_path_angle, "°")
    print_metric("Entry Altitude", entry.altitude, "km", ".0f")
    print_metric("Vehicle Mass", MoonFlybyConfig.VEHICLE_MASS_KG, "kg", ".0f")
    print_metric("Ballistic Coefficient", beta, "kg/m²", ".0f")
    print_metric("Lift-to-Drag Ratio", MoonFlybyConfig.LIFT_TO_DRAG)

    # Simulate entry
    print_subheader("Simulating Entry...")
    result = simulate_entry(
        entry=entry,
        ballistic_coef=beta,
        nose_radius_m=MoonFlybyConfig.NOSE_RADIUS_M,
        body='earth',
        lift_to_drag=MoonFlybyConfig.LIFT_TO_DRAG,
        dt=0.5
    )

    print_subheader("Entry Results")
    print_metric("Peak Deceleration", result.peak_deceleration_g, "g's")
    print_metric("Peak Heat Rate", result.peak_heat_rate_w_cm2, "W/cm²", ".0f")
    print_metric("Total Heat Load", result.total_heat_load_j_cm2, "J/cm²", ".0f")
    print_metric("Peak Decel Altitude", result.altitude_peak_decel, "km", ".0f")
    print_metric("Peak Heating Altitude", result.altitude_peak_heating, "km", ".0f")
    print_metric("Max Dynamic Pressure", result.max_dynamic_pressure / 1000, "kPa", ".1f")
    print_metric("Time to Peak Decel", result.time_to_peak_decel, "seconds", ".0f")

    # Compare to Apollo
    print_subheader("Apollo Comparison")
    print_metric("Apollo Peak Decel", "6-7", "g's")
    print_metric("Apollo Peak Heat Rate", "1000-1500", "W/cm²")
    print_metric("Apollo Entry FPA", "-6.5", "° (nominal)")

    # Parachute deployment
    chute = parachute_deployment_conditions(2.0, 10.0, 'earth')
    print_subheader("Parachute Deployment (Mach 2.0)")
    print_metric("Deployment Altitude", chute['altitude_km'], "km", ".0f")
    print_metric("Deployment Velocity", chute['velocity_m_s'], "m/s", ".0f")
    print_metric("Dynamic Pressure", chute['dynamic_pressure_pa'] / 1000, "kPa", ".1f")

    return result


def phase8_thermal_analysis(entry_result: EntryResult, mission: dict) -> dict:
    """Phase 8: Thermal environment analysis."""
    print_header("PHASE 8: THERMAL ANALYSIS")

    if entry_result is None:
        print("  Error: No entry result available")
        return None

    # TPS requirements
    print_subheader("Thermal Protection System Analysis")

    peak_heat = entry_result.peak_heat_rate_w_cm2
    total_heat = entry_result.total_heat_load_j_cm2

    # Estimate TPS thickness (simplified correlation)
    # For AVCOAT (Apollo): ~2-3 cm for ~1500 W/cm² peak
    tps_thickness_cm = 2.0 * (peak_heat / 1500.0)**0.5

    print_metric("Peak Heating Rate", peak_heat, "W/cm²", ".0f")
    print_metric("Total Heat Load", total_heat, "J/cm²", ".0f")
    print_metric("Estimated TPS Thickness", tps_thickness_cm, "cm")

    # Temperature estimates
    # Stagnation temperature estimate (simplified)
    v_entry = entry_result.trajectory[0].velocity * 1000  # m/s
    t_stag = 300 + 0.5 * v_entry**2 / 1000  # Rough estimate (K)

    print_metric("Stagnation Temperature (est)", t_stag, "K", ".0f")
    print_metric("TPS Recession (est)", peak_heat / 500, "mm", ".1f")

    # Apollo TPS reference
    print_subheader("Apollo TPS Reference")
    print("  Material: AVCOAT-5026-39/HC-G")
    print("  Thickness: 2-3 cm (varies by location)")
    print("  Max surface temp: ~2800 K")
    print("  Ablation rate: ~0.5 mm/s at peak heating")

    thermal_data = {
        'peak_heat_rate': peak_heat,
        'total_heat_load': total_heat,
        'tps_thickness_cm': tps_thickness_cm,
        'stagnation_temp_k': t_stag,
    }

    return thermal_data


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_trajectory_3d(trajectory_data: list, flyby_data: dict, mission: dict) -> None:
    """Plot 3D trajectory visualization."""
    print_subheader("Generating 3D Trajectory Plot...")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract trajectory points
    times = [p['time'] for p in trajectory_data]
    positions = np.array([p['r'] for p in trajectory_data]) * AU_TO_KM  # km

    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=2, label='Spacecraft')

    # Plot Earth position
    earth = BODIES['earth']
    time_start = ts.tt_jd(epoch_jd + times[0])
    time_end = ts.tt_jd(epoch_jd + times[-1])
    r_earth_start = body_position(earth, time_start) * AU_TO_KM
    r_earth_end = body_position(earth, time_end) * AU_TO_KM
    ax.scatter(*r_earth_start, c='blue', s=200, marker='o', label='Earth (start)')
    ax.scatter(*r_earth_end, c='cyan', s=200, marker='o', label='Earth (end)')

    # Plot Moon position
    moon = BODIES['moon']
    if flyby_data:
        time_flyby = ts.tt_jd(epoch_jd + flyby_data['periapsis_time'])
        r_moon_flyby = body_position(moon, time_flyby) * AU_TO_KM
        ax.scatter(*r_moon_flyby, c='gray', s=150, marker='o', label='Moon (flyby)')

    # Labels
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Circumlunar Free-Return Trajectory')
    ax.legend()

    if MoonFlybyConfig.SAVE_PLOTS:
        plt.savefig('/home/boss/data/space_state/moon_flyby_trajectory.png', dpi=150)
        print("  Saved: moon_flyby_trajectory.png")

    plt.close()


def plot_entry_profile(entry_result: EntryResult) -> None:
    """Plot entry trajectory profiles."""
    print_subheader("Generating Entry Profile Plots...")

    if entry_result is None or not entry_result.trajectory:
        print("  No entry data available")
        return

    trajectory = entry_result.trajectory

    times = [p.time for p in trajectory]
    altitudes = [p.altitude for p in trajectory]
    velocities = [p.velocity for p in trajectory]
    g_loads = [p.deceleration for p in trajectory]
    heat_rates = [p.heat_rate for p in trajectory]
    mach = [p.mach_number for p in trajectory]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Atmospheric Entry Profile (Apollo-like Lunar Return)', fontsize=14)

    # Altitude vs Time
    axes[0, 0].plot(times, altitudes, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Altitude (km)')
    axes[0, 0].set_title('Altitude Profile')
    axes[0, 0].grid(True, alpha=0.3)

    # Velocity vs Time
    axes[0, 1].plot(times, velocities, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (km/s)')
    axes[0, 1].set_title('Velocity Profile')
    axes[0, 1].grid(True, alpha=0.3)

    # G-load vs Time
    axes[0, 2].plot(times, g_loads, 'g-', linewidth=2)
    axes[0, 2].axhline(y=entry_result.peak_deceleration_g, color='r', linestyle='--',
                       label=f'Peak: {entry_result.peak_deceleration_g:.1f} g')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Deceleration (g)')
    axes[0, 2].set_title('G-Load Profile')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Heat Rate vs Time
    axes[1, 0].plot(times, heat_rates, 'm-', linewidth=2)
    axes[1, 0].axhline(y=entry_result.peak_heat_rate_w_cm2, color='r', linestyle='--',
                       label=f'Peak: {entry_result.peak_heat_rate_w_cm2:.0f} W/cm²')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Heat Rate (W/cm²)')
    axes[1, 0].set_title('Aerodynamic Heating')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Mach Number vs Time
    axes[1, 1].plot(times, mach, 'c-', linewidth=2)
    axes[1, 1].axhline(y=2.0, color='r', linestyle='--', label='Mach 2 (chute deploy)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Mach Number')
    axes[1, 1].set_title('Mach Number Profile')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Altitude vs Velocity (energy dissipation)
    axes[1, 2].plot(velocities, altitudes, 'k-', linewidth=2)
    axes[1, 2].set_xlabel('Velocity (km/s)')
    axes[1, 2].set_ylabel('Altitude (km)')
    axes[1, 2].set_title('Altitude-Velocity Profile')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if MoonFlybyConfig.SAVE_PLOTS:
        plt.savefig('/home/boss/data/space_state/entry_profile.png', dpi=150)
        print("  Saved: entry_profile.png")

    plt.close()


def plot_thermal_analysis(entry_result: EntryResult, thermal_data: dict) -> None:
    """Plot thermal analysis visualization."""
    print_subheader("Generating Thermal Analysis Plots...")

    if entry_result is None or thermal_data is None:
        print("  No thermal data available")
        return

    trajectory = entry_result.trajectory

    times = [p.time for p in trajectory]
    altitudes = [p.altitude for p in trajectory]
    heat_rates = [p.heat_rate for p in trajectory]
    dyn_press = [p.dynamic_pressure for p in trajectory]
    velocities = [p.velocity for p in trajectory]

    # Compute cumulative heat load
    dt = times[1] - times[0] if len(times) > 1 else 0.5
    cumulative_heat = np.cumsum(heat_rates) * dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Thermal Protection System Analysis', fontsize=14)

    # Heat Rate vs Altitude
    axes[0, 0].plot(altitudes, heat_rates, 'r-', linewidth=2)
    axes[0, 0].set_xlabel('Altitude (km)')
    axes[0, 0].set_ylabel('Heat Rate (W/cm²)')
    axes[0, 0].set_title('Heat Rate vs Altitude')
    axes[0, 0].invert_xaxis()
    axes[0, 0].grid(True, alpha=0.3)

    # Mark peak heating altitude
    peak_idx = np.argmax(heat_rates)
    axes[0, 0].axvline(x=altitudes[peak_idx], color='g', linestyle='--',
                       label=f'Peak at {altitudes[peak_idx]:.0f} km')
    axes[0, 0].legend()

    # Cumulative Heat Load
    axes[0, 1].plot(times, cumulative_heat, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Cumulative Heat (J/cm²)')
    axes[0, 1].set_title('Integrated Heat Load')
    axes[0, 1].grid(True, alpha=0.3)

    # Dynamic Pressure vs Altitude
    axes[1, 0].plot(altitudes, np.array(dyn_press) / 1000, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Altitude (km)')
    axes[1, 0].set_ylabel('Dynamic Pressure (kPa)')
    axes[1, 0].set_title('Dynamic Pressure Profile')
    axes[1, 0].invert_xaxis()
    axes[1, 0].grid(True, alpha=0.3)

    # Heat Rate vs Velocity
    axes[1, 1].plot(velocities, heat_rates, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Velocity (km/s)')
    axes[1, 1].set_ylabel('Heat Rate (W/cm²)')
    axes[1, 1].set_title('Heat Rate vs Velocity')
    axes[1, 1].grid(True, alpha=0.3)

    # Apollo reference line
    axes[1, 1].axhline(y=1500, color='r', linestyle='--', alpha=0.5,
                       label='Apollo peak (~1500 W/cm²)')
    axes[1, 1].legend()

    plt.tight_layout()

    if MoonFlybyConfig.SAVE_PLOTS:
        plt.savefig('/home/boss/data/space_state/thermal_analysis.png', dpi=150)
        print("  Saved: thermal_analysis.png")

    plt.close()


# =============================================================================
# Mission Validation
# =============================================================================

def validate_mission(mission: dict, flyby_data: dict, entry_state: dict,
                    entry_result: EntryResult, thermal_data: dict) -> dict:
    """Validate mission against Apollo reference data."""
    print_header("MISSION VALIDATION")

    validation = {
        'checks': [],
        'passed': 0,
        'failed': 0,
    }

    def check(name, value, ref_low, ref_high, unit=''):
        passed = ref_low <= value <= ref_high
        status = "PASS" if passed else "FAIL"
        validation['checks'].append({
            'name': name,
            'value': value,
            'reference': f"{ref_low}-{ref_high}",
            'status': status
        })
        if passed:
            validation['passed'] += 1
        else:
            validation['failed'] += 1
        print(f"  [{status}] {name}: {value:.2f} {unit} (ref: {ref_low}-{ref_high})")

    print_subheader("Apollo Reference Comparison")

    # TLI Delta-V
    if 'tli_delta_v_km_s' in mission:
        check("TLI ΔV", mission['tli_delta_v_km_s'] * 1000, 3000, 3300, "m/s")

    # C3
    if 'c3_achieved' in mission:
        check("Departure C3", mission['c3_achieved'], 8.0, 14.0, "km²/s²")

    # Flyby altitude
    if flyby_data and 'altitude_km' in flyby_data:
        check("Flyby Altitude", flyby_data['altitude_km'], 100, 500, "km")

    # Entry velocity
    if entry_state and 'velocity_km_s' in entry_state:
        check("Entry Velocity", entry_state['velocity_km_s'], 10.5, 11.5, "km/s")

    # Entry FPA
    if entry_state and 'fpa_deg' in entry_state:
        check("Entry FPA", abs(entry_state['fpa_deg']), 5.5, 8.0, "°")

    # Peak deceleration
    if entry_result:
        check("Peak Deceleration", entry_result.peak_deceleration_g, 4.0, 10.0, "g")

    # Peak heating
    if entry_result:
        check("Peak Heat Rate", entry_result.peak_heat_rate_w_cm2, 500, 2000, "W/cm²")

    print_subheader("Validation Summary")
    total = validation['passed'] + validation['failed']
    print(f"  Passed: {validation['passed']}/{total}")
    print(f"  Failed: {validation['failed']}/{total}")
    print(f"  Success Rate: {100*validation['passed']/total:.0f}%")

    return validation


# =============================================================================
# Main Mission Execution
# =============================================================================

def run_mission() -> dict:
    """Execute the complete Moon flyby and Earth reentry mission."""
    print("\n" + "=" * 70)
    print("  CIRCUMLUNAR FREE-RETURN AND EARTH REENTRY MISSION")
    print("  Apollo-Style Trajectory with Reentry Analysis")
    print("=" * 70)

    results = {}
    cfg = MoonFlybyConfig

    # -------------------------------------------------------------------------
    # MISSION CONTROLLER SETUP
    # -------------------------------------------------------------------------
    controller = None
    if cfg.ENABLE_FDR:
        controller = MissionController(
            mission_name=cfg.NAME,
            display_mode=cfg.DISPLAY_MODE,
            enable_fdr=True,
            fdr_interval=cfg.FDR_INTERVAL,
            display_interval=5.0,  # Update every 5 seconds
        )
        controller.set_bodies(BODIES)
        print(f"\n  FDR Enabled: Log directory = {controller.fdr.log_dir if controller.fdr else 'N/A'}")

    # Phase 1: Mission Design
    if controller:
        controller.begin_phase("MISSION DESIGN", "Computing free-return trajectory parameters")
    mission = phase1_mission_design()
    results['mission'] = mission

    # Phase 2: LEO Setup
    if controller:
        controller.begin_phase("LEO PARKING ORBIT", f"Altitude: {cfg.LEO_ALTITUDE_KM} km")
    prop = phase2_leo_setup(mission)

    # Attach controller to propagator
    if controller:
        controller.set_spacecraft(prop.ship)
        controller.set_start_time(prop.t)
        controller.begin_mission()
        prop.attach_controller(controller)

    # Phase 3: TLI Burn
    if controller:
        controller.begin_phase("TRANS-LUNAR INJECTION", f"Target C3: {mission['c3_km2s2']:.1f} km²/s²")
        controller.start_burn(prop.t, "TLI Burn")
    prop = phase3_tli_burn(prop, mission)
    if controller:
        controller.end_burn(prop.t, "TLI Burn")
        controller.log_event(prop.t, "TLI_COMPLETE", f"C3 achieved: {mission.get('c3_achieved', 0):.1f} km²/s²")
    results['post_tli'] = {'c3': mission.get('c3_achieved')}

    # Phase 4: Lunar Transit
    if controller:
        controller.begin_phase("TRANS-LUNAR COAST", "Coast to lunar encounter")
    prop = phase4_lunar_transit(prop, mission)

    # Phase 5: Lunar Flyby
    if controller:
        controller.begin_phase("LUNAR FLYBY", f"Target altitude: {cfg.FLYBY_ALTITUDE_KM} km")
    prop, flyby_data = phase5_lunar_flyby(prop, mission)
    if controller and flyby_data:
        controller.log_event(
            flyby_data['periapsis_time'], "LUNAR_PERIAPSIS",
            f"Flyby altitude: {flyby_data['altitude_km']:.0f} km, V∞: {flyby_data['v_infinity_km_s']:.2f} km/s"
        )
    results['flyby'] = flyby_data

    # Phase 6: Earth Return
    if controller:
        controller.begin_phase("EARTH RETURN", "Free-return coast to Earth")
    prop, trajectory_data, entry_state = phase6_earth_return(prop, mission)
    if controller and entry_state:
        controller.log_event(
            entry_state['time'], "ENTRY_INTERFACE",
            f"Altitude: {entry_state['altitude_km']:.0f} km, V: {entry_state['velocity_km_s']:.2f} km/s, FPA: {entry_state['fpa_deg']:.1f}°"
        )
    results['trajectory'] = trajectory_data
    results['entry_state'] = entry_state
    results['propagator'] = prop  # Store full propagator for visualization

    # Phase 7: Atmospheric Entry
    if controller:
        controller.begin_phase("ATMOSPHERIC ENTRY", "Simulating reentry heating and deceleration")
    entry_result = phase7_atmospheric_entry(entry_state, mission)
    if controller and entry_result:
        controller.log_event(
            prop.t, "ENTRY_COMPLETE",
            f"Peak g: {entry_result.peak_deceleration_g:.1f}, Peak heat: {entry_result.peak_heat_rate_w_cm2:.0f} W/cm²"
        )
    results['entry_result'] = entry_result

    # Phase 8: Thermal Analysis
    if controller:
        controller.begin_phase("THERMAL ANALYSIS", "TPS assessment and Apollo comparison")
    thermal_data = phase8_thermal_analysis(entry_result, mission)
    results['thermal'] = thermal_data

    # -------------------------------------------------------------------------
    # RECORD TRAJECTORY AND FINALIZE
    # -------------------------------------------------------------------------
    if controller and controller.fdr:
        # Record full trajectory from propagation result
        frames = controller.fdr.record_trajectory(prop.result, prop.ship, BODIES, downsample=5)
        print(f"\n  Recorded {frames} trajectory frames to FDR")

        # Force final update
        controller.update(prop.t, prop.y, force=True)

    # Validation
    validation = validate_mission(mission, flyby_data, entry_state,
                                  entry_result, thermal_data)
    results['validation'] = validation

    # -------------------------------------------------------------------------
    # FINALIZE MISSION AND GENERATE PLOTS
    # -------------------------------------------------------------------------
    if controller:
        mission_success = validation['passed'] >= validation['passed'] + validation['failed'] * 0.5
        finalize_result = controller.finalize(
            success=mission_success,
            objectives=[
                "LEO departure completed",
                "TLI burn executed",
                "Lunar flyby achieved",
                "Free-return trajectory verified",
                "Entry corridor analysis complete",
            ]
        )

        if finalize_result:
            print_header("MISSION DATA EXPORT")
            summary = finalize_result['summary']
            print(f"  Duration: {summary.duration_str}")
            print(f"  Total Delta-V: {summary.total_delta_v_mps:.1f} m/s")
            print(f"  Propellant Used: {summary.propellant_used_kg:,.0f} kg")
            print()
            print("  Saved Files:")
            for fmt, path in finalize_result['files'].items():
                print(f"    {fmt}: {path}")

        # Generate mission plots
        if controller.fdr:
            print_header("GENERATING MISSION PLOTS")
            plots = generate_mission_plots(controller.fdr, show=False)
            for name, path in plots.items():
                print(f"    {name}: {path}")

        results['controller'] = controller

    # Visualization
    if MoonFlybyConfig.SHOW_TRAJECTORY_3D and trajectory_data:
        plot_trajectory_3d(trajectory_data, flyby_data, mission)

    if MoonFlybyConfig.SHOW_THERMAL_PLOTS and entry_result:
        plot_entry_profile(entry_result)
        plot_thermal_analysis(entry_result, thermal_data)

    # Final Summary
    print_header("MISSION COMPLETE")
    print(f"  Total Checks Passed: {validation['passed']}")
    print(f"  Total Checks Failed: {validation['failed']}")

    return results


# =============================================================================
# Interactive Visualization
# =============================================================================

def visualize_mission_interactive(results: dict, mode: str = 'interactive'):
    """
    Launch interactive visualization for the Moon flyby mission.

    Args:
        results: Mission results dictionary
        mode: 'interactive', 'dashboard', or 'full'
    """
    from visual import (
        TrajectoryData,
        BodyData,
        EventMarker,
        create_interactive_trajectory_viewer,
        create_mission_dashboard,
        run_mission_viewer,
        trajectory_from_propagation_result,
    )

    print_header("INTERACTIVE MISSION VISUALIZATION")

    # Use full propagator trajectory if available (preferred - contains all phases)
    prop = results.get('propagator')
    if prop is not None and hasattr(prop, 'result') and prop.result.t.size > 0:
        print("  Using full propagation trajectory (all phases)")
        trajectory = trajectory_from_propagation_result(prop.result, name="Moon Flyby Spacecraft")
        trajectory.color = "cyan"
        times = trajectory.times
        positions = trajectory.positions
        print(f"  Trajectory points: {len(times)}")
        print(f"  Time span: {times[0]:.2f} to {times[-1]:.2f} days")
    else:
        # Fallback to manually collected trajectory data
        trajectory_data = results.get('trajectory', [])
        if not trajectory_data:
            print("  No trajectory data available for visualization")
            return

        print("  Using manually collected trajectory (phase 6 only)")
        times = np.array([p['time'] for p in trajectory_data])
        positions = np.array([p['r'] for p in trajectory_data]) * AU_TO_KM  # km
        velocities = np.array([p['v'] for p in trajectory_data]) * AU_TO_KM / DAY_TO_SEC if 'v' in trajectory_data[0] else None

        trajectory = TrajectoryData(
            times=times,
            positions=positions,
            velocities=velocities,
            name="Moon Flyby Spacecraft",
            color="cyan"
        )

    # Create body data for Earth and Moon
    earth_body = BodyData(
        name="Earth",
        positions=np.array([0, 0, 0]),  # Simplified
        radius_km=6378.137,
        color="blue"
    )

    moon_body = BodyData(
        name="Moon",
        positions=np.array([384400, 0, 0]),  # Simplified average distance
        radius_km=1737.4,
        color="gray"
    )

    # Create event markers (position is required)
    flyby_data = results.get('flyby', {})
    events = [
        EventMarker(time=times[0], position=positions[0], name="Launch", color="green"),
        EventMarker(time=times[-1], position=positions[-1], name="Earth Return", color="red"),
    ]

    if flyby_data and 'periapsis_time' in flyby_data:
        # Find closest trajectory point to flyby time
        flyby_idx = np.argmin(np.abs(times - flyby_data['periapsis_time']))
        events.append(EventMarker(
            time=flyby_data['periapsis_time'],
            position=positions[flyby_idx],
            name="Lunar Flyby",
            color="gray"
        ))

    if mode == 'interactive':
        print("  Creating interactive trajectory viewer...")
        fig = create_interactive_trajectory_viewer(
            trajectory=trajectory,
            bodies=[earth_body, moon_body],
            events=events,
            title="Circumlunar Free-Return Mission"
        )
        fig.show()

    elif mode == 'dashboard':
        print("  Creating mission dashboard...")
        fig = create_mission_dashboard(
            trajectory=trajectory,
            central_body='earth',
            show_elements=True,
        )
        fig.show()

    elif mode == 'full':
        print("  Launching interactive mission viewer...")
        print("  Open your browser to: http://127.0.0.1:8052")
        run_mission_viewer(
            trajectory=trajectory,
            bodies=[earth_body, moon_body],
            events=events,
            port=8052,
        )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Moon Flyby and Earth Reentry Mission")
    parser.add_argument('--visualize', choices=['none', 'static', 'interactive', 'dashboard', 'full'],
                       default='static', help='Visualization mode (default: static)')
    args = parser.parse_args()

    # Update config based on args
    if args.visualize == 'none':
        MoonFlybyConfig.SHOW_TRAJECTORY_3D = False
        MoonFlybyConfig.SHOW_THERMAL_PLOTS = False
    MoonFlybyConfig.INTERACTIVE_MODE = args.visualize

    results = run_mission()

    # Launch interactive visualization if requested
    if args.visualize in ['interactive', 'dashboard', 'full']:
        visualize_mission_interactive(results, mode=args.visualize)
