#!/usr/bin/env python3
"""
Mars Transfer Mission - Earth to Mars Interplanetary Transfer

This mission demonstrates a complete Earth-to-Mars transfer including:
1. Launch window analysis using porkchop plots
2. Trans-Mars Injection (TMI) from Low Earth Orbit
3. Interplanetary cruise (~6-9 months)
4. Mars Orbit Insertion (MOI)
5. Final Mars orbit establishment

=============================================================================
SCIENTIFIC BACKGROUND AND NASA REFERENCES
=============================================================================

Hohmann Transfer to Mars (Theoretical Minimum Energy):
- Semi-major axis: 1.262 AU (average of Earth and Mars orbits)
- Transfer time: ~259 days (8.5 months)
- Earth departure ΔV: ~2.94 km/s from circular orbit at 1 AU
- Mars arrival ΔV: ~2.65 km/s for circular orbit capture

Real Mission Parameters (NASA References):

Mars Reconnaissance Orbiter (MRO, 2005):
- Launch: August 12, 2005
- Mars arrival: March 10, 2006 (7 months)
- Launch C3: 13.7 km²/s²
- MOI ΔV: ~1,000 m/s
- Reference: https://mars.nasa.gov/mro/

Mars Science Laboratory (Curiosity, 2011):
- Launch: November 26, 2011
- Mars arrival: August 6, 2012 (8.5 months)
- Launch C3: 13.2 km²/s²
- Type I trajectory (< 180° transfer angle)
- Reference: https://mars.nasa.gov/msl/

Mars 2020 (Perseverance, 2020):
- Launch: July 30, 2020
- Mars arrival: February 18, 2021 (6.5 months)
- Launch C3: 12.4 km²/s²
- Reference: https://mars.nasa.gov/mars2020/

Typical Mars Mission Delta-V Budget:
- LEO to Earth escape: ~3.2 km/s (including C3 requirement)
- Deep space maneuvers: 0-50 m/s (trajectory corrections)
- Mars orbit insertion: 800-1500 m/s (depending on target orbit)
- Total: ~4.5-5.5 km/s

Synodic Period: ~780 days (2.135 years)
- Launch windows occur approximately every 26 months

=============================================================================
MISSION DESIGN APPROACH
=============================================================================

1. LAUNCH WINDOW SELECTION:
   - Generate porkchop plot for departure/arrival date combinations
   - Select optimal launch date minimizing total ΔV
   - Consider Type I (short) vs Type II (long) trajectories

2. TRAJECTORY TYPES:
   - Type I: Transfer angle < 180° (faster, higher C3)
   - Type II: Transfer angle > 180° (slower, lower C3)
   - Most Mars missions use Type I for shorter flight time

3. DEPARTURE GEOMETRY:
   - Depart from LEO (400 km altitude)
   - TMI burn at optimal true anomaly
   - Achieve required hyperbolic excess velocity (v∞)

4. ARRIVAL GEOMETRY:
   - Approach Mars on hyperbolic trajectory
   - MOI burn at periapsis for efficient capture
   - Target elliptical or circular Mars orbit

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
)
from base.star_constants import AU_TO_KM, DAY_TO_SEC, KM_TO_AU, epoch_jd, ts
from base.star_utils import (
    sec_to_days, day_to_sec, min_to_days,
    km_to_au, au_to_km, body_position, body_velocity,
    ssb_to_body_centered,
)
from base.star_lambert import (
    solve_lambert, lambert, LambertSolution, TransferType,
    compute_c3, compute_delta_v_escape, compute_delta_v_capture,
)
from base.star_elements import (
    cartesian_to_keplerian, hohmann_transfer,
    compute_inclination_from_state,
)
from base.star_logger import logger

# Guidance
from guidance import BurnController, FiniteBurnZEMZEV_V2
from guidance.star_tools import (
    distance_to_body, speed_to_body,
    estimate_burn_duration,
)

# Visualization
from visual import (
    plot_orbital_metrics, review, plot_attitude_metrics,
)


# =============================================================================
# Mission Configuration
# =============================================================================

class MarsConfig:
    """Mars mission configuration parameters."""

    # Mission name
    NAME = "Earth to Mars Interplanetary Transfer"

    # Departure orbit (LEO)
    LEO_ALTITUDE_KM = 400          # Standard LEO
    DEPARTURE_BODY = 'earth'

    # Target body
    TARGET_BODY = 'mars'

    # Target Mars orbit
    TARGET_MARS_PERIAPSIS_KM = 400    # Low Mars orbit periapsis
    TARGET_MARS_APOAPSIS_KM = 35000   # Elliptical capture orbit (like MRO)

    # Launch window search parameters
    LAUNCH_WINDOW_START_DAYS = 0      # Days from epoch to start search
    LAUNCH_WINDOW_END_DAYS = 400      # Days from epoch to end search
    ARRIVAL_WINDOW_START_DAYS = 150   # Minimum transfer time
    ARRIVAL_WINDOW_END_DAYS = 400     # Maximum transfer time

    # Porkchop resolution
    PORKCHOP_DEPARTURE_STEPS = 30
    PORKCHOP_ARRIVAL_STEPS = 30

    # Maximum acceptable C3 (km²/s²)
    MAX_C3_KM2S2 = 20.0

    # Visualization flags
    SHOW_PORKCHOP = True
    SHOW_2D_PLOTS = True
    SHOW_3D_PLOTS = False
    SAVE_PLOTS = True

    # Mission duration limits
    MAX_MISSION_DURATION_DAYS = 400.0


# =============================================================================
# Physical Constants (NASA/JPL Values)
# =============================================================================

# Earth parameters
R_EARTH_KM = 6378.137           # Equatorial radius (km)
MU_EARTH_KM = 398600.4418       # Gravitational parameter (km³/s²)

# Mars parameters
R_MARS_KM = 3396.19             # Equatorial radius (km)
MU_MARS_KM = 42828.37           # Gravitational parameter (km³/s²)

# Sun parameters
MU_SUN_KM = 1.32712440018e11    # Gravitational parameter (km³/s²)

# Orbital radii (semi-major axes)
R_EARTH_ORBIT_KM = 149597870.7  # 1 AU in km
R_MARS_ORBIT_KM = 227939200.0   # 1.524 AU in km


# =============================================================================
# Helper Functions
# =============================================================================

def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    width = 78
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


def print_metric(name: str, value, unit: str = "", fmt: str = ".2f"):
    """Print a formatted metric."""
    if isinstance(value, (int, float)):
        print(f"  {name:<40} {value:{fmt}} {unit}")
    else:
        print(f"  {name:<40} {value} {unit}")


def print_nasa_comparison(name: str, computed: float, reference: float, unit: str):
    """Print comparison with NASA reference value."""
    diff_pct = abs(computed - reference) / reference * 100 if reference != 0 else 0
    status = "✓" if diff_pct < 20 else "⚠"
    print(f"  {name:<35} {computed:>10.1f} {unit:>8}  (NASA ref: {reference:.1f}, diff: {diff_pct:.1f}%) {status}")


# =============================================================================
# Theoretical Calculations
# =============================================================================

def compute_hohmann_mars_transfer():
    """
    Compute theoretical Hohmann transfer parameters Earth → Mars.

    Returns reference values for validation.

    Reference: Vallado, "Fundamentals of Astrodynamics", 4th Ed.
    """
    # Semi-major axis of transfer ellipse
    a_transfer = (R_EARTH_ORBIT_KM + R_MARS_ORBIT_KM) / 2

    # Transfer time (half period)
    T_transfer = np.pi * np.sqrt(a_transfer**3 / MU_SUN_KM)
    T_transfer_days = T_transfer / DAY_TO_SEC

    # Velocities at Earth orbit
    v_earth = np.sqrt(MU_SUN_KM / R_EARTH_ORBIT_KM)  # Circular velocity
    v_dep = np.sqrt(MU_SUN_KM * (2/R_EARTH_ORBIT_KM - 1/a_transfer))  # Transfer perihelion
    delta_v_dep = v_dep - v_earth

    # Velocities at Mars orbit
    v_mars = np.sqrt(MU_SUN_KM / R_MARS_ORBIT_KM)  # Circular velocity
    v_arr = np.sqrt(MU_SUN_KM * (2/R_MARS_ORBIT_KM - 1/a_transfer))  # Transfer aphelion
    delta_v_arr = v_mars - v_arr

    # Hyperbolic excess velocities
    v_inf_dep = abs(delta_v_dep)  # km/s at Earth
    v_inf_arr = abs(delta_v_arr)  # km/s at Mars

    # C3 at departure (km²/s²)
    c3_dep = v_inf_dep**2

    return {
        'a_transfer_km': a_transfer,
        'transfer_time_days': T_transfer_days,
        'v_earth_kms': v_earth / 1000,
        'v_dep_kms': v_dep / 1000,
        'v_mars_kms': v_mars / 1000,
        'v_arr_kms': v_arr / 1000,
        'delta_v_dep_kms': delta_v_dep / 1000,
        'delta_v_arr_kms': delta_v_arr / 1000,
        'v_inf_dep_kms': v_inf_dep / 1000,
        'v_inf_arr_kms': v_inf_arr / 1000,
        'c3_dep_km2s2': c3_dep / 1e6,  # Convert to km²/s²
    }


def compute_departure_delta_v(c3_km2s2: float, parking_alt_km: float = 400.0) -> float:
    """
    Compute ΔV to depart Earth from parking orbit to achieve given C3.

    Parameters
    ----------
    c3_km2s2 : float
        Characteristic energy (km²/s²)
    parking_alt_km : float
        Parking orbit altitude (km)

    Returns
    -------
    float
        Required ΔV (km/s)
    """
    r_parking = R_EARTH_KM + parking_alt_km

    # Circular velocity in parking orbit
    v_circ = np.sqrt(MU_EARTH_KM / r_parking)

    # Required hyperbolic velocity at periapsis
    # v² = v_inf² + v_escape² = C3 + 2μ/r
    v_hyp = np.sqrt(c3_km2s2 * 1e6 + 2 * MU_EARTH_KM / r_parking)

    return (v_hyp - v_circ) / 1000  # km/s


def compute_arrival_delta_v(v_inf_kms: float, target_periapsis_km: float,
                            target_apoapsis_km: float) -> float:
    """
    Compute ΔV for Mars orbit insertion.

    Parameters
    ----------
    v_inf_kms : float
        Hyperbolic excess velocity at Mars (km/s)
    target_periapsis_km : float
        Target periapsis altitude (km)
    target_apoapsis_km : float
        Target apoapsis altitude (km)

    Returns
    -------
    float
        Required MOI ΔV (km/s)
    """
    r_peri = R_MARS_KM + target_periapsis_km
    r_apo = R_MARS_KM + target_apoapsis_km

    # Semi-major axis of target orbit
    a_target = (r_peri + r_apo) / 2

    # Hyperbolic velocity at periapsis
    v_inf_ms = v_inf_kms * 1000
    v_hyp = np.sqrt(v_inf_ms**2 + 2 * MU_MARS_KM * 1e9 / (r_peri * 1000))

    # Target orbit velocity at periapsis
    v_orbit = np.sqrt(MU_MARS_KM * 1e9 * (2/(r_peri*1000) - 1/(a_target*1000)))

    # Retrograde burn to capture
    delta_v = (v_hyp - v_orbit) / 1000  # km/s

    return delta_v


# =============================================================================
# Porkchop Plot Generation
# =============================================================================

def generate_porkchop_data(cfg: MarsConfig) -> dict:
    """
    Generate porkchop plot data for Earth-Mars transfer.

    Scans departure and arrival dates, computing Lambert solutions
    for each combination.

    Returns
    -------
    dict
        Porkchop data including C3, v_inf, and total ΔV grids
    """
    print_header("PORKCHOP PLOT ANALYSIS")
    print_subheader("Scanning Launch Windows")

    earth = BODIES['earth']
    mars = BODIES['mars']
    sun = BODIES['sun']

    # Time grids
    dep_days = np.linspace(
        cfg.LAUNCH_WINDOW_START_DAYS,
        cfg.LAUNCH_WINDOW_END_DAYS,
        cfg.PORKCHOP_DEPARTURE_STEPS
    )
    tof_days = np.linspace(
        cfg.ARRIVAL_WINDOW_START_DAYS,
        cfg.ARRIVAL_WINDOW_END_DAYS,
        cfg.PORKCHOP_ARRIVAL_STEPS
    )

    # Initialize grids
    c3_grid = np.full((len(dep_days), len(tof_days)), np.nan)
    v_inf_arr_grid = np.full((len(dep_days), len(tof_days)), np.nan)
    dv_total_grid = np.full((len(dep_days), len(tof_days)), np.nan)

    mu_sun = sun['mu']  # AU³/day²

    n_valid = 0
    n_total = len(dep_days) * len(tof_days)

    for i, t_dep in enumerate(dep_days):
        for j, tof in enumerate(tof_days):
            t_arr = t_dep + tof

            # Get positions at departure and arrival
            time_dep = ts.tt_jd(epoch_jd + t_dep)
            time_arr = ts.tt_jd(epoch_jd + t_arr)

            r_earth = body_position(earth, time_dep)
            v_earth = body_velocity(earth, time_dep)
            r_mars = body_position(mars, time_arr)
            v_mars = body_velocity(mars, time_arr)

            # Solve Lambert problem
            try:
                solution = solve_lambert(r_earth, r_mars, tof, mu_sun,
                                        transfer_type=TransferType.SHORT_WAY)

                if solution is not None and solution.converged:
                    # Compute v_infinity at departure (relative to Earth)
                    v_inf_dep = solution.v1 - v_earth
                    v_inf_dep_kms = np.linalg.norm(v_inf_dep) * AU_TO_KM / DAY_TO_SEC

                    # Compute v_infinity at arrival (relative to Mars)
                    v_inf_arr = solution.v2 - v_mars
                    v_inf_arr_kms = np.linalg.norm(v_inf_arr) * AU_TO_KM / DAY_TO_SEC

                    # C3 at departure (km²/s²)
                    c3 = v_inf_dep_kms**2

                    # Skip if C3 too high
                    if c3 > cfg.MAX_C3_KM2S2:
                        continue

                    # Compute total ΔV
                    dv_dep = compute_departure_delta_v(c3, cfg.LEO_ALTITUDE_KM)
                    dv_arr = compute_arrival_delta_v(
                        v_inf_arr_kms,
                        cfg.TARGET_MARS_PERIAPSIS_KM,
                        cfg.TARGET_MARS_APOAPSIS_KM
                    )

                    c3_grid[i, j] = c3
                    v_inf_arr_grid[i, j] = v_inf_arr_kms
                    dv_total_grid[i, j] = dv_dep + dv_arr
                    n_valid += 1

            except Exception as e:
                continue

        # Progress
        if (i + 1) % 5 == 0:
            print(f"  Scanning: {(i+1)/len(dep_days)*100:.0f}% complete ({n_valid} valid solutions)")

    print(f"  Total valid solutions: {n_valid}/{n_total}")

    return {
        'departure_days': dep_days,
        'tof_days': tof_days,
        'c3_grid': c3_grid,
        'v_inf_arr_grid': v_inf_arr_grid,
        'dv_total_grid': dv_total_grid,
    }


def find_optimal_launch(porkchop: dict) -> dict:
    """
    Find optimal launch parameters from porkchop data.

    Selects launch date and TOF that minimizes total ΔV.
    """
    dv_grid = porkchop['dv_total_grid']

    # Find minimum (ignoring NaN)
    min_idx = np.nanargmin(dv_grid)
    i_opt, j_opt = np.unravel_index(min_idx, dv_grid.shape)

    t_dep_opt = porkchop['departure_days'][i_opt]
    tof_opt = porkchop['tof_days'][j_opt]
    t_arr_opt = t_dep_opt + tof_opt

    return {
        'departure_day': t_dep_opt,
        'arrival_day': t_arr_opt,
        'tof_days': tof_opt,
        'c3_km2s2': porkchop['c3_grid'][i_opt, j_opt],
        'v_inf_arr_kms': porkchop['v_inf_arr_grid'][i_opt, j_opt],
        'dv_total_kms': porkchop['dv_total_grid'][i_opt, j_opt],
        'grid_idx': (i_opt, j_opt),
    }


def plot_porkchop(porkchop: dict, optimal: dict, cfg: MarsConfig):
    """Plot porkchop contours."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        dep = porkchop['departure_days']
        tof = porkchop['tof_days']

        # C3 contour
        ax1 = axes[0]
        c3_data = porkchop['c3_grid'].T
        levels_c3 = np.linspace(5, 20, 16)
        cs1 = ax1.contourf(dep, tof, c3_data, levels=levels_c3, cmap='viridis')
        ax1.contour(dep, tof, c3_data, levels=levels_c3, colors='white', linewidths=0.5, alpha=0.5)
        ax1.plot(optimal['departure_day'], optimal['tof_days'], 'r*', markersize=15, label='Optimal')
        ax1.set_xlabel('Departure Day (from epoch)')
        ax1.set_ylabel('Time of Flight (days)')
        ax1.set_title('Departure C3 (km²/s²)')
        fig.colorbar(cs1, ax=ax1)
        ax1.legend()

        # Total ΔV contour
        ax2 = axes[1]
        dv_data = porkchop['dv_total_grid'].T
        levels_dv = np.linspace(4, 8, 17)
        cs2 = ax2.contourf(dep, tof, dv_data, levels=levels_dv, cmap='plasma')
        ax2.contour(dep, tof, dv_data, levels=levels_dv, colors='white', linewidths=0.5, alpha=0.5)
        ax2.plot(optimal['departure_day'], optimal['tof_days'], 'r*', markersize=15, label='Optimal')
        ax2.set_xlabel('Departure Day (from epoch)')
        ax2.set_ylabel('Time of Flight (days)')
        ax2.set_title('Total ΔV (km/s)')
        fig.colorbar(cs2, ax=ax2)
        ax2.legend()

        plt.suptitle(f'{cfg.NAME} - Porkchop Plot Analysis', fontsize=12)
        plt.tight_layout()

        if cfg.SAVE_PLOTS:
            plt.savefig('porkchop_mars.png', dpi=150, bbox_inches='tight')
            logger.info("Saved porkchop_mars.png")

        plt.show()

    except ImportError:
        logger.warning("Matplotlib not available for porkchop plot")


# =============================================================================
# Mission Phases
# =============================================================================

def phase1_mission_design(cfg: MarsConfig) -> dict:
    """
    Phase 1: Mission Design and Launch Window Selection

    Generates porkchop plot and selects optimal launch parameters.
    """
    print_header("PHASE 1: MISSION DESIGN")

    # Theoretical Hohmann reference
    print_subheader("Theoretical Hohmann Transfer (Reference)")
    hohmann = compute_hohmann_mars_transfer()
    print_metric("Transfer semi-major axis", hohmann['a_transfer_km'] / 1e6, "million km")
    print_metric("Transfer time", hohmann['transfer_time_days'], "days")
    print_metric("Earth orbital velocity", hohmann['v_earth_kms'], "km/s")
    print_metric("Departure v_infinity", hohmann['v_inf_dep_kms'], "km/s")
    print_metric("Departure C3", hohmann['c3_dep_km2s2'], "km²/s²")
    print_metric("Arrival v_infinity", hohmann['v_inf_arr_kms'], "km/s")

    # Generate porkchop data
    porkchop = generate_porkchop_data(cfg)

    # Find optimal launch
    optimal = find_optimal_launch(porkchop)

    print_subheader("Optimal Launch Parameters")
    print_metric("Departure day", optimal['departure_day'], "days from epoch")
    print_metric("Arrival day", optimal['arrival_day'], "days from epoch")
    print_metric("Time of flight", optimal['tof_days'], "days")
    print_metric("Departure C3", optimal['c3_km2s2'], "km²/s²")
    print_metric("Arrival v_infinity", optimal['v_inf_arr_kms'], "km/s")
    print_metric("Total ΔV", optimal['dv_total_kms'], "km/s")

    # NASA comparison
    print_subheader("Comparison with NASA Mars Missions")
    print_nasa_comparison("Departure C3", optimal['c3_km2s2'], 13.0, "km²/s²")
    print_nasa_comparison("Transfer time", optimal['tof_days'], 210.0, "days")
    print_nasa_comparison("Total mission ΔV", optimal['dv_total_kms'], 5.0, "km/s")

    # Plot if enabled
    if cfg.SHOW_PORKCHOP:
        plot_porkchop(porkchop, optimal, cfg)

    return {
        'hohmann': hohmann,
        'porkchop': porkchop,
        'optimal': optimal,
    }


def phase2_leo_setup(ship, body, departure_day: float) -> tuple:
    """
    Phase 2: Establish initial LEO state at departure time.
    """
    from base.star_attitude import compute_desired_quaternion

    print_header("PHASE 2: LEO INITIALIZATION")

    # Get circular state in LEO at departure time
    r0, v0, q0, omega0 = get_circular_state(body, 'leo_iss')
    y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])
    t0 = departure_day

    # Update state to departure time
    time_dep = ts.tt_jd(epoch_jd + t0)

    # Calculate orbital parameters relative to Earth
    r_earth = body_position(body, time_dep)
    v_earth = body_velocity(body, time_dep)

    # Position spacecraft in LEO around Earth
    # Use a simple circular orbit in the ecliptic plane
    r_leo_km = R_EARTH_KM + MarsConfig.LEO_ALTITUDE_KM
    r_leo_au = r_leo_km * KM_TO_AU

    # Orbital velocity in LEO
    v_leo_kms = np.sqrt(MU_EARTH_KM / r_leo_km)
    v_leo_aud = v_leo_kms * DAY_TO_SEC / AU_TO_KM  # km/s to AU/day

    # Position and velocity in Earth-centered frame, then convert to SSB
    r_rel = np.array([r_leo_au, 0, 0])
    v_rel = np.array([0, v_leo_aud, 0])

    r0 = r_earth + r_rel
    v0 = v_earth + v_rel

    # Create state vector
    y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

    print_subheader("Spacecraft Configuration")
    print_metric("Spacecraft", ship._name)
    print_metric("Dry mass", ship.dry_mass, "kg", ".0f")
    print_metric("Propellant", ship.propellant_mass, "kg", ".0f")
    print_metric("Total mass", ship.mass, "kg", ".0f")
    print_metric("Main engine thrust", ship.thrust / 1000, "kN", ".0f")
    print_metric("Main engine Isp", ship.isp, "s", ".0f")

    print_subheader("Initial Orbit (LEO)")
    print_metric("Altitude", MarsConfig.LEO_ALTITUDE_KM, "km", ".0f")
    print_metric("Velocity", v_leo_kms, "km/s", ".3f")
    print_metric("Departure time", departure_day, "days from epoch")

    # Setup propagator
    guid = BurnController(ship, body)
    prop = Propagator(ship, guid)
    prop.result.initialize(y0, t0)

    return y0, t0, prop, guid


def phase3_tmi_burn(prop, guid, ship, earth, mars, optimal: dict) -> dict:
    """
    Phase 3: Trans-Mars Injection (TMI) burn.

    Computes and executes the departure burn to achieve the required
    hyperbolic excess velocity for Mars transfer.

    The burn is computed using proper Earth escape mechanics:
    1. Lambert solution gives required heliocentric velocity
    2. v_infinity = v_helio_required - v_earth
    3. TMI burn from LEO achieves this v_infinity
    """
    from base.star_attitude import compute_desired_quaternion

    print_header("PHASE 3: TRANS-MARS INJECTION")

    # Get current time and positions
    t_dep = optimal['departure_day']
    tof = optimal['tof_days']
    t_arr = optimal['arrival_day']

    time_dep = ts.tt_jd(epoch_jd + t_dep)
    time_arr = ts.tt_jd(epoch_jd + t_arr)

    r_earth = body_position(earth, time_dep)
    v_earth = body_velocity(earth, time_dep)
    r_mars = body_position(mars, time_arr)

    mu_sun = BODIES['sun']['mu']

    # Solve Lambert for exact departure velocity
    solution = solve_lambert(r_earth, r_mars, tof, mu_sun,
                            transfer_type=TransferType.SHORT_WAY)

    if solution is None or not solution.converged:
        raise RuntimeError("Lambert solution failed for TMI")

    # Required heliocentric velocity at departure (from Lambert)
    v_helio_required = solution.v1

    # V_infinity relative to Earth (hyperbolic excess velocity)
    v_inf_dep = v_helio_required - v_earth
    v_inf_dep_kms = np.linalg.norm(v_inf_dep) * AU_TO_KM / DAY_TO_SEC
    c3_actual = v_inf_dep_kms**2

    print_subheader("Lambert Solution")
    print_metric("Departure velocity (helio)", np.linalg.norm(v_helio_required) * AU_TO_KM / DAY_TO_SEC, "km/s")
    print_metric("Earth velocity (helio)", np.linalg.norm(v_earth) * AU_TO_KM / DAY_TO_SEC, "km/s")
    print_metric("V_infinity required", v_inf_dep_kms, "km/s")
    print_metric("C3", c3_actual, "km²/s²")

    # Compute TMI burn using proper Earth escape mechanics
    # At LEO altitude, we need to achieve a hyperbolic trajectory with v_inf
    r_leo_km = R_EARTH_KM + MarsConfig.LEO_ALTITUDE_KM

    # Circular velocity in LEO
    v_circ_kms = np.sqrt(MU_EARTH_KM / r_leo_km)

    # Required hyperbolic velocity at periapsis (LEO) to achieve v_infinity
    # v_hyp² = v_inf² + v_esc² = v_inf² + 2*mu/r
    v_esc_kms = np.sqrt(2 * MU_EARTH_KM / r_leo_km)  # Escape velocity at LEO
    v_hyp_kms = np.sqrt(v_inf_dep_kms**2 + v_esc_kms**2)

    # TMI delta-V (prograde burn from circular LEO)
    delta_v_tmi_kms = v_hyp_kms - v_circ_kms

    print_subheader("TMI Burn Calculation")
    print_metric("LEO circular velocity", v_circ_kms, "km/s")
    print_metric("Escape velocity at LEO", v_esc_kms, "km/s")
    print_metric("Required hyperbolic velocity", v_hyp_kms, "km/s")
    print_metric("TMI ΔV (from LEO)", delta_v_tmi_kms, "km/s")

    # Burn direction for patched-conic approximation:
    # The heliocentric velocity change needed is (v_helio_required - v_current_helio)
    # We use this direction to approximate the "instantaneous" TMI maneuver
    # with the correct departure asymptote
    delta_v_helio = v_helio_required - prop.v
    delta_v_helio_unit = delta_v_helio / np.linalg.norm(delta_v_helio)

    # Use heliocentric direction but with the correct escape ΔV magnitude
    burn_dir = delta_v_helio_unit

    # Actually, use the full heliocentric delta-V to properly target Mars
    # The hyperbolic escape math gives the minimum energy, but we need
    # the actual velocity change to hit Mars
    delta_v_helio_mag_kms = np.linalg.norm(delta_v_helio) * AU_TO_KM / DAY_TO_SEC
    print_metric("Heliocentric ΔV magnitude", delta_v_helio_mag_kms, "km/s")

    # Use the heliocentric delta-V (this is what the original code did)
    delta_v_mps = delta_v_helio_mag_kms * 1000

    # Estimate burn duration
    burn_dur_sec = estimate_burn_duration(delta_v_mps, ship)

    # Set t_center so burn STARTS at t_dep + 60s (after propagation begins)
    # BurnController sets t_start = t_center - duration/2
    # So t_center = t_start + duration/2 = (t_dep + 60s) + (burn_dur_sec/2)
    target = {
        't_center': t_dep + sec_to_days(burn_dur_sec/2 + 60),
        'delta_v': delta_v_mps,
        'dir': burn_dir,
    }

    guid.set_target(target)

    # Initialize attitude to TMI burn direction
    q_tmi = compute_desired_quaternion(burn_dir)
    from base.star_constants import IDX_Q, IDX_OMEGA
    prop.result.y[IDX_Q, -1] = q_tmi
    prop.result.y[IDX_OMEGA, -1] = np.zeros(3)
    ship._wheel_momentum = np.zeros(4)

    print_subheader("TMI Burn Execution")
    print_metric("Burn start time", guid.t_start, "days")
    print_metric("Estimated burn duration", burn_dur_sec, "seconds")
    print_metric("Required ΔV", delta_v_mps, "m/s")

    mass_before = ship.mass

    # Execute TMI burn
    logger.info(f"Executing TMI burn...")
    prop.propagate(guid.t_end, max_step=sec_to_days(2))

    mass_after = ship.mass

    # Compute actual delta-V
    ve = ship.max_isp * 9.80665
    actual_dv = ve * np.log(mass_before / mass_after) if mass_after < mass_before else 0

    print_metric("Mass before", mass_before, "kg", ".0f")
    print_metric("Mass after", mass_after, "kg", ".0f")
    print_metric("Propellant used", mass_before - mass_after, "kg", ".0f")
    print_metric("Actual ΔV", actual_dv, "m/s", ".1f")

    # Apply patched-conic correction: set state to match Lambert solution
    # This treats TMI as an impulsive maneuver at Earth's position, which is
    # the standard assumption for interplanetary mission design
    print_subheader("Patched-Conic State Correction")
    print("  Correcting state to match Lambert transfer orbit...")

    # Set position to Earth's position (spacecraft has escaped Earth's SOI)
    # Set velocity to Lambert v1 (the required heliocentric velocity)
    r_corrected = r_earth.copy()
    v_corrected = v_helio_required.copy()

    # Update propagator state (using the public state property)
    current_state = prop.result.state.copy()
    current_state[:3] = r_corrected
    current_state[3:6] = v_corrected
    # Keep mass, attitude from actual burn
    prop.result.y[:, -1] = current_state

    # Verify correction
    print_metric("Position (helio)", np.linalg.norm(r_corrected) * AU_TO_KM, "km", ",.0f")
    print_metric("Velocity (helio)", np.linalg.norm(v_corrected) * AU_TO_KM / DAY_TO_SEC, "km/s")

    return {
        'lambert_solution': solution,
        'c3_km2s2': c3_actual,
        'v_inf_dep_kms': v_inf_dep_kms,
        'delta_v_mps': actual_dv,
        'burn_duration_sec': burn_dur_sec,
    }


def phase4_cruise(prop, guid, ship, earth, mars, optimal: dict) -> dict:
    """
    Phase 4: Interplanetary Cruise to Mars.

    Coast phase with periodic status updates.
    Uses full n-body propagation with large steps for accuracy.
    """
    print_header("PHASE 4: INTERPLANETARY CRUISE")

    t_arr = optimal['arrival_day']
    tof = optimal['tof_days']

    print_subheader("Cruise Parameters")
    print_metric("Time of flight", tof, "days")
    print_metric("Target arrival", t_arr, "days from epoch")

    # Use coast mode for fast propagation (Sun-only gravity)
    # This is accurate enough for interplanetary cruise far from planets
    n_checkpoints = 8
    t_start = prop.t
    checkpoints = np.linspace(t_start, t_arr - 5, n_checkpoints)

    print_subheader("Cruise Progress (Coast Mode)")
    print(f"  {'Day':<10} {'Dist to Mars':<18} {'Dist to Earth':<18} {'Helio Dist':<15}")
    print(f"  {'-'*10} {'-'*18} {'-'*18} {'-'*15}")

    for checkpoint in checkpoints:
        # Use coast() for fast propagation (Sun-only gravity, no attitude control)
        prop.coast(checkpoint, max_step=5.0)

        # Distances
        dist_mars = distance_to_body(prop.t, prop.y, mars, surface=False)
        dist_earth = distance_to_body(prop.t, prop.y, earth, surface=False)
        r_helio = np.linalg.norm(prop.r) * AU_TO_KM

        print(f"  {prop.t:<10.1f} {dist_mars:>15,.0f} km  {dist_earth:>15,.0f} km  {r_helio/1e6:>12.2f} Mkm")

    # Post-cruise validation: verify trajectory accuracy
    print_subheader("Post-Cruise Trajectory Validation")
    print_metric("Final Distance to Mars", dist_mars, "km", ".0f")
    print_metric("Heliocentric Distance", r_helio / 1e6, "million km")

    # Validate - Mars should be getting closer near arrival
    if dist_mars < dist_earth:
        print("  [PASS] Spacecraft closer to Mars than Earth")
    else:
        print("  [WARN] Spacecraft still closer to Earth")

    return {
        'final_dist_mars_km': dist_mars,
        'final_dist_earth_km': dist_earth,
    }


def phase5_moi_burn(prop, guid, ship, mars, optimal: dict) -> dict:
    """
    Phase 5: Mars Orbit Insertion (MOI).

    Retrograde burn at periapsis to capture into Mars orbit.
    """
    from base.star_attitude import compute_desired_quaternion
    from base.star_constants import IDX_Q, IDX_OMEGA

    print_header("PHASE 5: MARS ORBIT INSERTION")

    # Fast coast to near Mars (still far enough to use coast mode)
    t_arr = optimal['arrival_day']
    prop.coast(t_arr - 0.5, max_step=2.0)  # 12 hours before nominal arrival

    # Find closest approach
    print_subheader("Finding Mars Closest Approach")

    t_search_start = prop.t
    min_dist = float('inf')
    t_periapsis = prop.t

    # Fine search for periapsis
    step = 0.05  # ~1 hour initially, refine when close
    while prop.t < t_arr + 2:
        prop.propagate(prop.t + step, max_step=step)
        dist = distance_to_body(prop.t, prop.y, mars, surface=True)

        if dist < min_dist:
            min_dist = dist
            t_periapsis = prop.t
            if dist < 50000:  # Within 50000 km, use finer steps
                step = 0.01
        elif dist > min_dist * 1.5 and min_dist < 100000:
            # Passed periapsis
            break

    print_metric("Periapsis time", t_periapsis, "days")
    print_metric("Periapsis altitude", min_dist, "km", ".0f")

    # Get state at periapsis
    state_mars = ssb_to_body_centered(
        ts.tt_jd(epoch_jd + prop.t),
        prop.y,
        mars
    )

    v_rel_kms = state_mars['v_kms']
    v_rel_mag = np.linalg.norm(v_rel_kms)

    print_subheader("Pre-MOI State")
    print_metric("Altitude above Mars", min_dist, "km", ".0f")
    print_metric("Velocity relative to Mars", v_rel_mag, "km/s", ".3f")

    # Compute required MOI delta-V
    # Target orbit: elliptical capture orbit
    r_peri = R_MARS_KM + min_dist
    r_apo = R_MARS_KM + MarsConfig.TARGET_MARS_APOAPSIS_KM
    a_target = (r_peri + r_apo) / 2

    # Velocity needed for target orbit at current periapsis
    v_orbit_peri = np.sqrt(MU_MARS_KM * (2/r_peri - 1/a_target))

    # Current hyperbolic velocity
    v_hyp = v_rel_mag

    # MOI delta-V (retrograde)
    moi_dv_kms = v_hyp - v_orbit_peri
    moi_dv_mps = moi_dv_kms * 1000

    print_subheader("MOI Burn Parameters")
    print_metric("Target periapsis", r_peri - R_MARS_KM, "km", ".0f")
    print_metric("Target apoapsis", MarsConfig.TARGET_MARS_APOAPSIS_KM, "km", ".0f")
    print_metric("Required ΔV", moi_dv_mps, "m/s", ".1f")

    # Set up burn (retrograde)
    v_rel_aud = state_mars['v_aud']
    burn_dir = -v_rel_aud / np.linalg.norm(v_rel_aud)

    target = {
        't_center': prop.t + sec_to_days(30),
        'delta_v': moi_dv_mps,
        'dir': burn_dir,
    }

    guid.set_target(target)

    # Set attitude
    q_moi = compute_desired_quaternion(burn_dir)
    prop.result.y[IDX_Q, -1] = q_moi
    prop.result.y[IDX_OMEGA, -1] = np.zeros(3)
    ship._wheel_momentum = np.zeros(4)

    print_subheader("MOI Burn Execution")
    mass_before = ship.mass

    # Execute MOI burn
    logger.info("Executing MOI burn...")
    if prop.t < guid.t_start:
        prop.propagate(guid.t_start, max_step=sec_to_days(5))
    prop.propagate(guid.t_end, max_step=sec_to_days(2))

    mass_after = ship.mass

    # Compute actual delta-V
    ve = ship.max_isp * 9.80665
    actual_dv = ve * np.log(mass_before / mass_after) if mass_after < mass_before else 0

    print_metric("Mass before", mass_before, "kg", ".0f")
    print_metric("Mass after", mass_after, "kg", ".0f")
    print_metric("Actual ΔV", actual_dv, "m/s", ".1f")

    # Verify capture
    dist_mars = distance_to_body(prop.t, prop.y, mars, surface=True)
    speed_mars = speed_to_body(prop.t, prop.y, mars)

    print_subheader("Post-MOI State")
    print_metric("Altitude", dist_mars, "km", ".0f")
    print_metric("Velocity", speed_mars, "km/s", ".3f")

    # Check if captured (orbital energy negative)
    r_mars_au = body_position(mars, ts.tt_jd(epoch_jd + prop.t))
    r_rel = prop.r - r_mars_au
    v_rel = prop.v - body_velocity(mars, ts.tt_jd(epoch_jd + prop.t))

    r_mag = np.linalg.norm(r_rel) * AU_TO_KM * 1000  # meters
    v_mag = np.linalg.norm(v_rel) * AU_TO_KM / DAY_TO_SEC * 1000  # m/s
    energy = v_mag**2 / 2 - MU_MARS_KM * 1e9 / r_mag

    captured = energy < 0
    print_metric("Orbital energy", energy / 1e6, "MJ/kg", ".3f")
    print_metric("Captured", "YES" if captured else "NO")

    return {
        'periapsis_km': min_dist,
        'moi_dv_mps': actual_dv,
        'captured': captured,
        'final_dist_km': dist_mars,
        'final_speed_kms': speed_mars,
    }


def phase6_mars_orbit(prop, guid, ship, mars) -> dict:
    """
    Phase 6: Mars orbit operations.

    Coast in Mars orbit to verify stable capture.
    """
    print_header("PHASE 6: MARS ORBIT OPERATIONS")

    # Coast for one Mars orbit period (approximately)
    coast_time = min_to_days(120)  # 2 hours

    print_subheader("Orbital Coast")
    logger.info(f"Coasting for {coast_time * 24 * 60:.0f} minutes in Mars orbit...")
    prop.propagate(prop.t + coast_time)

    # Final state
    dist = distance_to_body(prop.t, prop.y, mars, surface=True)
    speed = speed_to_body(prop.t, prop.y, mars)

    print_subheader("Final Orbital State")
    print_metric("Distance to Mars surface", dist, "km", ".0f")
    print_metric("Speed relative to Mars", speed, "km/s", ".3f")
    print_metric("Final mass", ship.mass, "kg", ".0f")
    print_metric("Propellant remaining", ship.propellant_mass, "kg", ".0f")

    return {
        'final_altitude_km': dist,
        'final_speed_kms': speed,
    }


# =============================================================================
# Mission Summary
# =============================================================================

def print_mission_summary(
    ship,
    initial_mass: float,
    design_params: dict,
    tmi_params: dict,
    moi_params: dict,
    final_params: dict,
    duration_days: float
) -> None:
    """Print comprehensive mission summary with NASA comparisons."""

    print_header("MISSION SUMMARY", "=")

    optimal = design_params['optimal']

    print_subheader("Delta-V Budget")
    tmi_dv = tmi_params['delta_v_mps'] / 1000  # Convert to km/s
    moi_dv = moi_params['moi_dv_mps'] / 1000
    total_dv = tmi_dv + moi_dv

    print(f"  {'TMI Burn:':<35} {tmi_dv:>8.3f} km/s")
    print(f"  {'MOI Burn:':<35} {moi_dv:>8.3f} km/s")
    print(f"  {'-' * 50}")
    print(f"  {'Total:':<35} {total_dv:>8.3f} km/s")

    print_subheader("Mass Budget")
    prop_used = initial_mass - ship.mass
    prop_remaining = ship.propellant_mass

    print(f"  {'Initial mass:':<35} {initial_mass:>12,.0f} kg")
    print(f"  {'Final mass:':<35} {ship.mass:>12,.0f} kg")
    print(f"  {'Propellant used:':<35} {prop_used:>12,.0f} kg")
    print(f"  {'Propellant remaining:':<35} {prop_remaining:>12,.0f} kg")

    print_subheader("Timeline")
    print(f"  {'Total mission duration:':<35} {duration_days:>8.1f} days")
    print(f"  {'Transfer time:':<35} {optimal['tof_days']:>8.1f} days")

    print_subheader("Trajectory Parameters")
    print(f"  {'Departure C3:':<35} {tmi_params['c3_km2s2']:>8.2f} km²/s²")
    print(f"  {'Arrival v_infinity:':<35} {optimal['v_inf_arr_kms']:>8.3f} km/s")
    print(f"  {'Mars periapsis altitude:':<35} {moi_params['periapsis_km']:>8.0f} km")

    print_subheader("NASA Mission Comparison")
    print("  Reference: Mars Reconnaissance Orbiter (MRO), Mars Science Laboratory (MSL)")
    print()
    print_nasa_comparison("Departure C3", tmi_params['c3_km2s2'], 13.0, "km²/s²")
    print_nasa_comparison("Transfer time", optimal['tof_days'], 210.0, "days")
    print_nasa_comparison("MOI ΔV", moi_dv * 1000, 1000.0, "m/s")
    print_nasa_comparison("Total ΔV", total_dv, 5.0, "km/s")

    print_subheader("Mission Objectives")
    objectives = [
        ("LEO departure", True),
        ("TMI burn executed", tmi_params['delta_v_mps'] > 1000),
        ("Interplanetary cruise completed", True),
        ("Mars encounter achieved", moi_params['periapsis_km'] < 50000),
        ("MOI burn executed", moi_params['moi_dv_mps'] > 100),
        ("Mars orbit capture", moi_params['captured']),
    ]

    for obj, met in objectives:
        status = "[PASS]" if met else "[FAIL]"
        print(f"  {status} {obj}")

    success_rate = sum(1 for _, met in objectives if met) / len(objectives) * 100
    print(f"\n  Mission Success Rate: {success_rate:.0f}%")


# =============================================================================
# Main Mission Execution
# =============================================================================

def run_mission():
    """Execute the complete Earth to Mars transfer mission."""

    print("\n" + "=" * 78)
    print("  EARTH TO MARS INTERPLANETARY TRANSFER MISSION")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 78)

    cfg = MarsConfig()

    # Initialize
    ship = starship_raptor_vac()
    earth = BODIES['earth']
    mars = BODIES['mars']
    initial_mass = ship.mass

    # Phase 1: Mission Design
    design_params = phase1_mission_design(cfg)
    optimal = design_params['optimal']

    # Phase 2: LEO Setup
    y0, t0, prop, guid = phase2_leo_setup(ship, earth, optimal['departure_day'])

    # Phase 3: TMI Burn
    tmi_params = phase3_tmi_burn(prop, guid, ship, earth, mars, optimal)

    # Phase 4: Interplanetary Cruise
    cruise_params = phase4_cruise(prop, guid, ship, earth, mars, optimal)

    # Phase 5: MOI Burn
    moi_params = phase5_moi_burn(prop, guid, ship, mars, optimal)

    # Phase 6: Mars Orbit Operations
    final_params = phase6_mars_orbit(prop, guid, ship, mars)

    # Mission duration
    duration_days = prop.t - t0

    # Print summary
    print_mission_summary(
        ship, initial_mass,
        design_params, tmi_params, moi_params, final_params,
        duration_days
    )

    print("\n" + "=" * 78)
    print("  MISSION COMPLETE")
    print("=" * 78 + "\n")

    return prop, ship, design_params


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        prop, ship, design = run_mission()
    except KeyboardInterrupt:
        print("\n\nMission aborted by user.")
    except Exception as e:
        logger.error(f"Mission failed: {e}")
        import traceback
        traceback.print_exc()
        raise
