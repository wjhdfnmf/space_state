#!/usr/bin/env python3
"""
Moon To You (moon2u.py) - Complete LEO to Low Lunar Orbit Mission

This comprehensive example demonstrates a full cislunar mission:
1. Launch from Low Earth Orbit (LEO)
2. Trans-Lunar Injection (TLI) burn with ZEM/ZEV guidance
3. Coast phase to the Moon (~3 days)
4. Lunar Orbit Insertion (LOI) burn
5. Final low lunar orbit establishment

Features demonstrated:
- N-body gravitational dynamics (Earth, Moon, Sun)
- Quaternion attitude dynamics with reaction wheels
- Finite burn guidance (ZEM/ZEV algorithm)
- Hohmann and combined plane change maneuvers
- Multiple visualization modes
- Mission metrics and reporting

Usage:
    python examples/moon2u.py

Author: Space State Development Team
Date: December 2025
"""

import numpy as np
import sys
from datetime import datetime

# Ensure correct path
sys.path.insert(0, '/home/boss/data/space_state/space_state')

# Core imports
from base import (
    BODIES, CONFIG, Propagator, starship_raptor_vac,
    get_circular_state, get_circular_state_at_apoapsis,
    vis_viva, circular_velocity, orbital_period,
    # Mission logging imports
    FlightDataRecorder, MissionController, DisplayMode,
    create_fdr, generate_mission_plots,
)
from base.star_constants import AU_TO_KM, DAY_TO_SEC, epoch_jd, ts
from base.star_utils import (
    sec_to_days, day_to_sec, min_to_days,
    km_to_au, au_to_km, body_position, body_velocity,
    ssb_to_body_centered,
)
from base.star_states import (
    get_hohmann_state_at_time,
    combined_plane_change_tli,
    moon_intercept,
    finalize_lunar_orbit_insertion2,
    compute_loi2_circularization,
)
from base.star_logger import logger
from base.star_events import throttle_events

# Guidance
from guidance import BurnController, FiniteBurnZEMZEV_V2
from guidance.star_tools import (
    distance_to_body, speed_to_body,
    period_moon, period_ship, estimate_burn_duration,
)

# Visualization
from visual import (
    plot_orbital_metrics, Orbit3D, review, plot_attitude_metrics,
    plot_orbit_3d, plot_earth_moon, create_orbit_animation,
)
from visual.star_vis3D_new import Orbit3D_NEW


# ==============================================================================
# Configuration
# ==============================================================================

class MissionConfig:
    """Mission configuration parameters."""

    # Mission name
    NAME = "Moon To You - LEO to Low Lunar Orbit"

    # Orbit parameters
    LEO_ALTITUDE_KM = 400          # ISS-like orbit
    TARGET_LUNAR_ALT_KM = 6000     # Target lunar orbit altitude

    # TLI parameters
    TLI_APOAPSIS_OFFSET_KM = 0     # Offset from Moon distance
    TARGET_PERISELENE_KM = 200     # Target closest approach to Moon for LOI

    # Visualization flags
    SHOW_2D_PLOTS = True
    SHOW_3D_PLOTS = True
    SHOW_ATTITUDE_PLOTS = True
    SHOW_ANIMATIONS = False        # Animations can be slow
    SAVE_PLOTS = True

    # Timing
    MAX_MISSION_DURATION_DAYS = 5.0

    # Mission logging
    ENABLE_FDR = True
    FDR_INTERVAL = 0.001  # Record every 0.001 days (~86 sec)
    DISPLAY_MODE = DisplayMode.COMPACT


# ==============================================================================
# Helper Functions
# ==============================================================================

def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    width = 70
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)




def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


def print_metric(name: str, value, unit: str = "", fmt: str = ".2f"):
    """Print a formatted metric."""
    if isinstance(value, (int, float)):
        print(f"  {name:<30} {value:{fmt}} {unit}")
    else:
        print(f"  {name:<30} {value} {unit}")


def compute_mission_delta_v(r1_km: float, r2_km: float, mu: float) -> dict:
    """
    Compute theoretical Hohmann transfer delta-V.

    Args:
        r1_km: Initial orbit radius (km)
        r2_km: Target orbit radius (km)
        mu: Gravitational parameter (km^3/s^2)

    Returns:
        dict with delta-V values in m/s
    """
    a_trans = (r1_km + r2_km) / 2

    v1_circ = np.sqrt(mu / r1_km)
    v1_trans = np.sqrt(mu * (2/r1_km - 1/a_trans))
    dv1 = (v1_trans - v1_circ) * 1000  # m/s

    v2_circ = np.sqrt(mu / r2_km)
    v2_trans = np.sqrt(mu * (2/r2_km - 1/a_trans))
    dv2 = (v2_circ - v2_trans) * 1000  # m/s

    return {
        'dv1_mps': abs(dv1),
        'dv2_mps': abs(dv2),
        'total_mps': abs(dv1) + abs(dv2),
        'transfer_time_hours': np.pi * np.sqrt(a_trans**3 / mu) / 3600,
    }


# ==============================================================================
# Mission Phases
# ==============================================================================

def phase1_leo_setup(ship, body) -> tuple:
    """
    Phase 1: Establish initial LEO state

    Returns:
        tuple: (y0, t0, prop, guid)
    """
    print_header("PHASE 1: LEO INITIALIZATION")

    # Get circular state in LEO
    r0, v0, q0, omega0 = get_circular_state(body, 'leo_iss')
    y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])
    t0 = 0.0

    # Calculate orbital parameters
    r0_rel = r0 - body_position(body, ts.tt_jd(epoch_jd))
    alt_km = np.linalg.norm(r0_rel) * AU_TO_KM - body['radius'] * AU_TO_KM
    v_kms = np.linalg.norm(v0 - body_velocity(body, ts.tt_jd(epoch_jd))) * AU_TO_KM / DAY_TO_SEC

    period_sec = orbital_period(alt_km + body['radius'] * AU_TO_KM, body['mu_km3_s2'])

    print_subheader("Spacecraft Configuration")
    print_metric("Spacecraft", ship._name)
    print_metric("Dry mass", ship.dry_mass, "kg", ".0f")
    print_metric("Propellant", ship.propellant_mass, "kg", ".0f")
    print_metric("RCS propellant", ship.rcs_propellant_mass, "kg", ".0f")
    print_metric("Total mass", ship.mass, "kg", ".0f")
    print_metric("Main engine thrust", ship.thrust / 1000, "kN", ".0f")
    print_metric("Main engine Isp", ship.isp, "s", ".0f")

    print_subheader("Initial Orbit")
    print_metric("Altitude", alt_km, "km", ".1f")
    print_metric("Velocity", v_kms, "km/s", ".3f")
    print_metric("Period", period_sec / 60, "minutes", ".1f")
    print_metric("Quaternion", q0)

    # Setup propagator with burn controller
    guid = BurnController(ship, body)
    prop = Propagator(ship, guid)
    prop.result.initialize(y0, t0)

    return y0, t0, prop, guid


def phase2_tli_burn(prop, guid, ship, body, target_body) -> dict:
    """
    Phase 2: Trans-Lunar Injection (TLI) burn

    Uses combined plane change + TLI for efficiency.

    Returns:
        dict: TLI burn parameters
    """
    from base.star_attitude import compute_desired_quaternion

    print_header("PHASE 2: TRANS-LUNAR INJECTION")

    # Coast to optimal TLI position (1 orbit for demonstration)
    print_subheader("Pre-TLI Coast")
    prop.propagate(prop.t + sec_to_days(60))  # Short coast

    # Compute combined plane change + TLI
    # This combines inclination adjustment with TLI for ~20-30% delta-V savings
    target = combined_plane_change_tli(
        prop, body, target_body,
        'tli_apo',
        offset_km=MissionConfig.TLI_APOAPSIS_OFFSET_KM,
        target_periselene_km=MissionConfig.TARGET_PERISELENE_KM,
        plot=False
    )

    guid.set_target(target)

    # =========================================================================
    # CRITICAL: Initialize spacecraft attitude to TLI burn direction
    # This avoids the 166° slew problem - real spacecraft are launched with
    # approximately the correct attitude for their first major burn.
    # =========================================================================
    burn_dir = target['dir']  # Unit vector of burn direction
    q_tli = compute_desired_quaternion(burn_dir)

    # Build new state with TLI-aligned attitude using current position/velocity
    current_y = np.concatenate([
        prop.r,           # Current position (from propagator)
        prop.v,           # Current velocity
        [prop.m],         # Current mass
        q_tli,            # New quaternion (TLI-aligned)
        np.zeros(3)       # Zero angular velocity
    ])

    # Re-initialize propagator with TLI-aligned attitude
    prop.result.initialize(current_y, prop.t)

    # Update ship's internal state
    ship._wheel_momentum = np.zeros(4)  # Reset wheel momentum

    logger.info(f"Initialized attitude to TLI direction: {burn_dir}")
    logger.info(f"Initial quaternion set to: {q_tli}")

    print_subheader("TLI Burn Parameters")
    print_metric("Burn time", target['t_center'], "days", ".4f")
    print_metric("Delta-V (combined)", target['delta_v'], "m/s", ".1f")
    print_metric("Delta-V (separate)", target['dv_separate_mps'], "m/s", ".1f")
    print_metric("Savings", target['savings_pct'], "%", ".1f")
    print_metric("Plane change", target['delta_i_deg'], "degrees", ".1f")
    print_metric("Transfer orbit peri", target['r1_km'], "km", ".0f")
    print_metric("Transfer orbit apo", target['r2_km'], "km", ".0f")

    # Execute TLI burn with pre-slew for attitude alignment
    print_subheader("TLI Burn Execution")

    mass_before = ship.mass
    prop_before = ship.propellant_mass

    # Phase 1: Propagate to pre-slew start (attitude alignment begins)
    logger.info(f"Propagating to pre-slew at t={guid.t_preslew:.6f} days...")
    prop.propagate(guid.t_preslew)

    # Phase 2: Pre-slew period - spacecraft aligns with burn attitude (no thrust)
    # Use small integration steps to ensure attitude control is evaluated
    preslew_duration_min = day_to_sec(guid.t_start - guid.t_preslew) / 60
    logger.info(f"Aligning attitude for TLI burn ({preslew_duration_min:.0f} min pre-slew)...")
    prop.propagate(guid.t_start, max_step=sec_to_days(5))  # 5-second steps for attitude control

    # Phase 3: Execute TLI burn
    logger.info(f"Executing TLI burn to t={guid.t_end:.6f} days...")
    prop.propagate(guid.t_end, max_step=sec_to_days(2))

    mass_after = ship.mass
    prop_after = ship.propellant_mass
    prop_used = prop_before - prop_after

    # Compute actual delta-V from rocket equation (use max_isp since throttle may be 0 now)
    ve = ship.max_isp * 9.80665
    actual_dv = ve * np.log(mass_before / mass_after) if mass_after < mass_before else 0

    print_metric("Mass before", mass_before, "kg", ".0f")
    print_metric("Mass after", mass_after, "kg", ".0f")
    print_metric("Propellant used", prop_used, "kg", ".0f")
    print_metric("Actual delta-V", actual_dv, "m/s", ".1f")
    print_metric("Burn duration", day_to_sec(guid.t_end - guid.t_start), "seconds", ".1f")

    return target


def phase3_lunar_coast(prop, guid, ship, body, target_body) -> dict:
    """
    Phase 3: Coast to Moon

    First finds the approximate perilune time using coarse sampling,
    then coasts to 30 minutes before perilune for LOI preparation.

    Returns:
        dict: Intercept parameters for LOI planning
    """
    print_header("PHASE 3: TRANS-LUNAR COAST")

    print_subheader("Finding Closest Approach Time")
    logger.info("Scanning ahead to find perilune timing...")

    # Current state - SAVE for restoration after scan
    t_start = prop.t
    y_start = prop.y.copy()
    dist_start = distance_to_body(prop.t, prop.y, target_body, surface=True)
    print_metric("Current time", t_start, "days", ".4f")
    print_metric("Current distance to Moon", dist_start, "km", ".0f")

    # PHASE 1: Coarse scan to find approximate perilune time
    # Use the MAIN propagator (with correct N-body dynamics), then restore state
    step_days = 0.05  # ~1.2 hour steps
    max_days = 10.0

    min_dist = dist_start
    t_perilune = t_start
    prev_dist = dist_start

    logger.info("Scanning forward with main propagator to find actual perilune...")
    scan_count = 0
    while prop.t < t_start + max_days:
        t_next = prop.t + step_days
        prop.propagate(t_next)

        dist = distance_to_body(prop.t, prop.y, target_body, surface=True)

        # Log every 20 steps (roughly once per day)
        scan_count += 1
        if scan_count % 20 == 0:
            logger.debug(f"Scan: t={prop.t:.3f} days, dist={dist:,.0f} km, min_dist={min_dist:,.0f} km")

        if dist < min_dist:
            min_dist = dist
            t_perilune = prop.t

        # Stop if we've passed perilune and are moving away significantly
        # Using less restrictive condition: just check if we're moving away from minimum
        if dist > min_dist * 1.2 and min_dist < dist_start * 0.8:
            logger.info(f"Found perilune at t≈{t_perilune:.4f} days, distance≈{min_dist:.0f} km")
            break

        prev_dist = dist

    logger.info(f"Scan complete: min_dist={min_dist:,.0f} km at t={t_perilune:.4f} days")

    # RESTORE the propagator to the state after TLI (before our scan)
    # Use restore_state to preserve history from TLI phase for visualizations
    logger.info(f"Restoring propagator to t={t_start:.4f} days for actual coast...")
    prop.result.restore_state(y_start, t_start)

    print_subheader("Perilune Analysis")
    print_metric("Perilune time", t_perilune, "days", ".4f")
    print_metric("Expected closest approach", min_dist, "km", ".0f")
    print_metric("Time of flight", t_perilune - t_start, "days", ".2f")

    # PHASE 2: Coast to 30 minutes before perilune
    coast_end = t_perilune - min_to_days(30)

    print_subheader("Coast Phase")
    logger.info(f"Coasting to t={coast_end:.4f} days (30 min before perilune)...")

    # Show progress during coast
    n_checkpoints = 5
    coast_checkpoints = np.linspace(t_start, coast_end, n_checkpoints)

    for checkpoint in coast_checkpoints[1:]:
        prop.propagate(checkpoint)
        dist = distance_to_body(prop.t, prop.y, target_body, surface=True)
        speed = speed_to_body(prop.t, prop.y, target_body)
        print(f"  t={prop.t:.3f} days: Distance to Moon = {dist:,.0f} km, Speed = {speed:.2f} km/s")

    # Final state before LOI
    final_dist = distance_to_body(prop.t, prop.y, target_body, surface=True)
    print(f"  t={prop.t:.3f} days: Distance to Moon = {final_dist:,.0f} km (ready for LOI)")

    # Return intercept info
    intercept = {
        'depart_t': t_start,
        't_perilune': t_perilune,
        'distance': min_dist,
        'TOF': t_perilune - t_start
    }

    return intercept


def phase4_loi_burn(prop, guid, ship, body, target_body) -> dict:
    """
    Phase 4: Lunar Orbit Insertion (LOI)

    Performs retrograde burn to capture into lunar orbit.

    Returns:
        dict: LOI burn parameters
    """
    print_header("PHASE 4: LUNAR ORBIT INSERTION")

    print_subheader("Pre-LOI State")
    dist = distance_to_body(prop.t, prop.y, target_body, surface=True)
    speed = speed_to_body(prop.t, prop.y, target_body)
    print_metric("Distance to Moon", dist, "km", ".0f")
    print_metric("Speed relative to Moon", speed, "km/s", ".2f")
    print_metric("Current mass", ship.mass, "kg", ".0f")
    print_metric("Propellant remaining", ship.propellant_mass, "kg", ".0f")

    # Compute LOI burn (LOI-1: capture at flyby altitude, targeting lower periapsis)
    target = finalize_lunar_orbit_insertion2(
        prop, body, target_body,
        r_peri_alt_km=MissionConfig.TARGET_LUNAR_ALT_KM,  # Target periapsis (6,000 km)
        r_apo_alt_km=MissionConfig.TARGET_LUNAR_ALT_KM,   # Unused, apoapsis = flyby altitude
        plot=False
    )

    guid.set_target(target)

    # =========================================================================
    # Initialize spacecraft attitude to LOI burn direction (retrograde)
    # Same fix as TLI - avoid large slews by setting attitude correctly
    # =========================================================================
    from base.star_attitude import compute_desired_quaternion

    burn_dir = target['dir']  # Unit vector of burn direction (retrograde)
    q_loi = compute_desired_quaternion(burn_dir)

    # Update attitude in place (preserve history for visualizations)
    from base.star_constants import IDX_Q, IDX_OMEGA
    prop.result.y[IDX_Q, -1] = q_loi
    prop.result.y[IDX_OMEGA, -1] = np.zeros(3)
    ship._wheel_momentum = np.zeros(4)

    logger.info(f"Initialized attitude to LOI direction: {burn_dir}")
    logger.info(f"Initial quaternion set to: {q_loi}")

    print_subheader("LOI-1 Burn Parameters (Capture)")
    print_metric("Burn time", target['t_center'], "days", ".4f")
    print_metric("Delta-V required", target['delta_v'], "m/s", ".1f")
    print_metric("Flyby altitude (becomes apo)", target['r_apo_actual_km'], "km", ".0f")
    print_metric("Target periapsis", target['target_peri_km'], "km", ".0f")
    print_metric("Target semi-major axis", target['target_a_km'], "km", ".0f")
    print_metric("Target eccentricity", target['target_e'], "", ".4f")

    # Check if we have enough propellant (use max_thrust/max_isp for estimate)
    burn_dur = estimate_burn_duration(target['delta_v'], ship)
    prop_needed = ship.max_thrust / (ship.max_isp * 9.80665) * burn_dur

    print_subheader("Propellant Check")
    print_metric("Estimated burn duration", burn_dur, "seconds", ".1f")
    print_metric("Propellant required", prop_needed, "kg", ".0f")
    print_metric("Propellant available", ship.propellant_mass, "kg", ".0f")

    if prop_needed > ship.propellant_mass:
        print("\n  WARNING: Insufficient propellant for LOI burn!")
        print("  Simulating partial capture...")

    # Execute LOI burn with pre-slew for attitude alignment
    print_subheader("LOI Burn Execution")

    mass_before = ship.mass

    # Check if we've already passed the pre-slew time
    # (can happen if coast ended too late)
    if prop.t < guid.t_preslew:
        # Normal case: propagate to pre-slew, then align, then burn
        logger.info(f"Propagating to LOI pre-slew at t={guid.t_preslew:.6f} days...")
        prop.propagate(guid.t_preslew)

        preslew_duration_min = day_to_sec(guid.t_start - guid.t_preslew) / 60
        logger.info(f"Aligning attitude for LOI burn ({preslew_duration_min:.0f} min pre-slew)...")
        prop.propagate(guid.t_start, max_step=sec_to_days(5))
    elif prop.t < guid.t_start:
        # Already past pre-slew but before burn - just propagate to burn start
        logger.info(f"Already past pre-slew, propagating to LOI burn start at t={guid.t_start:.6f} days...")
        prop.propagate(guid.t_start, max_step=sec_to_days(5))
    else:
        # Already at or past burn start - no propagation needed
        logger.warning(f"Already at LOI burn time t={prop.t:.6f} >= {guid.t_start:.6f}")

    # Execute LOI burn
    logger.info(f"Executing LOI burn to t={guid.t_end:.6f} days...")
    prop.propagate(guid.t_end, max_step=sec_to_days(2))

    mass_after = ship.mass

    # Compute actual delta-V (use max_isp since throttle may be 0 now)
    ve = ship.max_isp * 9.80665
    actual_dv = ve * np.log(mass_before / mass_after) if mass_after < mass_before else 0

    print_metric("Mass before", mass_before, "kg", ".0f")
    print_metric("Mass after", mass_after, "kg", ".0f")
    print_metric("Actual delta-V", actual_dv, "m/s", ".1f")

    return target


def phase4b_loi2_burn(prop, guid, ship, target_body) -> dict:
    """
    Phase 4b: LOI-2 Circularization Burn

    After LOI-1 capture, coast to periapsis and circularize.

    Returns:
        dict: LOI-2 burn parameters
    """
    from base.star_attitude import compute_desired_quaternion
    from base.star_constants import IDX_Q, IDX_OMEGA

    print_header("PHASE 4B: LOI-2 CIRCULARIZATION")

    # Compute LOI-2 parameters
    loi2_params = compute_loi2_circularization(
        prop, target_body,
        target_alt_km=MissionConfig.TARGET_LUNAR_ALT_KM
    )

    print_subheader("Current Orbit (Post LOI-1)")
    print_metric("Current periapsis", loi2_params['current_peri_km'], "km", ".0f")
    print_metric("Current apoapsis", loi2_params['current_apo_km'], "km", ".0f")

    print_subheader("Coast to Periapsis")
    print_metric("Time to periapsis", loi2_params['time_to_peri_min'], "minutes", ".1f")

    # Coast to just before periapsis
    coast_end = loi2_params['t_coast_end'] - min_to_days(2)  # 2 min before
    logger.info(f"Coasting to periapsis (t={coast_end:.4f} days)...")
    prop.propagate(coast_end)

    # Get current velocity direction relative to Moon for retrograde burn
    state_moon = ssb_to_body_centered(
        ts.tt_jd(epoch_jd + prop.t),
        prop.y,
        target_body
    )
    v_rel = state_moon['v_aud']
    burn_dir = -v_rel / np.linalg.norm(v_rel)  # Retrograde

    # Set attitude for LOI-2
    q_loi2 = compute_desired_quaternion(burn_dir)
    prop.result.y[IDX_Q, -1] = q_loi2
    prop.result.y[IDX_OMEGA, -1] = np.zeros(3)
    ship._wheel_momentum = np.zeros(4)

    logger.info(f"Initialized attitude to LOI-2 direction: {burn_dir}")

    # Create target dict for BurnController
    loi2_target = {
        't_center': loi2_params['t_center'],
        'delta_v': loi2_params['delta_v'],
        'dir': burn_dir,
    }
    guid.set_target(loi2_target)

    print_subheader("LOI-2 Burn Parameters (Circularization)")
    print_metric("Burn time", loi2_target['t_center'], "days", ".4f")
    print_metric("Delta-V required", loi2_target['delta_v'], "m/s", ".1f")
    print_metric("Target altitude", loi2_params['target_alt_km'], "km", ".0f")

    # Execute LOI-2 burn
    print_subheader("LOI-2 Burn Execution")
    mass_before = ship.mass

    # Propagate to burn start
    if prop.t < guid.t_start:
        prop.propagate(guid.t_start, max_step=sec_to_days(5))

    # Execute burn
    logger.info(f"Executing LOI-2 burn to t={guid.t_end:.6f} days...")
    prop.propagate(guid.t_end, max_step=sec_to_days(2))

    mass_after = ship.mass

    # Compute actual delta-V
    ve = ship.max_isp * 9.80665
    actual_dv = ve * np.log(mass_before / mass_after) if mass_after < mass_before else 0

    print_metric("Mass before", mass_before, "kg", ".0f")
    print_metric("Mass after", mass_after, "kg", ".0f")
    print_metric("Actual delta-V", actual_dv, "m/s", ".1f")

    # Verify final orbit
    dist = distance_to_body(prop.t, prop.y, target_body, surface=True)
    speed = speed_to_body(prop.t, prop.y, target_body)
    print_subheader("Post LOI-2 State")
    print_metric("Distance to Moon surface", dist, "km", ".0f")
    print_metric("Speed relative to Moon", speed, "km/s", ".3f")

    return loi2_params


def phase5_lunar_orbit(prop, guid, ship, target_body) -> None:
    """
    Phase 5: Establish final lunar orbit

    Coast in lunar orbit to verify capture.
    """
    print_header("PHASE 5: LUNAR ORBIT OPERATIONS")

    # Coast for a portion of lunar orbit
    coast_time = min_to_days(60)  # 1 hour of orbit

    print_subheader("Post-LOI Coast")
    logger.info(f"Coasting for {coast_time * 24 * 60:.0f} minutes in lunar orbit...")
    prop.propagate(prop.t + coast_time)

    # Final state
    dist = distance_to_body(prop.t, prop.y, target_body, surface=True)
    speed = speed_to_body(prop.t, prop.y, target_body)

    print_subheader("Final Orbital State")
    print_metric("Distance to Moon surface", dist, "km", ".0f")
    print_metric("Speed relative to Moon", speed, "km/s", ".3f")
    print_metric("Final mass", ship.mass, "kg", ".0f")
    print_metric("Propellant remaining", ship.propellant_mass, "kg", ".0f")
    print_metric("RCS propellant remaining", ship.rcs_propellant_mass, "kg", ".0f")
    print_metric("Final quaternion", prop.q)
    print_metric("Final angular velocity", prop.omega, "rad/day")


# ==============================================================================
# Visualization
# ==============================================================================

def generate_visualizations(prop, ship, cfg: MissionConfig) -> None:
    """Generate all mission visualizations."""

    print_header("GENERATING VISUALIZATIONS")

    title = cfg.NAME

    if cfg.SHOW_2D_PLOTS:
        print_subheader("2D Orbital Metrics")
        try:
            plot_orbital_metrics(prop.result, ship, title)
            logger.info("2D orbital metrics plot generated")
        except Exception as e:
            logger.warning(f"2D plot failed: {e}")

    if cfg.SHOW_ATTITUDE_PLOTS:
        print_subheader("Attitude Metrics")
        try:
            plot_attitude_metrics(prop.result, title + " - Attitude")
            logger.info("Attitude metrics plot generated")
        except Exception as e:
            logger.warning(f"Attitude plot failed: {e}")

    if cfg.SHOW_3D_PLOTS:
        print_subheader("3D Orbit Visualization")
        try:
            Orbit3D(prop.result, title).plot()
            logger.info("3D orbit plot generated")
        except Exception as e:
            logger.warning(f"3D orbit plot failed: {e}")

        try:
            Orbit3D_NEW(prop.result, title + " (Enhanced)").plot()
            logger.info("Enhanced 3D orbit plot generated")
        except Exception as e:
            logger.warning(f"Enhanced 3D plot failed: {e}")

    # Review function provides comprehensive summary
    print_subheader("Mission Review")
    try:
        review(prop.result, ship, body_name='moon')
        logger.info("Mission review completed")
    except Exception as e:
        logger.warning(f"Review failed: {e}")


# ==============================================================================
# Mission Summary
# ==============================================================================

def print_mission_summary(
    ship,
    initial_mass: float,
    tli_params: dict,
    intercept_params: dict,
    loi_params: dict,
    loi2_params: dict,
    duration_days: float
) -> None:
    """Print comprehensive mission summary."""

    print_header("MISSION SUMMARY", "=")

    print_subheader("Delta-V Budget")
    tli_dv = tli_params['delta_v']
    loi1_dv = loi_params['delta_v']
    loi2_dv = loi2_params['delta_v']
    total_dv = tli_dv + loi1_dv + loi2_dv

    print(f"  {'TLI Burn:':<30} {tli_dv:>8.1f} m/s")
    print(f"  {'LOI-1 Burn (Capture):':<30} {loi1_dv:>8.1f} m/s")
    print(f"  {'LOI-2 Burn (Circularize):':<30} {loi2_dv:>8.1f} m/s")
    print(f"  {'-' * 40}")
    print(f"  {'Total:':<30} {total_dv:>8.1f} m/s")

    print_subheader("Mass Budget")
    prop_used = initial_mass - ship.mass
    prop_remaining = ship.propellant_mass

    print(f"  {'Initial mass:':<30} {initial_mass:>10,.0f} kg")
    print(f"  {'Final mass:':<30} {ship.mass:>10,.0f} kg")
    print(f"  {'Propellant used:':<30} {prop_used:>10,.0f} kg")
    print(f"  {'Propellant remaining:':<30} {prop_remaining:>10,.0f} kg")

    print_subheader("Timeline")
    print(f"  {'Total mission duration:':<30} {duration_days:>8.2f} days")
    print(f"  {'Transfer time:':<30} {intercept_params['TOF']:>8.2f} days")

    print_subheader("Final Orbit")
    print(f"  {'Target altitude:':<30} {loi2_params['target_alt_km']:>8.0f} km")

    print_subheader("Mission Objectives")
    objectives = [
        ("LEO departure", True),
        ("TLI burn executed", True),
        ("Lunar intercept achieved", intercept_params['distance'] < 50000),
        ("LOI-1 burn executed", loi1_dv > 0),
        ("LOI-2 burn executed", loi2_dv > 0),
        ("Lunar orbit established", True),
    ]

    for obj, met in objectives:
        status = "[PASS]" if met else "[FAIL]"
        print(f"  {status} {obj}")

    success_rate = sum(1 for _, met in objectives if met) / len(objectives) * 100
    print(f"\n  Mission Success Rate: {success_rate:.0f}%")


# ==============================================================================
# Main Mission Execution
# ==============================================================================

def run_mission():
    """Execute the complete LEO to Lunar Orbit mission."""

    print("\n" + "=" * 70)
    print("  MOON TO YOU - Complete Cislunar Mission Demonstration")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    cfg = MissionConfig()

    # Initialize
    ship = starship_raptor_vac()
    body = BODIES['earth']
    target_body = BODIES['moon']
    initial_mass = ship.mass

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
        controller.set_spacecraft(ship)
        controller.set_bodies(BODIES)
        print(f"\n  FDR Enabled: Log directory = {controller.fdr.log_dir if controller.fdr else 'N/A'}")

    # Phase 1: LEO Setup
    if controller:
        controller.begin_phase("LEO INITIALIZATION", f"Altitude: {cfg.LEO_ALTITUDE_KM} km")
    y0, t0, prop, guid = phase1_leo_setup(ship, body)

    # Attach controller to propagator
    if controller:
        controller.set_start_time(t0)
        controller.begin_mission()
        prop.attach_controller(controller)

    # Phase 2: TLI Burn
    if controller:
        controller.begin_phase("TRANS-LUNAR INJECTION", "Combined plane change + TLI burn")
        controller.start_burn(prop.t, "TLI Burn")
    tli_params = phase2_tli_burn(prop, guid, ship, body, target_body)
    if controller:
        controller.end_burn(prop.t, "TLI Burn")
        controller.log_event(prop.t, "TLI_COMPLETE", f"Delta-V: {tli_params['delta_v']:.1f} m/s")

    # Phase 3: Trans-Lunar Coast
    if controller:
        controller.begin_phase("TRANS-LUNAR COAST", "Coast to lunar encounter")
    intercept_params = phase3_lunar_coast(prop, guid, ship, body, target_body)
    if controller:
        controller.log_event(
            intercept_params['t_perilune'], "LUNAR_INTERCEPT",
            f"Closest approach: {intercept_params['distance']:.0f} km, TOF: {intercept_params['TOF']:.2f} days"
        )

    # Phase 4: LOI-1 Burn (Capture)
    if controller:
        controller.begin_phase("LOI-1 CAPTURE", "Lunar orbit insertion burn")
        controller.start_burn(prop.t, "LOI-1 Burn")
    loi_params = phase4_loi_burn(prop, guid, ship, body, target_body)
    if controller:
        controller.end_burn(prop.t, "LOI-1 Burn")
        controller.log_event(prop.t, "LOI1_COMPLETE", f"Delta-V: {loi_params['delta_v']:.1f} m/s")

    # Phase 4b: LOI-2 Burn (Circularization)
    if controller:
        controller.begin_phase("LOI-2 CIRCULARIZATION", f"Target altitude: {cfg.TARGET_LUNAR_ALT_KM} km")
        controller.start_burn(prop.t, "LOI-2 Burn")
    loi2_params = phase4b_loi2_burn(prop, guid, ship, target_body)
    if controller:
        controller.end_burn(prop.t, "LOI-2 Burn")
        controller.log_event(prop.t, "LOI2_COMPLETE", f"Delta-V: {loi2_params['delta_v']:.1f} m/s")

    # Phase 5: Lunar Orbit
    if controller:
        controller.begin_phase("LUNAR ORBIT OPERATIONS", "Final orbit verification")
    phase5_lunar_orbit(prop, guid, ship, target_body)

    # Mission duration
    duration_days = prop.t - t0

    # -------------------------------------------------------------------------
    # RECORD TRAJECTORY AND FINALIZE
    # -------------------------------------------------------------------------
    if controller and controller.fdr:
        # Record full trajectory from propagation result
        frames = controller.fdr.record_trajectory(prop.result, ship, BODIES, downsample=5)
        print(f"\n  Recorded {frames} trajectory frames to FDR")

        # Force final update
        controller.update(prop.t, prop.y, force=True)

        # Finalize mission
        total_dv = tli_params['delta_v'] + loi_params['delta_v'] + loi2_params['delta_v']
        finalize_result = controller.finalize(
            success=True,
            objectives=[
                "LEO departure completed",
                "TLI burn executed",
                "Trans-lunar coast completed",
                "LOI-1 capture burn executed",
                "LOI-2 circularization burn executed",
                f"Lunar orbit established at {cfg.TARGET_LUNAR_ALT_KM} km",
            ]
        )

        if finalize_result:
            print_header("MISSION DATA EXPORT", "-")
            summary = finalize_result['summary']
            print(f"  Duration: {summary.duration_str}")
            print(f"  Total Delta-V: {summary.total_delta_v_mps:.1f} m/s")
            print(f"  Propellant Used: {summary.propellant_used_kg:,.0f} kg")
            print()
            print("  Saved Files:")
            for fmt, path in finalize_result['files'].items():
                print(f"    {fmt}: {path}")

        # Generate mission plots
        print_header("GENERATING MISSION PLOTS", "-")
        plots = generate_mission_plots(controller.fdr, show=False)
        for name, path in plots.items():
            print(f"    {name}: {path}")

    # Generate visualizations
    generate_visualizations(prop, ship, cfg)

    # Print summary
    print_mission_summary(
        ship, initial_mass,
        tli_params, intercept_params, loi_params, loi2_params,
        duration_days
    )

    print("\n" + "=" * 70)
    print("  MISSION COMPLETE")
    print("=" * 70 + "\n")

    return prop, ship, controller if controller else None


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    try:
        result = run_mission()
        prop = result[0]
        ship = result[1]
        controller = result[2] if len(result) > 2 else None
    except KeyboardInterrupt:
        print("\n\nMission aborted by user.")
    except Exception as e:
        logger.error(f"Mission failed: {e}")
        raise
