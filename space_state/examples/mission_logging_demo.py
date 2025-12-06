#!/usr/bin/env python3
"""
Mission Logging Demo - Flight Data Recorder and Console Display

Demonstrates the complete mission telemetry logging and display system:
- Flight Data Recorder (FDR) for comprehensive data logging
- Console Display for real-time flight computer style output
- Mission Controller for coordinated logging and display
- Automatic log file generation and mission documentation

This example runs a simple LEO to GTO transfer mission while
recording all telemetry data and displaying real-time metrics.

Author: Space State Development Team
"""

import numpy as np
import sys
import os

# Reduce logging noise
os.environ['LOG_LEVEL'] = 'WARNING'

# Ensure correct path
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base import (
    BODIES, CONFIG, Propagator,
    starship_raptor_vac, get_circular_state,
    vis_viva, orbital_period,
    # New logging/display imports
    FlightDataRecorder, MissionController, DisplayMode,
    create_fdr, create_mission_controller, generate_mission_plots,
)
from base.star_constants import AU_TO_KM, DAY_TO_SEC
from base.star_utils import sec_to_days, km_to_au
from base.star_attitude import compute_desired_quaternion
from base.star_constants import IDX_Q, IDX_OMEGA

from guidance import BurnController
from guidance.star_tools import estimate_burn_duration


# =============================================================================
# Mission Configuration
# =============================================================================

class DemoConfig:
    """Demo mission configuration."""
    MISSION_NAME = "LEO to GTO Transfer Demo"

    # Orbits
    LEO_ALTITUDE_KM = 400       # Starting orbit
    GTO_PERIGEE_KM = 400        # GTO perigee (same as LEO)
    GTO_APOGEE_KM = 35786       # GTO apogee (GEO altitude)

    # Earth parameters
    R_EARTH_KM = 6378.137
    MU_EARTH_KM = 398600.4418


# =============================================================================
# Main Demo
# =============================================================================

def run_logging_demo():
    """Run the mission logging demonstration."""

    print("\n" + "=" * 78)
    print("  MISSION LOGGING SYSTEM DEMONSTRATION")
    print("  LEO to GTO Transfer with Full Telemetry Recording")
    print("=" * 78 + "\n")

    cfg = DemoConfig()
    earth = BODIES['earth']

    # -------------------------------------------------------------------------
    # 1. SETUP SPACECRAFT
    # -------------------------------------------------------------------------
    print("--- SPACECRAFT SETUP ---\n")

    ship = starship_raptor_vac()
    print(f"  Spacecraft: {ship._name}")
    print(f"  Dry Mass: {ship.dry_mass:,.0f} kg")
    print(f"  Propellant: {ship.propellant_mass:,.0f} kg")
    print(f"  Total Mass: {ship.mass:,.0f} kg")
    print(f"  Thrust: {ship.max_thrust/1000:,.0f} kN")
    print(f"  Isp: {ship.max_isp:.0f} s")
    print()

    # -------------------------------------------------------------------------
    # 2. SETUP MISSION CONTROLLER (combines display + FDR)
    # -------------------------------------------------------------------------
    print("--- MISSION CONTROLLER SETUP ---\n")

    # Create mission controller with full telemetry recording
    # Use finer recording interval for better data resolution
    controller = MissionController(
        mission_name=cfg.MISSION_NAME,
        display_mode=DisplayMode.COMPACT,
        enable_fdr=True,
        fdr_interval=0.0001,  # ~8.6 sec base interval (10x faster during burns)
        display_interval=2.0,  # Update display every 2 seconds
    )
    controller.set_spacecraft(ship)
    controller.set_bodies(BODIES)

    print(f"  Mission: {cfg.MISSION_NAME}")
    print(f"  Display Mode: COMPACT")
    print(f"  FDR Enabled: True")
    print(f"  Log Directory: {controller.fdr.log_dir if controller.fdr else 'N/A'}")
    print()

    # -------------------------------------------------------------------------
    # 3. CALCULATE TRANSFER PARAMETERS
    # -------------------------------------------------------------------------
    print("--- TRANSFER CALCULATION ---\n")

    r_leo = cfg.R_EARTH_KM + cfg.LEO_ALTITUDE_KM
    r_gto_peri = cfg.R_EARTH_KM + cfg.GTO_PERIGEE_KM
    r_gto_apo = cfg.R_EARTH_KM + cfg.GTO_APOGEE_KM

    # Semi-major axis of GTO
    a_gto = (r_gto_peri + r_gto_apo) / 2

    # Velocities (km/s)
    v_leo = np.sqrt(cfg.MU_EARTH_KM / r_leo)
    v_gto_peri = np.sqrt(cfg.MU_EARTH_KM * (2/r_gto_peri - 1/a_gto))

    # Delta-V for transfer
    dv_transfer_kms = v_gto_peri - v_leo
    dv_transfer_mps = dv_transfer_kms * 1000

    print(f"  LEO Altitude: {cfg.LEO_ALTITUDE_KM:,.0f} km")
    print(f"  GTO Apogee: {cfg.GTO_APOGEE_KM:,.0f} km")
    print(f"  LEO Velocity: {v_leo:.3f} km/s")
    print(f"  GTO Perigee Velocity: {v_gto_peri:.3f} km/s")
    print(f"  Transfer Delta-V: {dv_transfer_mps:,.1f} m/s")
    print()

    # -------------------------------------------------------------------------
    # 4. INITIALIZE STATE AND PROPAGATOR
    # -------------------------------------------------------------------------
    print("--- PROPAGATOR SETUP ---\n")

    # Get initial state in LEO
    r0, v0, q0, omega0 = get_circular_state(earth, 'leo_iss')
    y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])
    t0 = 0.0

    # Create guidance controller
    guidance = BurnController(ship, earth)

    # Create propagator
    prop = Propagator(ship, guidance, bodies=BODIES)
    prop.result.initialize(y0, t0)

    # Attach mission controller to propagator
    prop.attach_controller(controller)

    print(f"  Initial Time: {t0} days")
    print(f"  Position (SSB): {r0}")
    print(f"  Velocity (SSB): {v0}")
    print()

    # Set mission start time
    controller.set_start_time(t0)

    # -------------------------------------------------------------------------
    # 5. RUN MISSION WITH LOGGING
    # -------------------------------------------------------------------------

    controller.begin_mission()

    # --- PHASE 1: Pre-burn coast ---
    controller.begin_phase("PRE-BURN COAST", "Establishing attitude for transfer burn")

    # Set up prograde burn direction
    v_current = prop.v
    burn_dir = v_current / np.linalg.norm(v_current)

    # Estimate burn duration
    burn_dur_sec = estimate_burn_duration(dv_transfer_mps, ship)
    print(f"  Estimated burn duration: {burn_dur_sec:.1f} seconds")

    # Schedule burn
    t_burn_start = t0 + sec_to_days(60)  # 60 seconds from now
    target = {
        't_center': t_burn_start + sec_to_days(burn_dur_sec/2),
        'delta_v': dv_transfer_mps,
        'dir': burn_dir,
    }
    guidance.set_target(target)

    # Set attitude for burn
    q_burn = compute_desired_quaternion(burn_dir)
    prop.result.y[IDX_Q, -1] = q_burn
    prop.result.y[IDX_OMEGA, -1] = np.zeros(3)

    # Coast to burn start
    prop.propagate(t_burn_start, max_step=sec_to_days(10))

    # --- PHASE 2: Transfer burn ---
    controller.begin_phase("TRANSFER BURN", f"Target Delta-V: {dv_transfer_mps:.0f} m/s")

    mass_before = ship.mass

    # Record burn start
    controller.start_burn(prop.t, "GTO Transfer")

    # Execute burn
    prop.propagate(guidance.t_end, max_step=sec_to_days(2))

    mass_after = ship.mass

    # Record burn end
    controller.end_burn(prop.t, "GTO Transfer")

    # Calculate actual delta-V
    ve = ship.max_isp * 9.80665
    actual_dv = ve * np.log(mass_before / mass_after) if mass_after < mass_before else 0

    controller.log_event(
        prop.t,
        "BURN_COMPLETE",
        f"Actual Delta-V: {actual_dv:.1f} m/s, Propellant: {mass_before - mass_after:.0f} kg"
    )

    # --- PHASE 3: Coast to apogee ---
    controller.begin_phase("COAST TO APOGEE", "Verifying GTO insertion")

    # Coast for a bit to verify orbit
    coast_time = sec_to_days(3600)  # 1 hour
    prop.propagate(prop.t + coast_time, max_step=sec_to_days(60))

    # --- PHASE 4: Mission complete ---
    controller.begin_phase("MISSION COMPLETE", "Transfer burn successful")

    # Record the full trajectory from propagation result
    if controller.fdr:
        frames = controller.fdr.record_trajectory(prop.result, ship, BODIES, downsample=5)
        print(f"  Recorded {frames} trajectory frames for validation")

    # Force a final telemetry update
    controller.update(prop.t, prop.y, force=True)

    # -------------------------------------------------------------------------
    # 6. FINALIZE AND SAVE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  MISSION DATA EXPORT")
    print("=" * 78 + "\n")

    # Finalize mission with objectives
    result = controller.finalize(
        success=True,
        objectives=[
            "LEO departure completed",
            "GTO transfer burn executed",
            "Coast to apogee verified",
        ]
    )

    if result:
        print("  Mission Summary:")
        summary = result['summary']
        print(f"    Duration: {summary.duration_str}")
        print(f"    Total Delta-V: {summary.total_delta_v_mps:.1f} m/s")
        print(f"    Propellant Used: {summary.propellant_used_kg:,.0f} kg")
        print(f"    Final Mass: {summary.final_mass_kg:,.0f} kg")
        print()

        print("  Saved Files:")
        for fmt, path in result['files'].items():
            print(f"    {fmt}: {path}")

    # -------------------------------------------------------------------------
    # 7. GENERATE MISSION PLOTS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("  GENERATING MISSION PLOTS")
    print("-" * 78 + "\n")

    if controller.fdr:
        plots = generate_mission_plots(controller.fdr, show=False)
        print("  Generated Plots:")
        for name, path in plots.items():
            print(f"    {name}: {path}")

    print("\n" + "=" * 78)
    print("  DEMO COMPLETE")
    print("=" * 78 + "\n")

    return prop, controller


# =============================================================================
# Alternative: Using FDR Directly (without display)
# =============================================================================

def run_fdr_only_demo():
    """
    Demonstrate using FDR directly without console display.

    Useful for:
    - Batch processing
    - Background simulations
    - When you only need data logging
    """
    print("\n--- FDR-ONLY DEMO ---\n")

    ship = starship_raptor_vac()
    earth = BODIES['earth']

    # Create FDR directly
    fdr = create_fdr(
        mission_name="FDR Only Test",
        ship=ship,
        start_time=0.0,
        record_interval=0.0001,  # More frequent recording
    )

    # Create propagator and attach FDR
    r0, v0, q0, omega0 = get_circular_state(earth, 'leo_iss')
    y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

    guidance = BurnController(ship, earth)
    prop = Propagator(ship, guidance, bodies=BODIES)
    prop.result.initialize(y0, 0.0)
    prop.attach_fdr(fdr)

    # Record mission start
    fdr.record_event(0.0, "mission_start", "FDR-only test mission started")

    # Run a short propagation
    prop.propagate(sec_to_days(300))  # 5 minutes

    # Record event
    fdr.record_event(prop.t, "checkpoint", "5-minute checkpoint reached")

    # More propagation
    prop.propagate(sec_to_days(600))  # 10 minutes total

    # Finalize and save
    summary = fdr.finalize(success=True)
    files = fdr.save()

    print(f"  FDR Log Directory: {fdr.log_dir}")
    print(f"  Telemetry Frames: {len(fdr.telemetry)}")
    print(f"  Events: {len(fdr.events)}")
    print()

    return fdr


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mission Logging Demo")
    parser.add_argument('--mode', choices=['full', 'fdr-only'],
                       default='full', help='Demo mode (default: full)')
    args = parser.parse_args()

    try:
        if args.mode == 'full':
            prop, controller = run_logging_demo()
        else:
            fdr = run_fdr_only_demo()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        import traceback
        print(f"\nDemo failed: {e}")
        traceback.print_exc()
        raise
