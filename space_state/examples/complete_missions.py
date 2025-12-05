#!/usr/bin/env python3
"""
Complete Mission Examples - space_state Astrodynamics Package

Demonstrates the full capabilities of the space_state package through
realistic mission scenarios similar to NASA missions and KSP gameplay.

Missions Included:
1. LEO to Mars Transfer (Mars 2020 style)
2. LEO to Moon Transfer (Apollo/Artemis style)
3. Earth Reentry (Crew Dragon style)
4. Mars EDL with parachute and thrusters (Perseverance style)
5. Moon Landing with thrusters (Apollo LM style)
6. GEO Satellite Deployment
7. L1/L2 Science Mission (JWST style)
8. Full mission visualization
"""

import numpy as np
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')
sys.path.insert(0, '/home/boss/data/space_state')

# Core imports
from base import (
    BODIES,
    cartesian_to_keplerian, keplerian_to_cartesian, OrbitalElements,
    vis_viva, circular_velocity, escape_velocity, orbital_period,
)
from base.star_constants import AU_TO_KM, DAY_TO_SEC

# Guidance imports
from guidance import (
    # Lambert solver
    lambert_solve,
    # Gravity assist
    compute_flyby, maximum_turn_angle,
    # Patched conics
    estimate_transfer_delta_v, compute_c3, SOI_RADII_AU,
    departure_v_infinity, heliocentric_to_planetocentric,
    # Station keeping
    estimate_geo_station_keeping_budget, sun_earth_lagrange_points,
    earth_moon_lagrange_points, lagrange_station_keeping_budget,
    # Entry/Descent
    ATMOSPHERES, simulate_entry, EntryConditions, mars_edl_profile,
    entry_corridor_bounds, peak_heating_rate, ballistic_coefficient,
    parachute_deployment_conditions,
    # Ground track
    GROUND_STATIONS, ground_track, find_visibility_windows,
    walker_constellation_coverage, sun_synchronous_inclination,
    # Launch windows
    LAUNCH_SITES, launch_azimuth, synodic_period, hohmann_phase_angle,
    interplanetary_launch_opportunities, gto_launch_parameters,
)

# Mission planning
from mission.mission_planner import (
    MissionPhase, MissionPlan, DeltaVBudget, PhaseType, ManeuverType,
)


# ============================================================================
# Constants
# ============================================================================

MU_SUN = 1.32712440041e11    # km³/s²
MU_EARTH = 398600.4418        # km³/s²
MU_MOON = 4902.8              # km³/s²
MU_MARS = 42828.37            # km³/s²

R_EARTH = 6378.137            # km
R_MOON = 1737.4               # km
R_MARS = 3389.5               # km

V_EARTH_ORBIT = 29.78         # km/s (Earth orbital velocity around Sun)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def print_subheader(title: str):
    """Print formatted subsection header."""
    print(f"\n--- {title} ---")


def print_phase(name: str, dv: float = None, duration: float = None):
    """Print mission phase info."""
    info = f"  [{name}]"
    if dv is not None:
        info += f" ΔV = {dv:.3f} km/s"
    if duration is not None:
        if duration > 1:
            info += f" | Duration = {duration:.1f} days"
        else:
            info += f" | Duration = {duration*24:.1f} hours"
    print(info)


# ============================================================================
# MISSION 1: LEO to Mars Transfer
# ============================================================================

def mission_leo_to_mars():
    """
    Mars Transfer Mission - Similar to Mars 2020/Perseverance

    Phase breakdown:
    1. Launch from Earth to LEO parking orbit
    2. Trans-Mars Injection (TMI)
    3. Interplanetary cruise
    4. Mars Orbit Insertion (MOI)
    5. Optional: Descent to Mars surface
    """
    print_header("MISSION 1: EARTH TO MARS TRANSFER")

    print("""
Mission Profile: Mars 2020 / Perseverance Style
- Launch: Atlas V 541 from Cape Canaveral
- Parking orbit: 167 x 167 km, 28.5° inclination
- Transfer: Type I Hohmann-like trajectory
- Mars arrival: Aerocapture or propulsive insertion
""")

    # ===== Phase 1: Launch to LEO =====
    print_subheader("Phase 1: Launch to LEO")

    r_parking = R_EARTH + 185  # km (typical parking orbit)
    v_parking = circular_velocity(r_parking, MU_EARTH)

    print(f"Parking orbit altitude: 185 km")
    print(f"Parking orbit velocity: {v_parking:.3f} km/s")
    print(f"Orbital period: {orbital_period(r_parking, MU_EARTH)/60:.1f} minutes")

    # Launch site
    ksc = LAUNCH_SITES['kennedy']
    # Due east launch achieves minimum inclination = launch site latitude
    print(f"Launch azimuth (due east): 90.0°")
    print(f"Minimum achievable inclination: {ksc.latitude:.1f}°")

    # ===== Phase 2: Trans-Mars Injection =====
    print_subheader("Phase 2: Trans-Mars Injection (TMI)")

    # Get transfer parameters
    transfer = estimate_transfer_delta_v('earth', 'mars')

    print(f"C3 (departure energy): {transfer['c3_departure']:.1f} km²/s²")
    print(f"V-infinity at departure: {transfer['v_inf_departure_km_s']:.2f} km/s")

    # Calculate TMI delta-V from parking orbit
    v_inf_dep = transfer['v_inf_departure_km_s']
    v_escape = escape_velocity(r_parking, MU_EARTH)
    v_tmi = np.sqrt(v_escape**2 + v_inf_dep**2)
    dv_tmi = v_tmi - v_parking

    print(f"Escape velocity at parking orbit: {v_escape:.3f} km/s")
    print(f"Required velocity for TMI: {v_tmi:.3f} km/s")
    print(f"TMI delta-V: {dv_tmi:.3f} km/s")

    # ===== Phase 3: Interplanetary Cruise =====
    print_subheader("Phase 3: Interplanetary Cruise")

    print(f"Transfer time: {transfer['transfer_time_days']:.0f} days ({transfer['transfer_time_days']/30:.1f} months)")
    print(f"Distance traveled: ~480 million km")

    # Trajectory correction maneuvers (typical)
    tcm_budget = 0.015  # km/s typical for Mars missions
    print(f"TCM budget (typical): {tcm_budget*1000:.0f} m/s")

    # ===== Phase 4: Mars Orbit Insertion =====
    print_subheader("Phase 4: Mars Orbit Insertion (MOI)")

    v_inf_arr = transfer['v_inf_arrival_km_s']
    print(f"Mars arrival V-infinity: {v_inf_arr:.2f} km/s")

    # Capture into 250 x 33,000 km orbit (typical science orbit)
    r_periapsis = R_MARS + 250
    r_apoapsis = R_MARS + 33000

    # Velocity at periapsis of arrival hyperbola
    v_periapsis_hyp = np.sqrt(v_inf_arr**2 + 2*MU_MARS/r_periapsis)

    # Velocity at periapsis of capture orbit
    a_capture = (r_periapsis + r_apoapsis) / 2
    v_periapsis_capture = np.sqrt(MU_MARS * (2/r_periapsis - 1/a_capture))

    dv_moi = v_periapsis_hyp - v_periapsis_capture

    print(f"Capture orbit: {r_periapsis-R_MARS:.0f} x {r_apoapsis-R_MARS:.0f} km")
    print(f"Hyperbolic periapsis velocity: {v_periapsis_hyp:.3f} km/s")
    print(f"Capture orbit periapsis velocity: {v_periapsis_capture:.3f} km/s")
    print(f"MOI delta-V: {dv_moi:.3f} km/s")

    # ===== Mission Summary =====
    print_subheader("Mission Delta-V Budget")

    total_dv = dv_tmi + tcm_budget + dv_moi

    print_phase("TMI", dv_tmi)
    print_phase("TCM", tcm_budget)
    print_phase("MOI", dv_moi)
    print(f"\n  TOTAL: {total_dv:.3f} km/s")

    # Propellant mass calculation
    dry_mass = 1025  # kg (Perseverance cruise stage dry mass approx)
    isp = 318  # s (typical bipropellant)

    budget = DeltaVBudget("Mars Transfer", total_dv, margin_percent=10)
    prop_mass = budget.fuel_mass(dry_mass, isp)

    print(f"\n  Propellant required (Isp={isp}s, 10% margin): {prop_mass:.0f} kg")

    return {
        'dv_tmi': dv_tmi,
        'dv_moi': dv_moi,
        'total_dv': total_dv,
        'transfer_days': transfer['transfer_time_days']
    }


# ============================================================================
# MISSION 2: LEO to Moon Transfer
# ============================================================================

def mission_leo_to_moon():
    """
    Lunar Transfer Mission - Apollo/Artemis Style

    Phases:
    1. Launch to LEO parking orbit
    2. Trans-Lunar Injection (TLI)
    3. Lunar coast
    4. Lunar Orbit Insertion (LOI)
    5. Optional: Descent to lunar surface
    """
    print_header("MISSION 2: EARTH TO MOON TRANSFER")

    print("""
Mission Profile: Apollo / Artemis Style
- Parking orbit: 185 km circular
- Transfer: Free-return capable trajectory
- Lunar orbit: 100 km circular (low lunar orbit)
""")

    # Earth-Moon distance
    D_MOON = 384400  # km (mean distance)

    # ===== Phase 1: Parking Orbit =====
    print_subheader("Phase 1: LEO Parking Orbit")

    r_parking = R_EARTH + 185
    v_parking = circular_velocity(r_parking, MU_EARTH)

    print(f"Parking orbit: {r_parking - R_EARTH:.0f} km altitude")
    print(f"Orbital velocity: {v_parking:.3f} km/s")

    # ===== Phase 2: Trans-Lunar Injection =====
    print_subheader("Phase 2: Trans-Lunar Injection (TLI)")

    # TLI puts spacecraft on transfer orbit to Moon
    # Apogee at Moon's distance
    r_apogee = D_MOON
    a_transfer = (r_parking + r_apogee) / 2

    # Velocity at perigee of transfer ellipse
    v_tli = np.sqrt(MU_EARTH * (2/r_parking - 1/a_transfer))
    dv_tli = v_tli - v_parking

    print(f"Transfer orbit apogee: {r_apogee:.0f} km")
    print(f"TLI velocity: {v_tli:.3f} km/s")
    print(f"TLI delta-V: {dv_tli:.3f} km/s")

    # Transfer time (half period of transfer ellipse)
    T_transfer = np.pi * np.sqrt(a_transfer**3 / MU_EARTH)  # seconds

    print(f"Transfer time: {T_transfer/3600:.1f} hours ({T_transfer/86400:.1f} days)")

    # ===== Phase 3: Mid-Course Corrections =====
    print_subheader("Phase 3: Mid-Course Corrections")

    mcc_budget = 0.025  # km/s (typical)
    print(f"MCC budget: {mcc_budget*1000:.0f} m/s")

    # ===== Phase 4: Lunar Orbit Insertion =====
    print_subheader("Phase 4: Lunar Orbit Insertion (LOI)")

    # Velocity relative to Moon at arrival (simplified)
    # At Moon's distance, transfer orbit velocity:
    v_at_moon = np.sqrt(MU_EARTH * (2/D_MOON - 1/a_transfer))

    # Moon's orbital velocity
    v_moon = np.sqrt(MU_EARTH / D_MOON)

    # Relative velocity (simplified - assuming same direction)
    v_rel = v_moon - v_at_moon  # Spacecraft slower than Moon
    v_approach = abs(v_rel)

    # More accurate: spacecraft arrives with ~0.8-1.0 km/s relative velocity
    v_approach = 0.9  # km/s (typical lunar approach velocity)

    print(f"Approach velocity relative to Moon: {v_approach:.2f} km/s")

    # LOI to 100 km circular orbit
    r_loi = R_MOON + 100
    v_circular_moon = circular_velocity(r_loi, MU_MOON)

    # Hyperbolic arrival velocity at periapsis
    v_periapsis_hyp = np.sqrt(v_approach**2 + 2*MU_MOON/r_loi)

    dv_loi = v_periapsis_hyp - v_circular_moon

    print(f"Lunar orbit altitude: {r_loi - R_MOON:.0f} km")
    print(f"Circular orbit velocity: {v_circular_moon:.3f} km/s")
    print(f"LOI delta-V: {dv_loi:.3f} km/s")

    # ===== Mission Summary =====
    print_subheader("Mission Delta-V Budget")

    total_dv = dv_tli + mcc_budget + dv_loi

    print_phase("TLI", dv_tli)
    print_phase("MCC", mcc_budget)
    print_phase("LOI", dv_loi)
    print(f"\n  TOTAL: {total_dv:.3f} km/s")

    # For return trip
    print_subheader("Return Trip (Trans-Earth Injection)")

    # TEI is approximately same as LOI
    dv_tei = dv_loi
    print(f"TEI delta-V: {dv_tei:.3f} km/s")
    print(f"Total round-trip: {total_dv + dv_tei:.3f} km/s")

    return {
        'dv_tli': dv_tli,
        'dv_loi': dv_loi,
        'total_dv': total_dv,
        'transfer_hours': T_transfer/3600
    }


# ============================================================================
# MISSION 3: Earth Reentry
# ============================================================================

def mission_earth_reentry():
    """
    Earth Atmospheric Reentry - Crew Dragon / Orion Style

    Phases:
    1. Deorbit burn from LEO
    2. Entry interface (120 km)
    3. Peak heating
    4. Peak deceleration
    5. Parachute deployment
    6. Splashdown
    """
    print_header("MISSION 3: EARTH ATMOSPHERIC REENTRY")

    print("""
Mission Profile: Crew Dragon / Orion Style Reentry
- Deorbit from 400 km LEO
- Lifting entry (L/D = 0.3)
- Entry corridor: -1° to -3° flight path angle
- Parachute deployment: Mach 2.0, 10 km altitude
""")

    # ===== Phase 1: Deorbit Burn =====
    print_subheader("Phase 1: Deorbit from LEO")

    r_leo = R_EARTH + 400  # ISS altitude
    v_leo = circular_velocity(r_leo, MU_EARTH)

    # Deorbit to entry interface at 120 km
    r_entry = R_EARTH + 120

    # Transfer ellipse with apogee at LEO, perigee at entry
    a_deorbit = (r_leo + r_entry) / 2
    v_deorbit = np.sqrt(MU_EARTH * (2/r_leo - 1/a_deorbit))

    dv_deorbit = v_leo - v_deorbit

    print(f"LEO altitude: {r_leo - R_EARTH:.0f} km")
    print(f"Entry interface: {r_entry - R_EARTH:.0f} km")
    print(f"Deorbit delta-V: {dv_deorbit*1000:.0f} m/s (retrograde)")

    # ===== Phase 2: Entry Conditions =====
    print_subheader("Phase 2: Entry Interface Conditions")

    # Velocity at entry interface
    v_entry = np.sqrt(MU_EARTH * (2/r_entry - 1/a_deorbit))

    print(f"Entry velocity: {v_entry:.3f} km/s ({v_entry*1000:.0f} m/s)")

    # Entry corridor
    steep, shallow = entry_corridor_bounds(v_entry, 'earth', max_decel_g=8.0)
    print(f"Entry corridor (8g limit): {steep:.1f}° to {shallow:.1f}°")

    # Flight path angle at entry (from ellipse geometry)
    # For near-circular to slightly elliptical, FPA is small
    fpa_entry = -2.0  # degrees (typical)
    print(f"Target flight path angle: {fpa_entry:.1f}°")

    # ===== Phase 3: Entry Simulation =====
    print_subheader("Phase 3: Atmospheric Entry")

    # Vehicle parameters (Crew Dragon-like)
    mass = 9500  # kg
    diameter = 4.0  # m
    cd = 1.3  # drag coefficient
    area = np.pi * (diameter/2)**2

    beta = ballistic_coefficient(mass, cd, area)
    print(f"Vehicle mass: {mass} kg")
    print(f"Ballistic coefficient: {beta:.0f} kg/m²")

    # Entry conditions
    entry = EntryConditions(
        velocity=v_entry,
        flight_path_angle=fpa_entry,
        altitude=120.0,
        latitude=0.0,
        longitude=0.0,
        azimuth=90.0
    )

    # Simulate entry
    result = simulate_entry(entry, beta, nose_radius_m=1.0, body='earth', dt=0.5)

    print(f"\nEntry Results:")
    print(f"  Peak deceleration: {result.peak_deceleration_g:.1f} g")
    print(f"  Peak heat rate: {result.peak_heat_rate_w_cm2:.0f} W/cm²")
    print(f"  Total heat load: {result.total_heat_load_j_cm2:.0f} J/cm²")

    # ===== Phase 4: Parachute Deployment =====
    print_subheader("Phase 4: Parachute Deployment")

    chute = parachute_deployment_conditions(mach_target=2.0, altitude_km=10.0, body='earth')

    print(f"Deployment altitude: 10 km")
    print(f"Deployment Mach: 2.0")
    print(f"Velocity at deployment: {chute['velocity_m_s']:.0f} m/s")
    print(f"Dynamic pressure: {chute['dynamic_pressure_pa']:.0f} Pa")

    # ===== Phase 5: Descent and Splashdown =====
    print_subheader("Phase 5: Descent Under Parachute")

    # Terminal velocity under main chutes
    v_terminal = 7.5  # m/s (typical for crewed capsule)
    print(f"Terminal velocity under mains: {v_terminal:.1f} m/s")
    print(f"Splashdown velocity: ~{v_terminal:.0f} m/s")

    # Total timeline
    print_subheader("Timeline Summary")
    print("  T-0:30:00  Deorbit burn")
    print("  T-0:00:00  Entry interface (120 km)")
    print("  T+0:04:00  Peak heating")
    print("  T+0:05:00  Peak deceleration")
    print("  T+0:08:00  Drogue chute deployment (8 km)")
    print("  T+0:10:00  Main chute deployment (2.5 km)")
    print("  T+0:15:00  Splashdown")

    return {
        'dv_deorbit': dv_deorbit,
        'v_entry': v_entry,
        'peak_g': result.peak_deceleration_g,
        'peak_heat': result.peak_heat_rate_w_cm2
    }


# ============================================================================
# MISSION 4: Mars EDL
# ============================================================================

def mission_mars_edl():
    """
    Mars Entry, Descent, and Landing - Perseverance Style

    Phases:
    1. Approach and entry interface
    2. Hypersonic entry (heat shield)
    3. Supersonic parachute deployment
    4. Heat shield jettison
    5. Powered descent (sky crane)
    6. Touchdown
    """
    print_header("MISSION 4: MARS EDL")

    print("""
Mission Profile: Perseverance / MSL Style EDL
- Entry velocity: 5.5 km/s
- Entry angle: -12° (steep for thin atmosphere)
- Supersonic parachute at Mach 2.0
- Sky crane powered descent
- Target: Jezero Crater
""")

    # ===== Mars Atmosphere =====
    print_subheader("Mars Atmospheric Model")

    mars_atm = ATMOSPHERES['mars']
    print(f"Surface density: {mars_atm.surface_density:.4f} kg/m³")
    print(f"Scale height: {mars_atm.scale_height:.1f} km")
    print(f"Entry altitude: {mars_atm.entry_altitude:.0f} km")
    print(f"(Earth surface density: {ATMOSPHERES['earth'].surface_density:.3f} kg/m³)")

    # ===== Full EDL Profile =====
    print_subheader("EDL Profile Generation")

    profile = mars_edl_profile(
        entry_velocity_km_s=5.5,
        entry_fpa_deg=-12.0,
        vehicle_mass_kg=1025,  # Rover + descent stage
        heatshield_diameter_m=4.5
    )

    print(f"\nBallistic coefficient: {profile['ballistic_coefficient_kg_m2']:.0f} kg/m²")

    # Entry results
    entry_result = profile['entry_result']
    print(f"\nEntry Phase:")
    print(f"  Peak deceleration: {entry_result.peak_deceleration_g:.1f} g")
    print(f"  Peak heat rate: {entry_result.peak_heat_rate_w_cm2:.0f} W/cm²")

    # Parachute deployment
    print(f"\nParachute Deployment:")
    chute = profile['parachute_deployment']
    print(f"  Altitude: {chute['altitude_km']:.0f} km")
    print(f"  Velocity: {chute['velocity_m_s']:.0f} m/s")
    print(f"  Mach: {chute['mach_number']:.1f}")

    # ===== Powered Descent =====
    print_subheader("Powered Descent Phase")

    # Skycrane descent parameters
    print("Sky Crane Maneuver:")
    print("  Backshell separation: ~2 km altitude")
    print("  Powered descent initiation: ~1.5 km altitude")
    print("  Rover lowering: ~20 m altitude")
    print("  Touchdown velocity: < 1 m/s")

    # Delta-V for powered descent
    # Rough estimate: decelerate from ~100 m/s to 0 + hover
    v_chute_terminal = 80  # m/s (Mars parachute terminal velocity)
    dv_descent = v_chute_terminal / 1000 + 0.05  # km/s, plus hover margin

    print(f"\n  Terminal velocity under chute: ~{v_chute_terminal} m/s")
    print(f"  Powered descent delta-V: ~{dv_descent*1000:.0f} m/s")

    # ===== Timeline =====
    print_subheader("EDL Timeline")
    timeline = profile['timeline']

    for event, time in timeline.items():
        event_name = event.replace('_', ' ').title()
        print(f"  E+{time:>5.0f}s  {event_name}")

    # ===== Comparison with Earth =====
    print_subheader("Mars vs Earth Entry Comparison")
    print(f"{'Parameter':<25} {'Mars':<15} {'Earth':<15}")
    print("-" * 55)
    print(f"{'Surface density (kg/m³)':<25} {mars_atm.surface_density:<15.4f} {ATMOSPHERES['earth'].surface_density:<15.3f}")
    print(f"{'Entry velocity (km/s)':<25} {'5.5':<15} {'7.8':<15}")
    print(f"{'Entry angle (deg)':<25} {'-12':<15} {'-2':<15}")
    print(f"{'Peak decel (g)':<25} {f'{entry_result.peak_deceleration_g:.1f}':<15} {'~4-8':<15}")
    print(f"{'Parachute useful?':<25} {'Marginal':<15} {'Yes':<15}")
    print(f"{'Powered descent needed?':<25} {'Yes':<15} {'No':<15}")

    return profile


# ============================================================================
# MISSION 5: Moon Landing
# ============================================================================

def mission_moon_landing():
    """
    Lunar Landing Mission - Apollo LM / Artemis HLS Style

    Phases:
    1. Lunar orbit (100 km circular)
    2. Descent Orbit Insertion (DOI)
    3. Powered Descent Initiation (PDI)
    4. Braking phase
    5. Approach phase
    6. Terminal descent
    7. Touchdown
    """
    print_header("MISSION 5: LUNAR LANDING")

    print("""
Mission Profile: Apollo LM / Artemis HLS Style
- Start: 100 km circular lunar orbit
- DOI to 15 x 100 km orbit
- PDI at perilune
- Fully propulsive descent (no atmosphere)
""")

    # ===== Phase 1: Starting Orbit =====
    print_subheader("Phase 1: Lunar Orbit")

    r_orbit = R_MOON + 100  # km
    v_orbit = circular_velocity(r_orbit, MU_MOON)
    T_orbit = orbital_period(r_orbit, MU_MOON)

    print(f"Orbit altitude: 100 km")
    print(f"Orbital velocity: {v_orbit:.3f} km/s ({v_orbit*1000:.0f} m/s)")
    print(f"Orbital period: {T_orbit/60:.1f} minutes")

    # ===== Phase 2: Descent Orbit Insertion =====
    print_subheader("Phase 2: Descent Orbit Insertion (DOI)")

    # Lower perilune to 15 km for PDI
    r_perilune = R_MOON + 15
    r_apolune = r_orbit

    a_doi = (r_perilune + r_apolune) / 2
    v_apolune_doi = np.sqrt(MU_MOON * (2/r_apolune - 1/a_doi))

    dv_doi = v_orbit - v_apolune_doi

    print(f"Descent orbit: {r_perilune-R_MOON:.0f} x {r_apolune-R_MOON:.0f} km")
    print(f"DOI delta-V: {dv_doi*1000:.0f} m/s (retrograde)")

    # ===== Phase 3: Powered Descent =====
    print_subheader("Phase 3: Powered Descent Initiation (PDI)")

    # Velocity at perilune
    v_perilune = np.sqrt(MU_MOON * (2/r_perilune - 1/a_doi))

    print(f"Perilune altitude: 15 km")
    print(f"Velocity at PDI: {v_perilune:.3f} km/s ({v_perilune*1000:.0f} m/s)")

    # Total delta-V needed to land from PDI
    # Must kill all horizontal velocity + gravity losses
    # Gravity loss ~10-15% of ideal delta-V
    gravity_loss_factor = 1.15

    dv_descent_ideal = v_perilune
    dv_descent = dv_descent_ideal * gravity_loss_factor

    print(f"\nPowered Descent:")
    print(f"  Ideal delta-V: {dv_descent_ideal*1000:.0f} m/s")
    print(f"  Gravity losses (~15%): {(dv_descent - dv_descent_ideal)*1000:.0f} m/s")
    print(f"  Total descent delta-V: {dv_descent*1000:.0f} m/s")

    # ===== Descent Phases =====
    print_subheader("Descent Phase Breakdown")

    print("""
  Braking Phase (PDI to 2 km):
    - High thrust, gravity turn
    - Reduce velocity from 1.7 km/s to ~100 m/s
    - Duration: ~8 minutes

  Approach Phase (2 km to 150 m):
    - Pitch up to near-vertical
    - Velocity: 100 m/s → 20 m/s
    - Duration: ~2 minutes

  Terminal Descent (150 m to surface):
    - Manual or automated landing
    - Velocity: 20 m/s → 0
    - Duration: ~1 minute
""")

    # ===== Ascent Budget =====
    print_subheader("Ascent Stage (for return)")

    # Ascent to rendezvous orbit
    r_rendezvous = r_orbit
    v_orbital = circular_velocity(r_rendezvous, MU_MOON)

    # Ascent gravity losses higher due to starting from rest
    gravity_loss_ascent = 1.20
    dv_ascent = v_orbital * gravity_loss_ascent

    print(f"Target orbit: {r_rendezvous - R_MOON:.0f} km circular")
    print(f"Orbital velocity: {v_orbital*1000:.0f} m/s")
    print(f"Ascent delta-V (with losses): {dv_ascent*1000:.0f} m/s")

    # ===== Total Budget =====
    print_subheader("Total Landing Mission Delta-V")

    total_dv = dv_doi + dv_descent + dv_ascent

    print_phase("DOI", dv_doi)
    print_phase("Powered Descent", dv_descent)
    print_phase("Ascent", dv_ascent)
    print(f"\n  TOTAL: {total_dv:.3f} km/s ({total_dv*1000:.0f} m/s)")

    # Propellant calculation
    dry_mass = 4700  # kg (Apollo LM ascent stage)
    isp = 311  # s (Apollo descent engine)

    budget = DeltaVBudget("Lunar Landing", total_dv, margin_percent=5)
    prop_mass = budget.fuel_mass(dry_mass, isp)

    print(f"\n  Propellant required (5% margin): {prop_mass:.0f} kg")

    return {
        'dv_doi': dv_doi,
        'dv_descent': dv_descent,
        'dv_ascent': dv_ascent,
        'total_dv': total_dv
    }


# ============================================================================
# MISSION 6: GEO Satellite Deployment
# ============================================================================

def mission_geo_satellite():
    """
    GEO Communications Satellite Deployment

    Phases:
    1. Launch to GTO
    2. GTO coast
    3. Apogee Kick Motor (AKM) for GEO insertion
    4. Station acquisition
    5. Station keeping
    """
    print_header("MISSION 6: GEO SATELLITE DEPLOYMENT")

    print("""
Mission Profile: Commercial GEO Comsat
- Launch: Falcon 9 / Ariane 6 to GTO
- Transfer: Geostationary Transfer Orbit
- Insertion: Electric or chemical propulsion
- Location: 75°W longitude slot
""")

    # ===== GTO Parameters =====
    print_subheader("Geostationary Transfer Orbit")

    ksc = LAUNCH_SITES['kennedy']
    gto = gto_launch_parameters(ksc, target_longitude=-75.0)

    # Standard GTO parameters
    gto_perigee = gto['perigee_km']
    gto_apogee = gto['apogee_km']

    print(f"Launch site: Kennedy Space Center (28.5°N)")
    print(f"GTO perigee: {gto_perigee:.0f} km")
    print(f"GTO apogee: {gto_apogee:.0f} km")

    # Calculate transfer time (half period of GTO)
    r_perigee = R_EARTH + gto_perigee
    r_apogee = R_EARTH + gto_apogee
    a_gto = (r_perigee + r_apogee) / 2
    T_gto = 2 * np.pi * np.sqrt(a_gto**3 / MU_EARTH)  # seconds
    transfer_time_hours = T_gto / 3600 / 2  # half period to apogee

    print(f"Transfer time to apogee: {transfer_time_hours:.1f} hours")

    # GTO orbital parameters
    a_gto = (r_perigee + r_apogee) / 2

    # Velocity at GTO apogee
    v_apogee_gto = np.sqrt(MU_EARTH * (2/r_apogee - 1/a_gto))

    print(f"\nVelocity at GTO apogee: {v_apogee_gto:.3f} km/s")

    # ===== GEO Insertion =====
    print_subheader("GEO Insertion Maneuver")

    # GEO parameters
    R_GEO = 42164  # km (GEO radius from Earth center)
    r_geo = R_GEO
    v_geo = circular_velocity(r_geo, MU_EARTH)

    print(f"GEO altitude: {r_geo - R_EARTH:.0f} km")
    print(f"GEO orbital velocity: {v_geo:.3f} km/s")
    print(f"GEO orbital period: {orbital_period(r_geo, MU_EARTH)/3600:.1f} hours")

    # Delta-V for circularization at apogee
    dv_circ = v_geo - v_apogee_gto

    print(f"\nApogee Kick Motor (AKM) delta-V: {dv_circ:.3f} km/s")
    print(f"  (This is for co-planar transfer from equatorial launch)")

    # If launched from KSC (28.5°), need plane change
    print_subheader("Plane Change (from KSC 28.5° inclination)")

    inc_launch = 28.5  # degrees
    dv_plane = 2 * v_geo * np.sin(np.radians(inc_launch/2))

    print(f"Launch inclination: {inc_launch}°")
    print(f"Plane change delta-V: {dv_plane:.3f} km/s")

    # Combined maneuver at apogee is more efficient
    dv_combined = np.sqrt(v_geo**2 + v_apogee_gto**2 -
                          2*v_geo*v_apogee_gto*np.cos(np.radians(inc_launch)))

    print(f"\nCombined circularization + plane change: {dv_combined:.3f} km/s")
    print(f"Savings vs separate maneuvers: {(dv_circ + dv_plane) - dv_combined:.3f} km/s")

    # ===== Station Keeping =====
    print_subheader("Station Keeping Budget")

    sk_budget = estimate_geo_station_keeping_budget(
        inclination_deadband=0.05,
        longitude_deadband=0.1,
        area_to_mass=0.02
    )

    print(f"Annual delta-V budget:")
    print(f"  North-South (inclination): {sk_budget.north_south:.1f} m/s/year")
    print(f"  East-West (longitude): {sk_budget.east_west:.1f} m/s/year")
    print(f"  Eccentricity control: {sk_budget.eccentricity:.1f} m/s/year")
    print(f"  TOTAL: {sk_budget.total:.1f} m/s/year")
    print(f"\nManeuvers per year: {sk_budget.maneuvers_per_year}")

    # 15-year mission lifetime
    lifetime_years = 15
    total_sk = sk_budget.total * lifetime_years / 1000  # km/s

    print(f"\n{lifetime_years}-year mission station keeping: {total_sk:.2f} km/s")

    # ===== Total Budget =====
    print_subheader("Total Mission Delta-V Budget")

    total_dv = dv_combined + total_sk

    print_phase("GEO Insertion (combined)", dv_combined)
    print_phase(f"Station Keeping ({lifetime_years}yr)", total_sk)
    print(f"\n  TOTAL: {total_dv:.3f} km/s")

    # Satellite mass calculation
    dry_mass = 2000  # kg (typical GEO comsat)

    # Chemical insertion
    isp_chem = 320  # s
    budget_chem = DeltaVBudget("GEO Chemical", total_dv, margin_percent=5)
    prop_chem = budget_chem.fuel_mass(dry_mass, isp_chem)

    # Electric propulsion (for comparison)
    isp_ep = 1500  # s
    budget_ep = DeltaVBudget("GEO Electric", total_dv, margin_percent=5)
    prop_ep = budget_ep.fuel_mass(dry_mass, isp_ep)

    print(f"\nPropellant comparison (dry mass {dry_mass} kg):")
    print(f"  Chemical (Isp={isp_chem}s): {prop_chem:.0f} kg")
    print(f"  Electric (Isp={isp_ep}s): {prop_ep:.0f} kg")

    return {
        'dv_insertion': dv_combined,
        'dv_sk_annual': sk_budget.total,
        'total_dv': total_dv
    }


# ============================================================================
# MISSION 7: L1/L2 Science Mission
# ============================================================================

def mission_lagrange_point():
    """
    L2 Science Mission - JWST Style

    Phases:
    1. Launch to trans-L2 trajectory
    2. Mid-course corrections
    3. L2 orbit insertion
    4. Station keeping at L2
    5. Science operations
    """
    print_header("MISSION 7: SUN-EARTH L2 SCIENCE MISSION")

    print("""
Mission Profile: JWST Style L2 Observatory
- Launch: Direct injection to L2 transfer
- Destination: Sun-Earth L2 Lagrange point
- Orbit: Halo orbit around L2
- Lifetime: 10+ years with station keeping
""")

    # ===== L2 Location =====
    print_subheader("Sun-Earth L2 Lagrange Point")

    se_points = sun_earth_lagrange_points()
    l2 = se_points['L2']

    print(f"L2 distance from Earth: {l2.distance_km:,.0f} km")
    print(f"L2 distance from Earth: {l2.distance_km/1e6:.2f} million km")
    print(f"L2 is {'stable' if l2.stable else 'unstable'} (requires station keeping)")

    # Also show L1 for comparison
    l1 = se_points['L1']
    print(f"\nL1 distance from Earth: {l1.distance_km:,.0f} km")

    # ===== Transfer Trajectory =====
    print_subheader("Transfer to L2")

    # L2 transfer is essentially a weak escape trajectory
    # C3 is very low, near zero
    c3_l2 = 0.5  # km²/s² (typical for L2 missions)

    r_parking = R_EARTH + 185
    v_parking = circular_velocity(r_parking, MU_EARTH)

    # Departure velocity
    v_inf = np.sqrt(c3_l2)
    v_escape = escape_velocity(r_parking, MU_EARTH)
    v_departure = np.sqrt(v_escape**2 + v_inf**2)

    dv_tli = v_departure - v_parking

    print(f"Departure C3: {c3_l2:.2f} km²/s²")
    print(f"V-infinity: {v_inf:.3f} km/s")
    print(f"Trans-L2 injection delta-V: {dv_tli:.3f} km/s")

    # Transfer time
    transfer_days = 30  # Typical for L2 transfers
    print(f"Transfer time: ~{transfer_days} days")

    # ===== Mid-Course Corrections =====
    print_subheader("Mid-Course Corrections")

    mcc_budget = 0.025  # km/s
    print(f"MCC budget: {mcc_budget*1000:.0f} m/s")
    print("(Used for trajectory cleanup and halo orbit targeting)")

    # ===== L2 Orbit Insertion =====
    print_subheader("Halo Orbit Insertion")

    # L2 orbit insertion is a small maneuver to enter halo orbit
    dv_loi = 0.010  # km/s (typical)
    print(f"Halo orbit insertion: {dv_loi*1000:.0f} m/s")

    # Halo orbit parameters
    print(f"\nHalo Orbit Parameters:")
    print(f"  Semi-major axis: ~800,000 km (perpendicular to Sun-Earth line)")
    print(f"  Orbital period: ~6 months")
    print(f"  This keeps JWST in perpetual shadow of Earth for cooling")

    # ===== Station Keeping =====
    print_subheader("L2 Station Keeping")

    sk_annual = lagrange_station_keeping_budget('L2', 'sun-earth')

    print(f"Annual station keeping: {sk_annual:.1f} m/s/year")
    print(f"(L2 is unstable - requires regular corrections)")

    # Compare with other Lagrange points
    print(f"\nStation Keeping Comparison (m/s/year):")
    for point in ['L1', 'L2', 'L4', 'L5']:
        sk = lagrange_station_keeping_budget(point, 'sun-earth')
        print(f"  {point}: {sk:.1f} m/s/year")

    # ===== Mission Lifetime Budget =====
    print_subheader("Mission Lifetime Budget")

    lifetime = 20  # years (JWST goal)

    total_sk = sk_annual * lifetime / 1000  # km/s
    total_dv = dv_tli + mcc_budget + dv_loi + total_sk

    print(f"Mission lifetime: {lifetime} years")
    print_phase("Trans-L2 Injection", dv_tli)
    print_phase("Mid-Course Corrections", mcc_budget)
    print_phase("Halo Orbit Insertion", dv_loi)
    print_phase(f"Station Keeping ({lifetime}yr)", total_sk)
    print(f"\n  TOTAL: {total_dv:.3f} km/s")

    # JWST-like mass analysis
    print_subheader("Propellant Analysis (JWST-like)")

    dry_mass = 6200  # kg (JWST without propellant)
    prop_mass = 240  # kg (JWST propellant load)
    isp = 220  # s (JWST hydrazine thrusters)

    # How long can JWST last?
    g0 = 9.80665
    ve = isp * g0  # m/s
    total_mass = dry_mass + prop_mass

    # Tsiolkovsky for available delta-V
    dv_available = ve * np.log(total_mass / dry_mass) / 1000  # km/s

    # Subtract transfer and insertion
    dv_for_sk = dv_available - dv_tli - mcc_budget - dv_loi
    if dv_for_sk < 0:
        print(f"(Using JWST values - most delta-V comes from Ariane 5)")
        dv_for_sk = 0.150  # JWST reserved for SK

    years_possible = (dv_for_sk * 1000) / sk_annual

    print(f"Propellant load: {prop_mass} kg")
    print(f"Available delta-V: ~{dv_available*1000:.0f} m/s")
    print(f"Delta-V for station keeping: ~{dv_for_sk*1000:.0f} m/s")
    print(f"Potential lifetime: ~{years_possible:.0f} years")

    return {
        'dv_transfer': dv_tli,
        'dv_sk_annual': sk_annual,
        'total_dv': total_dv
    }


# ============================================================================
# MISSION 8: Visualization Example
# ============================================================================

def demonstrate_visualization():
    """
    Demonstrate visualization capabilities for mission planning.
    """
    print_header("VISUALIZATION CAPABILITIES")

    print("""
The space_state package includes comprehensive visualization tools:

1. 3D Orbit Plotting (OrbitPlotter3D)
   - Bodies with proper colors and scaling
   - Trajectory lines with markers
   - Multiple trajectories on same plot
   - Dark mode for presentations

2. Ground Track Visualization
   - Latitude/longitude traces
   - Ground station visibility
   - Coverage footprints

3. Animation Support
   - Animated orbit propagation
   - Trajectory playback
   - Multi-body animations

4. Interactive Plots (Plotly)
   - Zoomable/rotatable 3D views
   - Hover information
   - Export to HTML
""")

    print_subheader("Available Visualization Functions")

    print("""
from visual import (
    # Core plotter class
    OrbitPlotter3D,

    # Static plots
    plot_orbit_3d,
    plot_trajectory_3d,
    plot_body,
    plot_reference_frame,

    # Animation
    create_orbit_animation,
    create_trajectory_animation,

    # Ground tracks
    plot_ground_track,

    # Multi-body
    plot_solar_system,
    plot_earth_moon,

    # Interactive (Plotly)
    create_interactive_orbit,
    create_interactive_trajectory,

    # Utilities
    set_equal_aspect_3d,
    create_sphere,
    BODY_COLORS,
    BODY_RADII_KM,
)
""")

    print_subheader("Example: Plotting LEO Orbit")

    print("""
# Example code:
import numpy as np
from visual import OrbitPlotter3D, BODY_RADII_KM

# Create plotter
plotter = OrbitPlotter3D(title="LEO Orbit", dark_mode=True)

# Add Earth
plotter.add_body('earth')

# Generate circular orbit positions
N = 100
r = BODY_RADII_KM['earth'] + 400  # 400 km altitude
theta = np.linspace(0, 2*np.pi, N)
positions = np.array([
    r * np.cos(theta),
    r * np.sin(theta),
    np.zeros(N)
])

# Add trajectory
plotter.add_trajectory(positions, color='cyan', label='ISS Orbit')

# Show or save
plotter.show()
# plotter.save('leo_orbit.png')
""")

    return True


# ============================================================================
# Gap Analysis and Roadmap
# ============================================================================

def analyze_gaps():
    """
    Analyze gaps between current implementation and full NASA/KSP capability.
    """
    print_header("FEATURE GAP ANALYSIS")

    print("""
Comparing space_state to NASA-grade simulation and KSP:

============ IMPLEMENTED (COMPLETE) ============
""")

    implemented = [
        "Orbital mechanics (Kepler, vis-viva, Hohmann transfers)",
        "Lambert solver for transfer orbit design",
        "Gravity assist / flyby calculations",
        "Patched conics and SOI transitions",
        "Station keeping budgets (GEO, Lagrange)",
        "Eclipse modeling and prediction",
        "Thermal analysis (solar flux, equilibrium temperature)",
        "Entry/Descent/Landing (Earth, Mars, Venus, Titan)",
        "Ground track and coverage analysis",
        "Launch window calculations",
        "3D visualization with animation",
        "Porkchop plot generation",
        "Monte Carlo uncertainty analysis",
        "Mission planning framework",
        "Trajectory optimization",
    ]

    for item in implemented:
        print(f"  [x] {item}")

    print("""

============ PARTIALLY IMPLEMENTED ============
""")

    partial = [
        ("Full n-body propagation", "Works, but could use adaptive SOI"),
        ("Attitude dynamics", "Quaternion propagation exists, needs controllers"),
        ("Finite burn maneuvers", "Basic support, needs better guidance"),
        ("Proximity operations", "Basic relative motion, needs full rendezvous"),
    ]

    for item, note in partial:
        print(f"  [~] {item}")
        print(f"      Note: {note}")

    print("""

============ NEEDED FOR NASA-GRADE ============
""")

    missing_nasa = [
        "High-fidelity force models (SRP, atmospheric drag, J2-J6)",
        "Precise ephemeris integration (DE440 direct)",
        "Ground station scheduling optimization",
        "Conjunction assessment and collision avoidance",
        "Orbit determination from tracking data",
        "Covariance propagation and maneuver planning",
        "Link budget and communication analysis",
        "Power system sizing with solar array pointing",
        "Detailed thermal network modeling",
        "Structural dynamics and flex modes",
        "Propulsion system modeling (tank pressurization, etc.)",
    ]

    for item in missing_nasa:
        print(f"  [ ] {item}")

    print("""

============ NEEDED FOR KSP-STYLE ============
""")

    missing_ksp = [
        "Real-time flight simulation with physics",
        "Interactive maneuver node planning",
        "Launch vehicle staging simulation",
        "Aerodynamic lift/drag during ascent",
        "Rocket equation with staging",
        "Surface operations (rovers, EVA)",
        "Resource management (fuel, power, life support)",
        "Contract/mission objective system",
        "Career/science mode progression",
        "Part assembly and craft design",
    ]

    for item in missing_ksp:
        print(f"  [ ] {item}")

    print("""

============ PRIORITY RECOMMENDATIONS ============

For NASA-like mission planning:
1. Add high-fidelity force models (J2, SRP, drag)
2. Implement orbit determination
3. Add conjunction assessment

For KSP-like gameplay:
1. Add real-time physics simulation
2. Implement staging and ascent guidance
3. Add interactive maneuver planning UI

Current package is excellent for:
- Mission feasibility studies
- Delta-V budgeting
- Transfer trajectory design
- Ground track analysis
- Thermal/power preliminary sizing
""")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all mission demonstrations."""
    print("\n" + "=" * 75)
    print("  SPACE_STATE COMPLETE MISSION DEMONSTRATIONS")
    print("  Astrodynamics for Mission Design")
    print("=" * 75)

    # Run each mission
    results = {}

    results['mars'] = mission_leo_to_mars()
    results['moon'] = mission_leo_to_moon()
    results['reentry'] = mission_earth_reentry()
    results['mars_edl'] = mission_mars_edl()
    results['moon_landing'] = mission_moon_landing()
    results['geo'] = mission_geo_satellite()
    results['l2'] = mission_lagrange_point()

    # Visualization demo
    demonstrate_visualization()

    # Gap analysis
    analyze_gaps()

    # Summary
    print_header("MISSION SUMMARY")

    print("""
Mission Type                    Total Delta-V    Duration
----------------------------------------------------------------""")
    print(f"LEO to Mars Transfer            {results['mars']['total_dv']:.2f} km/s        {results['mars']['transfer_days']:.0f} days")
    print(f"LEO to Moon Transfer            {results['moon']['total_dv']:.2f} km/s        {results['moon']['transfer_hours']:.0f} hours")
    print(f"Earth Reentry (from LEO)        {results['reentry']['dv_deorbit']:.3f} km/s       ~15 min EDL")
    print(f"Moon Landing (from LLO)         {results['moon_landing']['total_dv']:.2f} km/s        ~12 min descent")
    print(f"GEO Satellite (15yr life)       {results['geo']['total_dv']:.2f} km/s        15 years")
    print(f"L2 Science Mission (20yr)       {results['l2']['total_dv']:.2f} km/s        20 years")

    print("\n" + "=" * 75)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 75 + "\n")

    return results


if __name__ == "__main__":
    main()
