#!/usr/bin/env python3
"""
Gravity Assist Mission Example

Demonstrates the astrodynamics module capabilities by designing
a Venus-Earth-Earth gravity assist (VEEGA) trajectory similar to
Galileo's path to Jupiter.

This example shows:
1. Orbital element conversions
2. Interplanetary transfer calculations
3. Gravity assist flyby design
4. B-plane targeting
5. Patched conics trajectory assembly
"""

import numpy as np
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base import (
    BODIES,
    cartesian_to_keplerian, keplerian_to_cartesian, OrbitalElements,
    vis_viva, circular_velocity, escape_velocity, orbital_period,
)
from guidance import (
    compute_flyby, flyby_turn_angle, maximum_turn_angle,
    b_plane_from_periapsis, BPlaneTarget,
    powered_flyby_delta_v, design_flyby_for_target,
    gravity_assist_sequence_delta_v,
    estimate_transfer_delta_v, compute_c3, departure_v_infinity,
    heliocentric_to_planetocentric, SOI_RADII_AU,
)

# ============================================================================
# Constants
# ============================================================================

# Gravitational parameters (km³/s²)
MU_SUN = 1.32712440041e11
MU_EARTH = 398600.4418
MU_VENUS = 324858.592
MU_JUPITER = 126686534.0

# Body radii (km)
R_EARTH = 6378.137
R_VENUS = 6051.8
R_JUPITER = 71492.0

# Orbital velocities (km/s) - approximate
V_EARTH = 29.78
V_VENUS = 35.02
V_JUPITER = 13.07


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# ============================================================================
# Part 1: Orbital Elements Demonstration
# ============================================================================

def demonstrate_orbital_elements():
    """Demonstrate orbital element conversions."""
    print_header("PART 1: ORBITAL ELEMENTS")

    # Create a typical Earth departure orbit (parking orbit before TLI)
    print_subheader("Earth Parking Orbit")

    # Circular LEO at 300 km altitude
    r_leo = R_EARTH + 300  # km
    v_leo = circular_velocity(r_leo, MU_EARTH)

    print(f"Parking orbit altitude: 300 km")
    print(f"Orbital radius: {r_leo:.1f} km")
    print(f"Circular velocity: {v_leo:.3f} km/s")
    print(f"Orbital period: {orbital_period(r_leo, MU_EARTH)/60:.1f} minutes")

    # Define state vector
    r = np.array([r_leo, 0.0, 0.0])
    v = np.array([0.0, v_leo, 0.0])

    # Convert to Keplerian elements
    elements = cartesian_to_keplerian(r, v, MU_EARTH)

    print(f"\nKeplerian Elements:")
    print(f"  Semi-major axis: {elements.a:.1f} km")
    print(f"  Eccentricity: {elements.e:.6f}")
    print(f"  Inclination: {np.degrees(elements.i):.2f}°")
    print(f"  RAAN: {np.degrees(elements.raan):.2f}°")
    print(f"  Arg of Periapsis: {np.degrees(elements.omega):.2f}°")
    print(f"  True Anomaly: {np.degrees(elements.nu):.2f}°")

    # Create elliptical departure orbit
    print_subheader("Trans-Venus Injection")

    # Escape velocity and departure excess
    v_escape = escape_velocity(r_leo, MU_EARTH)
    v_inf_target = 3.5  # km/s (typical for Venus transfer)
    v_departure = np.sqrt(v_escape**2 + v_inf_target**2)

    print(f"Escape velocity at LEO: {v_escape:.3f} km/s")
    print(f"Target V-infinity: {v_inf_target:.2f} km/s")
    print(f"Required departure velocity: {v_departure:.3f} km/s")
    print(f"Delta-V for TVI: {v_departure - v_leo:.3f} km/s")

    # Create hyperbolic departure trajectory
    r_dep = np.array([r_leo, 0.0, 0.0])
    v_dep = np.array([0.0, v_departure, 0.0])

    elements_dep = cartesian_to_keplerian(r_dep, v_dep, MU_EARTH)

    print(f"\nDeparture Hyperbola:")
    print(f"  Semi-major axis: {elements_dep.a:.1f} km (negative = hyperbolic)")
    print(f"  Eccentricity: {elements_dep.e:.4f}")
    print(f"  C3: {v_inf_target**2:.2f} km²/s²")

    return elements


# ============================================================================
# Part 2: Gravity Assist Design
# ============================================================================

def demonstrate_gravity_assist():
    """Demonstrate gravity assist calculations."""
    print_header("PART 2: GRAVITY ASSIST DESIGN")

    # Venus flyby design
    print_subheader("Venus Flyby")

    # Typical Venus flyby parameters
    v_inf_venus = 4.5  # km/s (approach velocity)
    min_altitude = 300  # km (safe flyby altitude)
    rp_venus = R_VENUS + min_altitude

    # Maximum turn angle at minimum altitude
    max_turn = maximum_turn_angle(v_inf_venus, rp_venus, MU_VENUS)
    print(f"V-infinity at Venus: {v_inf_venus:.2f} km/s")
    print(f"Minimum flyby altitude: {min_altitude} km")
    print(f"Maximum turn angle: {np.degrees(max_turn):.1f}°")

    # Design actual flyby at 1000 km altitude
    rp_actual = R_VENUS + 1000
    v_inf_in = np.array([v_inf_venus, 0.0, 0.0])

    flyby = compute_flyby(v_inf_in, rp_actual, MU_VENUS)

    print(f"\nActual flyby (1000 km altitude):")
    print(f"  Periapsis: {flyby.periapsis:.1f} km")
    print(f"  Turn angle: {np.degrees(flyby.turn_angle):.1f}°")
    print(f"  Hyperbolic eccentricity: {flyby.e_hyperbolic:.3f}")
    print(f"  Delta-V gain: {flyby.delta_v_gain:.3f} km/s")

    # B-plane targeting
    print_subheader("B-Plane Targeting")

    b = b_plane_from_periapsis(rp_actual, v_inf_venus, MU_VENUS)
    print(f"B-plane impact parameter: {b:.1f} km")

    # Create B-plane target for specific inclination change
    target = BPlaneTarget.from_periapsis(rp_actual, v_inf_venus, MU_VENUS, theta=np.radians(30))
    print(f"B·T component: {target.b_dot_t:.1f} km")
    print(f"B·R component: {target.b_dot_r:.1f} km")
    print(f"B-plane angle: {np.degrees(target.theta):.1f}°")

    # Earth flyby for comparison
    print_subheader("Earth Flyby (for comparison)")

    v_inf_earth = 4.0  # km/s
    rp_earth = R_EARTH + 500

    flyby_earth = compute_flyby(np.array([v_inf_earth, 0.0, 0.0]), rp_earth, MU_EARTH)

    print(f"V-infinity: {v_inf_earth:.2f} km/s")
    print(f"Periapsis altitude: 500 km")
    print(f"Turn angle: {np.degrees(flyby_earth.turn_angle):.1f}°")
    print(f"Delta-V gain: {flyby_earth.delta_v_gain:.3f} km/s")

    return flyby


# ============================================================================
# Part 3: VEEGA Trajectory
# ============================================================================

def design_veega_trajectory():
    """Design a Venus-Earth-Earth Gravity Assist trajectory."""
    print_header("PART 3: VEEGA TRAJECTORY DESIGN")

    print("""
The VEEGA trajectory (Venus-Earth-Earth Gravity Assist) was used by
the Galileo spacecraft to reach Jupiter. It uses:
  1. Venus flyby to increase orbital energy
  2. First Earth flyby to further boost energy
  3. Second Earth flyby for final Jupiter injection
""")

    # Phase 1: Earth to Venus
    print_subheader("Phase 1: Earth to Venus Transfer")
    transfer_ev = estimate_transfer_delta_v('earth', 'venus')
    print(f"Transfer time: {transfer_ev['transfer_time_days']:.0f} days")
    print(f"Departure V∞: {transfer_ev['v_inf_departure_km_s']:.2f} km/s")
    print(f"Departure C3: {transfer_ev['c3_departure']:.1f} km²/s²")
    print(f"Venus arrival V∞: {transfer_ev['v_inf_arrival_km_s']:.2f} km/s")

    # Phase 2: Venus Flyby
    print_subheader("Phase 2: Venus Gravity Assist")
    v_inf_venus = transfer_ev['v_inf_arrival_km_s']
    rp_venus = R_VENUS + 2000  # 2000 km altitude

    flyby_venus = compute_flyby(
        np.array([v_inf_venus, 0.0, 0.0]),
        rp_venus,
        MU_VENUS
    )
    print(f"Flyby altitude: 2000 km")
    print(f"Turn angle: {np.degrees(flyby_venus.turn_angle):.1f}°")
    print(f"Delta-V gain: {flyby_venus.delta_v_gain:.2f} km/s")

    # Phase 3: Venus to Earth (first return)
    print_subheader("Phase 3: Return to Earth (First)")
    # Post-Venus, spacecraft has higher energy
    v_inf_post_venus = np.linalg.norm(flyby_venus.v_inf_out)
    print(f"Post-Venus V∞: {v_inf_post_venus:.2f} km/s")
    print(f"Transfer time to Earth: ~12 months")

    # Phase 4: First Earth Flyby
    print_subheader("Phase 4: First Earth Gravity Assist")
    v_inf_earth1 = 5.0  # km/s (typical after Venus encounter)
    rp_earth1 = R_EARTH + 500

    flyby_earth1 = compute_flyby(
        np.array([v_inf_earth1, 0.0, 0.0]),
        rp_earth1,
        MU_EARTH
    )
    print(f"Arrival V∞: {v_inf_earth1:.2f} km/s")
    print(f"Flyby altitude: 500 km")
    print(f"Turn angle: {np.degrees(flyby_earth1.turn_angle):.1f}°")
    print(f"Delta-V gain: {flyby_earth1.delta_v_gain:.2f} km/s")

    # Phase 5: Second Earth Flyby (after 2-year loop)
    print_subheader("Phase 5: Second Earth Gravity Assist")
    v_inf_earth2 = 8.0  # km/s (higher energy after first Earth flyby)
    rp_earth2 = R_EARTH + 300

    flyby_earth2 = compute_flyby(
        np.array([v_inf_earth2, 0.0, 0.0]),
        rp_earth2,
        MU_EARTH
    )
    print(f"Arrival V∞: {v_inf_earth2:.2f} km/s")
    print(f"Flyby altitude: 300 km")
    print(f"Turn angle: {np.degrees(flyby_earth2.turn_angle):.1f}°")
    print(f"Delta-V gain: {flyby_earth2.delta_v_gain:.2f} km/s")

    # Phase 6: Earth to Jupiter
    print_subheader("Phase 6: Earth to Jupiter Cruise")
    transfer_ej = estimate_transfer_delta_v('earth', 'jupiter')
    print(f"Transfer time: {transfer_ej['transfer_time_days']:.0f} days ({transfer_ej['transfer_time_days']/365:.1f} years)")
    print(f"Jupiter arrival V∞: {transfer_ej['v_inf_arrival_km_s']:.2f} km/s")

    # Summary
    print_subheader("MISSION SUMMARY")

    total_dv_gain = (flyby_venus.delta_v_gain +
                    flyby_earth1.delta_v_gain +
                    flyby_earth2.delta_v_gain)

    print(f"""
Total gravity assist delta-V: {total_dv_gain:.2f} km/s

This "free" delta-V from gravity assists allows:
  - Smaller launch vehicle (lower C3 requirement)
  - More payload mass to Jupiter
  - Reduced propellant requirements

Without gravity assists, a direct Earth-Jupiter transfer would require
a much larger C3 (~80 km²/s²) versus VEEGA's ~15 km²/s².
""")

    # Gravity assist sequence analysis
    print_subheader("Gravity Assist Sequence Analysis")

    sequence = gravity_assist_sequence_delta_v(
        body_sequence=['venus', 'earth', 'earth'],
        v_inf_sequence=[v_inf_venus, v_inf_earth1, v_inf_earth2],
        rp_sequence=[rp_venus, rp_earth1, rp_earth2],
        mu_sequence=[MU_VENUS, MU_EARTH, MU_EARTH]
    )

    for i, fb in enumerate(sequence['flybys']):
        print(f"  {fb['body'].capitalize():8s}: Turn {fb['turn_angle_deg']:5.1f}°, ΔV gain {fb['delta_v_gain']:.2f} km/s")

    print(f"\n  Total ΔV from gravity assists: {sequence['total_delta_v_gain']:.2f} km/s")

    return sequence


# ============================================================================
# Part 4: Powered Flyby Analysis
# ============================================================================

def analyze_powered_flyby():
    """Analyze powered gravity assist options."""
    print_header("PART 4: POWERED GRAVITY ASSIST")

    print("""
A powered gravity assist combines a flyby with a propulsive maneuver
at periapsis, taking advantage of the Oberth effect for maximum
efficiency.
""")

    print_subheader("Oberth Effect Demonstration")

    # Compare delta-V efficiency at different altitudes
    v_inf_in = 5.0   # km/s
    v_inf_out = 8.0  # km/s (want to increase)

    altitudes = [200, 500, 1000, 2000, 5000]

    print(f"Goal: Increase V∞ from {v_inf_in} to {v_inf_out} km/s")
    print(f"Direct delta-V (no Oberth): {v_inf_out - v_inf_in:.2f} km/s")
    print(f"\nDelta-V required at different flyby altitudes:")
    print(f"{'Altitude (km)':<15} {'Periapsis (km)':<15} {'Delta-V (km/s)':<15} {'Savings':<15}")
    print("-" * 60)

    for alt in altitudes:
        rp = R_EARTH + alt
        dv = powered_flyby_delta_v(v_inf_in, v_inf_out, rp, MU_EARTH)
        savings = (v_inf_out - v_inf_in) - dv
        print(f"{alt:<15} {rp:<15.0f} {dv:<15.3f} {savings:<15.3f}")

    print(f"\nLower altitude = higher velocity at periapsis = more Oberth benefit")


# ============================================================================
# Part 5: SOI Transitions
# ============================================================================

def demonstrate_soi_transitions():
    """Demonstrate sphere of influence concepts."""
    print_header("PART 5: SPHERE OF INFLUENCE")

    print_subheader("SOI Radii of Solar System Bodies")

    bodies = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
    km_per_au = 149597870.7

    print(f"{'Body':<12} {'SOI (AU)':<12} {'SOI (km)':<15} {'SOI (body radii)':<20}")
    print("-" * 60)

    body_radii = {
        'mercury': 2439.7, 'venus': 6051.8, 'earth': 6378.1,
        'mars': 3389.5, 'jupiter': 71492, 'saturn': 58232,
        'uranus': 25362, 'neptune': 24622
    }

    for body in bodies:
        soi_au = SOI_RADII_AU.get(body, 0)
        soi_km = soi_au * km_per_au
        r_body = body_radii.get(body, 1)
        soi_radii = soi_km / r_body

        print(f"{body.capitalize():<12} {soi_au:<12.4f} {soi_km:<15,.0f} {soi_radii:<20.1f}")

    print_subheader("Frame Transformation Example")

    # Spacecraft approaching Earth
    r_sc_helio = np.array([1.005, 0.02, 0.0])  # AU
    v_sc_helio = np.array([-2.0, 28.0, 0.0])   # km/s (approaching)

    r_earth = np.array([1.0, 0.0, 0.0])
    v_earth = np.array([0.0, 29.78, 0.0])

    r_ec, v_ec = heliocentric_to_planetocentric(r_sc_helio, v_sc_helio, r_earth, v_earth)

    print(f"Spacecraft heliocentric position: {r_sc_helio} AU")
    print(f"Spacecraft heliocentric velocity: {v_sc_helio} km/s")
    print(f"\nEarth-centered position: {r_ec} AU")
    print(f"Earth-centered velocity: {v_ec} km/s")
    print(f"\nDistance to Earth: {np.linalg.norm(r_ec) * km_per_au:.0f} km")
    print(f"Earth SOI radius: {SOI_RADII_AU['earth'] * km_per_au:.0f} km")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("    ASTRODYNAMICS MODULE DEMONSTRATION")
    print("    Gravity Assist Mission Design")
    print("=" * 70)

    # Part 1: Orbital Elements
    demonstrate_orbital_elements()

    # Part 2: Gravity Assist
    demonstrate_gravity_assist()

    # Part 3: VEEGA Trajectory
    design_veega_trajectory()

    # Part 4: Powered Flyby
    analyze_powered_flyby()

    # Part 5: SOI Transitions
    demonstrate_soi_transitions()

    print("\n" + "=" * 70)
    print("    DEMONSTRATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
