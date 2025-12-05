#!/usr/bin/env python3
"""
Visualization Demonstration - space_state Package

This script demonstrates the visualization capabilities of the space_state package.
Run this to generate sample plots for validation.

Outputs:
- leo_orbit_3d.png: 3D plot of LEO orbit around Earth
- lunar_transfer.png: Earth-Moon transfer trajectory
- mars_transfer.png: Earth-Mars transfer arc
- ground_track.png: Ground track for ISS-like orbit

Usage:
    python visualization_demo.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')
sys.path.insert(0, '/home/boss/data/space_state')

# Import visualization tools
from visual import (
    OrbitPlotter3D, create_sphere, set_equal_aspect_3d,
    BODY_COLORS, BODY_RADII_KM,
)

# Import orbital mechanics
from base import (
    circular_velocity, orbital_period, vis_viva,
    cartesian_to_keplerian, keplerian_to_cartesian,
)
from base.star_constants import BODIES

# Constants
R_EARTH = 6378.137  # km
R_MOON = 1737.4     # km
MU_EARTH = 398600.4418  # km³/s²
MU_MOON = 4902.8    # km³/s²


def demo_leo_orbit():
    """Generate 3D visualization of LEO orbit."""
    print("Generating LEO orbit visualization...")

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Dark mode styling
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Create Earth sphere
    x, y, z = create_sphere(R_EARTH)
    ax.plot_surface(x, y, z, color='royalblue', alpha=0.8, linewidth=0)

    # Generate ISS-like orbit (400 km, 51.6° inclination)
    altitude = 400  # km
    inclination = np.radians(51.6)
    r = R_EARTH + altitude
    N = 500

    # Orbital period
    T = orbital_period(r, MU_EARTH)

    # Generate orbit points
    theta = np.linspace(0, 2*np.pi, N)
    x_orbit = r * np.cos(theta)
    y_orbit = r * np.sin(theta) * np.cos(inclination)
    z_orbit = r * np.sin(theta) * np.sin(inclination)

    # Plot orbit
    ax.plot(x_orbit, y_orbit, z_orbit, 'cyan', linewidth=2, label='ISS Orbit')

    # Mark current position
    ax.scatter([x_orbit[0]], [y_orbit[0]], [z_orbit[0]], c='red', s=100, label='Spacecraft')

    # Set equal aspect
    max_range = r * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # Labels
    ax.set_xlabel('X (km)', color='white')
    ax.set_ylabel('Y (km)', color='white')
    ax.set_zlabel('Z (km)', color='white')
    ax.tick_params(colors='white')
    ax.set_title(f'ISS Orbit: {altitude} km altitude, {np.degrees(inclination):.1f}° inclination',
                 color='white', fontsize=14)

    # Add info text
    v = circular_velocity(r, MU_EARTH)
    ax.text2D(0.02, 0.98, f'Orbital velocity: {v:.2f} km/s\nPeriod: {T/60:.1f} min',
              transform=ax.transAxes, color='white', fontsize=10, verticalalignment='top')

    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

    plt.savefig('leo_orbit_3d.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print("  Saved: leo_orbit_3d.png")


def demo_lunar_transfer():
    """Generate Earth-Moon transfer visualization."""
    print("Generating lunar transfer visualization...")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Earth
    x, y, z = create_sphere(R_EARTH * 3)  # Scale up for visibility
    ax.plot_surface(x, y, z, color='royalblue', alpha=0.8, linewidth=0)

    # Moon position (scaled)
    moon_distance = 384400  # km
    moon_scale = 1  # Keep real scale
    moon_x = moon_distance
    moon_y = 0
    moon_z = 0

    x_moon, y_moon, z_moon = create_sphere(R_MOON * 15, center=np.array([moon_x, moon_y, moon_z]))
    ax.plot_surface(x_moon, y_moon, z_moon, color='gray', alpha=0.9, linewidth=0)

    # LEO parking orbit (scaled up for visibility)
    r_leo = R_EARTH + 185
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(r_leo * 5 * np.cos(theta), r_leo * 5 * np.sin(theta), np.zeros(100),
            'cyan', linewidth=1, alpha=0.5, label='Parking Orbit')

    # Transfer ellipse
    # a = (r_leo + moon_distance) / 2 for Hohmann-like
    N = 500
    t_param = np.linspace(0, np.pi, N)  # Half ellipse

    # Simplified transfer trajectory
    a_transfer = (r_leo + moon_distance) / 2
    e_transfer = (moon_distance - r_leo) / (moon_distance + r_leo)

    r_transfer = a_transfer * (1 - e_transfer**2) / (1 + e_transfer * np.cos(t_param))
    x_transfer = r_transfer * np.cos(t_param)
    y_transfer = r_transfer * np.sin(t_param)

    ax.plot(x_transfer, y_transfer, np.zeros(N), 'lime', linewidth=2, label='Transfer Trajectory')

    # Mark key points
    ax.scatter([r_leo], [0], [0], c='cyan', s=100, marker='o', label='TLI')
    ax.scatter([moon_x], [moon_y], [moon_z], c='yellow', s=100, marker='*', label='Moon')

    # Set view
    max_range = moon_distance * 1.2
    ax.set_xlim(-max_range * 0.3, max_range)
    ax.set_ylim(-max_range * 0.5, max_range * 0.5)
    ax.set_zlim(-max_range * 0.3, max_range * 0.3)

    ax.set_xlabel('X (km)', color='white')
    ax.set_ylabel('Y (km)', color='white')
    ax.set_zlabel('Z (km)', color='white')
    ax.tick_params(colors='white')
    ax.set_title('Earth-Moon Transfer Trajectory', color='white', fontsize=14)

    # Transfer info
    transfer_time = np.pi * np.sqrt(a_transfer**3 / MU_EARTH) / 3600  # hours
    ax.text2D(0.02, 0.98, f'Transfer time: {transfer_time:.1f} hours\nDistance: {moon_distance:,} km',
              transform=ax.transAxes, color='white', fontsize=10, verticalalignment='top')

    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    ax.view_init(elev=30, azim=45)

    plt.savefig('lunar_transfer.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print("  Saved: lunar_transfer.png")


def demo_ground_track():
    """Generate ground track visualization."""
    print("Generating ground track visualization...")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Generate ISS-like ground track
    altitude = 400  # km
    inclination = 51.6  # degrees
    r = R_EARTH + altitude
    T = orbital_period(r, MU_EARTH)  # seconds

    # Earth rotation rate
    omega_earth = 2 * np.pi / 86164  # rad/s (sidereal day)

    # Simulate 3 orbits
    num_orbits = 3
    N = 1000
    time = np.linspace(0, num_orbits * T, N)

    # Orbital angular velocity
    omega_orbit = 2 * np.pi / T

    # Generate trajectory
    lats = []
    lons = []

    for t in time:
        # True anomaly (simplified circular)
        nu = omega_orbit * t

        # Position in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)

        # Rotate to Earth-fixed frame (simplified)
        # Latitude from inclination
        lat = np.degrees(np.arcsin(np.sin(np.radians(inclination)) * np.sin(nu)))

        # Longitude (RAAN + orbital position - Earth rotation)
        lon_inertial = np.degrees(nu)  # Simplified
        lon = (lon_inertial - np.degrees(omega_earth * t)) % 360
        if lon > 180:
            lon -= 360

        lats.append(lat)
        lons.append(lon)

    lats = np.array(lats)
    lons = np.array(lons)

    # Plot world map background (simplified)
    ax.set_facecolor('darkblue')

    # Plot coastlines (very simplified - just a rectangle for continents)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    # Ground track - handle wrap-around
    # Split track at longitude discontinuities
    lon_diff = np.abs(np.diff(lons))
    split_indices = np.where(lon_diff > 180)[0] + 1

    # Plot each segment
    segments = np.split(np.arange(len(lons)), split_indices)
    for i, seg in enumerate(segments):
        if len(seg) > 1:
            ax.plot(lons[seg], lats[seg], 'yellow', linewidth=1.5,
                   label='Ground Track' if i == 0 else '')

    # Mark current position
    ax.scatter([lons[0]], [lats[0]], c='red', s=100, zorder=5, label='Spacecraft')

    # Ground station markers (DSN)
    stations = {
        'Goldstone': (35.4, -116.9),
        'Canberra': (-35.4, 149.0),
        'Madrid': (40.4, -4.2),
    }
    for name, (lat, lon) in stations.items():
        ax.scatter([lon], [lat], c='lime', s=80, marker='^', zorder=5)
        ax.annotate(name, (lon, lat), textcoords="offset points",
                   xytext=(5, 5), color='lime', fontsize=8)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    ax.set_title(f'ISS Ground Track - {num_orbits} Orbits ({altitude} km, {inclination}° inc)',
                 fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # Add orbit info
    ax.text(0.98, 0.02, f'Period: {T/60:.1f} min\nVelocity: {circular_velocity(r, MU_EARTH):.2f} km/s',
            transform=ax.transAxes, ha='right', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig('ground_track.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ground_track.png")


def demo_interplanetary():
    """Generate Mars transfer visualization (simplified)."""
    print("Generating Mars transfer visualization...")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # AU to km (for display we'll use AU)
    AU = 1.496e8  # km

    # Orbital radii (AU)
    r_earth = 1.0
    r_mars = 1.524

    # Sun
    ax.scatter([0], [0], c='yellow', s=200, zorder=10, label='Sun')

    # Earth orbit
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(r_earth * np.cos(theta), r_earth * np.sin(theta),
            'cyan', linewidth=1, alpha=0.5, label="Earth's Orbit")

    # Mars orbit
    ax.plot(r_mars * np.cos(theta), r_mars * np.sin(theta),
            'red', linewidth=1, alpha=0.5, label="Mars' Orbit")

    # Hohmann transfer
    a_transfer = (r_earth + r_mars) / 2
    e_transfer = (r_mars - r_earth) / (r_mars + r_earth)

    # Transfer arc (half ellipse)
    t_param = np.linspace(0, np.pi, 100)
    r_transfer = a_transfer * (1 - e_transfer**2) / (1 + e_transfer * np.cos(t_param))
    x_transfer = r_transfer * np.cos(t_param)
    y_transfer = r_transfer * np.sin(t_param)

    ax.plot(x_transfer, y_transfer, 'lime', linewidth=2, label='Transfer Trajectory')

    # Departure point (Earth)
    ax.scatter([r_earth], [0], c='blue', s=100, zorder=5, marker='o', label='Earth at Departure')

    # Arrival point (Mars)
    # Mars position at arrival (after transfer time)
    # Transfer time ≈ π * sqrt(a³/μ_sun)
    mars_angle_at_arrival = np.pi - np.radians(44.3)  # Hohmann phase angle
    ax.scatter([r_mars * np.cos(mars_angle_at_arrival)],
               [r_mars * np.sin(mars_angle_at_arrival)],
               c='red', s=100, zorder=5, marker='o', label='Mars at Arrival')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    ax.set_xlabel('X (AU)', color='white', fontsize=12)
    ax.set_ylabel('Y (AU)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.set_title('Earth-Mars Hohmann Transfer', color='white', fontsize=14)

    # Transfer info
    MU_SUN = 1.327e11  # km³/s²
    transfer_time = np.pi * np.sqrt((a_transfer * AU)**3 / MU_SUN) / 86400  # days
    ax.text(0.02, 0.98, f'Transfer time: {transfer_time:.0f} days\nΔV (dep): ~3.6 km/s\nΔV (arr): ~2.1 km/s',
            transform=ax.transAxes, color='white', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='white'))

    ax.legend(loc='lower right', facecolor='black', edgecolor='white', labelcolor='white')

    plt.savefig('mars_transfer.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print("  Saved: mars_transfer.png")


def main():
    """Run all visualization demos."""
    print("\n" + "=" * 60)
    print("  SPACE_STATE VISUALIZATION DEMONSTRATION")
    print("=" * 60 + "\n")

    demo_leo_orbit()
    demo_lunar_transfer()
    demo_ground_track()
    demo_interplanetary()

    print("\n" + "=" * 60)
    print("  VISUALIZATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - leo_orbit_3d.png    : ISS-like LEO orbit")
    print("  - lunar_transfer.png  : Earth-Moon transfer")
    print("  - ground_track.png    : ISS ground track")
    print("  - mars_transfer.png   : Earth-Mars Hohmann transfer")
    print("\nOpen these files to validate the visualizations.")


if __name__ == "__main__":
    main()
