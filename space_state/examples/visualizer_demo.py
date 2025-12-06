#!/usr/bin/env python3
"""
Visualizer Demo Script

Demonstrates the interactive visualization capabilities of space_state.
Run this script to launch an interactive mission viewer in your browser.

Usage:
    python visualizer_demo.py [--mode MODE]

Modes:
    orbit     - Simple orbit visualization with time slider
    dashboard - Full mission dashboard with multiple panels
    static    - Generate static plots (no web server)
"""

import sys
import os
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual import (
    TrajectoryData,
    BodyData,
    EventMarker,
    create_interactive_trajectory_viewer,
    create_mission_dashboard,
    create_attitude_visualizer,
    create_interactive_ground_track,
    create_mission_timeline,
    run_mission_viewer,
    plot_orbit_3d,
)

def generate_demo_orbit(n_points=500, orbits=3):
    """Generate a simple elliptical orbit for demonstration."""
    # Orbital parameters (LEO-like)
    a = 7000  # km
    e = 0.1
    i = 45 * np.pi / 180  # 45 deg inclination

    t = np.linspace(0, orbits * 2 * np.pi, n_points)

    # Orbital radius
    r = a * (1 - e**2) / (1 + e * np.cos(t))

    # Position in orbital plane
    x_orb = r * np.cos(t)
    y_orb = r * np.sin(t)

    # Rotate to 3D (inclination only for simplicity)
    x = x_orb
    y = y_orb * np.cos(i)
    z = y_orb * np.sin(i)

    # Velocity (simplified)
    v_mag = np.sqrt(398600 / r)  # vis-viva
    vx = -v_mag * np.sin(t)
    vy = v_mag * np.cos(t) * np.cos(i)
    vz = v_mag * np.cos(t) * np.sin(i)

    # Time in days
    T_orbital = 2 * np.pi * np.sqrt(a**3 / 398600) / 86400  # days
    time_days = np.linspace(0, orbits * T_orbital, n_points)

    # Mass (constant for this demo)
    mass = np.ones(n_points) * 1000

    # Quaternions (simple rotation)
    q = np.zeros((n_points, 4))
    q[:, 0] = 1.0  # Identity quaternion

    return TrajectoryData(
        times=time_days,
        positions=np.column_stack([x, y, z]),
        velocities=np.column_stack([vx, vy, vz]),
        masses=mass,
        quaternions=q,
        name="Demo Spacecraft",
        color="cyan"
    )


def generate_transfer_trajectory(n_points=300):
    """Generate a Hohmann transfer demonstration."""
    # LEO to GEO transfer
    r1 = 7000   # LEO radius km
    r2 = 42164  # GEO radius km

    # Transfer ellipse parameters
    a_transfer = (r1 + r2) / 2
    e_transfer = (r2 - r1) / (r2 + r1)

    # Full transfer (180 deg)
    theta = np.linspace(0, np.pi, n_points)
    r = a_transfer * (1 - e_transfer**2) / (1 + e_transfer * np.cos(theta))

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(n_points)

    # Time
    T_transfer = np.pi * np.sqrt(a_transfer**3 / 398600) / 86400  # days
    time_days = np.linspace(0, T_transfer, n_points)

    # Velocity (Keplerian)
    v = np.sqrt(398600 * (2/r - 1/a_transfer))
    vx = -v * np.sin(theta)
    vy = v * np.cos(theta)
    vz = np.zeros(n_points)

    mass = np.linspace(1000, 800, n_points)  # Burn at start

    q = np.zeros((n_points, 4))
    for i, angle in enumerate(theta):
        q[i] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]

    return TrajectoryData(
        times=time_days,
        positions=np.column_stack([x, y, z]),
        velocities=np.column_stack([vx, vy, vz]),
        masses=mass,
        quaternions=q,
        name="Hohmann Transfer",
        color="orange"
    )


def create_demo_events(trajectory: TrajectoryData):
    """Create demonstration events with positions from trajectory."""
    times = trajectory.times
    pos = trajectory.positions

    # Find trajectory indices for events
    launch_idx = 0
    burn_idx = np.argmin(np.abs(times - 0.001))
    apogee_idx = np.argmin(np.abs(times - 0.15))
    insert_idx = np.argmin(np.abs(times - 0.19))

    return [
        EventMarker(time=0.0, position=pos[launch_idx], name="Launch", color="green"),
        EventMarker(time=0.001, position=pos[burn_idx], name="TMI Burn", color="orange"),
        EventMarker(time=0.15, position=pos[apogee_idx], name="Apogee", color="yellow"),
        EventMarker(time=0.19, position=pos[insert_idx], name="GEO Insertion", color="red"),
    ]


def demo_orbit_viewer():
    """Run the interactive orbit viewer demo."""
    print("Generating demo orbit...")
    trajectory = generate_demo_orbit()

    earth = BodyData(
        name="Earth",
        positions=np.array([0, 0, 0]),
        radius_km=6371.0,
        color="blue"
    )

    print("Creating interactive viewer...")
    fig = create_interactive_trajectory_viewer(
        trajectory=trajectory,
        bodies=[earth],
        title="Interactive Orbit Demo"
    )

    fig.show()


def demo_mission_dashboard():
    """Run the full mission dashboard demo."""
    print("Generating demo trajectory...")
    trajectory = generate_transfer_trajectory()

    print("Creating mission dashboard...")
    fig = create_mission_dashboard(
        trajectory=trajectory,
        central_body='earth',
        show_elements=True,
    )

    fig.show()


def demo_static_plots():
    """Generate static plots (no web server)."""
    print("Generating demo orbit...")
    trajectory = generate_demo_orbit()

    print("Creating 3D orbit plot...")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(trajectory.positions[:, 0],
            trajectory.positions[:, 1],
            trajectory.positions[:, 2],
            'b-', linewidth=1, label='Orbit')

    # Plot Earth
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    r = 6371
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    ax.plot_surface(x, y, z, alpha=0.3, color='blue')

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Demo Orbit Visualization')
    ax.legend()

    # Equal aspect ratio
    max_range = np.max(np.abs(trajectory.positions)) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    plt.savefig('demo_orbit.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_orbit.png")

    # Attitude plot
    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))

    time_hrs = trajectory.times * 24

    # Quaternion components
    axes[0, 0].plot(time_hrs, trajectory.quaternions[:, 0], label='q0 (scalar)')
    axes[0, 0].plot(time_hrs, trajectory.quaternions[:, 1], label='q1')
    axes[0, 0].plot(time_hrs, trajectory.quaternions[:, 2], label='q2')
    axes[0, 0].plot(time_hrs, trajectory.quaternions[:, 3], label='q3')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Quaternion')
    axes[0, 0].set_title('Attitude Quaternion')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Position magnitude
    r_mag = np.linalg.norm(trajectory.positions, axis=1)
    axes[0, 1].plot(time_hrs, r_mag)
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Altitude (km)')
    axes[0, 1].set_title('Orbital Radius')
    axes[0, 1].grid(True, alpha=0.3)

    # Velocity magnitude
    v_mag = np.linalg.norm(trajectory.velocities, axis=1)
    axes[1, 0].plot(time_hrs, v_mag)
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Velocity (km/s)')
    axes[1, 0].set_title('Orbital Velocity')
    axes[1, 0].grid(True, alpha=0.3)

    # Mass
    axes[1, 1].plot(time_hrs, trajectory.masses)
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Mass (kg)')
    axes[1, 1].set_title('Spacecraft Mass')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_metrics.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_metrics.png")

    plt.show()


def demo_full_viewer():
    """Run the full interactive mission viewer (Dash app)."""
    print("Generating trajectory data...")
    trajectory = generate_transfer_trajectory()
    events = create_demo_events(trajectory)

    earth = BodyData(
        name="Earth",
        positions=np.array([0, 0, 0]),
        radius_km=6371.0,
        color="blue"
    )

    print("\nLaunching Mission Viewer...")
    print("Open your browser to: http://127.0.0.1:8051")
    print("Press Ctrl+C to stop\n")

    run_mission_viewer(
        trajectory=trajectory,
        bodies=[earth],
        events=events,
        port=8051,
    )


def main():
    parser = argparse.ArgumentParser(description="Space State Visualizer Demo")
    parser.add_argument('--mode', choices=['orbit', 'dashboard', 'static', 'full'],
                       default='static',
                       help='Visualization mode (default: static)')
    args = parser.parse_args()

    print("=" * 60)
    print("Space State Visualizer Demo")
    print("=" * 60)

    if args.mode == 'orbit':
        demo_orbit_viewer()
    elif args.mode == 'dashboard':
        demo_mission_dashboard()
    elif args.mode == 'static':
        demo_static_plots()
    elif args.mode == 'full':
        demo_full_viewer()

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
