"""
Tests for 3D Visualization Module
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from visual.orbit_3d import (
    OrbitPlotter3D,
    create_sphere,
    set_equal_aspect_3d,
    BODY_COLORS,
    BODY_RADII_KM,
)


class TestCreateSphere:
    """Test sphere creation utility."""

    def test_sphere_dimensions(self):
        """Test sphere mesh has correct dimensions."""
        x, y, z = create_sphere(radius=1.0, n_points=20)

        assert x.shape == (20, 20)
        assert y.shape == (20, 20)
        assert z.shape == (20, 20)

    def test_sphere_radius(self):
        """Test sphere has correct radius."""
        radius = 5.0
        x, y, z = create_sphere(radius=radius)

        # Points should be at distance = radius from center
        r = np.sqrt(x**2 + y**2 + z**2)
        np.testing.assert_allclose(r, radius, rtol=1e-10)

    def test_sphere_with_center(self):
        """Test sphere with offset center."""
        center = np.array([10.0, 20.0, 30.0])
        x, y, z = create_sphere(radius=1.0, center=center)

        # Points should be at distance = 1 from center
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        np.testing.assert_allclose(r, 1.0, rtol=1e-10)


class TestSetEqualAspect:
    """Test equal aspect ratio utility."""

    def test_sets_equal_limits(self):
        """Test that limits are set correctly."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.array([0, 10])
        y = np.array([0, 5])
        z = np.array([0, 2])

        set_equal_aspect_3d(ax, x, y, z)

        # All ranges should be equal (= max range = 10)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        assert xlim[1] - xlim[0] == ylim[1] - ylim[0]
        assert ylim[1] - ylim[0] == zlim[1] - zlim[0]

        plt.close(fig)


class TestBodyConstants:
    """Test body color and radius constants."""

    def test_body_colors_exist(self):
        """Test common bodies have colors defined."""
        assert 'earth' in BODY_COLORS
        assert 'moon' in BODY_COLORS
        assert 'mars barycenter' in BODY_COLORS

    def test_body_radii_exist(self):
        """Test common bodies have radii defined."""
        assert 'earth' in BODY_RADII_KM
        assert 'moon' in BODY_RADII_KM

    def test_earth_radius_correct(self):
        """Test Earth radius is approximately correct."""
        assert 6300 < BODY_RADII_KM['earth'] < 6400

    def test_moon_radius_correct(self):
        """Test Moon radius is approximately correct."""
        assert 1700 < BODY_RADII_KM['moon'] < 1800


class TestOrbitPlotter3D:
    """Test OrbitPlotter3D class."""

    def test_initialization(self):
        """Test plotter initialization."""
        plotter = OrbitPlotter3D(title="Test", dark_mode=True)

        assert plotter.title == "Test"
        assert plotter.dark_mode
        assert plotter.fig is not None
        assert plotter.ax is not None

        plt.close(plotter.fig)

    def test_add_body(self):
        """Test adding body to plotter."""
        plotter = OrbitPlotter3D()
        plotter.add_body('earth')

        assert len(plotter._bodies) == 1
        assert plotter._bodies[0]['name'] == 'earth'

        plt.close(plotter.fig)

    def test_add_trajectory(self):
        """Test adding trajectory to plotter."""
        plotter = OrbitPlotter3D()

        # Create simple circular trajectory
        t = np.linspace(0, 2*np.pi, 100)
        positions = np.array([
            7000 * np.cos(t),
            7000 * np.sin(t),
            np.zeros_like(t)
        ])

        plotter.add_trajectory(positions, label='Test Orbit')

        assert len(plotter._trajectories) == 1
        assert plotter._trajectories[0]['label'] == 'Test Orbit'

        plt.close(plotter.fig)

    def test_add_marker(self):
        """Test adding marker to plotter."""
        plotter = OrbitPlotter3D()
        plotter.add_marker(np.array([1000, 0, 0]), label='Test Marker')

        assert len(plotter._markers) == 1
        assert plotter._markers[0]['label'] == 'Test Marker'

        plt.close(plotter.fig)

    def test_add_multiple_elements(self):
        """Test adding multiple elements."""
        plotter = OrbitPlotter3D()

        plotter.add_body('earth')
        plotter.add_body('moon', position=np.array([384400, 0, 0]))

        t = np.linspace(0, 2*np.pi, 50)
        traj1 = np.array([7000 * np.cos(t), 7000 * np.sin(t), np.zeros_like(t)])
        traj2 = np.array([10000 * np.cos(t), 10000 * np.sin(t), 1000 * np.ones_like(t)])

        plotter.add_trajectory(traj1, label='LEO')
        plotter.add_trajectory(traj2, label='Transfer', color='red')

        assert len(plotter._bodies) == 2
        assert len(plotter._trajectories) == 2

        plt.close(plotter.fig)

    def test_trajectory_shape_handling(self):
        """Test trajectory handles both (3,n) and (n,3) shapes."""
        plotter = OrbitPlotter3D()

        # Shape (3, n)
        traj1 = np.random.randn(3, 100) * 1000
        plotter.add_trajectory(traj1, label='Shape 3xN')

        # Shape (n, 3)
        traj2 = np.random.randn(100, 3) * 1000
        plotter.add_trajectory(traj2, label='Shape Nx3')

        # Both should be stored as (3, n)
        assert plotter._trajectories[0]['positions'].shape[0] == 3
        assert plotter._trajectories[1]['positions'].shape[0] == 3

        plt.close(plotter.fig)


class TestOrbitPlotterRendering:
    """Test OrbitPlotter3D rendering (non-interactive)."""

    def test_render_simple_orbit(self):
        """Test rendering a simple orbit."""
        plotter = OrbitPlotter3D(title="Simple Orbit")
        plotter.add_body('earth')

        t = np.linspace(0, 2*np.pi, 100)
        positions = np.array([
            7000 * np.cos(t),
            7000 * np.sin(t),
            np.zeros_like(t)
        ])
        plotter.add_trajectory(positions)

        # Draw elements (don't show)
        plotter._draw_bodies()
        plotter._draw_trajectories()

        # Should not raise
        plt.close(plotter.fig)

    def test_render_with_markers(self):
        """Test rendering with markers."""
        plotter = OrbitPlotter3D()

        plotter.add_marker(np.array([7000, 0, 0]), label='Point 1')
        plotter.add_marker(np.array([0, 7000, 0]), label='Point 2')

        plotter._draw_markers()

        plt.close(plotter.fig)


class TestVisualizationIntegration:
    """Test visualization with trajectory data."""

    def test_circular_orbit_visualization(self):
        """Test visualizing a circular orbit."""
        # Generate circular orbit data
        n_points = 200
        t = np.linspace(0, 2*np.pi, n_points)
        r = 6771  # LEO radius in km

        positions = np.array([
            r * np.cos(t),
            r * np.sin(t),
            np.zeros_like(t)
        ])

        plotter = OrbitPlotter3D(title="LEO Circular Orbit")
        plotter.add_body('earth')
        plotter.add_trajectory(positions, color='cyan', label='LEO')

        # Verify data was added correctly
        assert len(plotter._trajectories) == 1
        assert plotter._trajectories[0]['positions'].shape == (3, n_points)

        plt.close(plotter.fig)

    def test_elliptical_orbit_visualization(self):
        """Test visualizing an elliptical orbit."""
        # Generate elliptical orbit
        n_points = 200
        a = 20000  # Semi-major axis
        e = 0.5    # Eccentricity
        t = np.linspace(0, 2*np.pi, n_points)

        r = a * (1 - e**2) / (1 + e * np.cos(t))
        positions = np.array([
            r * np.cos(t),
            r * np.sin(t),
            np.zeros_like(t)
        ])

        plotter = OrbitPlotter3D(title="Elliptical Orbit")
        plotter.add_body('earth')
        plotter.add_trajectory(positions, color='orange', label='Ellipse')

        # Mark periapsis and apoapsis
        plotter.add_marker(positions[:, 0], color='lime', label='Periapsis')
        plotter.add_marker(positions[:, n_points//2], color='red', label='Apoapsis')

        assert len(plotter._markers) == 2

        plt.close(plotter.fig)

    def test_inclined_orbit_visualization(self):
        """Test visualizing an inclined orbit."""
        n_points = 200
        t = np.linspace(0, 2*np.pi, n_points)
        r = 8000  # km
        inc = np.radians(45)  # 45 degree inclination

        # Rotate orbit plane
        x = r * np.cos(t)
        y = r * np.sin(t) * np.cos(inc)
        z = r * np.sin(t) * np.sin(inc)

        positions = np.array([x, y, z])

        plotter = OrbitPlotter3D(title="45° Inclined Orbit")
        plotter.add_body('earth')
        plotter.add_trajectory(positions, label='Inclined')
        plotter.add_reference_frame(scale=10000)

        plt.close(plotter.fig)


class TestGroundTrackData:
    """Test ground track calculation (not plotting)."""

    def test_lat_lon_calculation(self):
        """Test latitude/longitude calculation from positions."""
        # Create position pointing to North Pole
        pos_north = np.array([[0], [0], [7000]])

        r = np.linalg.norm(pos_north, axis=0)
        lat = np.degrees(np.arcsin(pos_north[2] / r))
        lon = np.degrees(np.arctan2(pos_north[1], pos_north[0]))

        assert lat[0] == 90.0  # North pole
        assert lon[0] == 0.0

    def test_equatorial_position(self):
        """Test equatorial position lat/lon."""
        # Position on equator at 0° longitude
        pos_eq = np.array([[7000], [0], [0]])

        r = np.linalg.norm(pos_eq, axis=0)
        lat = np.degrees(np.arcsin(pos_eq[2] / r))

        assert abs(lat[0]) < 0.001  # Should be ~0°


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
