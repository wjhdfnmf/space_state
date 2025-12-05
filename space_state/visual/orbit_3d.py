# visual/orbit_3d.py
"""
Enhanced 3D Orbit Visualization

Provides comprehensive 3D visualization capabilities:
- Static orbit plots
- Animated trajectories
- Multi-body visualization
- Interactive plots (Plotly)
- Ground tracks
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base.star_constants import AU_TO_KM, BODIES


# ==================== UTILITIES ====================

def create_sphere(
    radius: float,
    center: np.ndarray = None,
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sphere mesh for plotting.

    Args:
        radius: Sphere radius
        center: Center coordinates [x, y, z]
        n_points: Resolution (points per dimension)

    Returns:
        Tuple of (x, y, z) mesh arrays
    """
    if center is None:
        center = np.zeros(3)

    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)

    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    return x, y, z


def set_equal_aspect_3d(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Set equal aspect ratio for 3D plot.

    Args:
        ax: Matplotlib 3D axes
        x, y, z: Data arrays
    """
    max_range = max(
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ) / 2.0

    mid_x = (x.max() + x.min()) / 2.0
    mid_y = (y.max() + y.min()) / 2.0
    mid_z = (z.max() + z.min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


# ==================== BODY COLORS ====================

BODY_COLORS = {
    'sun': '#FFD700',
    'mercury barycenter': '#8C7853',
    'venus barycenter': '#FFA500',
    'earth': '#1E90FF',
    'moon': '#C0C0C0',
    'mars barycenter': '#CD5C5C',
    'jupiter barycenter': '#DAA520',
    'saturn barycenter': '#F4A460',
    'uranus barycenter': '#87CEEB',
    'neptune barycenter': '#4169E1',
}

BODY_RADII_KM = {
    'sun': 696340,
    'mercury barycenter': 2439.7,
    'venus barycenter': 6051.8,
    'earth': 6371.0,
    'moon': 1737.4,
    'mars barycenter': 3389.5,
    'jupiter barycenter': 69911,
    'saturn barycenter': 58232,
    'uranus barycenter': 25362,
    'neptune barycenter': 24622,
}


# ==================== ORBIT PLOTTER CLASS ====================

class OrbitPlotter3D:
    """
    Comprehensive 3D orbit plotter with multiple features.

    Example:
        plotter = OrbitPlotter3D()
        plotter.add_body('earth')
        plotter.add_trajectory(positions, color='cyan', label='Spacecraft')
        plotter.show()
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 10),
        dark_mode: bool = True,
        title: str = "3D Orbit View",
    ):
        """
        Initialize orbit plotter.

        Args:
            figsize: Figure size (width, height)
            dark_mode: Use dark background
            title: Plot title
        """
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.title = title
        self.dark_mode = dark_mode

        self._trajectories = []
        self._bodies = []
        self._markers = []
        self._all_points = []

        if dark_mode:
            self.ax.set_facecolor('black')
            self.fig.patch.set_facecolor('black')

    def add_body(
        self,
        name: str,
        position: np.ndarray = None,
        scale: float = 1.0,
        alpha: float = 1.0,
    ):
        """
        Add celestial body to plot.

        Args:
            name: Body name (e.g., 'earth', 'moon')
            position: Position [x, y, z] in km (default: origin)
            scale: Radius scaling factor
            alpha: Transparency
        """
        if position is None:
            position = np.zeros(3)

        color = BODY_COLORS.get(name, '#808080')
        radius = BODY_RADII_KM.get(name, 1000) * scale

        self._bodies.append({
            'name': name,
            'position': position,
            'radius': radius,
            'color': color,
            'alpha': alpha,
        })

        self._all_points.append(position)

    def add_trajectory(
        self,
        positions: np.ndarray,
        color: str = 'cyan',
        label: str = None,
        linewidth: float = 1.5,
        alpha: float = 0.8,
        show_start: bool = True,
        show_end: bool = True,
    ):
        """
        Add trajectory to plot.

        Args:
            positions: Position array (3, n_points) or (n_points, 3) in km
            color: Line color
            label: Legend label
            linewidth: Line width
            alpha: Transparency
            show_start: Show start marker
            show_end: Show end marker
        """
        # Ensure shape is (3, n)
        if positions.shape[0] != 3:
            positions = positions.T

        self._trajectories.append({
            'positions': positions,
            'color': color,
            'label': label,
            'linewidth': linewidth,
            'alpha': alpha,
            'show_start': show_start,
            'show_end': show_end,
        })

        self._all_points.extend([positions[:, 0], positions[:, -1]])

    def add_marker(
        self,
        position: np.ndarray,
        color: str = 'white',
        marker: str = 'o',
        size: float = 50,
        label: str = None,
    ):
        """
        Add marker to plot.

        Args:
            position: Marker position [x, y, z] in km
            color: Marker color
            marker: Marker style
            size: Marker size
            label: Legend label
        """
        self._markers.append({
            'position': position,
            'color': color,
            'marker': marker,
            'size': size,
            'label': label,
        })

        self._all_points.append(position)

    def add_reference_frame(
        self,
        origin: np.ndarray = None,
        scale: float = 1000.0,
        labels: List[str] = None,
    ):
        """
        Add reference frame axes.

        Args:
            origin: Frame origin [x, y, z]
            scale: Axis length in km
            labels: Axis labels [x, y, z]
        """
        if origin is None:
            origin = np.zeros(3)
        if labels is None:
            labels = ['X', 'Y', 'Z']

        colors = ['red', 'green', 'blue']
        for i, (c, lbl) in enumerate(zip(colors, labels)):
            end = origin.copy()
            end[i] += scale
            self.ax.plot(
                [origin[0], end[0]],
                [origin[1], end[1]],
                [origin[2], end[2]],
                color=c, linewidth=2, label=lbl
            )

    def _draw_bodies(self):
        """Draw all celestial bodies."""
        for body in self._bodies:
            x, y, z = create_sphere(body['radius'], body['position'])
            self.ax.plot_surface(
                x, y, z,
                color=body['color'],
                alpha=body['alpha'],
                shade=True,
                zorder=0,
            )

    def _draw_trajectories(self):
        """Draw all trajectories."""
        for traj in self._trajectories:
            pos = traj['positions']
            self.ax.plot(
                pos[0], pos[1], pos[2],
                color=traj['color'],
                linewidth=traj['linewidth'],
                alpha=traj['alpha'],
                label=traj['label'],
            )

            if traj['show_start']:
                self.ax.scatter(
                    [pos[0, 0]], [pos[1, 0]], [pos[2, 0]],
                    color='lime', s=80, marker='o', zorder=5,
                )

            if traj['show_end']:
                self.ax.scatter(
                    [pos[0, -1]], [pos[1, -1]], [pos[2, -1]],
                    color='red', s=80, marker='o', zorder=5,
                )

    def _draw_markers(self):
        """Draw all markers."""
        for m in self._markers:
            self.ax.scatter(
                [m['position'][0]],
                [m['position'][1]],
                [m['position'][2]],
                color=m['color'],
                marker=m['marker'],
                s=m['size'],
                label=m['label'],
                zorder=10,
            )

    def show(
        self,
        elev: float = 30,
        azim: float = 45,
        equal_aspect: bool = True,
        show_grid: bool = False,
        show_axes: bool = True,
        save_path: str = None,
    ):
        """
        Display the plot.

        Args:
            elev: Elevation angle (degrees)
            azim: Azimuth angle (degrees)
            equal_aspect: Force equal aspect ratio
            show_grid: Show grid lines
            show_axes: Show axes
            save_path: Optional path to save figure
        """
        # Draw all elements
        self._draw_bodies()
        self._draw_trajectories()
        self._draw_markers()

        # Set view
        self.ax.view_init(elev=elev, azim=azim)

        # Equal aspect
        if equal_aspect and self._all_points:
            all_pts = np.array(self._all_points)
            set_equal_aspect_3d(
                self.ax,
                all_pts[:, 0] if len(all_pts.shape) > 1 else all_pts,
                all_pts[:, 1] if len(all_pts.shape) > 1 else np.zeros_like(all_pts),
                all_pts[:, 2] if len(all_pts.shape) > 1 else np.zeros_like(all_pts),
            )

        # Styling
        text_color = 'white' if self.dark_mode else 'black'
        self.ax.set_title(self.title, color=text_color, fontsize=14, pad=20)

        if not show_grid:
            self.ax.grid(False)

        if not show_axes:
            self.ax.axis('off')
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False

        # Legend
        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            self.ax.legend(
                facecolor='black' if self.dark_mode else 'white',
                labelcolor='white' if self.dark_mode else 'black',
                loc='upper left',
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=self.fig.get_facecolor())

        plt.show()


# ==================== CONVENIENCE FUNCTIONS ====================

def plot_orbit_3d(
    positions: np.ndarray,
    body: str = 'earth',
    title: str = "Orbit",
    **kwargs
):
    """
    Quick orbit plot around a body.

    Args:
        positions: Trajectory positions (3, n) or (n, 3) in km
        body: Central body name
        title: Plot title
        **kwargs: Additional arguments for OrbitPlotter3D
    """
    plotter = OrbitPlotter3D(title=title, **kwargs)
    plotter.add_body(body)
    plotter.add_trajectory(positions, label='Orbit')
    plotter.show()


def plot_trajectory_3d(
    positions: np.ndarray,
    velocities: np.ndarray = None,
    times: np.ndarray = None,
    title: str = "Trajectory",
    color_by: str = 'time',
    **kwargs
):
    """
    Plot trajectory with color mapping.

    Args:
        positions: Position array (3, n) in km
        velocities: Optional velocity array (3, n)
        times: Optional time array
        title: Plot title
        color_by: 'time', 'velocity', or 'altitude'
    """
    if positions.shape[0] != 3:
        positions = positions.T

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    n_points = positions.shape[1]

    # Determine colors
    if color_by == 'time' and times is not None:
        colors = plt.cm.viridis((times - times.min()) / (times.max() - times.min()))
    elif color_by == 'velocity' and velocities is not None:
        v_mag = np.linalg.norm(velocities, axis=0)
        colors = plt.cm.plasma((v_mag - v_mag.min()) / (v_mag.max() - v_mag.min()))
    elif color_by == 'altitude':
        r = np.linalg.norm(positions, axis=0)
        colors = plt.cm.coolwarm((r - r.min()) / (r.max() - r.min()))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

    # Plot segments with colors
    for i in range(n_points - 1):
        ax.plot(
            positions[0, i:i+2],
            positions[1, i:i+2],
            positions[2, i:i+2],
            color=colors[i],
            linewidth=2,
        )

    # Start/end markers
    ax.scatter(*positions[:, 0], color='lime', s=100, marker='o', label='Start')
    ax.scatter(*positions[:, -1], color='red', s=100, marker='o', label='End')

    ax.set_title(title, color='white', fontsize=14)
    ax.legend(facecolor='black', labelcolor='white')

    set_equal_aspect_3d(ax, positions[0], positions[1], positions[2])
    ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_body(
    ax: plt.Axes,
    name: str,
    position: np.ndarray = None,
    scale: float = 1.0,
):
    """
    Add body to existing axes.

    Args:
        ax: Matplotlib 3D axes
        name: Body name
        position: Position [x, y, z]
        scale: Radius scale factor
    """
    if position is None:
        position = np.zeros(3)

    color = BODY_COLORS.get(name, '#808080')
    radius = BODY_RADII_KM.get(name, 1000) * scale

    x, y, z = create_sphere(radius, position)
    ax.plot_surface(x, y, z, color=color, alpha=1.0, shade=True)


def plot_reference_frame(
    ax: plt.Axes,
    origin: np.ndarray = None,
    scale: float = 1000.0,
    labels: bool = True,
):
    """
    Add reference frame to axes.

    Args:
        ax: Matplotlib 3D axes
        origin: Frame origin
        scale: Axis length
        labels: Show axis labels
    """
    if origin is None:
        origin = np.zeros(3)

    colors = ['r', 'g', 'b']
    axis_labels = ['X', 'Y', 'Z']

    for i, (c, lbl) in enumerate(zip(colors, axis_labels)):
        end = origin.copy()
        end[i] += scale
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            c, linewidth=2,
        )
        if labels:
            ax.text(end[0], end[1], end[2], lbl, color=c, fontsize=12)


# ==================== ANIMATION ====================

def create_orbit_animation(
    positions: np.ndarray,
    times: np.ndarray = None,
    body: str = 'earth',
    interval: int = 50,
    trail_length: int = 100,
    save_path: str = None,
) -> FuncAnimation:
    """
    Create animated orbit visualization.

    Args:
        positions: Position array (3, n) in km
        times: Optional time array
        body: Central body name
        interval: Animation interval (ms)
        trail_length: Length of trailing line
        save_path: Optional path to save animation

    Returns:
        FuncAnimation object
    """
    if positions.shape[0] != 3:
        positions = positions.T

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Add body
    plot_body(ax, body)

    # Initialize plot elements
    line, = ax.plot([], [], [], 'cyan', linewidth=1.5, alpha=0.6)
    point, = ax.plot([], [], [], 'o', color='yellow', markersize=8)

    set_equal_aspect_3d(ax, positions[0], positions[1], positions[2])
    ax.axis('off')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    def animate(i):
        start = max(0, i - trail_length)
        line.set_data(positions[0, start:i], positions[1, start:i])
        line.set_3d_properties(positions[2, start:i])

        point.set_data([positions[0, i]], [positions[1, i]])
        point.set_3d_properties([positions[2, i]])

        return line, point

    n_frames = positions.shape[1]
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=interval, blit=True
    )

    if save_path:
        anim.save(save_path, writer='pillow', fps=30)

    return anim


def create_trajectory_animation(
    positions: np.ndarray,
    velocities: np.ndarray = None,
    interval: int = 50,
    save_path: str = None,
) -> FuncAnimation:
    """
    Create trajectory animation with velocity vectors.

    Args:
        positions: Position array (3, n)
        velocities: Optional velocity array (3, n)
        interval: Animation interval (ms)
        save_path: Optional save path

    Returns:
        FuncAnimation object
    """
    return create_orbit_animation(positions, interval=interval, save_path=save_path)


# ==================== GROUND TRACKS ====================

def plot_ground_track(
    positions: np.ndarray,
    body_radius: float = 6371.0,
    title: str = "Ground Track",
    show_continents: bool = True,
):
    """
    Plot ground track on 2D map projection.

    Args:
        positions: Position array (3, n) in km (body-centered)
        body_radius: Body radius in km
        title: Plot title
        show_continents: Show continent outlines (Earth only)
    """
    if positions.shape[0] != 3:
        positions = positions.T

    # Convert to lat/lon
    r = np.linalg.norm(positions, axis=0)
    lat = np.degrees(np.arcsin(positions[2] / r))
    lon = np.degrees(np.arctan2(positions[1], positions[0]))

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot ground track
    ax.scatter(lon, lat, c=np.arange(len(lon)), cmap='viridis', s=1, alpha=0.7)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


# ==================== MULTI-BODY VIEWS ====================

def plot_solar_system(
    trajectories: Dict[str, np.ndarray] = None,
    bodies: List[str] = None,
    scale: str = 'inner',
    title: str = "Solar System",
):
    """
    Plot solar system view.

    Args:
        trajectories: Dict of {name: positions} arrays
        bodies: List of body names to show
        scale: 'inner' (Mercury-Mars) or 'outer' (full system)
        title: Plot title
    """
    if bodies is None:
        if scale == 'inner':
            bodies = ['sun', 'mercury barycenter', 'venus barycenter', 'earth', 'mars barycenter']
        else:
            bodies = list(BODY_COLORS.keys())

    plotter = OrbitPlotter3D(title=title, figsize=(14, 12))

    for body in bodies:
        plotter.add_body(body, scale=0.001 if body == 'sun' else 10.0)

    if trajectories:
        colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))
        for (name, pos), color in zip(trajectories.items(), colors):
            plotter.add_trajectory(pos, color=mcolors.to_hex(color), label=name)

    plotter.show()


def plot_earth_moon(
    earth_pos: np.ndarray = None,
    moon_pos: np.ndarray = None,
    trajectory: np.ndarray = None,
    title: str = "Earth-Moon System",
):
    """
    Plot Earth-Moon system.

    Args:
        earth_pos: Earth position (optional)
        moon_pos: Moon position (optional)
        trajectory: Spacecraft trajectory
        title: Plot title
    """
    plotter = OrbitPlotter3D(title=title)

    # Earth at origin
    plotter.add_body('earth', position=np.zeros(3) if earth_pos is None else earth_pos)

    # Moon
    if moon_pos is None:
        moon_pos = np.array([384400, 0, 0])  # Mean distance
    plotter.add_body('moon', position=moon_pos, scale=1.0)

    if trajectory is not None:
        plotter.add_trajectory(trajectory, color='cyan', label='Spacecraft')

    plotter.show()


# ==================== INTERACTIVE (PLOTLY) ====================

def create_interactive_orbit(
    positions: np.ndarray,
    body: str = 'earth',
    title: str = "Interactive Orbit",
) -> 'go.Figure':
    """
    Create interactive orbit plot with Plotly.

    Args:
        positions: Position array (3, n) in km
        body: Central body name
        title: Plot title

    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not available. Install with: pip install plotly")
        return None

    if positions.shape[0] != 3:
        positions = positions.T

    # Create figure
    fig = go.Figure()

    # Add body sphere
    radius = BODY_RADII_KM.get(body, 1000)
    color = BODY_COLORS.get(body, '#808080')

    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=0.9,
        name=body.title(),
    ))

    # Add trajectory
    fig.add_trace(go.Scatter3d(
        x=positions[0], y=positions[1], z=positions[2],
        mode='lines',
        line=dict(color='cyan', width=3),
        name='Trajectory',
    ))

    # Start/end markers
    fig.add_trace(go.Scatter3d(
        x=[positions[0, 0]], y=[positions[1, 0]], z=[positions[2, 0]],
        mode='markers',
        marker=dict(color='lime', size=8),
        name='Start',
    ))

    fig.add_trace(go.Scatter3d(
        x=[positions[0, -1]], y=[positions[1, -1]], z=[positions[2, -1]],
        mode='markers',
        marker=dict(color='red', size=8),
        name='End',
    ))

    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            bgcolor='rgb(20, 20, 40)',
        ),
        paper_bgcolor='rgb(20, 20, 40)',
        font=dict(color='white'),
    )

    return fig


def create_interactive_trajectory(
    positions: np.ndarray,
    times: np.ndarray = None,
    data: Dict[str, np.ndarray] = None,
    title: str = "Interactive Trajectory",
) -> 'go.Figure':
    """
    Create interactive trajectory with data plots.

    Args:
        positions: Position array (3, n)
        times: Time array
        data: Additional data to plot {name: values}
        title: Plot title

    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not available")
        return None

    if positions.shape[0] != 3:
        positions = positions.T

    n_data = len(data) if data else 0

    fig = make_subplots(
        rows=1 + n_data, cols=1,
        specs=[[{'type': 'scatter3d'}]] + [[{'type': 'scatter'}]] * n_data,
        row_heights=[0.6] + [0.4/n_data] * n_data if n_data > 0 else [1.0],
    )

    # 3D trajectory
    fig.add_trace(go.Scatter3d(
        x=positions[0], y=positions[1], z=positions[2],
        mode='lines',
        line=dict(color='cyan', width=2),
        name='Trajectory',
    ), row=1, col=1)

    # Data plots
    if data and times is not None:
        for i, (name, values) in enumerate(data.items(), start=2):
            fig.add_trace(go.Scatter(
                x=times, y=values,
                mode='lines',
                name=name,
            ), row=i, col=1)

    fig.update_layout(title=title, height=600 + 200 * n_data)

    return fig
