# visual/interactive_mission.py
"""
Interactive Mission Visualization Suite

Provides comprehensive interactive visualization tools with:
- Time slider control for trajectory playback
- Dynamic focus switching between bodies/spacecraft
- Multi-panel dashboards
- Real-time telemetry display
- Attitude visualization

Uses Plotly/Dash for web-based interactivity.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from dash import Dash, html, dcc, Input, Output, State, callback, no_update
    from dash.exceptions import PreventUpdate
    import dash_bootstrap_components as dbc
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    try:
        from dash import Dash, html, dcc, Input, Output, State, callback, no_update
        from dash.exceptions import PreventUpdate
        HAS_DASH = True
    except ImportError:
        pass

# Constants
AU_TO_KM = 149597870.7
DAY_TO_SEC = 86400.0


# ==================== DATA STRUCTURES ====================

@dataclass
class TrajectoryData:
    """Container for trajectory visualization data."""
    times: np.ndarray                    # Time array (days)
    positions: np.ndarray                # Position (n, 3) in km
    velocities: Optional[np.ndarray] = None  # Velocity (n, 3) in km/s
    masses: Optional[np.ndarray] = None  # Mass array (kg)
    quaternions: Optional[np.ndarray] = None  # Attitude (n, 4)
    angular_velocities: Optional[np.ndarray] = None  # Angular velocity (n, 3)

    # Metadata
    name: str = "Spacecraft"
    color: str = "cyan"

    @property
    def n_points(self) -> int:
        return len(self.times)

    @property
    def duration_days(self) -> float:
        return self.times[-1] - self.times[0]

    def get_state_at_time(self, t: float) -> Dict[str, Any]:
        """Interpolate state at given time."""
        idx = np.searchsorted(self.times, t)
        idx = min(idx, self.n_points - 1)

        state = {
            'time': self.times[idx],
            'position': self.positions[idx],
        }

        if self.velocities is not None:
            state['velocity'] = self.velocities[idx]
        if self.masses is not None:
            state['mass'] = self.masses[idx]
        if self.quaternions is not None:
            state['quaternion'] = self.quaternions[idx]
        if self.angular_velocities is not None:
            state['angular_velocity'] = self.angular_velocities[idx]

        return state


@dataclass
class BodyData:
    """Container for celestial body data."""
    name: str
    positions: np.ndarray  # (n, 3) in km or single position
    radius_km: float
    color: str = "#808080"

    def get_position_at_index(self, idx: int) -> np.ndarray:
        """Get body position at time index."""
        if self.positions.ndim == 1:
            return self.positions
        return self.positions[min(idx, len(self.positions) - 1)]


@dataclass
class EventMarker:
    """Marker for mission events."""
    time: float  # days
    position: np.ndarray  # (3,) in km
    name: str
    color: str = "yellow"
    symbol: str = "diamond"
    size: int = 12


# ==================== INTERACTIVE 3D TRAJECTORY ====================

def create_interactive_trajectory_viewer(
    trajectory: TrajectoryData,
    bodies: List[BodyData] = None,
    events: List[EventMarker] = None,
    title: str = "Mission Trajectory Viewer",
) -> 'go.Figure':
    """
    Create interactive 3D trajectory viewer with time slider.

    Args:
        trajectory: Spacecraft trajectory data
        bodies: List of celestial bodies
        events: List of event markers
        title: Plot title

    Returns:
        Plotly Figure with animation controls
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly required. Install with: pip install plotly")

    bodies = bodies or []
    events = events or []

    # Create frames for animation
    n_frames = min(trajectory.n_points, 200)  # Limit frames for performance
    frame_indices = np.linspace(0, trajectory.n_points - 1, n_frames, dtype=int)

    frames = []
    for i, idx in enumerate(frame_indices):
        frame_data = []

        # Trajectory trace up to current time
        frame_data.append(go.Scatter3d(
            x=trajectory.positions[:idx+1, 0],
            y=trajectory.positions[:idx+1, 1],
            z=trajectory.positions[:idx+1, 2],
            mode='lines',
            line=dict(color=trajectory.color, width=3),
            name=f'{trajectory.name} Path',
        ))

        # Current spacecraft position
        frame_data.append(go.Scatter3d(
            x=[trajectory.positions[idx, 0]],
            y=[trajectory.positions[idx, 1]],
            z=[trajectory.positions[idx, 2]],
            mode='markers',
            marker=dict(color='yellow', size=10, symbol='diamond'),
            name=trajectory.name,
        ))

        # Velocity vector (if available)
        if trajectory.velocities is not None:
            v = trajectory.velocities[idx]
            v_scale = np.linalg.norm(trajectory.positions[idx]) * 0.1  # Scale for visibility
            v_norm = v / (np.linalg.norm(v) + 1e-10) * v_scale

            frame_data.append(go.Scatter3d(
                x=[trajectory.positions[idx, 0], trajectory.positions[idx, 0] + v_norm[0]],
                y=[trajectory.positions[idx, 1], trajectory.positions[idx, 1] + v_norm[1]],
                z=[trajectory.positions[idx, 2], trajectory.positions[idx, 2] + v_norm[2]],
                mode='lines',
                line=dict(color='red', width=4),
                name='Velocity',
            ))

        # Bodies at current time
        for body in bodies:
            body_pos = body.get_position_at_index(idx)

            # Create sphere for body
            u = np.linspace(0, 2*np.pi, 20)
            v_angles = np.linspace(0, np.pi, 20)
            x_sphere = body.radius_km * np.outer(np.cos(u), np.sin(v_angles)) + body_pos[0]
            y_sphere = body.radius_km * np.outer(np.sin(u), np.sin(v_angles)) + body_pos[1]
            z_sphere = body.radius_km * np.outer(np.ones_like(u), np.cos(v_angles)) + body_pos[2]

            frame_data.append(go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                colorscale=[[0, body.color], [1, body.color]],
                showscale=False,
                opacity=0.9,
                name=body.name,
            ))

        frames.append(go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(
                title=f"{title}<br>Day {trajectory.times[idx]:.2f}"
            )
        ))

    # Create initial figure
    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames,
    )

    # Add event markers (static)
    for event in events:
        fig.add_trace(go.Scatter3d(
            x=[event.position[0]],
            y=[event.position[1]],
            z=[event.position[2]],
            mode='markers+text',
            marker=dict(color=event.color, size=event.size, symbol=event.symbol),
            text=[event.name],
            textposition='top center',
            name=event.name,
        ))

    # Add slider and play button
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X (km)', showbackground=False),
            yaxis=dict(title='Y (km)', showbackground=False),
            zaxis=dict(title='Z (km)', showbackground=False),
            bgcolor='rgb(20, 20, 40)',
        ),
        paper_bgcolor='rgb(20, 20, 40)',
        font=dict(color='white'),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.1,
                xanchor='right',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=50, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                        )],
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                        )],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0,
                yanchor='top',
                xanchor='left',
                currentvalue=dict(
                    font=dict(size=12),
                    prefix='Time: ',
                    suffix=' days',
                    visible=True,
                    xanchor='right',
                ),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[[str(i)], dict(
                            frame=dict(duration=0, redraw=True),
                            mode='immediate',
                        )],
                        label=f'{trajectory.times[frame_indices[i]]:.1f}',
                        method='animate',
                    )
                    for i in range(n_frames)
                ],
            ),
        ],
    )

    return fig


# ==================== MISSION DASHBOARD ====================

def create_mission_dashboard(
    trajectory: TrajectoryData,
    central_body: str = 'earth',
    show_elements: bool = True,
) -> 'go.Figure':
    """
    Create multi-panel mission dashboard.

    Panels:
    - 3D trajectory view
    - Altitude vs time
    - Velocity vs time
    - Orbital elements (if show_elements)

    Args:
        trajectory: Trajectory data
        central_body: Name of central body for altitude calc
        show_elements: Show orbital elements panel

    Returns:
        Plotly Figure with subplots
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly required")

    n_rows = 3 if show_elements else 2

    fig = make_subplots(
        rows=n_rows, cols=2,
        specs=[
            [{'type': 'scatter3d', 'rowspan': 2}, {'type': 'xy'}],
            [None, {'type': 'xy'}],
            [{'type': 'xy', 'colspan': 2}, None] if show_elements else [None, None],
        ],
        subplot_titles=['3D Trajectory', 'Altitude', 'Velocity', 'Orbital Elements'] if show_elements
                       else ['3D Trajectory', 'Altitude', None, 'Velocity'],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    # 3D Trajectory
    fig.add_trace(go.Scatter3d(
        x=trajectory.positions[:, 0],
        y=trajectory.positions[:, 1],
        z=trajectory.positions[:, 2],
        mode='lines',
        line=dict(color=trajectory.color, width=2),
        name='Trajectory',
    ), row=1, col=1)

    # Start/end markers
    fig.add_trace(go.Scatter3d(
        x=[trajectory.positions[0, 0]],
        y=[trajectory.positions[0, 1]],
        z=[trajectory.positions[0, 2]],
        mode='markers',
        marker=dict(color='lime', size=8),
        name='Start',
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[trajectory.positions[-1, 0]],
        y=[trajectory.positions[-1, 1]],
        z=[trajectory.positions[-1, 2]],
        mode='markers',
        marker=dict(color='red', size=8),
        name='End',
    ), row=1, col=1)

    # Altitude
    altitude = np.linalg.norm(trajectory.positions, axis=1)
    fig.add_trace(go.Scatter(
        x=trajectory.times,
        y=altitude,
        mode='lines',
        line=dict(color='cyan'),
        name='Altitude',
    ), row=1, col=2)

    # Velocity
    if trajectory.velocities is not None:
        velocity_mag = np.linalg.norm(trajectory.velocities, axis=1)
        fig.add_trace(go.Scatter(
            x=trajectory.times,
            y=velocity_mag,
            mode='lines',
            line=dict(color='orange'),
            name='Velocity',
        ), row=2, col=2)

    # Mass (if available)
    if show_elements and trajectory.masses is not None:
        fig.add_trace(go.Scatter(
            x=trajectory.times,
            y=trajectory.masses,
            mode='lines',
            line=dict(color='green'),
            name='Mass',
        ), row=3, col=1)

    fig.update_layout(
        title=f'{trajectory.name} Mission Dashboard',
        height=800,
        paper_bgcolor='rgb(30, 30, 50)',
        plot_bgcolor='rgb(30, 30, 50)',
        font=dict(color='white'),
    )

    # Update axes
    fig.update_xaxes(title_text='Time (days)', row=1, col=2)
    fig.update_yaxes(title_text='Altitude (km)', row=1, col=2)
    fig.update_xaxes(title_text='Time (days)', row=2, col=2)
    fig.update_yaxes(title_text='Velocity (km/s)', row=2, col=2)

    return fig


# ==================== ATTITUDE VISUALIZER ====================

def create_attitude_visualizer(
    quaternions: np.ndarray,
    times: np.ndarray,
    target_quaternions: Optional[np.ndarray] = None,
    title: str = "Attitude Visualization",
) -> 'go.Figure':
    """
    Create attitude visualization with quaternion display.

    Shows:
    - 3D spacecraft body frame orientation
    - Quaternion components over time
    - Pointing error (if target provided)

    Args:
        quaternions: Quaternion array (n, 4) [w, x, y, z]
        times: Time array
        target_quaternions: Optional target attitude
        title: Plot title

    Returns:
        Plotly Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly required")

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'scatter3d'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}],
        ],
        subplot_titles=['Body Frame', 'Quaternion Components', 'Angular Rates', 'Pointing Error'],
    )

    # Quaternion components
    q_labels = ['q_w', 'q_x', 'q_y', 'q_z']
    colors = ['white', 'red', 'green', 'blue']

    for i, (label, color) in enumerate(zip(q_labels, colors)):
        fig.add_trace(go.Scatter(
            x=times,
            y=quaternions[:, i],
            mode='lines',
            line=dict(color=color),
            name=label,
        ), row=1, col=2)

    # Body frame visualization (initial orientation)
    def quat_to_dcm(q):
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])

    # Draw body axes
    axis_length = 1.0
    axis_colors = ['red', 'green', 'blue']
    axis_names = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)']

    q0 = quaternions[0]
    R = quat_to_dcm(q0)

    for i, (color, name) in enumerate(zip(axis_colors, axis_names)):
        axis = R[:, i] * axis_length
        fig.add_trace(go.Scatter3d(
            x=[0, axis[0]],
            y=[0, axis[1]],
            z=[0, axis[2]],
            mode='lines+markers',
            line=dict(color=color, width=6),
            marker=dict(size=4),
            name=name,
        ), row=1, col=1)

    # Pointing error (if target provided)
    if target_quaternions is not None:
        # Calculate angular error between actual and target
        errors = []
        for q_actual, q_target in zip(quaternions, target_quaternions):
            # Quaternion error: q_err = q_target^(-1) * q_actual
            q_target_inv = np.array([q_target[0], -q_target[1], -q_target[2], -q_target[3]])
            # Simplified error calculation using dot product
            dot = np.abs(np.dot(q_actual, q_target))
            error_rad = 2 * np.arccos(np.clip(dot, 0, 1))
            errors.append(np.degrees(error_rad))

        fig.add_trace(go.Scatter(
            x=times,
            y=errors,
            mode='lines',
            line=dict(color='yellow'),
            name='Pointing Error',
        ), row=2, col=2)

    fig.update_layout(
        title=title,
        height=700,
        paper_bgcolor='rgb(30, 30, 50)',
        plot_bgcolor='rgb(30, 30, 50)',
        font=dict(color='white'),
    )

    return fig


# ==================== GROUND TRACK VISUALIZER ====================

def create_interactive_ground_track(
    positions_ecef: np.ndarray,
    times: np.ndarray,
    body_radius_km: float = 6371.0,
    title: str = "Interactive Ground Track",
) -> 'go.Figure':
    """
    Create interactive ground track with time slider.

    Args:
        positions_ecef: ECEF positions (n, 3) in km
        times: Time array
        body_radius_km: Body radius for lat/lon conversion
        title: Plot title

    Returns:
        Plotly Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly required")

    # Convert to lat/lon
    r = np.linalg.norm(positions_ecef, axis=1)
    lat = np.degrees(np.arcsin(positions_ecef[:, 2] / r))
    lon = np.degrees(np.arctan2(positions_ecef[:, 1], positions_ecef[:, 0]))

    # Create frames for animation
    n_frames = min(len(times), 100)
    frame_indices = np.linspace(0, len(times) - 1, n_frames, dtype=int)

    frames = []
    for i, idx in enumerate(frame_indices):
        frames.append(go.Frame(
            data=[
                # Full track (faded)
                go.Scattergeo(
                    lon=lon[:idx+1],
                    lat=lat[:idx+1],
                    mode='lines',
                    line=dict(color='cyan', width=2),
                    name='Ground Track',
                ),
                # Current position
                go.Scattergeo(
                    lon=[lon[idx]],
                    lat=[lat[idx]],
                    mode='markers',
                    marker=dict(color='yellow', size=12, symbol='star'),
                    name='Spacecraft',
                ),
            ],
            name=str(i),
        ))

    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames,
    )

    # Add ground stations
    stations = {
        'Goldstone': (35.4, -116.9),
        'Canberra': (-35.4, 149.0),
        'Madrid': (40.4, -4.2),
    }

    for name, (lat_s, lon_s) in stations.items():
        fig.add_trace(go.Scattergeo(
            lon=[lon_s],
            lat=[lat_s],
            mode='markers+text',
            marker=dict(color='lime', size=10, symbol='triangle-up'),
            text=[name],
            textposition='top center',
            name=name,
        ))

    fig.update_layout(
        title=title,
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(50, 50, 80)',
            showocean=True,
            oceancolor='rgb(30, 30, 60)',
            showlakes=True,
            lakecolor='rgb(30, 30, 60)',
            showcountries=True,
            countrycolor='rgb(100, 100, 150)',
            coastlinecolor='rgb(100, 100, 150)',
            bgcolor='rgb(20, 20, 40)',
        ),
        paper_bgcolor='rgb(20, 20, 40)',
        font=dict(color='white'),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.1,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(frame=dict(duration=100, redraw=True))],
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(frame=dict(duration=0, redraw=False))],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0,
                yanchor='top',
                xanchor='left',
                currentvalue=dict(prefix='Time: ', suffix=' days'),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[[str(i)], dict(frame=dict(duration=0, redraw=True))],
                        label=f'{times[frame_indices[i]]:.2f}',
                        method='animate',
                    )
                    for i in range(n_frames)
                ],
            ),
        ],
    )

    return fig


# ==================== MISSION TIMELINE ====================

def create_mission_timeline(
    events: List[Dict[str, Any]],
    title: str = "Mission Timeline",
) -> 'go.Figure':
    """
    Create interactive mission timeline (Gantt-style).

    Args:
        events: List of event dicts with keys:
            - name: Event name
            - start: Start time (days)
            - end: End time (days)
            - color: Event color
            - phase: Mission phase name
        title: Plot title

    Returns:
        Plotly Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly required")

    fig = go.Figure()

    # Group events by phase
    phases = {}
    for event in events:
        phase = event.get('phase', 'Unknown')
        if phase not in phases:
            phases[phase] = []
        phases[phase].append(event)

    y_pos = 0
    y_labels = []

    for phase_name, phase_events in phases.items():
        for event in phase_events:
            fig.add_trace(go.Bar(
                x=[event['end'] - event['start']],
                y=[y_pos],
                base=[event['start']],
                orientation='h',
                name=event['name'],
                marker=dict(color=event.get('color', 'cyan')),
                text=event['name'],
                textposition='inside',
                hovertemplate=(
                    f"<b>{event['name']}</b><br>"
                    f"Start: {event['start']:.2f} days<br>"
                    f"End: {event['end']:.2f} days<br>"
                    f"Duration: {event['end'] - event['start']:.2f} days"
                    "<extra></extra>"
                ),
            ))
            y_labels.append(event['name'])
            y_pos += 1

    fig.update_layout(
        title=title,
        xaxis=dict(title='Mission Time (days)'),
        yaxis=dict(
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels,
        ),
        barmode='overlay',
        showlegend=False,
        height=max(400, len(events) * 40),
        paper_bgcolor='rgb(30, 30, 50)',
        plot_bgcolor='rgb(30, 30, 50)',
        font=dict(color='white'),
    )

    return fig


# ==================== DASH APPLICATION ====================

def create_mission_viewer_app(
    trajectory: TrajectoryData,
    bodies: List[BodyData] = None,
    events: List[EventMarker] = None,
) -> 'Dash':
    """
    Create full-featured mission viewer Dash application.

    Features:
    - Interactive 3D view with time slider
    - Telemetry plots
    - Event timeline
    - Dynamic body focus

    Args:
        trajectory: Spacecraft trajectory
        bodies: Celestial bodies
        events: Mission events

    Returns:
        Dash application
    """
    if not HAS_DASH:
        raise ImportError("Dash required. Install with: pip install dash")

    bodies = bodies or []
    events = events or []

    try:
        app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    except:
        app = Dash(__name__)

    # Serialize data for client-side storage
    traj_json = {
        'times': trajectory.times.tolist(),
        'positions': trajectory.positions.tolist(),
        'velocities': trajectory.velocities.tolist() if trajectory.velocities is not None else None,
        'masses': trajectory.masses.tolist() if trajectory.masses is not None else None,
        'name': trajectory.name,
    }

    bodies_json = [
        {
            'name': b.name,
            'positions': b.positions.tolist(),
            'radius_km': b.radius_km,
            'color': b.color,
        }
        for b in bodies
    ]

    app.layout = html.Div([
        # Hidden data store
        dcc.Store(id='trajectory-data', data=traj_json),
        dcc.Store(id='bodies-data', data=bodies_json),

        # Header
        html.H1(
            f"{trajectory.name} Mission Viewer",
            style={'textAlign': 'center', 'color': 'white', 'padding': '20px'}
        ),

        # Main content
        html.Div([
            # Left panel - 3D view
            html.Div([
                dcc.Graph(
                    id='trajectory-3d',
                    style={'height': '600px'},
                ),
            ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            # Right panel - Telemetry
            html.Div([
                # Time display
                html.Div([
                    html.H3("Mission Time", style={'color': 'white'}),
                    html.Div(id='time-display', style={
                        'fontSize': '24px',
                        'color': 'cyan',
                        'padding': '10px',
                        'backgroundColor': 'rgba(0,0,0,0.3)',
                        'borderRadius': '5px',
                    }),
                ], style={'padding': '10px'}),

                # State display
                html.Div([
                    html.H3("Spacecraft State", style={'color': 'white'}),
                    html.Div(id='state-display', style={
                        'color': 'white',
                        'padding': '10px',
                        'backgroundColor': 'rgba(0,0,0,0.3)',
                        'borderRadius': '5px',
                        'fontFamily': 'monospace',
                    }),
                ], style={'padding': '10px'}),

                # Altitude plot
                dcc.Graph(
                    id='altitude-plot',
                    style={'height': '250px'},
                ),

            ], style={'width': '38%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        ]),

        # Time slider
        html.Div([
            html.Label("Mission Time", style={'color': 'white'}),
            dcc.Slider(
                id='time-slider',
                min=0,
                max=trajectory.n_points - 1,
                value=0,
                step=1,
                marks={
                    0: f'{trajectory.times[0]:.1f}d',
                    trajectory.n_points // 4: f'{trajectory.times[trajectory.n_points // 4]:.1f}d',
                    trajectory.n_points // 2: f'{trajectory.times[trajectory.n_points // 2]:.1f}d',
                    3 * trajectory.n_points // 4: f'{trajectory.times[3 * trajectory.n_points // 4]:.1f}d',
                    trajectory.n_points - 1: f'{trajectory.times[-1]:.1f}d',
                },
            ),
        ], style={'padding': '20px'}),

        # Playback controls
        html.Div([
            html.Button('Play', id='play-btn', n_clicks=0, style={'marginRight': '10px'}),
            html.Button('Pause', id='pause-btn', n_clicks=0, style={'marginRight': '10px'}),
            html.Button('Reset', id='reset-btn', n_clicks=0),
            dcc.Interval(id='playback-interval', interval=100, disabled=True),
        ], style={'padding': '10px', 'textAlign': 'center'}),

        # Focus selector
        html.Div([
            html.Label("Focus:", style={'color': 'white', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='focus-selector',
                options=[
                    {'label': 'Spacecraft', 'value': 'spacecraft'},
                    {'label': 'Origin', 'value': 'origin'},
                ] + [{'label': b.name, 'value': b.name} for b in bodies],
                value='origin',
                style={'width': '200px', 'display': 'inline-block'},
            ),
        ], style={'padding': '10px'}),

    ], style={'backgroundColor': 'rgb(30, 30, 50)', 'minHeight': '100vh'})

    # Callbacks
    @app.callback(
        [Output('trajectory-3d', 'figure'),
         Output('time-display', 'children'),
         Output('state-display', 'children'),
         Output('altitude-plot', 'figure')],
        [Input('time-slider', 'value'),
         Input('focus-selector', 'value')],
        [State('trajectory-data', 'data'),
         State('bodies-data', 'data')],
    )
    def update_display(time_idx, focus, traj_data, bodies_data):
        # Get current state
        positions = np.array(traj_data['positions'])
        times = np.array(traj_data['times'])
        velocities = np.array(traj_data['velocities']) if traj_data['velocities'] else None

        current_pos = positions[time_idx]
        current_time = times[time_idx]
        current_vel = velocities[time_idx] if velocities is not None else None

        # 3D Figure
        fig_3d = go.Figure()

        # Trajectory up to current time
        fig_3d.add_trace(go.Scatter3d(
            x=positions[:time_idx+1, 0],
            y=positions[:time_idx+1, 1],
            z=positions[:time_idx+1, 2],
            mode='lines',
            line=dict(color='cyan', width=3),
            name='Trajectory',
        ))

        # Current position
        fig_3d.add_trace(go.Scatter3d(
            x=[current_pos[0]],
            y=[current_pos[1]],
            z=[current_pos[2]],
            mode='markers',
            marker=dict(color='yellow', size=10, symbol='diamond'),
            name='Spacecraft',
        ))

        # Bodies
        for body in bodies_data:
            body_pos = np.array(body['positions'])
            if body_pos.ndim > 1:
                body_pos = body_pos[min(time_idx, len(body_pos) - 1)]

            # Simplified sphere representation
            fig_3d.add_trace(go.Scatter3d(
                x=[body_pos[0]],
                y=[body_pos[1]],
                z=[body_pos[2]],
                mode='markers',
                marker=dict(color=body['color'], size=20),
                name=body['name'],
            ))

        # Set camera based on focus
        camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))
        if focus == 'spacecraft':
            camera = dict(
                eye=dict(x=current_pos[0]/1e6 + 0.5, y=current_pos[1]/1e6 + 0.5, z=current_pos[2]/1e6 + 0.5),
                center=dict(x=current_pos[0]/1e6, y=current_pos[1]/1e6, z=current_pos[2]/1e6),
            )

        fig_3d.update_layout(
            scene=dict(
                aspectmode='data',
                camera=camera,
                bgcolor='rgb(20, 20, 40)',
            ),
            paper_bgcolor='rgb(30, 30, 50)',
            font=dict(color='white'),
            margin=dict(l=0, r=0, t=0, b=0),
        )

        # Time display
        time_text = f"Day {current_time:.3f}"

        # State display
        altitude = np.linalg.norm(current_pos)
        velocity = np.linalg.norm(current_vel) if current_vel is not None else 0

        state_text = html.Div([
            html.P(f"Position: [{current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f}] km"),
            html.P(f"Altitude: {altitude:,.0f} km"),
            html.P(f"Velocity: {velocity:.3f} km/s"),
        ])

        # Altitude plot
        fig_alt = go.Figure()
        altitudes = np.linalg.norm(positions, axis=1)

        fig_alt.add_trace(go.Scatter(
            x=times,
            y=altitudes,
            mode='lines',
            line=dict(color='cyan'),
            name='Altitude',
        ))

        # Current time marker
        fig_alt.add_vline(x=current_time, line_color='yellow', line_dash='dash')

        fig_alt.update_layout(
            xaxis=dict(title='Time (days)'),
            yaxis=dict(title='Altitude (km)'),
            paper_bgcolor='rgb(30, 30, 50)',
            plot_bgcolor='rgb(30, 30, 50)',
            font=dict(color='white'),
            margin=dict(l=50, r=20, t=20, b=50),
        )

        return fig_3d, time_text, state_text, fig_alt

    @app.callback(
        [Output('playback-interval', 'disabled'),
         Output('time-slider', 'value', allow_duplicate=True)],
        [Input('play-btn', 'n_clicks'),
         Input('pause-btn', 'n_clicks'),
         Input('reset-btn', 'n_clicks'),
         Input('playback-interval', 'n_intervals')],
        [State('time-slider', 'value'),
         State('trajectory-data', 'data')],
        prevent_initial_call=True,
    )
    def control_playback(play_clicks, pause_clicks, reset_clicks, n_intervals, current_val, traj_data):
        from dash import ctx

        n_points = len(traj_data['times'])

        if ctx.triggered_id == 'play-btn':
            return False, current_val
        elif ctx.triggered_id == 'pause-btn':
            return True, current_val
        elif ctx.triggered_id == 'reset-btn':
            return True, 0
        elif ctx.triggered_id == 'playback-interval':
            new_val = (current_val + 1) % n_points
            return False, new_val

        return True, current_val

    return app


def run_mission_viewer(
    trajectory: TrajectoryData,
    bodies: List[BodyData] = None,
    events: List[EventMarker] = None,
    port: int = 8051,
    debug: bool = False,
):
    """
    Run the interactive mission viewer.

    Args:
        trajectory: Spacecraft trajectory
        bodies: Celestial bodies
        events: Mission events
        port: Web server port
        debug: Enable debug mode
    """
    app = create_mission_viewer_app(trajectory, bodies, events)
    print(f"Starting mission viewer at http://localhost:{port}")
    app.run(port=port, debug=debug)


# ==================== UTILITY FUNCTIONS ====================

def trajectory_from_propagation_result(result, name: str = "Spacecraft") -> TrajectoryData:
    """
    Create TrajectoryData from PropagationResult.

    Args:
        result: PropagationResult object from propagator
        name: Trajectory name

    Returns:
        TrajectoryData
    """
    # Extract data from result
    times = result.t
    positions = result.y[:3, :].T * AU_TO_KM  # Convert AU to km
    velocities = result.y[3:6, :].T * AU_TO_KM / DAY_TO_SEC  # Convert AU/day to km/s
    masses = result.y[6, :] if result.y.shape[0] > 6 else None
    quaternions = result.y[7:11, :].T if result.y.shape[0] > 10 else None
    angular_velocities = result.y[11:14, :].T if result.y.shape[0] > 13 else None

    return TrajectoryData(
        times=times,
        positions=positions,
        velocities=velocities,
        masses=masses,
        quaternions=quaternions,
        angular_velocities=angular_velocities,
        name=name,
    )


# ==================== EXPORTS ====================

__all__ = [
    # Data structures
    'TrajectoryData',
    'BodyData',
    'EventMarker',
    # Interactive viewers
    'create_interactive_trajectory_viewer',
    'create_mission_dashboard',
    'create_attitude_visualizer',
    'create_interactive_ground_track',
    'create_mission_timeline',
    # Dash application
    'create_mission_viewer_app',
    'run_mission_viewer',
    # Utilities
    'trajectory_from_propagation_result',
]
