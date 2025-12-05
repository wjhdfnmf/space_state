# visual/maneuver_planner.py
"""
Interactive Maneuver Planning UI

Provides a web-based interface for planning spacecraft maneuvers:
- Interactive orbit visualization
- Maneuver node placement
- Delta-V calculations
- Real-time trajectory preview

Uses Dash/Plotly for web interface.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from dash import Dash, html, dcc, Input, Output, State, callback
    HAS_DASH = True
except ImportError:
    HAS_DASH = False


# ==================== MANEUVER NODE ====================

@dataclass
class ManeuverNode:
    """Represents a planned maneuver."""
    name: str
    t: float               # Time (days)
    delta_v: np.ndarray    # Delta-V [prograde, radial, normal] in m/s

    @property
    def magnitude(self) -> float:
        """Total delta-V magnitude in m/s."""
        return np.linalg.norm(self.delta_v)

    @property
    def prograde(self) -> float:
        """Prograde component in m/s."""
        return self.delta_v[0]

    @property
    def radial(self) -> float:
        """Radial component in m/s."""
        return self.delta_v[1]

    @property
    def normal(self) -> float:
        """Normal component in m/s."""
        return self.delta_v[2]


# ==================== MANEUVER CALCULATIONS ====================

def hohmann_delta_v(r1_km: float, r2_km: float, mu_km3_s2: float) -> Tuple[float, float]:
    """
    Calculate Hohmann transfer delta-V.

    Args:
        r1_km: Initial orbit radius (km)
        r2_km: Final orbit radius (km)
        mu_km3_s2: Gravitational parameter (km³/s²)

    Returns:
        (dv1_mps, dv2_mps): Delta-V at departure and arrival (m/s)
    """
    a_transfer = (r1_km + r2_km) / 2

    v1_circular = np.sqrt(mu_km3_s2 / r1_km) * 1000  # m/s
    v1_transfer = np.sqrt(mu_km3_s2 * (2/r1_km - 1/a_transfer)) * 1000  # m/s
    dv1 = v1_transfer - v1_circular

    v2_circular = np.sqrt(mu_km3_s2 / r2_km) * 1000  # m/s
    v2_transfer = np.sqrt(mu_km3_s2 * (2/r2_km - 1/a_transfer)) * 1000  # m/s
    dv2 = v2_circular - v2_transfer

    return dv1, dv2


def plane_change_delta_v(v_mps: float, angle_deg: float) -> float:
    """
    Calculate plane change delta-V.

    Args:
        v_mps: Orbital velocity (m/s)
        angle_deg: Inclination change (degrees)

    Returns:
        Delta-V for plane change (m/s)
    """
    return 2 * v_mps * np.sin(np.radians(angle_deg) / 2)


def combined_maneuver_delta_v(
    r1_km: float,
    r2_km: float,
    inc_change_deg: float,
    mu_km3_s2: float
) -> float:
    """
    Calculate optimal combined altitude and plane change.

    Performs plane change at apogee for efficiency.

    Returns:
        Total delta-V (m/s)
    """
    dv1, dv2_hohmann = hohmann_delta_v(r1_km, r2_km, mu_km3_s2)

    # Plane change at apogee (lower velocity = less delta-V)
    v_apogee = np.sqrt(mu_km3_s2 * (2/max(r1_km, r2_km) - 2/(r1_km + r2_km))) * 1000
    dv_plane = plane_change_delta_v(v_apogee, inc_change_deg)

    # Combined with circularization
    dv2 = np.sqrt(dv2_hohmann**2 + dv_plane**2 - 2*dv2_hohmann*dv_plane*np.cos(np.radians(inc_change_deg)))

    return abs(dv1) + abs(dv2)


# ==================== ORBIT PREVIEW ====================

def preview_orbit(
    r0_km: float,
    mu_km3_s2: float,
    dv_prograde_mps: float = 0,
    dv_radial_mps: float = 0,
    n_points: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Preview orbit after applying delta-V.

    Args:
        r0_km: Initial circular orbit radius (km)
        mu_km3_s2: Gravitational parameter
        dv_prograde_mps: Prograde delta-V (m/s)
        dv_radial_mps: Radial delta-V (m/s)
        n_points: Number of points for orbit plot

    Returns:
        Dict with 'x', 'y', 'apoapsis_km', 'periapsis_km', 'period_hours'
    """
    # Initial velocity (circular)
    v0 = np.sqrt(mu_km3_s2 / r0_km) * 1000  # m/s

    # Apply delta-V
    v_new = np.sqrt((v0 + dv_prograde_mps)**2 + dv_radial_mps**2) / 1000  # km/s

    # Calculate orbital elements
    r = r0_km
    v = v_new

    # Specific energy and semi-major axis
    energy = v**2 / 2 - mu_km3_s2 / r
    if abs(energy) < 1e-10:
        a = np.inf  # Parabolic
    else:
        a = -mu_km3_s2 / (2 * energy)

    # Eccentricity
    h = r * (v0 + dv_prograde_mps) / 1000  # Angular momentum (km²/s)
    e = np.sqrt(1 + 2 * energy * h**2 / mu_km3_s2**2)
    e = max(0, min(e, 0.999))  # Clamp

    if a > 0:
        apoapsis = a * (1 + e)
        periapsis = a * (1 - e)
        period = 2 * np.pi * np.sqrt(a**3 / mu_km3_s2) / 3600  # hours
    else:
        apoapsis = np.inf
        periapsis = a * (1 - e) if a < 0 else r
        period = np.inf

    # Generate orbit path
    theta = np.linspace(0, 2*np.pi, n_points)
    if a > 0 and e < 1:
        r_orbit = a * (1 - e**2) / (1 + e * np.cos(theta))
        x = r_orbit * np.cos(theta)
        y = r_orbit * np.sin(theta)
    else:
        # Just draw initial orbit for hyperbolic
        x = r0_km * np.cos(theta)
        y = r0_km * np.sin(theta)

    return {
        'x': x,
        'y': y,
        'apoapsis_km': apoapsis,
        'periapsis_km': periapsis,
        'period_hours': period,
        'semi_major_axis_km': a,
        'eccentricity': e,
    }


# ==================== PLOTLY FIGURES ====================

def create_maneuver_figure(
    initial_orbit_km: float = 6771,  # LEO
    mu_km3_s2: float = 398600.4,     # Earth
    body_radius_km: float = 6371,    # Earth
    dv_prograde: float = 0,
    dv_radial: float = 0,
) -> 'go.Figure':
    """
    Create interactive maneuver planning figure.

    Args:
        initial_orbit_km: Initial circular orbit radius (km)
        mu_km3_s2: Gravitational parameter
        body_radius_km: Central body radius
        dv_prograde: Prograde delta-V (m/s)
        dv_radial: Radial delta-V (m/s)

    Returns:
        Plotly Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly not available")

    # Initial orbit
    theta = np.linspace(0, 2*np.pi, 100)
    x_init = initial_orbit_km * np.cos(theta)
    y_init = initial_orbit_km * np.sin(theta)

    # Preview orbit after maneuver
    preview = preview_orbit(initial_orbit_km, mu_km3_s2, dv_prograde, dv_radial)

    fig = go.Figure()

    # Central body
    body_theta = np.linspace(0, 2*np.pi, 50)
    fig.add_trace(go.Scatter(
        x=body_radius_km * np.cos(body_theta),
        y=body_radius_km * np.sin(body_theta),
        fill='toself',
        fillcolor='lightblue',
        line=dict(color='blue'),
        name='Earth',
    ))

    # Initial orbit (dashed)
    fig.add_trace(go.Scatter(
        x=x_init, y=y_init,
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Initial Orbit',
    ))

    # New orbit
    fig.add_trace(go.Scatter(
        x=preview['x'], y=preview['y'],
        mode='lines',
        line=dict(color='cyan', width=2),
        name='New Orbit',
    ))

    # Maneuver point
    fig.add_trace(go.Scatter(
        x=[initial_orbit_km], y=[0],
        mode='markers',
        marker=dict(color='red', size=12, symbol='diamond'),
        name='Maneuver',
    ))

    # Add delta-V vector
    if dv_prograde != 0 or dv_radial != 0:
        scale = 500  # Scale for visualization
        fig.add_trace(go.Scatter(
            x=[initial_orbit_km, initial_orbit_km + dv_radial*scale],
            y=[0, dv_prograde*scale],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            name=f'ΔV: {np.sqrt(dv_prograde**2 + dv_radial**2):.1f} m/s',
        ))

    # Annotations
    annotations = [
        f"Initial: {initial_orbit_km - body_radius_km:.0f} km altitude",
        f"Apoapsis: {preview['apoapsis_km'] - body_radius_km:.0f} km",
        f"Periapsis: {preview['periapsis_km'] - body_radius_km:.0f} km",
        f"Period: {preview['period_hours']:.2f} hours",
        f"ΔV: {np.sqrt(dv_prograde**2 + dv_radial**2):.1f} m/s",
    ]

    fig.update_layout(
        title=f"Maneuver Planning - {' | '.join(annotations[-2:])}",
        xaxis=dict(
            title='X (km)',
            scaleanchor='y',
            scaleratio=1,
        ),
        yaxis=dict(title='Y (km)'),
        showlegend=True,
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text='<br>'.join(annotations),
                showarrow=False,
                align='left',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1,
            )
        ],
    )

    return fig


# ==================== DASH APPLICATION ====================

def create_maneuver_planner_app(
    initial_altitude_km: float = 400,
    body_name: str = 'earth',
) -> 'Dash':
    """
    Create interactive maneuver planning Dash application.

    Args:
        initial_altitude_km: Initial orbit altitude (km)
        body_name: Central body name

    Returns:
        Dash application
    """
    if not HAS_DASH:
        raise ImportError("Dash not available. Install with: pip install dash")

    from base.star_constants import BODIES

    body = BODIES.get(body_name, BODIES['earth'])
    body_radius_km = body['radius'] * 149597870.7  # AU to km
    mu = body['mu_km3_s2']
    initial_orbit_km = body_radius_km + initial_altitude_km

    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Maneuver Planner", style={'textAlign': 'center'}),

        html.Div([
            html.Div([
                html.Label("Prograde ΔV (m/s):"),
                dcc.Slider(
                    id='prograde-slider',
                    min=-500, max=3000, value=0, step=10,
                    marks={i: f'{i}' for i in range(-500, 3001, 500)},
                ),
                html.Div(id='prograde-value'),
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px'}),

            html.Div([
                html.Label("Radial ΔV (m/s):"),
                dcc.Slider(
                    id='radial-slider',
                    min=-200, max=200, value=0, step=5,
                    marks={i: f'{i}' for i in range(-200, 201, 100)},
                ),
                html.Div(id='radial-value'),
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px'}),
        ]),

        dcc.Graph(id='orbit-plot', style={'height': '600px'}),

        html.Div([
            html.H3("Orbital Parameters"),
            html.Div(id='orbit-info'),
        ], style={'padding': '20px'}),

        html.Div([
            html.H3("Common Maneuvers"),
            html.Button("Hohmann to GEO", id='btn-geo', n_clicks=0),
            html.Button("Raise 100 km", id='btn-raise', n_clicks=0),
            html.Button("Lower 100 km", id='btn-lower', n_clicks=0),
            html.Button("Reset", id='btn-reset', n_clicks=0),
        ], style={'padding': '20px'}),
    ])

    @app.callback(
        [Output('orbit-plot', 'figure'),
         Output('orbit-info', 'children'),
         Output('prograde-value', 'children'),
         Output('radial-value', 'children')],
        [Input('prograde-slider', 'value'),
         Input('radial-slider', 'value')],
    )
    def update_plot(prograde, radial):
        fig = create_maneuver_figure(
            initial_orbit_km=initial_orbit_km,
            mu_km3_s2=mu,
            body_radius_km=body_radius_km,
            dv_prograde=prograde,
            dv_radial=radial,
        )

        preview = preview_orbit(initial_orbit_km, mu, prograde, radial)

        info = html.Ul([
            html.Li(f"Semi-major axis: {preview['semi_major_axis_km']:.1f} km"),
            html.Li(f"Eccentricity: {preview['eccentricity']:.4f}"),
            html.Li(f"Apoapsis altitude: {preview['apoapsis_km'] - body_radius_km:.1f} km"),
            html.Li(f"Periapsis altitude: {preview['periapsis_km'] - body_radius_km:.1f} km"),
            html.Li(f"Period: {preview['period_hours']:.2f} hours"),
            html.Li(f"Total ΔV: {np.sqrt(prograde**2 + radial**2):.1f} m/s"),
        ])

        return fig, info, f"Prograde: {prograde} m/s", f"Radial: {radial} m/s"

    @app.callback(
        [Output('prograde-slider', 'value'),
         Output('radial-slider', 'value')],
        [Input('btn-geo', 'n_clicks'),
         Input('btn-raise', 'n_clicks'),
         Input('btn-lower', 'n_clicks'),
         Input('btn-reset', 'n_clicks')],
        prevent_initial_call=True,
    )
    def preset_maneuvers(geo_clicks, raise_clicks, lower_clicks, reset_clicks):
        from dash import ctx

        button_id = ctx.triggered_id

        if button_id == 'btn-geo':
            # Hohmann to GEO from LEO
            dv1, _ = hohmann_delta_v(initial_orbit_km, 42164, mu)
            return dv1, 0
        elif button_id == 'btn-raise':
            # Raise orbit by 100 km
            dv1, _ = hohmann_delta_v(initial_orbit_km, initial_orbit_km + 100, mu)
            return dv1, 0
        elif button_id == 'btn-lower':
            # Lower orbit by 100 km
            dv1, _ = hohmann_delta_v(initial_orbit_km, max(body_radius_km + 100, initial_orbit_km - 100), mu)
            return dv1, 0
        else:
            return 0, 0

    return app


def run_maneuver_planner(port: int = 8050, debug: bool = False):
    """
    Run the interactive maneuver planner.

    Args:
        port: Port number for web server
        debug: Enable debug mode
    """
    app = create_maneuver_planner_app()
    print(f"Starting maneuver planner at http://localhost:{port}")
    app.run(port=port, debug=debug)


# ==================== EXPORTS ====================

__all__ = [
    'ManeuverNode',
    'hohmann_delta_v',
    'plane_change_delta_v',
    'combined_maneuver_delta_v',
    'preview_orbit',
    'create_maneuver_figure',
    'create_maneuver_planner_app',
    'run_maneuver_planner',
]


if __name__ == '__main__':
    run_maneuver_planner(debug=True)
