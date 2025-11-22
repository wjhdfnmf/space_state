# star_vis2D.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from star_utils import au_to_km, apd_kps, body_position, body_velocity, au2day2_mjkg
from star_constants import BODIES, ts, epoch_jd

def plot_orbital_metrics(
    result,
    bodies_to_plot,
    title="Orbital Visualizer 2D"
):
    t_days = result.t
    t_hours = t_days # * 24.0

    r_au = result.y[:3, :].T      # (N, 3) position in SSB
    v_au_day = result.y[3:6, :].T # (N, 3) velocity in SSB
    m_kg = result.y[6, :].T      # (N,) mass in kg

    r_km = au_to_km(r_au)
    v_kms = apd_kps(v_au_day)

    # Precompute body positions over time
    body_pos_au = {}
    body_velocity_au_day = {}
    for body in bodies_to_plot:
        if body in BODIES and BODIES[body]['skyfield'] is not None:
            body_pos_au[body] = np.array([
                body_position(BODIES[body], ts.tt_jd(epoch_jd + t)) for t in t_days])
            body_velocity_au_day[body] = np.array(
                [body_velocity(BODIES[body], ts.tt_jd(epoch_jd + t)) for t in t_days])

    # ------------------------------------------------------------------
    # 1. Earth-Centered Altitude (from surface)
    # ------------------------------------------------------------------
    if 'earth' in bodies_to_plot and 'earth' in body_pos_au:
        earth_pos_au = body_pos_au['earth']
        dr_earth = r_au - earth_pos_au
        dist_earth_center_km = au_to_km(np.linalg.norm(dr_earth, axis=1))
        altitude_earth_km = dist_earth_center_km - au_to_km(BODIES['earth']['radius'])  # FROM SURFACE

    # ------------------------------------------------------------------
    # 2. Moon Altitude (from surface)
    # ------------------------------------------------------------------
    altitude_moon_km = None
    if 'moon' in bodies_to_plot and 'moon' in body_pos_au:
        moon_pos_au = body_pos_au['moon']
        dr_moon = r_au - moon_pos_au
        dist_moon_center_km = au_to_km(np.linalg.norm(dr_moon, axis=1))
        altitude_moon_km = dist_moon_center_km - au_to_km(BODIES['moon']['radius'])  # FROM SURFACE

    # ------------------------------------------------------------------
    # 3. Orbital Elements (Earth-Centered, Inertial)
    # ------------------------------------------------------------------
    # At each time step
    earth_center_au = body_pos_au['earth']
    dr_earth = r_au - earth_center_au
    v_earth_center = body_velocity_au_day['earth']

    v_rel_au_day = v_au_day - v_earth_center  # Relative to Earth center

    h_vec = np.cross(dr_earth, v_rel_au_day, axis=1)
    h = np.linalg.norm(h_vec, axis=1)
    v2 = np.sum(v_rel_au_day**2, axis=1)
    r_norm = np.linalg.norm(dr_earth, axis=1) + 1e-20

    mu = BODIES['earth']['mu']  # AU³/day²

    energy = 0.5 * v2 - mu / r_norm     # Specific orbital energy (AU²/day²)
    energymjkg = au2day2_mjkg(energy)   # Convert to MJ/kg
    a = -mu / (2 * energy + 1e-30)                  # Semi-major axis (au)
    e = np.sqrt(1 + (2 * energy * h**2) / mu**2)    # Eccentricity (0=circ, <1=ellip, 1=parabolic, >1=hyperbolic)

    apo_km = au_to_km(a * (1 + e) - BODIES['earth']['radius'])
    peri_km = au_to_km(a * (1 - e) - BODIES['earth']['radius'])

    # ------------------------------------------------------------------
    # 4. Thrust (placeholder)
    # ------------------------------------------------------------------
    thrust_mN_per_kg = np.zeros_like(t_hours)

    # ------------------------------------------------------------------
    # PLOTLY — 6 Subplots
    # ------------------------------------------------------------------

    tiles = {
        'earth_altitude': {
            'title': "Altitude above Earth [km]",
            'yaxis_title': "Altitude [km]",
            'lines': [
                {'name': "Earth Altitude", 'y': altitude_earth_km, 'line': dict(color="#00CED1")},
            ],
            'enabled': 'earth' in bodies_to_plot,
        },
        'moon_altitude': {
            'title': "Altitude above Moon [km]",
            'yaxis_title': "Altitude [km]",
            'lines': [
                {'name': "Moon Altitude", 'y': altitude_moon_km, 'line': dict(color="#FFD700")},
            ],
            'enabled': 'moon' in bodies_to_plot,
        },
        'eccentricity': {
            'title': "Eccentricity (Earth-Centered) 0=circ, <1=ellip, 1=parabolic, >1=hyperbolic",
            'yaxis_title': "Eccentricity",
            'lines': [
                {'name': "Eccentricity", 'y': e, 'line': dict(color="#FF69B4")},
            ],
            'enabled': True,
        },
        'apo_periapsis': {
            'title': "Apoapsis & Periapsis [km]",
            'yaxis_title': "Distance [km]",
            'lines': [
                {'name': "Apoapsis", 'y': apo_km, 'line': dict(color="#FF4500")},
                {'name': "Periapsis", 'y': peri_km, 'line': dict(color="#32CD32")}
            ],
            'enabled': True,
        },
        'specific_energy': {
            'title': "Specific Energy (Earth) [MJ/kg]",
            'yaxis_title': "Energy [MJ/kg]",
            'lines': [
                {'name': "Specific Energy", 'y': energymjkg, 'line': dict(color="#FFFF00")},
            ],
            'enabled': True,
        },
        'thrust': {
            'title': "Thrust [mN/kg]",
            'yaxis_title': "Thrust [mN/kg]",
            'lines': [
                {'name': "Thrust", 'y': thrust_mN_per_kg, 'line': dict(color="#FF0000")},
            ],
            'enabled': False,
        }
    }

    active_tiles = [tile for tile in tiles.values() if tile['enabled']]
    titles = [tile['title'] for tile in active_tiles]
    n_rows = len(active_tiles)

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=titles
    )

    for each in active_tiles:
        for line in each['lines']:
            fig.add_trace(
                go.Scatter(x=t_hours, y=line['y'], name=line['name'], line=line['line']),
                row=active_tiles.index(each) + 1, col=1
            )
            fig.update_yaxes(title_text=each['yaxis_title'], row=active_tiles.index(each) + 1, col=1)

    # Layout
    fig.update_layout(
        height=1000,
        title_text=title,
        title_x=0.5,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color="white"),
        legend=dict(bgcolor="#1a1a1a", bordercolor="gray")
    )

    fig.update_xaxes(title_text="Time [hours]", row=6, col=1, color="white")
    fig.update_yaxes(color="white")

    fig.show()

    #return fig  # Return for interactive use; or use fig.show()
    