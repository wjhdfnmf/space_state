# base/star_states.py

import numpy as np
from typing import Tuple, Literal
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import time
from scipy.integrate import solve_ivp

from .star_logger import logger
from .star_config import CONFIG
from .star_constants import epoch_jd, ts, world_time, BODIES
from .star_propagator import Propagator
from .star_utils import body_position, body_velocity, km_to_au, au_to_km, apd_kps, day_to_sec, apd_to_mps, min_to_days, ssb_to_body_centered
from .star_orbits import ALL_ORBITS
from guidance.star_tools import time_to_apoapsis, get_apoapsis_state, vis_viva, find_nodes, distance_to_body
from visual.star_plane_vis import plot_orbital_plane
import matplotlib.pyplot as plt

OrbitName = Literal[
    'leo_200', 'geo', 'tli_apo',
    'lunar_100km',
]

# 
# Get Circular State
#
def get_circular_state(BODY, orbit: OrbitName) -> Tuple[np.ndarray, np.ndarray]:
    """Return (r_ssb, v_ssb) for circular equatorial orbit at specified altitude."""
    alt = ALL_ORBITS[orbit].altitude_km
    altitude = km_to_au(alt)

    # Radial position and circular velocity in BODY-centered frame
    r0 = np.array([BODY['radius'] + altitude, 0.0, 0.0])    # [r, 0, 0]         AU
    v_circ = np.sqrt(BODY['mu'] / np.linalg.norm(r0))
    v0 = np.array([0.0, v_circ, 0.0])                       # [0, v_circ, 0]    AU/day

    # Convert to SSB frame
    r0_ssb = body_position(BODY, ts.tt_jd(epoch_jd)) + r0
    v0_ssb = body_velocity(BODY, ts.tt_jd(epoch_jd)) + v0

    return r0_ssb, v0_ssb

#
# Get Circular State at Apoapsis
#
def get_circular_state_at_apoapsis(BODY, t: float, y0: np.ndarray) -> dict:
    """Return circular orbit state at apoapsis from current state around BODY."""

    # Relative position and velocity to BODY
    r_rel = y0[:3] - body_position(BODY, ts.tt_jd(epoch_jd + t))
    v_rel = y0[3:6] - body_velocity(BODY, ts.tt_jd(epoch_jd + t))

    # Time to apoapsis (for SSB conversion)
    t_to_apo = time_to_apoapsis(r_rel, v_rel, BODY['mu'])
    t_apo = t + t_to_apo

    if t_to_apo < 0:
        raise ValueError("Already past apoapsis!")
    if t_to_apo == np.inf:
        raise ValueError("No apoapsis for parabolic/hyperbolic orbit. Cannot circularize.")

    # Exact apoapsis state relative to BODY
    apo_state = get_apoapsis_state(r_rel, v_rel, BODY['mu'])

    r_apo_rel = apo_state["r_apo"]       # Position at apoapsis relative to BODY
    r_target = np.linalg.norm(r_apo_rel) # Circular orbit radius at apoapsis
    
    # Circular velocity
    v_circ_mag = np.sqrt(BODY['mu'] / r_target)
    h_unit = np.cross(r_apo_rel, apo_state["v_apo"])
    h_unit /= np.linalg.norm(h_unit)
    v_circ_dir = np.cross(h_unit, r_apo_rel)
    v_circ_dir /= np.linalg.norm(v_circ_dir)
    v_circ_rel = v_circ_dir * v_circ_mag

    # Delta-V to circularize
    v_delta = np.linalg.norm(v_circ_rel - apo_state['v_apo'])

    # State of BODY at apoapsis time
    r_body_apo = body_position(BODY, ts.tt_jd(epoch_jd + t_apo))
    v_body_apo = body_velocity(BODY, ts.tt_jd(epoch_jd + t_apo))

    # SSB frame
    r_apo_ssb = r_body_apo + r_apo_rel
    v_apo_ssb = v_body_apo + v_circ_rel

    y_apo = np.concatenate([r_apo_ssb, v_apo_ssb, [0.0]])

    return {
        # target
        't_center': t_apo,
        'delta_v': apd_to_mps(v_delta),
        # info
        'r': r_apo_ssb,
        'v': v_apo_ssb,
        'y_transfer': y_apo,
        't_to': t_to_apo,
        'r_rel_km': au_to_km(r_apo_rel),
        'v_rel_kms': apd_kps(v_circ_rel),
        't_to_sec': day_to_sec(t_to_apo),
    }

#
# Hohmann Transfer State at Target Time
#
def get_hohmann_state_at_time(
        prop: Propagator, # current state
        body: dict,
        t_target: float,            # days
        target_orbit: OrbitName,
        use_prop: bool = False,
        max_step: float = CONFIG.max_step,
        offset_km: float = 0.0,
    ) -> dict:
    """Return a new state along the current orbit that applies the Hohmann transfer at time t"""
    
    # Propagate to target time to get future state
    if use_prop:
        prop_hist = prop
    else:
        prop_hist = Propagator()
        prop_hist.result.initialize(prop.y, prop.t)
    
    prop_hist.propagate(t_target, max_step=max_step)

    # Relative position and velocity to BODY at target time
    earth_pos = body_position(body, ts.tt_jd(epoch_jd + t_target))
    earth_vel = body_velocity(body, ts.tt_jd(epoch_jd + t_target))
    r_rel = prop_hist.r - earth_pos
    v_rel = prop_hist.v - earth_vel

    # Initial and target orbit radii
    r1_au = np.linalg.norm(r_rel)                                                         # Initial orbit radius
    r2_au = km_to_au(ALL_ORBITS[target_orbit].altitude_km + offset_km) + body['radius']   # Target orbit radius

    # Compute ΔV to enter Hohmann transfer orbit (prograde)
    a_trans = (r1_au + r2_au) / 2 # Semi-major axis of transfer orbit
    v_current = np.linalg.norm(v_rel)
    v_trans_peri = vis_viva(body['mu'], r1_au, a_trans)
    dv = v_trans_peri - v_current  # AU/day

    # Apply Δv in current velocity direction
    v_dir_rel = v_rel / np.linalg.norm(v_rel)
    v_transfer_rel = v_rel + dv * v_dir_rel

    # Final transfer state in SSB frame
    v_transfer = v_transfer_rel + earth_vel  # AU/day in SSB frame
    y_transfer = np.concatenate([prop_hist.r, v_transfer, [prop_hist.m]])

    # Return new target state
    return {
        # target
        't_center': t_target,
        'delta_v': apd_to_mps(dv),
        # info
        'y_transfer': y_transfer,
        'r1_km': au_to_km(r1_au),
        'r2_km': au_to_km(r2_au),
        'prop': prop,
    }

def inclination_adjustment(prop: Propagator, body: dict, target_body: dict, plot: bool = False) -> dict:
    """Adjust transfer velocity v_rel to match target body's orbital plane (aligns h)."""

    r_ssb = prop.r
    v_ssb = prop.v
    t0 = prop.t
    
    # Relative state at t0
    r_body0 = body_position(body, world_time(t0))
    v_body0 = body_velocity(body, world_time(t0))
    r_target0 = body_position(target_body, world_time(t0)) - r_body0
    v_target0 = body_velocity(target_body, world_time(t0)) - v_body0
    r_rel0 = r_ssb - r_body0
    v_rel0 = v_ssb - v_body0

    # Find nodes of intersection of the two orbital planes
    nodes = find_nodes(r_rel0, v_rel0, r_target0, v_target0, t0, body, t_span=1.0)

    t_node = nodes['ascending_t']

    # Propagate to t_node
    y0 = np.concatenate([r_ssb, v_ssb, [1.0]])
    prop = Propagator()
    prop.result.initialize(y0, t0)
    prop.propagate(t_node)

    # Relative state at t_node
    r_body = body_position(body, world_time(t_node))
    v_body = body_velocity(body, world_time(t_node))
    r_rel = prop.result.state[:3] - r_body
    v_rel = prop.result.state[3:6] - v_body
    r_target = body_position(target_body, world_time(t_node)) - r_body
    v_target = body_velocity(target_body, world_time(t_node)) - v_body
    
    # Compute inclination difference
    h_rel_hat = np.cross(r_rel, v_rel) / np.linalg.norm(np.cross(r_rel, v_rel))
    h_target_hat = np.cross(r_target, v_target) / np.linalg.norm(np.cross(r_target, v_target))

    cos_di = np.clip(np.dot(h_rel_hat, h_target_hat), -1.0, 1.0)
    delta_i = np.arccos(cos_di)

    cross = np.cross(h_rel_hat, h_target_hat)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-10:
        logger.debug("No inclination adjustment needed (planes aligned).")
        return  # No adjustment needed
    
    axis = cross / cross_norm

    # Enforce r in target plane (project if drift)
    h_t_norm = np.dot(r_rel, h_target_hat)
    if abs(h_t_norm) > 1e-10:  # AU
        logger.warning(f"r offset from plane: {h_t_norm:.2e} AU; projecting.")
        r_rel -= h_t_norm * h_target_hat

    # Rotate v_rel around axis by delta_i (preserves |v| for pure plane change)
    rot_pos = R.from_rotvec(delta_i * axis)
    v_test_pos = rot_pos.apply(v_rel)
    h_test_pos = np.cross(r_rel, v_test_pos) / np.linalg.norm(np.cross(r_rel, v_test_pos))
    cos_pos = np.dot(h_test_pos, h_target_hat)

    rot_neg = R.from_rotvec(-delta_i * axis)
    v_test_neg = rot_neg.apply(v_rel)
    h_test_neg = np.cross(r_rel, v_test_neg) / np.linalg.norm(np.cross(r_rel, v_test_neg))
    cos_neg = np.dot(h_test_neg, h_target_hat)

    if cos_pos >= cos_neg:
        v_final = v_test_pos
        delta_i_signed = delta_i
    else:
        v_final = v_test_neg
        delta_i_signed = -delta_i

    delta_i_deg = np.degrees(delta_i_signed)

    # Δv
    dv_vec = v_final - v_rel  # AU/day
    dv_mag = np.linalg.norm(dv_vec)  # AU/day
    dv_dir = dv_vec / dv_mag

    # Validate
    h_new_hat = np.cross(r_rel, v_final) / np.linalg.norm(np.cross(r_rel, v_final))
    residual_deg = np.degrees(np.arccos(np.clip(np.dot(h_new_hat, h_target_hat), -1.0, 1.0)))

    if plot: plot_orbital_plane(r_rel, v_final, r_target, v_target)
    
    return {
        # target
        't_center': t_node,
        'delta_v': apd_to_mps(dv_mag),
        'dir': dv_dir,
        # info
        'delta_i_deg': delta_i_deg,
        'residual_deg': residual_deg,
    }
    

# Escape from Earth (C3 = 0 → parabolic)
def escape_state_earth(BODY: dict, t: float, y0: np.ndarray) -> np.ndarray:
    """Modify velocity at current point to achieve parabolic escape (v_esc)."""
    r_rel = y0[:3] - body_position(BODY, ts.tt_jd(epoch_jd + t))
    v_rel = y0[3:6] - body_velocity(BODY, ts.tt_jd(epoch_jd + t))
    r = np.linalg.norm(r_rel)
    v_esc = vis_viva(BODY['mu'], r, 0) # Escape velocity
    v_dir = v_rel / np.linalg.norm(v_rel)
    v_new = v_dir * v_esc
    v = v_new + body_velocity(BODY, ts.tt_jd(epoch_jd + t))
    return np.concatenate([y0[:3], v, y0[6:]])


# Moon Intercept Analysis for departure time optimization
def moon_intercept(
        prop: Propagator,
        body: dict,
        target_body: dict,
        time_forward: float = 0.1,
        grip_size: int = 1_000,
        plot: bool = False,
) -> dict:
    
    t0 = prop.t
    y0 = prop.y

    distance_tracking = []

    prop_hist = Propagator()
    prop_hist.result.initialize(y0, t0)

    logger.debug(f"Starting Moon intercept analysis...")

    #for t in np.linspace(t0, t0 + time_forward, grip_size):
    for t in tqdm(np.linspace(t0, t0 + time_forward, grip_size), desc="Processing", unit="steps"):

        t1 = get_hohmann_state_at_time(prop_hist, body, t, 'tli_apo', use_prop=True)
        t2 = get_circular_state_at_apoapsis(body, t, t1['y_transfer'])

        distance = distance_to_body(t2['t_center'], t2['y_transfer'], target_body, surface=True)
        distance_tracking.append((t, distance))

    # Extract for plotting
    ts_departure, distances_km = zip(*distance_tracking)

    # Summary stats
    min_dist = min(distances_km)
    min_idx = distances_km.index(min_dist)
    t_closest = ts_departure[min_idx]

    # Flight statistics
    prop_hist = Propagator()
    prop_hist.result.initialize(y0, t0)

    t1 = get_hohmann_state_at_time(prop_hist, body, t_closest, 'tli_apo')
    t2 = get_circular_state_at_apoapsis(body, t_closest, t1['y_transfer'])


    distance = distance_to_body(t2['t_center'], t2['y_transfer'], target_body, surface=True)
    logger.debug(f"Closest approach at t={t2['t_center']:.4f} days with distance {distance:.2f} km")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(ts_departure, distances_km, 'o-', linewidth=2, markersize=4, color='blue')
        plt.xlabel('Departure Time (days from t0)')
        plt.ylabel('Distance to Moon at Apoapsis (km)')
        plt.title('Departure Time vs Moon Intercept Distance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('departure_vs_distance.png', dpi=300, bbox_inches='tight')  # Save plot
        plt.show()  # Or display

    TOF = t2['t_center'] - t_closest
    delta_v = t1['delta_v'] + t2['delta_v']

    return {
        "depart_t": t_closest,
        "distance":min_dist,
        "TOF": TOF,
        "delta_v": delta_v
    }

def finalize_lunar_orbit_insertion(
        prop: Propagator,
        time_forward: float = 2.0,
        grip_size: int = 1_000,
        plot: bool = True,
) -> dict:
    """Determine burn to enter lunar orbit when distance to Moon is minimized."""

    body = BODIES['earth']
    target_body = BODIES['moon']

    t0 = prop.t
    y0 = prop.y

    distance_tracking = []

    prop_hist = Propagator()
    prop_hist.result.initialize(y0, t0)

    logger.debug(f"Starting Moon intercept analysis...")

    for t in tqdm(np.linspace(t0, t0 + time_forward, grip_size), desc="Processing", unit="steps"):

        prop_hist.propagate(t)

        distance = distance_to_body(prop_hist.t, prop_hist.y, target_body, surface=True)
        distance_tracking.append((t, distance))

    # Extract for plotting
    times_tested, distances_km = zip(*distance_tracking)

    # Summary stats
    min_dist = min(distances_km)
    min_idx = distances_km.index(min_dist)
    t_closest = times_tested[min_idx]

    # Flight statistics
    prop_hist = Propagator()
    prop_hist.result.initialize(y0, t0)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(times_tested, distances_km, 'o-', linewidth=2, markersize=4, color='blue')
        plt.xlabel('Coast Time (days from t0)')
        plt.ylabel('Distance to Moon at Apoapsis (km)')
        plt.title('Coast Time vs Moon Intercept Distance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('departure_vs_distance.png', dpi=300, bbox_inches='tight')  # Save plot
        plt.show()  # Or display

    logger.debug(f"Closest approach at t={t_closest:.4f} days with distance {min_dist:.2f} km")

    # Determine circularization at closest approach
    prop_hist.propagate(t_closest)

    t_burn = prop_hist.t
    r_ssb = prop_hist.r
    v_ssb = prop_hist.v

    # Relative position and velocity to BODY at target time
    r_body = body_position(target_body, ts.tt_jd(epoch_jd + t_burn))
    v_body = body_velocity(target_body, ts.tt_jd(epoch_jd + t_burn))
    r_rel = r_ssb - r_body
    v_rel = v_ssb - v_body
    
    # Circular velocity
    v_circ_mag = np.sqrt(target_body['mu'] / np.linalg.norm(r_rel))
    h_unit = np.cross(r_rel, v_rel)
    h_unit /= np.linalg.norm(h_unit)
    v_circ_dir = np.cross(h_unit, r_rel)
    v_circ_dir /= np.linalg.norm(v_circ_dir)
    v_circ_rel = v_circ_dir * v_circ_mag

    # Delta-V to circularize
    v_delta = np.linalg.norm(v_circ_rel - v_rel)
    dv_dir = (v_circ_rel - v_rel) / v_delta
    logger.debug(f"LOI burn at t={t_burn:.4f} days with Δv={apd_to_mps(v_delta):.2f} m/s")

    return {
        # target
        't_center': t_burn,
        'delta_v': apd_to_mps(v_delta),
        'dir': dv_dir,
        # info
    }

def finalize_lunar_orbit_insertion2(
    prop: Propagator,
    body: dict,  # Unused; kept for API compat
    target_body: dict,  # Moon
    r_peri_alt_km: float = 100.0,  # Perilune altitude (km); Apollo ~60 NM ~111 km
    r_apo_alt_km: float = 1000.0,  # Apocynthion for elliptical safety (km); Apollo ~170 NM ~315 km
    plot: bool = False,
) -> dict:
    """
    Determine LOI-1 burn for elliptical lunar capture at perilune (Apollo-style).
    Returns: {'t_burn': float (days), 'delta_v_mps': float, 'thrust_dir': np.array(3,),
              'burn_duration_days': float, 'target_a_km': float, 'target_e': float}
    Advisory: For direct circular, set r_apo_alt_km = r_peri_alt_km (higher Δv ~1 km/s vs. ~0.9 km/s split).
    """
    # Current SSB state
    t0 = prop.result.time  # Fix: Use prop.result per star_structures.py
    y0_ssb = prop.result.state
    time_obj = ts.tt_jd(epoch_jd + t0)

    # Moon-relative initial state (advisory: switch frame for accuracy)
    state_moon = ssb_to_body_centered(time_obj, y0_ssb, target_body)
    r0_rel = state_moon['r_au']  # AU, Moon-centered
    v0_rel = state_moon['v_aud']  # AU/day

    mu_moon = target_body['mu']  # AU^3/day^2
    r_moon = target_body['radius']  # AU

    # Target orbit params (elliptical)
    r_peri = r_moon + km_to_au(r_peri_alt_km)
    r_apo = r_moon + km_to_au(r_apo_alt_km)
    a_target = (r_peri + r_apo) / 2
    e_target = (r_apo - r_peri) / (r_apo + r_peri)
    v_peri_target = np.sqrt(mu_moon * (2 / r_peri - 1 / a_target))  # AU/day, vis-viva

    logger.info(f"Target elliptical: peri={r_peri_alt_km} km, apo={r_apo_alt_km} km, a={au_to_km(a_target):.0f} km, e={e_target:.3f}")

    # Propagate ~1 hour forward in Moon frame to find perilune
    tf = t0 + min_to_days(60)
    y0_rel = np.concatenate([r0_rel, v0_rel])

    def radial_velocity_event(t: float, y: np.ndarray) -> float:
        """Event for radial velocity zero (perilune/apolune)."""
        r, v = y[:3], y[3:]
        r_norm = np.linalg.norm(r)
        if r_norm == 0:
            return 1.0  # Avoid div0
        return np.dot(v, r) / r_norm  # vr; zero at peri/apo

    radial_velocity_event.terminal = False
    radial_velocity_event.direction = -1  # Approach from positive vr (outbound) to negative (inbound) for perilune

    def relative_ode(t_rel: float, y_rel: np.ndarray) -> np.ndarray:
        """2-body ODE in Moon frame (ignores Earth perturbations; advisory: use full n-body for <1% error)."""
        r, v = y_rel[:3], y_rel[3:]
        r_norm = np.linalg.norm(r)
        if r_norm == 0:
            return np.zeros(6)
        return np.concatenate([v, -mu_moon * r / r_norm**3])

    sol = solve_ivp(
        relative_ode, [t0, tf], y0_rel,
        method='DOP853', rtol=1e-10, atol=1e-12,
        events=radial_velocity_event, dense_output=True, max_step=min_to_days(1)
    )

    if len(sol.t_events[0]) == 0:
        # Fallback: dense sample and argmin |r| (more robust)
        t_dense = np.linspace(t0, tf, 1000)
        r_dense = sol.sol(t_dense)[:3].T
        dists = np.linalg.norm(r_dense, axis=1)
        peri_idx = np.argmin(dists)
        t_peri = t_dense[peri_idx]
        logger.warning("No radial vel event; used dense argmin.")
    else:
        t_peri = np.sort(sol.t_events[0])[0]  # First perilune

    # State at perilune
    y_peri_rel = sol.sol(t_peri)
    r_peri_rel = y_peri_rel[:3]
    v_peri_rel = y_peri_rel[3:]
    r_peri_norm = np.linalg.norm(r_peri_rel)
    v_peri_mag = np.linalg.norm(v_peri_rel)  # Incoming hyperbolic speed AU/day

    # Burn: Retrograde to match v_peri_target (scalar Δv; dir opposite v)
    delta_v_apd = v_peri_mag - v_peri_target  # Positive for decel
    delta_v_mps = apd_kps(delta_v_apd) * 1000  # km/s to m/s? Wait, apd_kps is AU/day -> km/s
    thrust_dir = -v_peri_rel / v_peri_mag  # Unit vector, retrograde

    # Duration estimate (needs ship; advisory: pass ship or use default)
    # burn_duration_sec = estimate_burn_duration(delta_v_mps, ship)  # Placeholder
    # burn_duration_days = sec_to_days(burn_duration_sec)
    burn_duration_days = 0.0  # TODO: Integrate ship

    if plot:
        import matplotlib.pyplot as plt
        t_sample = np.linspace(t0, tf, 500)
        sol_sample = sol.sol(t_sample)
        dist_sample = np.linalg.norm(sol_sample[:3].T, axis=1)
        plt.figure(figsize=(8, 5))
        plt.plot(t_sample - t0, dist_sample)
        plt.axvline(t_peri - t0, color='r', ls='--', label=f'Perilune t={t_peri-t0:.3f} days')
        plt.xlabel('Time from now (days)')
        plt.ylabel('Distance to Moon (AU)')
        plt.title('Approach to Perilune')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    logger.info(f"LOI-1 at t={t_peri:.4f} days: Δv={delta_v_mps:.0f} m/s, r_peri={au_to_km(r_peri_norm):.0f} km")

    return {
        # target
        't_center': t_peri,
        'delta_v': delta_v_mps,
        'dir': thrust_dir,
        # info
        'burn_duration_days': burn_duration_days,
        'target_a_km': au_to_km(a_target),
        'target_e': e_target,
        'r_peri_actual_km': au_to_km(r_peri_norm),
    }