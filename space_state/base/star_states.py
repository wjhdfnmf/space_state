# base/star_states.py

import numpy as np
from typing import Tuple
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import time

from .star_logger import logger
from .star_config import CONFIG
from .star_constants import epoch_jd, ts
from .star_propagator import Propagator
from .star_utils import body_position, body_velocity, km_to_au, au_to_km, apd_kps, day_to_sec, apd_to_mps
from .star_orbits import ALL_ORBITS
from guidance.star_tools import time_to_apoapsis, get_apoapsis_state
import matplotlib.pyplot as plt

from typing import Literal

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

    return {
        'r': r_apo_ssb,
        'v': v_apo_ssb,
        't_to': t_to_apo,
        't_apo': t_apo,
        'delta_v': v_delta,
        'delta_v_mps': apd_to_mps(v_delta),
        'r_rel_km': au_to_km(r_apo_rel),
        'v_rel_kms': apd_kps(v_circ_rel),
        't_to_sec': day_to_sec(t_to_apo),
    }

#
# Hohmann Transfer State at Target Time
#
def get_hohmann_state_at_time(
        BODY,
        t0: float,                  # Initial time days
        y0: np.ndarray,             # Initial state SSB frame (au, au/day)
        t_target: float,            # days
        target_orbit: OrbitName,
        prop: Propagator = None,
        max_step: float = CONFIG.max_step,
        offset_km: float = 0.0,
    ) -> dict:
    """Return a new state along the current orbit that applies the Hohmann transfer at time t"""
    
    # Propagate to target time to get future state
    if prop is None:
        prop = Propagator()
        prop.result.initialize(y0, t0)
    
    prop.propagate(t_target, max_step=max_step)
    y_target = prop.result.state

    # Relative position and velocity to BODY at target time
    earth_pos = body_position(BODY, ts.tt_jd(epoch_jd + t_target))
    earth_vel = body_velocity(BODY, ts.tt_jd(epoch_jd + t_target))
    r_rel = y_target[:3] - earth_pos
    v_rel = y_target[3:6] - earth_vel

    # Initial and target orbit radii
    r1_au = np.linalg.norm(r_rel)                                                         # Initial orbit radius
    r2_au = km_to_au(ALL_ORBITS[target_orbit].altitude_km + offset_km) + BODY['radius']   # Target orbit radius

    # Compute ΔV to enter Hohmann transfer orbit (prograde)
    a_trans = (r1_au + r2_au) / 2 # Semi-major axis of transfer orbit
    v_current = np.linalg.norm(v_rel)
    v_trans_peri = np.sqrt(BODY['mu'] * (2/r1_au - 1/a_trans))
    dv = v_trans_peri - v_current  # AU/day

    # Apply Δv in current velocity direction
    v_dir_rel = v_rel / np.linalg.norm(v_rel)
    v_transfer_rel = v_rel + dv * v_dir_rel

    # Final transfer state in SSB frame
    v_transfer = v_transfer_rel + earth_vel  # AU/day in SSB frame
    y_transfer = np.concatenate([y_target[:3], v_transfer, y_target[6:]])

    # Return new target state
    return {
        'r_target': y_target[:3],
        'y_target': y_target,
        't_target': t_target,
        'v_transfer': v_transfer,
        'y_transfer': y_transfer,
        'v_transfer_rel': v_transfer_rel,
        'r1_au': r1_au,
        'r2_au': r2_au,
        'r1_km': au_to_km(r1_au),
        'r2_km': au_to_km(r2_au),
        'delta_v': dv,
        'delta_v_mps': apd_to_mps(dv),
        'prop': prop,
    }


def hohmann_setup(BODY, orbit_p: OrbitName, orbit_a: OrbitName = None, t0: float = 0.0) -> dict:

    periapsis = ALL_ORBITS[orbit_p].altitude_km
    apoapsis = periapsis * 6 if orbit_a is None else ALL_ORBITS[orbit_a].altitude_km

    # Initial state: 400 km LEO
    r0, v0 = get_circular_state(BODY, orbit_p)

    # Reference Earth state
    earth_pos = body_position(BODY, ts.tt_jd(epoch_jd + t0))
    earth_vel = body_velocity(BODY, ts.tt_jd(epoch_jd + t0))

    # Target: GEO at apoapsis
    r1_au = np.linalg.norm(r0 - earth_pos)     # Current orbit radius
    r2_au = km_to_au(apoapsis) + BODY['radius']   # GEO radius

    # Compute Δv (prograde)
    a_trans = (r1_au + r2_au) / 2
    v_circ_leo = np.sqrt(BODY['mu'] / r1_au)
    v_trans_peri = np.sqrt(BODY['mu'] * (2/r1_au - 1/a_trans))
    dv = v_trans_peri - v_circ_leo  # AU/day

    # Relative state to Earth
    r0_rel = r0 - earth_pos  # AU
    v0_rel = v0 - earth_vel  # AU/day

    # Apply Δv1 in current velocity direction
    v_dir_rel = v0_rel / np.linalg.norm(v0_rel)
    v_transfer_rel = v0_rel + dv * v_dir_rel
    v_transfer = v_transfer_rel + earth_vel  # AU/day in SSB frame

    return {
        'r0': r0,
        'v_transfer': v_transfer,
        'delta_v': dv,
        'delta_v_mps': apd_to_mps(dv),

    }

# def inclination_adjustment():

    # Account for target inclination
    # if target_body is not None:
    #     t_apo = t_target + time_to_apoapsis(r_rel, v_transfer_rel, BODY['mu'])

    #     body_apo_pos = body_position(BODY, ts.tt_jd(epoch_jd + t_apo))
    #     body_apo_vel = body_velocity(BODY, ts.tt_jd(epoch_jd + t_apo))
    #     target_apo_pos = body_position(target_body, ts.tt_jd(epoch_jd + t_apo))
    #     target_apo_vel = body_velocity(target_body, ts.tt_jd(epoch_jd + t_apo))

    #     r_target_rel = target_apo_pos - body_apo_pos
    #     v_target_rel = target_apo_vel - body_apo_vel
    #     h_target = np.cross(r_target_rel, v_target_rel)
    #     h_target_norm = h_target / np.linalg.norm(h_target)

    #     h_sc = np.cross(r_rel, v_transfer_rel)
    #     h_sc_norm = h_sc / np.linalg.norm(h_sc)

    #     cos_i = abs(np.dot(h_sc_norm, h_target_norm))
    #     i_diff_rad = np.arccos(np.clip(cos_i, -1.0, 1.0))
    #     i_diff_deg = np.degrees(i_diff_rad)

    #     v_transfer_original = v_transfer_rel.copy()
    #     if i_diff_deg > 1e-6:
    #         rot = R.align_vectors([h_target_norm], [h_sc_norm])[0]  # Returns rotation matrix
    #         # Apply to direction only
    #         v_dir = v_transfer_rel / np.linalg.norm(v_transfer_rel)
    #         v_dir_adj = rot.apply(v_dir)
    #         v_transfer_rel = np.linalg.norm(v_transfer_original) * v_dir_adj

    #         # Verify adjusted inclination
    #         h_sc_adj  = np.cross(r_rel, v_transfer_rel)
    #         h_sc_adj_norm = h_sc_adj / np.linalg.norm(h_sc_adj)
    #         cos_i_adj = abs(np.dot(h_sc_adj_norm, h_target_norm))
    #         i_adj_rad = np.arccos(np.clip(cos_i_adj, -1.0, 1.0))
    #         i_adj_deg = np.degrees(i_adj_rad)

    #         dv_normal = np.linalg.norm(v_transfer_rel - v_transfer_original)
    #         dv_normal_mps = apd_to_mps(dv_normal)
    #         dv += dv_normal  # Adjust delta-v for inclination change

    #         logger.debug(f" Initial inclination difference: {i_diff_deg:.6f} deg")
    #         logger.debug(f" Adjusted inclination difference: {i_adj_deg:.6f} deg")
    #         logger.debug(f" Normal Δv for inclination change: {dv_normal_mps:.2f} m/s")
    #     else:
    #         v_dir_adj = np.zeros(3)
    #         dv_normal_mps = 0.0
    #         i_adj_deg = 0.0
    # else:
    #     v_dir_adj = np.zeros(3)
    #     dv_normal_mps = 0.0
    #     i_adj_deg = 0.0



# Escape from Earth (C3 = 0 → parabolic)
def escape_state_earth(BODY: dict, t: float, y0: np.ndarray) -> np.ndarray:
    """Modify velocity at current point to achieve parabolic escape (v_esc)."""
    r_rel = y0[:3] - body_position(BODY, ts.tt_jd(epoch_jd + t))
    v_rel = y0[3:6] - body_velocity(BODY, ts.tt_jd(epoch_jd + t))
    r = np.linalg.norm(r_rel)
    v_esc = np.sqrt(2 * BODY['mu'] / r)
    v_dir = v_rel / np.linalg.norm(v_rel)
    v_new = v_dir * v_esc
    v = v_new + body_velocity(BODY, ts.tt_jd(epoch_jd + t))
    return np.concatenate([y0[:3], v, y0[6:]])


# Moon Intercept Analysis for departure time optimization
def moon_intercept(
        t0: float,
        y0: np.ndarray,
        current_body: dict,
        target_body: dict,
        time_forward: float = 0.1,
        grip_size: int = 1_000,
        plot: bool = True,
) -> dict:

    distance_tracking = []

    prop = Propagator()
    prop.result.initialize(y0, 0.0)

    logger.debug(f"Starting Moon intercept analysis...")
    start_time = time.time()

    #for t in np.linspace(t0, t0 + time_forward, grip_size):
    for t in tqdm(np.linspace(t0, t0 + time_forward, grip_size), desc="Processing", unit="steps"):

        target_1 = get_hohmann_state_at_time(current_body, t0, y0, t, 'tli_apo', target_body=target_body, prop=prop)
        y = np.concatenate([target_1['r_target'], target_1['v_transfer'], [0.0]])
        prop = target_1['prop']

        target_2 = get_circular_state_at_apoapsis(current_body, t, y)

        distance_vec = target_2['r'] - body_position(target_body, ts.tt_jd(epoch_jd + target_2['t_apo']))
        distance = np.linalg.norm(distance_vec) - target_body['radius']
        distance_km = au_to_km(distance)

        distance_tracking.append((t, distance_km))

        #logger.debug(f"Departure t={t:.10f} days: Distance to Moon at Apoapsis = {distance_km:.2f} km")
    

    # Final timing
    exec_time_sec = time.time() - start_time
    elapsed_min = exec_time_sec / 60
    print(f"Elapsed: {elapsed_min:.2f} minutes")

    # Extract for plotting
    ts_departure, distances_km = zip(*distance_tracking)

    # Summary stats
    min_dist = min(distances_km)
    min_idx = distances_km.index(min_dist)
    t_closest = ts_departure[min_idx]

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(ts_departure, distances_km, 'o-', linewidth=2, markersize=4, color='blue')
        plt.xlabel('Departure Time (days from t0)')
        plt.ylabel('Distance to Moon at Apoapsis (km)')
        plt.title('Departure Time vs Moon Intercept Distance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('departure_vs_distance.png', dpi=300, bbox_inches='tight')  # Save plot
        plt.show(block=False)  # Or display
        plt.pause(0.1)

    # Flight statistics
    target_1 = get_hohmann_state_at_time(current_body, t0, y0, t_closest, 'tli_apo', target_body=target_body)
    y = np.concatenate([target_1['r_target'], target_1['v_transfer'], [0.0]])
    target_2 = get_circular_state_at_apoapsis(current_body, t_closest, y)
    distance_vec = target_2['r'] - body_position(target_body, ts.tt_jd(epoch_jd + target_2['t_apo']))

    TOF = target_2['t_apo'] - t_closest
    delta_v = target_1['delta_v_mps'] + target_2['delta_v_mps']

    logger.debug(f" Distance to Moon at closest approach: {distance_vec} AU ({au_to_km(np.linalg.norm(distance_vec)):.2f} km)")

    return {
        "depart_t": t_closest,
        "distance":min_dist,
        "TOF": TOF,
        "delta_v": delta_v
    }