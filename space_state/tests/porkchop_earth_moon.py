from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

from base.star_constants import epoch_jd, ts, BODIES
from base.star_utils import body_position, au_to_km
from base.star_states import get_circular_state_at_apoapsis, get_hohmann_state_at_time, get_orbiting_state
from base.star_logger import logger
from base.star_propagator import Propagator
from visual import plot_orbital_metrics, Orbit3D, review


def test_moon_intercept():

    logger.info(f"--- Lunar Transfer Investigation ---")
    title = "Lunar Transfer Investigation"

    target_body = BODIES['moon']

    body = BODIES['earth']
    t0 = 0.0

    r0, v0 = get_orbiting_state(body, 'leo_iss')
    y0 = np.concatenate([r0, v0, [0.0]])

    distance_tracking = []

    for t in np.linspace(t0, t0 + 10, 100):

        logger.info(f"--- Launch Time: {t:.2f} days ---")

        target_1 = get_hohmann_state_at_time(body, t0, y0, t, 'tli_apo', max_step=1.0)
        y = np.concatenate([target_1['r_target'], target_1['v_transfer'], [0.0]]) 

        target_2 = get_circular_state_at_apoapsis(body, t, y)
        t_apo = target_2['t_apo']

        TOF = t_apo - t
        delta_v = target_1['delta_v_mps'] + target_2['delta_v_mps']

        distance = target_2['r'] - body_position(target_body, ts.tt_jd(epoch_jd + t_apo))
        distance_km = au_to_km(np.linalg.norm(distance))

        logger.info(f"Time of Flight (TOF): {TOF:.2f} days")
        logger.info(f"Total Delta-V for Lunar Transfer: {delta_v:.2f} m/s")
        logger.info(f"Distance to Moon at Apoapsis: {distance_km:.2f} km")
        distance_tracking.append((t, TOF, delta_v, distance_km))
    

    # Extract for plotting
    ts_departure, _, _, distances_km = zip(*distance_tracking)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ts_departure, distances_km, 'o-', linewidth=2, markersize=4, color='blue')
    plt.xlabel('Departure Time (days from t0)')
    plt.ylabel('Distance to Moon at Apoapsis (km)')
    plt.title('Departure Time vs Moon Intercept Distance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('departure_vs_distance.png', dpi=300, bbox_inches='tight')  # Save plot
    plt.show()  # Or display

    # Summary stats
    min_dist = min(distances_km)
    min_idx = distances_km.index(min_dist)
    logger.info(f"Minimum distance: {min_dist:.0f} km at t={ts_departure[min_idx]:.2f} days")