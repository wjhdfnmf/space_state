from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

from base.star_constants import epoch_jd, ts, BODIES
from base.star_utils import body_position, au_to_km, sec_to_days
from base.star_ship import starship_raptor_vac
from base.star_states import get_circular_state_at_apoapsis, get_hohmann_state_at_time, get_orbiting_state
from base.star_logger import logger
from guidance.star_tools import estimate_burn_duration
from base.star_propagator import Propagator
from visual import plot_orbital_metrics, Orbit3D, review


def moon_intercept(
        t0,
        y0,
        ship = starship_raptor_vac(),
        current_body = BODIES['earth'],
        target_body = BODIES['moon'],
        scrub_t = 0.001,
        scrub_d = 500,
        grid_size = 20
) -> dict:

    distance_tracking = []

    for t in np.linspace(t0, t0 + 0.1, 200):

        target_1 = get_hohmann_state_at_time(current_body, t0, y0, t, 'tli_apo', max_step=1.0)
        y = np.concatenate([target_1['r_target'], target_1['v_transfer'], [0.0]]) 

        target_2 = get_circular_state_at_apoapsis(current_body, t, y)
        t_apo = target_2['t_apo']

        # TOF = t_apo - t
        # delta_v = target_1['delta_v_mps'] + target_2['delta_v_mps']

        distance = target_2['r'] - body_position(target_body, ts.tt_jd(epoch_jd + t_apo))
        distance_km = au_to_km(np.linalg.norm(distance))

        distance_tracking.append((t, distance_km))
    

    # Extract for plotting
    ts_departure, distances_km = zip(*distance_tracking)
    # # Plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(ts_departure, distances_km, 'o-', linewidth=2, markersize=4, color='blue')
    # plt.xlabel('Departure Time (days from t0)')
    # plt.ylabel('Distance to Moon at Apoapsis (km)')
    # plt.title('Departure Time vs Moon Intercept Distance')
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig('departure_vs_distance.png', dpi=300, bbox_inches='tight')  # Save plot
    # plt.show()  # Or display

    # Summary stats
    min_dist = min(distances_km)
    min_idx = distances_km.index(min_dist)
    t_closest = ts_departure[min_idx]
    # logger.info(f"Minimum distance: {min_dist:.0f} km at t={t_closest} days")


    # t_early = t_closest - scrub_t
    # t_late = t_closest + scrub_t

    # ts_departure = np.linspace(t_early, t_late, grid_size)
    # offsets_km = np.linspace(-scrub_d, scrub_d, grid_size)
    # T, D = np.meshgrid(ts_departure, offsets_km)
    # distances_km = np.zeros_like(T)

    # for i, t in enumerate(ts_departure):
    #     for j, d in enumerate(offsets_km):
    #         #logger.debug(f"t={t} days, offset={d} km")

    #         target_1 = get_hohmann_state_at_time(current_body, t0, y0, t, 'tli_apo', max_step=1.0, offset_km=d)
    #         y = np.concatenate([target_1['r_target'], target_1['v_transfer'], [0.0]]) 

    #         target_2 = get_circular_state_at_apoapsis(current_body, t, y)
    #         t_apo = target_2['t_apo']

    #         burn_time_sec = estimate_burn_duration(
    #             target_2['delta_v_mps'],
    #             ship,
    #         )

    #         burn_time = sec_to_days(burn_time_sec)

    #         inspection_time = t_apo + burn_time

    #         r_au = target_2['r']
    #         r_moon_au = body_position(target_body, ts.tt_jd(epoch_jd + inspection_time))

    #         distance_vec_au = r_au - r_moon_au
    #         distances_km[i, j] = au_to_km(np.linalg.norm(distance_vec_au))

    # Plot contours
    # fig, ax = plt.subplots(figsize=(10, 6))
    # contour = ax.contourf(T, D, distances_km, levels=20, cmap='viridis', extend='both')
    # ax.contour(T, D, distances_km, levels=10, colors='white', alpha=0.5, linewidths=0.5)  # Contours
    # cbar = plt.colorbar(contour, ax=ax, label='Distance to Moon at Apoapsis (km)')
    # ax.set_xlabel('Departure Time t (days from t0)')
    # ax.set_ylabel('Offset d (km)')
    # ax.set_title('Departure Time vs Moon Distance (Offset d)')
    # ax.grid(True, alpha=0.3)

    # # Highlight minimum
    # min_idx = np.unravel_index(np.argmin(distances_km), distances_km.shape)
    # min_t, min_d = ts_departure[min_idx[1]], offsets_km[min_idx[0]]
    # min_dist = distances_km[min_idx]
    # ax.plot(min_t, min_d, 'r*', markersize=15, label=f'Min: {min_dist:.0f} km at t={min_t:.2f}, d={min_d:.0f}')
    # ax.legend()

    # plt.tight_layout()
    # plt.savefig('departure_vs_distance_fine.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # logger.info(f"Minimum distance: {min_dist} km at t={min_t} days, offset d={min_d} km")

    return {
        "depart_t": min_t,
        "offset": min_d,
        "distance":min_dist
    }