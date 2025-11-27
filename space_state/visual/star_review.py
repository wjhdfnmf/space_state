# visual/star_review.py
import numpy as np

from base.star_logger import logger
from base.star_utils import au_to_km
from base.star_constants import BODIES, ts, epoch_jd
from base.star_structures import PropagationResult

def review(result: PropagationResult, ship, body_name: str = "earth"):
    if result.t.size < 2:
        logger.warning("log_orbit_summary: result has fewer than 2 points")
        return

    body = BODIES[body_name]
    radius_km = au_to_km(body["radius"])

    # Earth-centered position in km
    earth_pos_au = np.array([
        body["skyfield"].at(ts.tt_jd(epoch_jd + t)).position.au
        for t in result.t
    ]).T  # (3, N)

    r_ssb_au = result.y[:3, :]                       # (3, N)
    r_rel_au = r_ssb_au - earth_pos_au
    r_norm_km = au_to_km(np.linalg.norm(r_rel_au, axis=0))

    altitude_km = r_norm_km - radius_km

    tof_hours = (result.t[-1] - result.t[0]) * 24.0
    tof_days = tof_hours / 24.0

    logger.info("------------------------------------------")
    logger.info(f"       ORBIT SUMMARY — {body_name.upper()}")
    logger.info("------------------------------------------")
    logger.info(f"Time of Flight       : {tof_days:.6f} days  ({tof_hours:.2f} hours)")
    logger.info(f"Initial Altitude     : {altitude_km[0]:8.1f} km")
    logger.info(f"Final Altitude       : {altitude_km[-1]:8.1f} km")
    logger.info(f"Minimum Altitude     : {altitude_km.min():8.1f} km")
    logger.info(f"Maximum Altitude     : {altitude_km.max():8.1f} km")
    logger.info(f"Mean Altitude        : {altitude_km.mean():8.1f} km")
    logger.info(f"Perigee Safety       : {altitude_km.min() - 100:.1f} km above 100 km limit")
    logger.info(f"Total Points         : {len(result.t)}")
    logger.info("------------------------------------------")
    logger.info(f"Final Mass           : {ship.mass:.1f} kg")
    logger.info(f"Propellant Used      : {ship.initial_propellant_mass - ship.propellant_mass:.1f} kg")
    logger.info(f"Remaining Propellant : {ship.propellant_mass:.1f} kg")
    logger.info("------------------------------------------")

    # Bonus: detect if re-entry risk
    if altitude_km.min() < 80:
        logger.warning(f"PERIGEE BELOW 80 km → Atmospheric re-entry likely!")
    elif altitude_km.min() < 120:
        logger.warning(f"Low perigee ({altitude_km.min():.1f} km) → significant drag expected")