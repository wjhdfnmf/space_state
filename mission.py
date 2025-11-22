# ------------------------------------------------------------------
# mission.py
# ------------------------------------------------------------------

import numpy as np

from star_logger import logger
from star_config import DEFAULT_CONFIG, EARTH_LEO, EARTH_GSO, MOON_ORBIT
from star_utils import km_to_au, au_to_km, apd_kps, days_to_seconds
from star_utils import body_position, body_velocity
from star_structures import StateVector
from star_constants import BODIES, epoch_jd, ts
from star_dynamics import propagate_orbit
from star_states import get_orbiting_initial_state
from star_ship import starship_raptor_vac

from star_vis2D import plot_orbital_metrics
from star_validate import validate_orbit

# ------------------------------------------------------------------
# Simulation Parameters - USER CONFIGURABLE
# ------------------------------------------------------------------

duration_days = 1                      # Propagation duration in days
ship = starship_raptor_vac()            # Spacecraft definition
body = ['earth', EARTH_LEO]             # Body to orbit and initial altitude above surface in km
all_bodies = DEFAULT_CONFIG.all_bodies      # Perturbing bodies

# ------------------------------------------------------------------
# Initial State - SSB Frame (r, v, t)
# ------------------------------------------------------------------

BODY = BODIES[body[0]]
state0 = get_orbiting_initial_state(BODY, body[1])
state0.set_mass(ship.total_mass)
time0 = ts.tt_jd(epoch_jd)

logger.info(f"Initial State: {state0}")
logger.info(f"Initial Altitude: {au_to_km(np.linalg.norm(state0.r - body_position(BODY, time0)) - BODY['radius']):.1f} km"
            f", Velocity: {apd_kps(np.linalg.norm(state0.v - body_velocity(BODY, time0))):.3f} km/s"
            f", Simulation Duration: {duration_days:.1f} days")

# ------------------------------------------------------------------
# 
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Orbit Propagation
# ------------------------------------------------------------------

result = propagate_orbit(
    state0=state0,
    t_duration=duration_days,
    ship=ship,
    bodies=all_bodies
)

# # ------------------------------------------------------------------
# # Validation
# # ------------------------------------------------------------------
validate_orbit(
    result,
    tolerance_km=1.0,
    tolerance_kms=0.01
)

# # ------------------------------------------------------------------
# # Visual Plots
# # ------------------------------------------------------------------

plot_orbital_metrics(
    result,
    bodies_to_plot=['earth', 'moon'],
    title="400 km LEO with Moon Perturbation"
)