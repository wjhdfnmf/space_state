# tests/guidance_controller.py
"""
Test Guidance Controller for Orbital Maneuvers
"""

import numpy as np

from base.star_utils import sec_to_days, day_to_sec, ssb_to_body_centered, body_position, body_velocity, min_to_days
from base.star_propagator import Propagator
from base.star_ship import starship_raptor_vac
from base.star_states import get_circular_state_at_apoapsis, get_circular_state, get_hohmann_state_at_time, moon_intercept, inclination_adjustment, finalize_lunar_orbit_insertion, finalize_lunar_orbit_insertion2
from base.star_constants import BODIES, world_time
from base.star_logger import logger
from guidance import BurnController
from guidance.star_tools import distance_to_body, period_moon, period_ship, speed_to_body
from visual import plot_orbital_metrics, Orbit3D, review
from visual.star_vis3D_new import Orbit3D_NEW

def test_guided_insertion():
    
    title = "Orbiting the Moon"
    ship = starship_raptor_vac()            # Spacecraft
    body = BODIES['moon']                  # Central body

    ## ------------------------------------------------------
    ## Initial state in SSB frame (au, au/day, days)
    r0, v0 = get_circular_state(body, 'lunar_100km')
    y0 = np.concatenate([r0, v0, [ship.mass]])
    t0 = 0.0    # Initial time
    tf = 1.0    # Final time delta

    # Setup Propagator and Guidance
    guid = BurnController(ship, body)
    prop = Propagator(ship, guid)
    prop.result.initialize(y0, 0.0)

    ## ------------------------------------------------------
    ## Finish propagation to final time
    prop.propagate(tf)

    ## Validate final state
    plot_orbital_metrics(prop.result, ship, title)
    Orbit3D(prop.result,title).plot()
    review(prop.result, ship, body_name='moon')

    Orbit3D_NEW(prop.result,title).plot()