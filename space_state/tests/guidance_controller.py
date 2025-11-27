# tests/test_guidance_controller.py
"""
Unit test for FiniteBurnZEMZEV_V2 guidance controller.
"""

import numpy as np

from base.star_utils import sec_to_days
from base.star_propagator import Propagator
from base.star_ship import starship_raptor_vac
from base.star_states import get_circular_state_at_apoapsis, hohmann_setup, get_orbiting_state, get_hohmann_state_at_time
from base.star_constants import BODIES
from base.star_logger import logger
from guidance.star_tools import estimate_burn_duration
from guidance import FiniteBurnZEMZEV_V2, BurnController
from visual import plot_orbital_metrics, Orbit3D, review

def test_guided_insertion():
    """Full guided insertion burn to GEO."""
    
    title = "Guided GEO Insertion Burn Test"
    ship = starship_raptor_vac()            # Spacecraft
    body = BODIES['earth']                  # Central body

    ## ------------------------------------------------------
    ## Initial state in SSB frame (au, au/day, days)
    r0, v0 = get_orbiting_state(body, 'leo_iss')
    y0 = np.concatenate([r0, v0, [ship.mass]])
    t0 = 0.0

    ship.log()

    # Setup Propagator and Guidance
    guidance = BurnController(ship, body)
    prop = Propagator(ship, guidance)
    prop.result.initialize(y0, 0.0)

    ## Determine transfer time (TODO: Intercept the Moon)
    t1 = 0.2
    tf = 2.0

    ## ------------------------------------------------------
    ## Hohmann Transfer to GEO Apoapsis
    target = get_hohmann_state_at_time(body, t0, y0, t1, 'geo')
    guidance.set_target(target['delta_v_mps'], target['t_burn'])

    prop.propagate(guidance.t_start)
    prop.propagate(guidance.t_end, max_step=sec_to_days(2))
    
    ## ------------------------------------------------------
    ## Circularization at Apoapsis

    # Setup guidance to target circular at apoapsis
    t = prop.result.time
    y = prop.result.state
    target = get_circular_state_at_apoapsis(body, t, y)
    guidance.set_target(target['delta_v_mps'], target['t_burn'])

    prop.propagate(guidance.t_start)
    prop.propagate(guidance.t_end, max_step=sec_to_days(2))

    ## ------------------------------------------------------
    ## Finish propagation to final time
    prop.propagate(tf)

    ## ------------------------------------------------------
    ## Validate final state
    plot_orbital_metrics(prop.result, ship, title)
    Orbit3D(prop.result,title).plot()
    review(prop.result, ship)