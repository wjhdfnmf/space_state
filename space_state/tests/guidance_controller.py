# tests/test_guidance_controller.py
"""
Unit test for FiniteBurnZEMZEV_V2 guidance controller.
"""

import numpy as np

from base.star_utils import sec_to_days, apd_to_mps
from base.star_propagator import Propagator
from base.star_ship import starship_raptor_vac
from base.star_states import get_circular_state_at_apoapsis, hohmann_setup, get_orbiting_state, get_hohmann_state_at_time, moon_intercept
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
    r0, v0 = get_orbiting_state(body, 'leo_200')
    y0 = np.concatenate([r0, v0, [ship.mass]])
    t0 = 0.0

    ship.log()

    # Setup Propagator and Guidance
    guidance = BurnController(ship, body)
    prop = Propagator(ship, guidance)
    prop.result.initialize(y0, 0.0)

    ## Determine transfer time (TODO: Intercept the Moon)
    #intercept = moon_intercept(t0, y0, body, BODIES['moon'])
    #t1 = intercept['depart_t']

    #logger.info(f"Optimal departure time for Moon intercept: {t1} days, TOF: {intercept['TOF']} days, Distance: {intercept['distance']:.2f} km")
    t1 = 0.3

    ## ------------------------------------------------------
    ## Hohmann Transfer to GEO Apoapsis
    #target = get_hohmann_state_at_time(body, t0, y0, t1, 'geo', target_body=BODIES['moon'], verbose=True)
    target = get_hohmann_state_at_time(body, t0, y0, t1, 'geo', verbose=True)

    #logger.debug(f" Inclination adjustment {apd_to_mps(target_r['v_transfer_rel'] - target['v_transfer_rel'])} m/s")
    #logger.debug(f" Inclination adjustment {apd_to_mps(np.linalg.norm(target_r['v_transfer_rel'] - target['v_transfer_rel']))} m/s")

    guidance.set_target(target['delta_v_mps'], target['t_burn'])
    logger.info(f"Hohmann transfer to GEO apoapsis burn at t={target['t_burn']} days, ΔV={target['delta_v_mps']} m/s")
    logger.info(f"  r1 = {target['r1_km']} km, r2 = {target['r2_km']} km")
    logger.info(f"  t_start = {guidance.t_start} days, t_end = {guidance.t_end} days")

    prop.propagate(guidance.t_start) # Propagate to burn start
    prop.propagate(guidance.t_end, max_step=sec_to_days(2)) # Propagate to burn end
    
    ## ------------------------------------------------------
    ## Circularization at Apoapsis

    #Setup guidance to target circular at apoapsis
    t = prop.result.time
    y = prop.result.state
    target = get_circular_state_at_apoapsis(body, t, y)
    guidance.set_target(target['delta_v_mps'], target['t_burn'])
    logger.info(f"Circularization burn at t={target['t_burn']} days, ΔV={target['delta_v_mps']} m/s")
    logger.info(f"  r_apo = {target['r_rel_km']} km")
    logger.info(f"  t_apo = {target['t_apo']} days")

    prop.propagate(guidance.t_start) # Propagate to burn start
    prop.propagate(guidance.t_end, max_step=sec_to_days(2)) # Propagate to burn end

    ## ------------------------------------------------------
    ## Finish propagation to final time
    prop.propagate(guidance.t_end + 1)
    #prop.propagate(1.0)

    ## ------------------------------------------------------
    ## Validate final state
    plot_orbital_metrics(prop.result, ship, title)
    Orbit3D(prop.result,title).plot()
    review(prop.result, ship)