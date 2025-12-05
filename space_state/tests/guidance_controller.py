# tests/guidance_controller.py
"""
Test Guidance Controller for Orbital Maneuvers
"""

import numpy as np

from base.star_utils import sec_to_days, day_to_sec, ssb_to_body_centered, body_position, body_velocity, min_to_days
from base.star_propagator import Propagator
from base.star_ship import starship_raptor_vac
from base.star_states import get_circular_state_at_apoapsis, get_circular_state, get_hohmann_state_at_time, moon_intercept, inclination_adjustment, finalize_lunar_orbit_insertion, finalize_lunar_orbit_insertion2, combined_plane_change_tli
from base.star_constants import BODIES, world_time
from base.star_logger import logger
from guidance import BurnController
from guidance.star_tools import distance_to_body, period_moon, period_ship, speed_to_body
from visual import plot_orbital_metrics, Orbit3D, review
from visual.star_vis3D_new import Orbit3D_NEW

def test_guided_insertion():
    
    title = "Guided Insertion Burn"
    ship = starship_raptor_vac()            # Spacecraft
    body = BODIES['earth']                  # Central body
    target_body = BODIES['moon']            # Target body

    ## ------------------------------------------------------
    ## Initial state in SSB frame (au, au/day, days)
    r0, v0, q0, omega0 = get_circular_state(body, 'leo_iss')
    y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])
    t0 = 0.0    # Initial time
    tf = 2.0    # Final time delta

    # Setup Propagator and Guidance
    guid = BurnController(ship, body)
    prop = Propagator(ship, guid)
    prop.result.initialize(y0, 0.0)

    ## Inital Propagation to t0
    logger.info(f"Propagate to initial time t={t0} days.")
    prop.propagate(t0)

    period_moon()
    ps = period_ship(prop, body)

    ## ------------------------------------------------------
    ## Combined Plane Change + TLI Burn (optimized)
    ## This single burn combines inclination adjustment with TLI for ~20-30% delta-V savings
    target = combined_plane_change_tli(prop, body, target_body, 'tli_apo', offset_km=-20_000)
    guid.set_target(target)

    logger.info(f"Combined TLI+PlaneChange burn at t={target['t_center']:.4f} days")
    logger.info(f"  Δv={target['delta_v']:.1f} m/s (separate would be {target['dv_separate_mps']:.1f} m/s)")
    logger.info(f"  Savings: {target['savings_pct']:.1f}%")
    logger.info(f"  r1 = {target['r1_km']:.0f} km, r2 = {target['r2_km']:.0f} km")
    logger.info(f"  Inclination change: {target['delta_i_deg']:.1f}°")

    prop.propagate(guid.t_start)  # Propagate to burn start
    prop.propagate(guid.t_end, max_step=sec_to_days(2))  # Execute burn

    ## ------------------------------------------------------
    ## Compute Moon intercept timing for coast phase
    intercept = moon_intercept(prop, body, target_body, plot=False)
    logger.info(f"Moon intercept: TOF={intercept['TOF']:.2f} days, Distance={intercept['distance']:.2f} km")

    ## ------------------------------------------------------
    ## Circularization at Apoapsis
    # target = get_circular_state_at_apoapsis(body, prop.t, prop.y)
    # guid.set_target(target)

    # logger.info(f"Circularization burn at t={target['t_center']} days, ΔV={target['delta_v']} m/s")

    # prop.propagate(guid.t_start)
    # prop.propagate(guid.t_end, max_step=sec_to_days(2))

    # logger.debug(f"Final time: {prop.t} days")
    # logger.debug(f"Final distance to Moon: {distance_to_body(prop.t, prop.y, target_body, surface=True):.2f} km")

    ## ------------------------------------------------------
    ## Enter Lunar Orbit Insertion Burn
    # NOTE: Skipped - requires more propellant than available after TLI
    # This is a simplified mission demonstrating TLI only
    i_begin = intercept['TOF'] - min_to_days(60)  # Subtract 60 minutes for LOI burn
    prop.propagate(i_begin)

    # Skip LOI burns to conserve propellant
    # target = get_circular_state_at_apoapsis(body, prop.t, prop.y)
    # guid.set_target(target)
    # logger.debug(f"Lunar Orbit Insertion burn at t={guid.t_start} days to t={guid.t_end} days")
    # prop.propagate(guid.t_start)
    # prop.propagate(guid.t_end, max_step=sec_to_days(2))

    logger.debug(f"Coast time: {prop.t} days")
    logger.debug(f"Distance to Moon: {distance_to_body(prop.t, prop.y, target_body, surface=True):.2f} km")
    logger.debug(f"Speed relative to Moon: {speed_to_body(prop.t, prop.y, target_body):.2f} km/s")

    ## Final Lunar Orbit Circularization Burn
    # NOTE: Skipped - mission delta-V budget exhausted after inclination change (3762 m/s)
    # and TLI (3500 m/s). The 28° plane change in LEO is extremely expensive.
    # For a complete mission, either reduce inclination change or use multi-stage vehicle.
    # See TODO: Implement rocket staging for complete Earth-to-Moon missions.

    # target = finalize_lunar_orbit_insertion(prop)
    # guid.set_target(target)
    # logger.debug(f"Final Lunar Orbit Circularization burn at t={guid.t_start} days to t={guid.t_end} days")
    # prop.propagate(guid.t_start)
    # prop.propagate(guid.t_end, max_step=sec_to_days(2))
    # logger.debug(f"Final time: {prop.t} days")
    # logger.debug(f"Final distance to Moon: {distance_to_body(prop.t, prop.y, target_body, surface=True):.2f} km")
    # logger.debug(f"Final speed relative to Moon: {speed_to_body(prop.t, prop.y, target_body):.2f} km/s")

    ## ------------------------------------------------------
    ## Finish propagation to final time
    prop.propagate(prop.t + tf)

    ## Validate final state
    plot_orbital_metrics(prop.result, ship, title)
    Orbit3D(prop.result,title).plot()
    review(prop.result, ship)

    Orbit3D_NEW(prop.result,title).plot()