# tests/orbit_moon.py
"""
Test Guidance Controller for Lunar Orbital Maneuvers

This test demonstrates attitude control with reaction wheels and RCS thrusters
during a simple Hohmann transfer around the Moon.
"""

import numpy as np

from base.star_utils import sec_to_days, day_to_sec, ssb_to_body_centered, body_position, body_velocity, min_to_days
from base.star_propagator import Propagator
from base.star_ship import starship_raptor_vac
from base.star_states import get_circular_state_at_apoapsis, get_circular_state, get_hohmann_state_at_time
from base.star_constants import BODIES, world_time
from base.star_logger import logger
from guidance import BurnController
from guidance.star_tools import distance_to_body, period_moon, period_ship, speed_to_body
from visual import plot_orbital_metrics, Orbit3D, review, plot_attitude_metrics
from visual.star_vis3D_new import Orbit3D_NEW

def test_guided_insertion():

    title = "Orbiting the Moon"
    ship = starship_raptor_vac()            # Spacecraft
    body = BODIES['moon']                   # Central body

    # Log initial ship state
    logger.info("=" * 60)
    logger.info("LUNAR ORBIT TEST - Attitude Control Demonstration")
    logger.info("=" * 60)
    logger.info(f"Ship: {ship._name}")
    logger.info(f"  Dry mass: {ship.dry_mass:,.0f} kg")
    logger.info(f"  Main propellant: {ship.propellant_mass:,.0f} kg")
    logger.info(f"  RCS propellant: {ship.rcs_propellant_mass:,.0f} kg")
    logger.info(f"  Total mass: {ship.mass:,.0f} kg")
    logger.info(f"  Inertia tensor (diagonal): {np.diag(ship.inertia)}")
    logger.info(f"  Initial wheel momentum: {ship.wheel_momentum}")

    ## ------------------------------------------------------
    ## Initial state in SSB frame (au, au/day, days)
    # get_circular_state now returns (r, v, q, omega)
    r0, v0, q0, omega0 = get_circular_state(body, 'lunar_50km')
    y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

    logger.info(f"Initial quaternion: {q0}")
    logger.info(f"Initial angular velocity: {omega0} rad/day")

    t0 = 0.0    # Initial time
    tf = 0.005  # Final time delta (0.005 days = ~7 minutes, quick coast test)

    # Setup Propagator and Guidance
    guid = BurnController(ship, body)
    prop = Propagator(ship, guid)
    prop.result.initialize(y0, 0.0)

    logger.info(f"\nPropagating to initial time t={t0} days...")
    prop.propagate(t0)

    ## ------------------------------------------------------
    ## Simple Hohmann transfer: 50km -> 100km orbit
    logger.info("\n" + "-" * 60)
    logger.info("Setting up Hohmann transfer: 50km -> 100km lunar orbit")
    logger.info("-" * 60)

    target = get_hohmann_state_at_time(prop, body, t0, 'lunar_100km')
    guid.set_target(target)

    logger.info(f"Delta-V required: {target['delta_v']:.2f} m/s")
    logger.info(f"Burn window: t_start={guid.t_start:.6f} days, t_end={guid.t_end:.6f} days")
    logger.info(f"Burn duration: {day_to_sec(guid.t_end - guid.t_start):.2f} seconds")

    # Log mass before burn
    logger.info(f"\nMass before burn:")
    logger.info(f"  Main propellant: {ship.propellant_mass:,.0f} kg")
    logger.info(f"  RCS propellant: {ship.rcs_propellant_mass:,.0f} kg")

    # Propagate through burn
    logger.info(f"\nPropagating to burn start at t={guid.t_start:.6f} days...")
    prop.propagate(guid.t_start)

    logger.info(f"Executing burn with 2-second max step...")
    prop.propagate(guid.t_end, max_step=sec_to_days(2))

    # Log mass after burn
    logger.info(f"\nMass after burn:")
    logger.info(f"  Main propellant: {ship.propellant_mass:,.0f} kg")
    logger.info(f"  RCS propellant: {ship.rcs_propellant_mass:,.0f} kg")
    logger.info(f"  Wheel momentum: {ship.wheel_momentum}")

    ## ------------------------------------------------------
    ## Finish propagation to final time
    logger.info(f"\nCoasting to final time t={guid.t_end + tf:.6f} days...")
    prop.propagate(guid.t_end + tf)

    # Final state
    logger.info("\n" + "=" * 60)
    logger.info("FINAL STATE")
    logger.info("=" * 60)
    logger.info(f"Final quaternion: {prop.q}")
    logger.info(f"Final angular velocity: {prop.omega} rad/day")
    logger.info(f"Final wheel momentum: {ship.wheel_momentum}")
    logger.info(f"Final main propellant: {ship.propellant_mass:,.0f} kg")
    logger.info(f"Final RCS propellant: {ship.rcs_propellant_mass:,.0f} kg")

    ## Validate final state
    plot_orbital_metrics(prop.result, ship, title)
    plot_attitude_metrics(prop.result, title + " - Attitude")
    Orbit3D(prop.result, title).plot()
    review(prop.result, ship, body_name='moon')

    Orbit3D_NEW(prop.result, title).plot()


if __name__ == '__main__':
    test_guided_insertion()