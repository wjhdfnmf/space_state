
# test_orbit_propogation.py
import pytest
import numpy as np
from typing import Callable, Tuple
from scipy.integrate import solve_ivp

from finite_burn_zemzev import FiniteBurnZEMZEV
from star_dynamics import n_body_acceleration
from star_utils import sec_to_days, au_to_km, body_position, body_velocity, km_to_au, mps2_apd2, convert_gm, calculate_eccentricity, apd_to_mps
from star_constants import BODIES, ts, epoch_jd, c
from star_ship import starship_raptor_vac
from star_config import SECONDS_PER_DAY, DEFAULT_CONFIG
from star_structures import PropagationResult, combine_propagations
from star_states import get_orbiting_initial_state
from star_logger import logger
from star_vis2D import plot_orbital_metrics
from star_guidance import time_to_apoapsis, estimate_burn_duration
from star_dynamics import eom_factory, n_body_acceleration, propagate_orbit

#ALL_BODIES = DEFAULT_CONFIG.all_bodies
ALL_BODIES = ['earth']

def solve(result: PropagationResult, eom: Callable, t_end: float) -> PropagationResult:

    sol = solve_ivp(
        fun=eom,
        t_span=[result.state.t, t_end],
        y0=result.state.y,
        method="DOP853",
        rtol=1e-9, atol=1e-9,
        max_step=60 / SECONDS_PER_DAY,
        events=None,
        dense_output=False,
    )

    r_final = PropagationResult(
        t=sol.t,
        y=sol.y,
        events=[],
        conservation={}
    )
    y1 = sol.y[:, -1]
    t1 = sol.t[-1]

    state = state.update_state(
        r=y1[0:3],
        v=y1[3:6],
        t=t1
    )
    
    if r_init is None:
        return r_final
    return combine_propagations([r_init, r_final])



def test_initial():
    # Initial state: 400 km LEO
    ship = starship_raptor_vac()
    state = get_orbiting_initial_state(BODIES['earth'], 400, ship) # SSB frame
    eom = eom_factory(ALL_BODIES, ship)

    t_span = [state.t, state.t + 1]  # 1 day coast
    result = solve(t_span, state, eom)

    plot_orbital_metrics(
        result,
        bodies_to_plot=['earth'],
        title="Initial Coast Test"
    )


if __name__ == "__main__":
    test_initial()