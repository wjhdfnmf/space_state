# ------------------------------------------------------------------
# mission.py
# ------------------------------------------------------------------

import numpy as np

from base import *
from guidance import FiniteBurnZEMZEV_V2, BurnController
from space_state.base.star_events import throttle_events
from visual import plot_orbital_metrics, Orbit3D, review

# ------------------------------------------------------------------
# Simulation Parameters - USER CONFIGURABLE
# ------------------------------------------------------------------

duration_days = 0.5
ship = starship_raptor_vac()
body = BODIES['earth']
ALL_BODIES = CONFIG.all_bodies

# ------------------------------------------------------------------
# Initial State - SSB Frame (r, v, t)
# ------------------------------------------------------------------

#r0, v0 = get_circular_state(body, 'leo_iss') # SSB frame
r0, v0 = hohmann_setup(body, 'leo_iss', 'geo')  # SSB frame
y0 = np.concatenate([r0, v0, [ship.mass]])
t0 = 0.0 # days

# ------------------------------------------------------------------
# Guidance Setup
# ------------------------------------------------------------------

target = get_circular_state_at_apoapsis(body, t0, y0)

guidance = FiniteBurnZEMZEV_V2(
    target['t_apo'],
    target['r_apo'],
    target['v_apo'],
    ship,
    ALL_BODIES
)

# guidance = BurnController(
#     t0 + 0.1,        # burn start time (days)
#     60,               # planned burn duration (seconds)
#     np.array([0.0, 0.0, 1.0]),  # thrust direction
#     1.0              # throttle
# )

# ------------------------------------------------------------------
# Orbit Propagation
# ------------------------------------------------------------------

prop = Propagator(ship, ALL_BODIES, guidance)
prop.result.initialize(y0, t0)
prop.result.describe_state()

# Propagate
tf = prop.result.time + duration_days # days
events = throttle_events(ship, guidance) # events

prop.propagate(tf, events)

# # ------------------------------------------------------------------
# # Visual Plots
# # ------------------------------------------------------------------

plot_orbital_metrics(
    prop.result,
    bodies_to_plot=ALL_BODIES,
    title="Initial Coast Test",
    ship=ship
)

Orbit3D(
    result= prop.result,
    title = "Initial Coast Test Visualization"
).plot()

review(prop.result)