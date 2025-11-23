# ------------------------------------------------------------------
# mission.py
# ------------------------------------------------------------------

import numpy as np

from base import *
from guidance import FiniteBurnZEMZEV
from visual import plot_orbital_metrics, Orbit3D, review

# ------------------------------------------------------------------
# Simulation Parameters - USER CONFIGURABLE
# ------------------------------------------------------------------

duration_days = 1
ship = starship_raptor_vac()
body = BODIES['earth']
orbit = ORBITS['earth_leo']
ALL_BODIES = CONFIG.all_bodies

# ------------------------------------------------------------------
# Initial State - SSB Frame (r, v, t)
# ------------------------------------------------------------------

#r0, v0 = get_orbiting_state(BODIES['earth'], ORBITS['earth_leo']) # SSB frame
r0, v0 = hohmann_setup(body, orbit)  # SSB frame
y0 = np.concatenate([r0, v0, [ship.total_mass]])

# ------------------------------------------------------------------
# Guidance Setup
# ------------------------------------------------------------------

guidance = FiniteBurnZEMZEV(
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0]),
    ship,
    ALL_BODIES
)

# ------------------------------------------------------------------
# Orbit Propagation
# ------------------------------------------------------------------

prop = Propagator(ship, ALL_BODIES, guidance)
prop.result.initialize(y0, 0.0)
prop.result.describe_state()

# Propagate for 1 day
prop.propagate(prop.result.time + 1)

# # ------------------------------------------------------------------
# # Visual Plots
# # ------------------------------------------------------------------

plot_orbital_metrics(
    prop.result,
    bodies_to_plot=ALL_BODIES,
    title="Initial Coast Test"
)

Orbit3D(
    result= prop.result,
    title = "Initial Coast Test Visualization"
).plot()

review(prop.result)