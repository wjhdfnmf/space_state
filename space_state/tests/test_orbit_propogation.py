
# test_orbit_propogation.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # adds project root

import numpy as np

from guidance import FiniteBurnZEMZEV
from star_constants import BODIES, ORBITS
from star_ship import starship_raptor_vac
from star_config import DEFAULT_CONFIG
from star_states import get_orbiting_state, hohmann_setup
from star_vis2D import plot_orbital_metrics
from star_vis3D import Orbit3D
from star_dynamics import Propagator
from star_validate import validate_orbit

ALL_BODIES = DEFAULT_CONFIG.all_bodies

def test_initial():
    # Initial state: 400 km LEO
    ship = starship_raptor_vac()
    #r0, v0 = get_orbiting_state(BODIES['earth'], ORBITS['earth_leo']) # SSB frame
    r0, v0 = hohmann_setup(BODIES['earth'], ORBITS['earth_leo'])  # SSB frame
    y0 = np.concatenate([r0, v0, [ship.total_mass]])

    ## ------------------------------------

    # Guidance
    guidance = FiniteBurnZEMZEV(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        ship,
        ALL_BODIES
    )

    # Propagator
    prop = Propagator(ship, ALL_BODIES, guidance)
    prop.result.initialize(y0, 0.0)
    prop.result.describe_state()

    # Propagate for 1 day
    prop.propagate(prop.result.time + 1)

    ## ------------------------------------

    # Validate
    validate_orbit(prop.result)

    # Plot results
    plot_orbital_metrics(
        prop.result,
        bodies_to_plot=ALL_BODIES,
        title="Initial Coast Test"
    )

    Orbit3D(
        result= prop.result,
        title = "Initial Coast Test Visualization"
    ).plot()


if __name__ == "__main__":
    test_initial()