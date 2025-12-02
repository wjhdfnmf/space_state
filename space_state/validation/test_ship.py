# validation/test_ship.py

import pytest
import numpy as np

from base.star_ship import starship_raptor_vac

from base.star_logger import logger
from base.star_utils import mps2_apd2, mps2_to_kps2
from base.star_config import CONFIG


# ------------------- TESTS FOR STARSHIP WITH RAPTOR VACUUM THRUSTERS -------------------

dry_mass = 120_000.0                # kg
propellant_mass = 1_100_000.0       # kg
mass_initial = dry_mass + propellant_mass  # kg

number_of_thrusters = 3
max_throttle = 1.0
max_thrust = 2.6e6 * number_of_thrusters  # N
max_isp = 380.0                     # s
max_ve = max_isp * CONFIG.g0        # m/s
max_mdot = max_thrust / max_ve      # kg/s
max_mdot_kgpd = max_mdot * 86400    # kg/day

dummy_direction = np.array([0.0, 0.0, 1.0])         # Default aft
temp_vector = dummy_direction * max_thrust / mass_initial  # m/s²
thrust_vector = mps2_apd2(temp_vector)              # N
thrust_vector_km_s2 = mps2_to_kps2(temp_vector)     # N to km·kg/s²

depleted_mass = dry_mass - 500.0    # kg


def test_starship_raptor_vac():
    """Test Starship with Raptor Vacuum thrusters initialization."""

    # Initialize the Starship with Raptor Vacuum thrusters
    ship = starship_raptor_vac()

    # Base assertions
    assert ship.mass == pytest.approx(mass_initial, rel=1e-5)
    assert ship.dry_mass == pytest.approx(dry_mass, rel=1e-5)
    assert ship.propellant_mass == pytest.approx(propellant_mass, rel=1e-5)
    assert len(ship.thrusters) == number_of_thrusters

    # Zero thrust properties
    assert ship.thrust == pytest.approx(0.0, abs=1e-5)
    assert ship.isp == pytest.approx(0.0, abs=1e-5)

    # Max thrust properties
    ship.set_throttle(max_throttle)
    assert ship.throttle == max_throttle   
    assert ship.isp == pytest.approx(max_isp, rel=1e-5)
    assert ship.thrust == pytest.approx(max_thrust, rel=1e-5)

    # Update thrust direction
    ship.thrust_direction = dummy_direction
    assert np.array_equal(ship.thrust_direction, dummy_direction)
    assert ship.get_thrust_vector().shape == (3,)
    assert np.array_equal(ship.get_thrust_vector(), thrust_vector)
    assert ship.get_thrust_vector_km_s2().shape == (3,)
    assert np.array_equal(ship.get_thrust_vector_km_s2(), thrust_vector_km_s2)
    assert ship.ve == pytest.approx(max_ve, rel=1e-5)
    assert ship.mdot == pytest.approx(max_mdot, rel=1e-5)
    assert ship.mdot_kgpd == pytest.approx(max_mdot_kgpd, rel=1e-5)

    # Update propellant mass and check mass
    ship.update_propellant(dry_mass)        # Deplete all propellant
    assert ship.propellant_mass == pytest.approx(0.0, abs=1e-5)
    assert ship.mass == pytest.approx(dry_mass, rel=1e-5)
    assert ship.active == False             # Inactive when no propellant

    # Attempt to consume more propellant than available
    ship.update_propellant(depleted_mass)   # Below dry mass
    assert ship.propellant_mass == pytest.approx(0.0, abs=1e-5)
    assert ship.mass == pytest.approx(dry_mass, rel=1e-5)

    # Attempt to add propellant (should not change)
    ship.update_propellant(mass_initial)    # Above dry mass
    assert ship.propellant_mass == pytest.approx(0.0, abs=1e-5)
    assert ship.mass == pytest.approx(dry_mass, rel=1e-5)

    logger.info("Starship Raptor Vacuum test passed.")