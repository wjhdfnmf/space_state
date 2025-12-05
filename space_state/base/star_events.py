# base/star_events.py
"""
Propagator Event Functions

Provides event functions for use with scipy.integrate.solve_ivp to detect:
- Engine ignition/cutoff
- Propellant depletion
- Periapsis/apoapsis crossing
- SOI transitions
- Altitude thresholds

Event Function Protocol:
- event(t, y) -> float: Returns value that crosses zero at event
- event.terminal: bool - If True, integration stops at event
- event.direction: int - +1 (rising), -1 (falling), 0 (either)
"""

from typing import Callable, List, Optional
import numpy as np

from .star_utils import body_position, body_velocity, ssb_to_body_centered


# =============================================================================
# Orbital Event Factories
# =============================================================================

def periapsis_event(
    body: dict,
    terminal: bool = True,
) -> Callable:
    """
    Create event function that triggers at periapsis passage.

    Periapsis occurs when radial velocity (r · v) crosses zero from positive
    to negative (approaching then receding).

    Parameters
    ----------
    body : dict
        Body dictionary with 'skyfield' key for position/velocity lookup
    terminal : bool
        If True, integration stops at periapsis

    Returns
    -------
    Callable
        Event function for use with solve_ivp

    Example
    -------
    >>> from base.star_constants import BODIES
    >>> peri_event = periapsis_event(BODIES['moon'])
    >>> # Use in propagator: events=[peri_event]
    """
    from .star_constants import epoch_jd, ts

    def event(t: float, y: np.ndarray) -> float:
        """Radial velocity relative to body (zero at peri/apo)."""
        time_obj = ts.tt_jd(epoch_jd + t)

        # Position and velocity relative to body
        r_body = body_position(body, time_obj)
        v_body = body_velocity(body, time_obj)

        r_rel = y[:3] - r_body
        v_rel = y[3:6] - v_body

        r_mag = np.linalg.norm(r_rel)
        if r_mag < 1e-12:
            return 0.0

        # Radial velocity: positive = moving away, negative = approaching
        v_radial = np.dot(r_rel, v_rel) / r_mag
        return v_radial

    event.terminal = terminal
    event.direction = -1  # Trigger when going from positive (outbound) to negative (inbound)
    event.name = f"Periapsis ({body.get('name', 'body')})"

    return event


def apoapsis_event(
    body: dict,
    terminal: bool = True,
) -> Callable:
    """
    Create event function that triggers at apoapsis passage.

    Apoapsis occurs when radial velocity crosses zero from negative to positive.

    Parameters
    ----------
    body : dict
        Body dictionary
    terminal : bool
        If True, integration stops at apoapsis

    Returns
    -------
    Callable
        Event function for use with solve_ivp
    """
    from .star_constants import epoch_jd, ts

    def event(t: float, y: np.ndarray) -> float:
        """Radial velocity relative to body."""
        time_obj = ts.tt_jd(epoch_jd + t)

        r_body = body_position(body, time_obj)
        v_body = body_velocity(body, time_obj)

        r_rel = y[:3] - r_body
        v_rel = y[3:6] - v_body

        r_mag = np.linalg.norm(r_rel)
        if r_mag < 1e-12:
            return 0.0

        v_radial = np.dot(r_rel, v_rel) / r_mag
        return v_radial

    event.terminal = terminal
    event.direction = +1  # Trigger when going from negative (inbound) to positive (outbound)
    event.name = f"Apoapsis ({body.get('name', 'body')})"

    return event


def altitude_event(
    body: dict,
    altitude_km: float,
    direction: int = 0,
    terminal: bool = True,
) -> Callable:
    """
    Create event function that triggers at specific altitude.

    Parameters
    ----------
    body : dict
        Body dictionary with 'radius' in AU
    altitude_km : float
        Target altitude above body surface (km)
    direction : int
        +1 = ascending through altitude
        -1 = descending through altitude
        0 = either direction
    terminal : bool
        If True, integration stops at altitude crossing

    Returns
    -------
    Callable
        Event function for use with solve_ivp
    """
    from .star_constants import epoch_jd, ts
    from .star_utils import km_to_au

    target_radius = body['radius'] + km_to_au(altitude_km)

    def event(t: float, y: np.ndarray) -> float:
        """Distance from body center minus target radius."""
        time_obj = ts.tt_jd(epoch_jd + t)
        r_body = body_position(body, time_obj)
        r_rel = y[:3] - r_body
        r_mag = np.linalg.norm(r_rel)
        return r_mag - target_radius

    event.terminal = terminal
    event.direction = direction
    event.name = f"Altitude {altitude_km:.0f} km ({body.get('name', 'body')})"

    return event


def distance_event(
    body: dict,
    distance_km: float,
    direction: int = 0,
    terminal: bool = True,
    surface: bool = False,
) -> Callable:
    """
    Create event function that triggers at specific distance from body.

    Parameters
    ----------
    body : dict
        Body dictionary
    distance_km : float
        Target distance (km). If surface=True, this is altitude; otherwise center distance.
    direction : int
        +1 = increasing distance, -1 = decreasing, 0 = either
    terminal : bool
        If True, integration stops at distance crossing
    surface : bool
        If True, distance is from surface; if False, from center

    Returns
    -------
    Callable
        Event function
    """
    from .star_constants import epoch_jd, ts
    from .star_utils import km_to_au

    if surface:
        target_radius = body['radius'] + km_to_au(distance_km)
    else:
        target_radius = km_to_au(distance_km)

    def event(t: float, y: np.ndarray) -> float:
        time_obj = ts.tt_jd(epoch_jd + t)
        r_body = body_position(body, time_obj)
        r_rel = y[:3] - r_body
        r_mag = np.linalg.norm(r_rel)
        return r_mag - target_radius

    event.terminal = terminal
    event.direction = direction
    event.name = f"Distance {distance_km:.0f} km ({body.get('name', 'body')})"

    return event


def soi_entry_event(
    body: dict,
    terminal: bool = True,
) -> Callable:
    """
    Create event function that triggers when entering a body's sphere of influence.

    Parameters
    ----------
    body : dict
        Body dictionary with 'soi' radius in AU
    terminal : bool
        If True, integration stops at SOI entry

    Returns
    -------
    Callable
        Event function
    """
    from .star_constants import epoch_jd, ts

    soi_radius = body.get('soi', 0.0)

    def event(t: float, y: np.ndarray) -> float:
        time_obj = ts.tt_jd(epoch_jd + t)
        r_body = body_position(body, time_obj)
        r_rel = y[:3] - r_body
        r_mag = np.linalg.norm(r_rel)
        return r_mag - soi_radius

    event.terminal = terminal
    event.direction = -1  # Entering = decreasing distance
    event.name = f"SOI Entry ({body.get('name', 'body')})"

    return event


def soi_exit_event(
    body: dict,
    terminal: bool = True,
) -> Callable:
    """
    Create event function that triggers when exiting a body's sphere of influence.

    Parameters
    ----------
    body : dict
        Body dictionary with 'soi' radius in AU
    terminal : bool
        If True, integration stops at SOI exit

    Returns
    -------
    Callable
        Event function
    """
    from .star_constants import epoch_jd, ts

    soi_radius = body.get('soi', 0.0)

    def event(t: float, y: np.ndarray) -> float:
        time_obj = ts.tt_jd(epoch_jd + t)
        r_body = body_position(body, time_obj)
        r_rel = y[:3] - r_body
        r_mag = np.linalg.norm(r_rel)
        return r_mag - soi_radius

    event.terminal = terminal
    event.direction = +1  # Exiting = increasing distance
    event.name = f"SOI Exit ({body.get('name', 'body')})"

    return event


def ascending_node_event(
    body: dict,
    reference_plane: np.ndarray = None,
    terminal: bool = False,
) -> Callable:
    """
    Create event function that triggers at ascending node.

    Ascending node is where the orbit crosses the reference plane
    from below (negative z) to above (positive z).

    Parameters
    ----------
    body : dict
        Body dictionary for frame center
    reference_plane : np.ndarray
        Normal vector of reference plane (default: z-axis = equatorial)
    terminal : bool
        If True, integration stops at node crossing

    Returns
    -------
    Callable
        Event function
    """
    from .star_constants import epoch_jd, ts

    if reference_plane is None:
        reference_plane = np.array([0, 0, 1])

    reference_plane = reference_plane / np.linalg.norm(reference_plane)

    def event(t: float, y: np.ndarray) -> float:
        time_obj = ts.tt_jd(epoch_jd + t)
        r_body = body_position(body, time_obj)
        r_rel = y[:3] - r_body

        # Height above reference plane
        return np.dot(r_rel, reference_plane)

    event.terminal = terminal
    event.direction = +1  # Ascending = crossing from below
    event.name = f"Ascending Node ({body.get('name', 'body')})"

    return event


def descending_node_event(
    body: dict,
    reference_plane: np.ndarray = None,
    terminal: bool = False,
) -> Callable:
    """
    Create event function that triggers at descending node.

    Parameters
    ----------
    body : dict
        Body dictionary for frame center
    reference_plane : np.ndarray
        Normal vector of reference plane
    terminal : bool
        If True, integration stops at node crossing

    Returns
    -------
    Callable
        Event function
    """
    from .star_constants import epoch_jd, ts

    if reference_plane is None:
        reference_plane = np.array([0, 0, 1])

    reference_plane = reference_plane / np.linalg.norm(reference_plane)

    def event(t: float, y: np.ndarray) -> float:
        time_obj = ts.tt_jd(epoch_jd + t)
        r_body = body_position(body, time_obj)
        r_rel = y[:3] - r_body
        return np.dot(r_rel, reference_plane)

    event.terminal = terminal
    event.direction = -1  # Descending = crossing from above
    event.name = f"Descending Node ({body.get('name', 'body')})"

    return event


# =============================================================================
# Propulsion Events
# =============================================================================

def throttle_events(ship, guidance) -> list[Callable]:
    """
    Returns two terminal events:
      1. Engine ON  (when guidance activates)
      2. Engine OFF (when guidance deactivates or propellant depleted)
      3. Propellant Depletion (when propellant runs low)
    Both are terminal = True → integrator stops exactly at switch
    """
    
    # Engine ON event
    def engine_on(t, y):
        # Called every step — return 0 when guidance wants to burn
        if guidance is None:
            return 1.0
        try:
            vgo, dir, throttle = guidance.update(t, y)
            return throttle - 0.01  # crosses zero when throttle > 1%
        except:
            return 1.0
    engine_on.terminal = True
    engine_on.direction = +1  # only trigger when going from off → on
    engine_on.name = "Engine Ignition"

    # Engine OFF event
    def engine_off(t, y):
        if guidance is None:
            return -1.0
        try:
            vgo, dir, throttle = guidance.update(t, y)
            # Trigger when throttle drops or propellant gone
            propellant_left = y[6] - ship.dry_mass
            return 0.01 - throttle if propellant_left > 100 else -1.0
        except:
            return -1.0
    engine_off.terminal = True
    engine_off.direction = -1  # only when going from on → off
    engine_off.name = "Engine Cutoff"

    # Propellant depletion event
    def propellant_depletion(t, y):
        return y[6] - ship.dry_mass - 100  # 100 kg safety margin
    propellant_depletion.terminal = True
    propellant_depletion.direction = -1
    propellant_depletion.name = "Propellant Depletion"

    return [engine_on, engine_off, propellant_depletion]