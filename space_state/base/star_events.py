# base/star_events.py
from typing import Callable

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