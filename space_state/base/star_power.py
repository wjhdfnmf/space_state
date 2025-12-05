# base/star_power.py
"""
Spacecraft Power System Model

Models:
- Solar panel power generation
- Battery state of charge
- Eclipse detection
- Power budget and load management
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .star_constants import BODIES
from .star_logger import logger


# Physical constants
SOLAR_CONSTANT = 1361.0  # W/m² at 1 AU
AU_KM = 1.496e8  # km per AU


class PowerMode(Enum):
    """Spacecraft power modes."""
    NORMAL = "normal"
    LOW_POWER = "low_power"
    SAFE = "safe"
    ECLIPSE = "eclipse"


@dataclass
class SolarPanel:
    """
    Solar panel model.

    Power output: P = η * A * S * cos(θ) * (1/r)²

    Where:
        η = efficiency (0.2-0.3 for modern panels)
        A = panel area (m²)
        S = solar constant (1361 W/m²)
        θ = sun angle (0° = sun-facing)
        r = distance from Sun (AU)
    """
    area: float = 100.0           # m²
    efficiency: float = 0.28      # ~28% for modern multi-junction
    degradation_rate: float = 0.02  # 2% per year
    age_days: float = 0.0         # Mission elapsed time

    def power_output(
        self,
        distance_au: float,
        sun_angle_rad: float = 0.0,
    ) -> float:
        """
        Calculate power output.

        Args:
            distance_au: Distance from Sun (AU)
            sun_angle_rad: Angle between panel normal and Sun direction

        Returns:
            Power output (W)
        """
        if distance_au <= 0:
            return 0.0

        # Degradation factor
        years = self.age_days / 365.25
        degradation = (1 - self.degradation_rate) ** years

        # Solar flux at distance
        flux = SOLAR_CONSTANT * (1.0 / distance_au) ** 2

        # Cosine loss from angle
        cos_factor = max(0.0, np.cos(sun_angle_rad))

        # Total power
        power = self.efficiency * self.area * flux * cos_factor * degradation

        return power

    def max_power(self, distance_au: float = 1.0) -> float:
        """Maximum power at given distance (sun-facing)."""
        return self.power_output(distance_au, sun_angle_rad=0.0)


@dataclass
class Battery:
    """
    Battery model with state of charge tracking.

    Supports charge/discharge with efficiency losses.
    """
    capacity_wh: float = 10000.0     # Watt-hours
    charge_efficiency: float = 0.95   # 95% charge efficiency
    discharge_efficiency: float = 0.95  # 95% discharge efficiency
    max_charge_rate: float = 2000.0   # W (C/5 rate for 10 kWh)
    max_discharge_rate: float = 5000.0  # W
    min_soc: float = 0.1              # 10% minimum SOC (protect battery)

    _current_soc: float = 1.0         # State of charge (0-1)

    @property
    def soc(self) -> float:
        """Current state of charge (0-1)."""
        return self._current_soc

    @property
    def energy_available(self) -> float:
        """Available energy (Wh)."""
        usable_soc = max(0, self._current_soc - self.min_soc)
        return usable_soc * self.capacity_wh * self.discharge_efficiency

    def charge(self, power: float, duration_hours: float) -> float:
        """
        Charge battery.

        Args:
            power: Charging power (W)
            duration_hours: Time duration

        Returns:
            Actual energy stored (Wh)
        """
        # Limit to max charge rate
        power = min(power, self.max_charge_rate)

        # Energy with efficiency
        energy = power * duration_hours * self.charge_efficiency

        # Calculate new SOC
        new_soc = self._current_soc + energy / self.capacity_wh
        new_soc = min(1.0, new_soc)

        actual_energy = (new_soc - self._current_soc) * self.capacity_wh
        self._current_soc = new_soc

        return actual_energy

    def discharge(self, power: float, duration_hours: float) -> float:
        """
        Discharge battery.

        Args:
            power: Power demand (W)
            duration_hours: Time duration

        Returns:
            Actual energy delivered (Wh)
        """
        # Limit to max discharge rate
        power = min(power, self.max_discharge_rate)

        # Energy needed from battery (accounting for efficiency)
        energy_needed = power * duration_hours / self.discharge_efficiency

        # Check available energy
        available = (self._current_soc - self.min_soc) * self.capacity_wh
        energy_used = min(energy_needed, available)

        # Update SOC
        self._current_soc -= energy_used / self.capacity_wh
        self._current_soc = max(self.min_soc, self._current_soc)

        # Actual energy delivered
        return energy_used * self.discharge_efficiency


@dataclass
class PowerLoad:
    """A power-consuming subsystem."""
    name: str
    power_w: float         # Nominal power draw (W)
    essential: bool = True  # True = cannot be shed in low power mode
    enabled: bool = True    # Current state


@dataclass
class PowerSystem:
    """
    Complete spacecraft power system.

    Manages solar panels, batteries, and power distribution.
    """
    panels: List[SolarPanel] = field(default_factory=list)
    batteries: List[Battery] = field(default_factory=list)
    loads: List[PowerLoad] = field(default_factory=list)

    _mode: PowerMode = PowerMode.NORMAL
    _in_eclipse: bool = False

    @classmethod
    def typical_spacecraft(cls) -> "PowerSystem":
        """Create typical deep-space spacecraft power system."""
        return cls(
            panels=[SolarPanel(area=50, efficiency=0.28)],
            batteries=[Battery(capacity_wh=5000)],
            loads=[
                PowerLoad("avionics", 150, essential=True),
                PowerLoad("attitude_control", 100, essential=True),
                PowerLoad("thermal", 200, essential=True),
                PowerLoad("comm", 100, essential=False),
                PowerLoad("payload", 300, essential=False),
            ]
        )

    @property
    def mode(self) -> PowerMode:
        return self._mode

    @property
    def in_eclipse(self) -> bool:
        return self._in_eclipse

    def total_load(self, mode: Optional[PowerMode] = None) -> float:
        """Total power consumption in current mode."""
        if mode is None:
            mode = self._mode

        total = 0.0
        for load in self.loads:
            if not load.enabled:
                continue
            if mode == PowerMode.LOW_POWER and not load.essential:
                continue
            if mode == PowerMode.SAFE and not load.essential:
                continue
            total += load.power_w

        return total

    def essential_load(self) -> float:
        """Power for essential loads only."""
        return sum(l.power_w for l in self.loads if l.essential and l.enabled)

    def solar_power(self, distance_au: float, sun_angles: List[float] = None) -> float:
        """
        Total solar panel power generation.

        Args:
            distance_au: Distance from Sun
            sun_angles: List of sun angles for each panel (or None for all sun-facing)

        Returns:
            Total power (W)
        """
        if self._in_eclipse:
            return 0.0

        if sun_angles is None:
            sun_angles = [0.0] * len(self.panels)

        total = 0.0
        for i, panel in enumerate(self.panels):
            angle = sun_angles[i] if i < len(sun_angles) else 0.0
            total += panel.power_output(distance_au, angle)

        return total

    def battery_soc(self) -> float:
        """Average battery state of charge."""
        if not self.batteries:
            return 0.0
        return sum(b.soc for b in self.batteries) / len(self.batteries)

    def battery_capacity(self) -> float:
        """Total battery capacity (Wh)."""
        return sum(b.capacity_wh for b in self.batteries)

    def update(
        self,
        duration_hours: float,
        distance_au: float = 1.0,
        in_eclipse: bool = False,
        sun_angles: List[float] = None,
    ) -> Dict:
        """
        Update power system state over a time step.

        Args:
            duration_hours: Time step duration
            distance_au: Distance from Sun
            in_eclipse: Whether spacecraft is in eclipse
            sun_angles: Sun angles for each panel

        Returns:
            Dict with power balance info
        """
        self._in_eclipse = in_eclipse

        # Solar power generation
        solar_gen = self.solar_power(distance_au, sun_angles)

        # Power consumption
        power_demand = self.total_load()

        # Power balance
        power_balance = solar_gen - power_demand

        # Handle power balance
        if power_balance >= 0:
            # Excess power - charge batteries
            excess = power_balance
            for battery in self.batteries:
                charged = battery.charge(excess, duration_hours)
                excess -= charged / duration_hours  # Reduce remaining excess

            # Return to normal mode if batteries healthy
            if self.battery_soc() > 0.5 and self._mode != PowerMode.NORMAL:
                self._mode = PowerMode.NORMAL
                logger.debug("Power: Returning to NORMAL mode")

        else:
            # Deficit - discharge batteries
            deficit = abs(power_balance)
            for battery in self.batteries:
                discharged = battery.discharge(deficit, duration_hours)
                deficit -= discharged / duration_hours

            # Check if we need to shed loads
            if deficit > 0 or self.battery_soc() < 0.3:
                if self._mode == PowerMode.NORMAL:
                    self._mode = PowerMode.LOW_POWER
                    logger.debug("Power: Entering LOW_POWER mode")

            if self.battery_soc() < 0.15:
                self._mode = PowerMode.SAFE
                logger.warning("Power: Entering SAFE mode (low battery)")

        # Set eclipse mode
        if in_eclipse and self._mode == PowerMode.NORMAL:
            self._mode = PowerMode.ECLIPSE

        return {
            "solar_power_w": solar_gen,
            "load_w": power_demand,
            "balance_w": power_balance,
            "battery_soc": self.battery_soc(),
            "mode": self._mode.value,
            "in_eclipse": in_eclipse,
        }


def detect_eclipse(
    r_sc: np.ndarray,
    r_body: np.ndarray,
    body_radius_au: float,
) -> bool:
    """
    Detect if spacecraft is in shadow of a body.

    Uses simple cylindrical shadow model.

    Args:
        r_sc: Spacecraft position (AU, SSB frame)
        r_body: Body position (AU, SSB frame)
        body_radius_au: Body radius (AU)

    Returns:
        True if in eclipse
    """
    # Sun at origin (SSB)
    r_sun = np.zeros(3)

    # Vector from Sun to body
    sun_to_body = r_body - r_sun
    d_body = np.linalg.norm(sun_to_body)

    if d_body < 1e-10:
        return False

    # Vector from Sun to spacecraft
    sun_to_sc = r_sc - r_sun
    d_sc = np.linalg.norm(sun_to_sc)

    # Check if spacecraft is behind body (further from Sun)
    proj = np.dot(sun_to_sc, sun_to_body) / d_body
    if proj < d_body:
        return False  # Spacecraft is not behind body

    # Perpendicular distance from Sun-body line
    perp = np.linalg.norm(sun_to_sc - (proj / d_body) * sun_to_body)

    # In shadow if within body's radius (cylindrical approximation)
    return perp < body_radius_au


def detect_earth_eclipse(r_sc: np.ndarray, r_earth: np.ndarray) -> bool:
    """Check if spacecraft is in Earth's shadow."""
    earth_radius_au = BODIES['earth']['radius']
    return detect_eclipse(r_sc, r_earth, earth_radius_au)


def detect_moon_eclipse(r_sc: np.ndarray, r_moon: np.ndarray) -> bool:
    """Check if spacecraft is in Moon's shadow."""
    moon_radius_au = BODIES['moon']['radius']
    return detect_eclipse(r_sc, r_moon, moon_radius_au)
