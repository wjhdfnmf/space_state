# star_ship.py
import numpy as np
from typing import List
from dataclasses import dataclass

from .star_logger import logger
from .star_config import CONFIG
from .star_utils import mps2_apd2, kgps_kgpd


class Thruster:
    """Fixed-direction, throttleable"""
    def __init__(self, thrust_max: float, isp: float):
        """
        Args:
            thrust_max: N (100% throttle, vacuum)
            isp: seconds (vacuum)
        """
        self.__thrust_max = float(thrust_max)
        self.__isp = float(isp)
        self._throttle = 0.0  # 0.0 to 1.0

    @property
    def _thrust(self) -> float:
        """Current thrust in N."""
        if self._throttle == 0.0:
            return 0.0
        return self.__thrust_max * self._throttle
    
    @property
    def _thrust_max(self) -> float:
        """Maximum thrust in N."""
        return self.__thrust_max
    
    @property
    def _isp(self) -> float:
        """Specific impulse [s] at current throttle."""
        return self.__isp  # Constant for now
    
    @property
    def _velocity_exhaust(self) -> float:
        """Exhaust velocity [m/s] at current ISP."""
        return self.__isp * CONFIG.g0  # m/s

    @property
    def _mdot(self) -> float:
        """Mass flow rate [kg/s] at current throttle."""
        if self._throttle == 0.0:
            return 0.0
        return self._thrust / (self._velocity_exhaust)  # kg/s
    
    def _set_throttle(self, level: float):
        """Set throttle level (0.0 to 1.0)."""
        self._throttle = np.clip(float(level), 0.0, 1.0)


@dataclass
class Spacecraft:
    """
    Spacecraft with fixed-aft thrusters, perfect pointing via guidance.
    Thrust direction commanded in inertial frame.
    """
    def __init__(self, name: str, dry_mass: float, propellant_mass: float, thrusters: List[Thruster]):
        self._name = name
        self._dry_mass = dry_mass                   # kg 
        self._propellant_mass = propellant_mass     # kg
        self._thrusters = thrusters          
        self._thrust_direction = np.array([0.0, 0.0, 1.0])  # Default aft
        self._initial_propellant = propellant_mass

    @property
    def dry_mass(self) -> float:
        return self._dry_mass

    @property
    def propellant_mass(self) -> float:
        return self._propellant_mass
    
    @property
    def initial_propellant_mass(self) -> float:
        return self._initial_propellant
    
    @property
    def total_mass(self) -> float:
        return self.dry_mass + self.propellant_mass

    @property
    def thrusters(self) -> List[Thruster]:
        return self._thrusters

    @property
    def thrust_direction(self) -> np.ndarray:
        return self._thrust_direction

    @thrust_direction.setter
    def thrust_direction(self, direction: np.ndarray):
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Thrust direction cannot be zero vector")
        self._thrust_direction = direction / norm

    @property
    def total_thrust(self) -> float:
        return sum(t._thrust for t in self.thrusters)
    
    @property
    def max_thrust(self) -> float:
        return sum(t._thrust_max for t in self.thrusters)

    @property
    def total_isp(self) -> float:
        # Weighted average by thrust
        return sum(t._isp * t._thrust for t in self.thrusters) / self.total_thrust

    @property ##
    def total_mdot(self) -> float:
        if not self.active:
            return 0.0
        return kgps_kgpd(sum(t._mdot for t in self.thrusters)) # kg/s
    
    @property
    def active(self) -> bool:
        return self.total_thrust > 0 and self.propellant_mass > 0
    
    def set_throttle(self, level: float): ###
        """Set throttle level (0.0 to 1.0) for all thrusters."""
        if level < 0.0 or level > 1.0:
            raise ValueError("Throttle level must be between 0.0 and 1.0")
        for thruster in self.thrusters:
            thruster._set_throttle(level)

    def get_thrust_vector(self) -> np.ndarray: ###
        """Total thrust acceleration in AU/day²."""
        if not self.active:
            return np.zeros(3)
        acc_m_s2 = self.total_thrust / self.total_mass  # N / kg = m/s²
        return mps2_apd2(acc_m_s2) * self.thrust_direction # m/s² → AU/day²
    
    def update_propellant(self, mass: float): ###
        """Consume propellant over time interval dt_sec."""
        remaining = mass - self.dry_mass
        if remaining < 0.0:
            logger.warning(f"{self._name}: Propellant exhausted!")
        self._propellant_mass = max(0.0, remaining)

    def log(self):
        logger.info(f"Spacecraft: {self._name}")
        logger.info(f"  Dry Mass: {self.dry_mass:.1f} kg")
        logger.info(f"  Propellant Mass: {self.propellant_mass:.1f} kg")
        logger.info(f"  Max Thrust: {self.max_thrust:.1f} N")
    
    def __str__(self):
        return f"Spacecraft(name={self._name}, mass={self.total_mass:.1f} kg, thrust={self.max_thrust:.1f} N)"
# ————————————————————————
# Real-world examples
# aft thrusters only - perfect pointing assumed - unlimited throttle range
# ————————————————————————


def centaur_rl10() -> Spacecraft:
    """ULA Centaur III (single RL10C-1-1)."""
    rl10 = Thruster(thrust_max=1.103e5, isp=465.5)
    return Spacecraft(
        name="Centaur",
        dry_mass=2_300,
        propellant_mass=20_700,
        thrusters=[rl10]
    )


def starship_raptor_vac() -> Spacecraft:
    """SpaceX Starship (3 vacuum Raptors)."""
    raptor = Thruster(thrust_max=2.6e6, isp=380.0)
    return Spacecraft(
        name="Starship",
        dry_mass=120_000,
        propellant_mass=1_100_000,
        thrusters=[raptor] * 3
    )


def agena_engine() -> Spacecraft:
    """Lockheed Agena (Bell 8096)."""
    bell = Thruster(thrust_max=7.1e4, isp=300.0)
    return Spacecraft(
        name="Agena",
        dry_mass=670,
        propellant_mass=6_000,
        thrusters=[bell]
    )