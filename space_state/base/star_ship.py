# base/star_ship.py
import numpy as np
from typing import List
from dataclasses import dataclass

from .star_logger import logger
from .star_config import CONFIG
from .star_constants import g0
from .star_utils import mps2_apd2, kgps_kgpd, mps2_to_kps2


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
        return self.__isp * g0  # m/s

    @property
    def _mdot(self) -> float:
        """Mass flow rate [kg/s] at current throttle."""
        if self._throttle == 0.0:
            return 0.0
        return self._thrust / (self._velocity_exhaust)  # kg/s
    
    @property
    def _max_mdot(self) -> float:
        """Maximum mass flow rate [kg/s] at full throttle."""
        return self.__thrust_max / (self._velocity_exhaust)  # kg/s
    
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

        # Attitude control properties
        self._inertia = np.eye(3)  # Default to unit matrix (kg·m²)
        self._wheel_momentum = np.zeros(4)  # 4 reaction wheels (N·m·s)
        self._wheel_torques = np.zeros(4)  # Current wheel torques (N·m) - set by attitude_control
        self._rcs_propellant_mass = 0.0  # kg
        self._rcs_mdot_kgpd = 0.0  # RCS mass flow rate (kg/day) - set by attitude_control

    @property ###
    def dry_mass(self) -> float:
        return self._dry_mass

    @property ###
    def propellant_mass(self) -> float:
        return self._propellant_mass
    
    @property
    def initial_propellant_mass(self) -> float:
        return self._initial_propellant
    
    @property ###
    def mass(self) -> float:
        return self.dry_mass + self.propellant_mass + self.rcs_propellant_mass

    @property
    def thrusters(self) -> List[Thruster]:
        return self._thrusters

    @property
    def thrust_direction(self) -> np.ndarray:
        return self._thrust_direction

    @thrust_direction.setter
    def thrust_direction(self, direction: np.ndarray):
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            raise ValueError("Thrust direction cannot be zero vector")
        self._thrust_direction = direction / norm

    @property
    def thrust(self) -> float:
        return sum(t._thrust for t in self.thrusters)
    
    @property
    def max_thrust(self) -> float:
        return sum(t._thrust_max for t in self.thrusters)

    @property
    def isp(self) -> float:
        # Weighted average by thrust
        if self.thrust == 0:
            return 0.0
        return sum(t._isp * t._thrust for t in self.thrusters) / self.thrust

    @property
    def max_isp(self) -> float:
        # Weighted average by max thrust
        return sum(t._isp * t._thrust_max for t in self.thrusters) / self.max_thrust

    @property ##
    def mdot(self) -> float:
        if not self.active:
            return 0.0
        return sum(t._mdot for t in self.thrusters) # kg/s
    
    @property
    def max_mdot(self) -> float:
        return sum(t._max_mdot for t in self.thrusters)  # kg/s

    @property
    def mdot_kgpd(self) -> float:
        """Mass flow rate in kg/day."""
        return kgps_kgpd(self.mdot)
    
    @property
    def ve(self) -> float: ###
        # Weighted average by thrust
        if self.thrust == 0:
            return 0.0
        return sum(t._velocity_exhaust * t._thrust for t in self.thrusters) / self.thrust
    
    @property
    def max_ve(self) -> float:
        # Weighted average by max thrust
        return sum(t._velocity_exhaust * t._thrust_max for t in self.thrusters) / self.max_thrust

    @property
    def active(self) -> bool:
        return self.thrust > 0 and self.propellant_mass > 0

    @property
    def throttle(self) -> float:
        if not self.active:
            return 0.0
        # Assuming all thrusters have the same throttle level
        return sum(t._throttle for t in self.thrusters) / len(self.thrusters) if self.thrusters else 0.0
    
    @throttle.setter
    def throttle(self, level: float):
        if level < 0.0 or level > 1.0:
            raise ValueError("Throttle level must be between 0.0 and 1.0")
        for thruster in self.thrusters:
            thruster._set_throttle(level)

    @property
    def thrust_vector(self) -> np.ndarray: ###
        """Total thrust acceleration in AU/day²."""
        if not self.active:
            return np.zeros(3)
        acc_m_s2 = self.thrust / self.mass  # N / kg = m/s²
        return mps2_apd2(acc_m_s2) * self.thrust_direction # m/s² → AU/day²
    
    @property
    def thrust_vector_km_s2(self) -> np.ndarray: ###
        """Total thrust acceleration in km/s²."""
        if not self.active:
            return np.zeros(3)
        acc_m_s2 = self.thrust / self.mass  # N / kg = m/s²
        return mps2_to_kps2(acc_m_s2) * self.thrust_direction  # m/s² → km/s²

    def thrust_vector_from_direction(self, direction: np.ndarray) -> np.ndarray:
        """
        Compute thrust vector in AU/day² for given inertial direction.

        Args:
            direction: Unit vector in inertial frame

        Returns:
            Thrust acceleration (AU/day²)
        """
        if not self.active:
            return np.zeros(3)
        acc_m_s2 = self.thrust / self.mass  # N / kg = m/s²
        return mps2_apd2(acc_m_s2) * direction

    @property
    def inertia(self) -> np.ndarray:
        """Inertia tensor (3×3 matrix) in kg·m²."""
        return self._inertia

    @inertia.setter
    def inertia(self, I: np.ndarray):
        """Set inertia tensor."""
        if I.shape != (3, 3):
            raise ValueError("Inertia must be 3×3 matrix")
        self._inertia = I

    @property
    def wheel_momentum(self) -> np.ndarray:
        """Reaction wheel momentum vector (4,) in N·m·s."""
        return self._wheel_momentum

    @property
    def rcs_propellant_mass(self) -> float:
        """RCS propellant mass in kg."""
        return self._rcs_propellant_mass

    @property
    def initial_rcs_propellant_mass(self) -> float:
        """Initial RCS propellant mass in kg."""
        if not hasattr(self, '_initial_rcs_propellant'):
            self._initial_rcs_propellant = self._rcs_propellant_mass
        return self._initial_rcs_propellant

    def consume_rcs_propellant(self, dm_kg: float) -> float:
        """
        Consume RCS propellant.

        Args:
            dm_kg: Mass to consume (positive value, kg)

        Returns:
            Actual mass consumed (may be less if tank depleted)
        """
        if dm_kg <= 0:
            return 0.0

        actual_consumed = min(dm_kg, self._rcs_propellant_mass)
        self._rcs_propellant_mass -= actual_consumed

        if self._rcs_propellant_mass <= 0:
            if actual_consumed > 0:
                logger.warning(f"RCS propellant depleted for {self._name}!")
            self._rcs_propellant_mass = 0.0

        return actual_consumed

    def update_propellant(self, mass: float, bypass: bool = False):
        """
        Update main propellant mass from total spacecraft mass.

        Note: This is called by the propagator with total mass from state vector.
        RCS propellant is tracked separately via consume_rcs_propellant().

        Args:
            mass: Total spacecraft mass (kg) from state vector
        """
        # Total mass = dry_mass + propellant_mass + rcs_propellant_mass
        # So propellant_mass = mass - dry_mass - rcs_propellant_mass
        remaining = mass - self.dry_mass - self._rcs_propellant_mass
        self._propellant_mass = max(0.0, remaining)

    def log(self):
        logger.info(f"Spacecraft: {self._name}")
        logger.info(f"  Dry Mass: {self.dry_mass:.1f} kg")
        logger.info(f"  Propellant Mass: {self.propellant_mass:.1f} kg")
        logger.info(f"  Max Thrust: {self.max_thrust:.1f} N")
        logger.info(f"  Max ISP: {self.max_isp:.1f} s")
    
    def __str__(self):
        return f"Spacecraft(name={self._name}, mass={self.total_mass:.1f} kg, thrust={self.max_thrust:.1f} N)"

    def copy(self) -> 'Spacecraft':
        return Spacecraft(
            name=self._name,
            dry_mass=self._dry_mass,
            propellant_mass=self._propellant_mass,
            thrusters=self._thrusters.copy()
        )


# ————————————————————————
# Staged Spacecraft
# Multi-stage rockets with jettison capability
# ————————————————————————

@dataclass
class Stage:
    """
    A single rocket stage with its own propellant and engines.

    Attributes:
        name: Stage identifier (e.g., "S1", "S2", "Booster")
        dry_mass: Stage dry mass in kg (structure, engines, tanks)
        propellant_mass: Initial propellant mass in kg
        thrusters: List of Thruster objects for this stage
        jettisoned: Whether this stage has been jettisoned
    """
    name: str
    dry_mass: float
    propellant_mass: float
    thrusters: List[Thruster]
    jettisoned: bool = False
    _current_propellant: float = None

    def __post_init__(self):
        if self._current_propellant is None:
            self._current_propellant = self.propellant_mass

    @property
    def current_propellant(self) -> float:
        return self._current_propellant

    @current_propellant.setter
    def current_propellant(self, value: float):
        self._current_propellant = max(0.0, value)

    @property
    def total_mass(self) -> float:
        """Current total mass of stage (dry + propellant)."""
        if self.jettisoned:
            return 0.0
        return self.dry_mass + self._current_propellant

    @property
    def thrust(self) -> float:
        """Current thrust from all thrusters on this stage."""
        if self.jettisoned or self._current_propellant <= 0:
            return 0.0
        return sum(t._thrust for t in self.thrusters)

    @property
    def max_thrust(self) -> float:
        """Maximum thrust at full throttle."""
        return sum(t._thrust_max for t in self.thrusters)

    @property
    def isp(self) -> float:
        """Weighted average ISP."""
        if self.thrust == 0:
            return sum(t._isp for t in self.thrusters) / len(self.thrusters) if self.thrusters else 0.0
        return sum(t._isp * t._thrust for t in self.thrusters) / self.thrust

    @property
    def mdot(self) -> float:
        """Mass flow rate in kg/s."""
        if self.jettisoned or self._current_propellant <= 0:
            return 0.0
        return sum(t._mdot for t in self.thrusters)

    @property
    def is_depleted(self) -> bool:
        """True if propellant exhausted."""
        return self._current_propellant <= 0

    def set_throttle(self, level: float):
        """Set throttle for all thrusters on this stage."""
        for t in self.thrusters:
            t._set_throttle(level)


class StagedSpacecraft(Spacecraft):
    """
    Multi-stage spacecraft with automatic staging.

    Stages are stored in order from first (bottom) to last (top/payload).
    When a stage is depleted, it's jettisoned and the next stage activates.
    """

    def __init__(self, name: str, stages: List[Stage], payload_mass: float = 0.0):
        """
        Args:
            name: Spacecraft name
            stages: List of Stage objects (first stage = index 0)
            payload_mass: Non-jettisoned payload mass (kg)
        """
        self._stages = stages
        self._payload_mass = payload_mass
        self._current_stage_idx = 0
        self._staging_events = []  # Log of staging events

        # Initialize as Spacecraft with combined properties
        super().__init__(
            name=name,
            dry_mass=self._compute_dry_mass(),
            propellant_mass=self._compute_propellant_mass(),
            thrusters=self._get_active_thrusters()
        )

        # Attitude control defaults (can be overridden)
        self._inertia = np.diag([500_000, 500_000, 100_000])
        self._rcs_propellant_mass = 2_000

    @property
    def stages(self) -> List[Stage]:
        return self._stages

    @property
    def current_stage(self) -> Stage:
        """Currently active stage."""
        if self._current_stage_idx < len(self._stages):
            return self._stages[self._current_stage_idx]
        return None

    @property
    def current_stage_idx(self) -> int:
        return self._current_stage_idx

    @property
    def num_stages(self) -> int:
        return len(self._stages)

    @property
    def payload_mass(self) -> float:
        return self._payload_mass

    def _compute_dry_mass(self) -> float:
        """Total dry mass of un-jettisoned stages + payload."""
        mass = self._payload_mass
        for stage in self._stages:
            if not stage.jettisoned:
                mass += stage.dry_mass
        return mass

    def _compute_propellant_mass(self) -> float:
        """Total propellant in un-jettisoned stages."""
        return sum(s.current_propellant for s in self._stages if not s.jettisoned)

    def _get_active_thrusters(self) -> List[Thruster]:
        """Get thrusters from currently active stage."""
        if self.current_stage is None:
            return []
        return self.current_stage.thrusters

    @property
    def dry_mass(self) -> float:
        return self._compute_dry_mass()

    @property
    def propellant_mass(self) -> float:
        return self._compute_propellant_mass()

    @property
    def mass(self) -> float:
        return self.dry_mass + self.propellant_mass + self.rcs_propellant_mass

    @property
    def thrust(self) -> float:
        if self.current_stage is None:
            return 0.0
        return self.current_stage.thrust

    @property
    def max_thrust(self) -> float:
        if self.current_stage is None:
            return 0.0
        return self.current_stage.max_thrust

    @property
    def isp(self) -> float:
        if self.current_stage is None:
            return 0.0
        return self.current_stage.isp

    @property
    def mdot(self) -> float:
        if self.current_stage is None:
            return 0.0
        return self.current_stage.mdot

    @property
    def active(self) -> bool:
        return self.current_stage is not None and not self.current_stage.is_depleted

    @property
    def throttle(self) -> float:
        """Get current throttle level."""
        if self.current_stage is None:
            return 0.0
        thrusters = self.current_stage.thrusters
        if not thrusters:
            return 0.0
        return sum(t._throttle for t in thrusters) / len(thrusters)

    @throttle.setter
    def throttle(self, level: float):
        """Set throttle for current stage."""
        if self.current_stage is not None:
            self.current_stage.set_throttle(level)

    def check_staging(self, t: float) -> bool:
        """
        Check if staging is needed and perform if so.

        Args:
            t: Current simulation time (days)

        Returns:
            True if a staging event occurred
        """
        if self.current_stage is None:
            return False

        if self.current_stage.is_depleted and not self.current_stage.jettisoned:
            self._perform_staging(t)
            return True
        return False

    def _perform_staging(self, t: float):
        """Jettison current stage and advance to next."""
        stage = self.current_stage
        if stage is None:
            return

        # Log the event
        event = {
            't': t,
            'stage_name': stage.name,
            'stage_idx': self._current_stage_idx,
            'jettisoned_mass': stage.dry_mass,
        }
        self._staging_events.append(event)

        logger.info(f"STAGING: Jettisoning {stage.name} at t={t:.4f} days")
        logger.info(f"  Jettisoned mass: {stage.dry_mass:,.0f} kg")

        # Jettison the stage
        stage.jettisoned = True

        # Advance to next stage
        self._current_stage_idx += 1

        if self._current_stage_idx < len(self._stages):
            new_stage = self._stages[self._current_stage_idx]
            logger.info(f"  Now active: {new_stage.name} ({new_stage.current_propellant:,.0f} kg propellant)")
        else:
            logger.info("  All stages expended - coasting on payload only")

    def consume_propellant(self, dm_kg: float):
        """
        Consume propellant from current stage.

        Args:
            dm_kg: Propellant mass to consume (positive value, kg)
        """
        if dm_kg <= 0 or self.current_stage is None:
            return

        stage = self.current_stage
        actual = min(dm_kg, stage.current_propellant)
        stage.current_propellant -= actual

    def update_propellant(self, total_mass: float, bypass: bool = False):
        """
        Update propellant from total spacecraft mass.
        Called by propagator after integration step.
        """
        if bypass:
            return

        # Current stage propellant = total_mass - (all dry masses) - (other stage propellants) - RCS
        other_propellant = sum(s.current_propellant for s in self._stages
                               if not s.jettisoned and s != self.current_stage)
        current_propellant = total_mass - self.dry_mass - other_propellant - self.rcs_propellant_mass

        if self.current_stage is not None:
            self.current_stage.current_propellant = max(0.0, current_propellant)

    def get_staging_summary(self) -> str:
        """Return summary of all stages."""
        lines = [f"Staged Spacecraft: {self._name}"]
        lines.append(f"  Payload mass: {self._payload_mass:,.0f} kg")
        lines.append(f"  Total mass: {self.mass:,.0f} kg")
        lines.append(f"  Current stage: {self._current_stage_idx + 1}/{len(self._stages)}")
        lines.append("")
        for i, stage in enumerate(self._stages):
            status = "JETTISONED" if stage.jettisoned else "ACTIVE" if i == self._current_stage_idx else "pending"
            lines.append(f"  Stage {i+1} ({stage.name}): {status}")
            lines.append(f"    Dry mass: {stage.dry_mass:,.0f} kg")
            lines.append(f"    Propellant: {stage.current_propellant:,.0f}/{stage.propellant_mass:,.0f} kg")
            lines.append(f"    Max thrust: {stage.max_thrust/1e6:.2f} MN")
            lines.append(f"    ISP: {stage.isp:.0f} s")
        return "\n".join(lines)

    def copy(self) -> 'StagedSpacecraft':
        """Create a copy of this staged spacecraft."""
        new_stages = []
        for s in self._stages:
            new_stage = Stage(
                name=s.name,
                dry_mass=s.dry_mass,
                propellant_mass=s.propellant_mass,
                thrusters=s.thrusters.copy(),
                jettisoned=s.jettisoned,
            )
            new_stage.current_propellant = s.current_propellant
            new_stages.append(new_stage)

        new_ship = StagedSpacecraft(
            name=self._name,
            stages=new_stages,
            payload_mass=self._payload_mass,
        )
        new_ship._current_stage_idx = self._current_stage_idx
        new_ship._inertia = self._inertia.copy()
        new_ship._rcs_propellant_mass = self._rcs_propellant_mass
        return new_ship


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
    """SpaceX Starship (3 vacuum Raptors + RCS + Attitude Control).

    Propellant budget includes ~5% margin for realistic pointing losses
    from attitude control (cos(θ) efficiency penalty during burns).
    """
    raptor = Thruster(thrust_max=2.6e6, isp=380.0)
    ship = Spacecraft(
        name="Starship",
        dry_mass=120_000,
        propellant_mass=1_500_000,  # Adequate margin for complex multi-burn missions
        thrusters=[raptor] * 3
    )

    # Add attitude control properties
    ship._inertia = np.diag([500_000, 500_000, 100_000])  # kg·m² (Ixx, Iyy, Izz)
    ship._rcs_propellant_mass = 2_000  # kg

    return ship


def agena_engine() -> Spacecraft:
    """Lockheed Agena (Bell 8096)."""
    bell = Thruster(thrust_max=7.1e4, isp=300.0)
    return Spacecraft(
        name="Agena",
        dry_mass=670,
        propellant_mass=6_000,
        thrusters=[bell]
    )


def starship_full_stack() -> StagedSpacecraft:
    """
    SpaceX Starship + Super Heavy full stack (two-stage).

    Stage 1 (Super Heavy): 33 Raptor engines, provides initial boost
    Stage 2 (Starship): 3 vacuum Raptors + 3 sea-level for orbital ops

    This configuration provides ~14 km/s delta-V capability, enough for:
    - Earth orbit insertion
    - Trans-lunar injection
    - Lunar orbit insertion
    - Return to Earth (with refueling)
    """
    # Super Heavy booster (Stage 1)
    # 33 Raptor engines, sea-level optimized
    raptor_sl = Thruster(thrust_max=2.3e6, isp=330.0)  # Sea-level Raptor
    stage1 = Stage(
        name="Super Heavy",
        dry_mass=200_000,      # ~200 tonnes dry
        propellant_mass=3_400_000,  # ~3400 tonnes propellant
        thrusters=[raptor_sl] * 33,
    )

    # Starship upper stage (Stage 2)
    # 3 vacuum Raptors + 3 sea-level (simplified to 6 vacuum)
    raptor_vac = Thruster(thrust_max=2.6e6, isp=380.0)  # Vacuum Raptor
    stage2 = Stage(
        name="Starship",
        dry_mass=120_000,      # ~120 tonnes dry
        propellant_mass=1_200_000,  # ~1200 tonnes propellant
        thrusters=[raptor_vac] * 6,
    )

    ship = StagedSpacecraft(
        name="Starship Full Stack",
        stages=[stage1, stage2],
        payload_mass=100_000,  # ~100 tonnes payload capacity
    )

    # Attitude control (on Starship stage)
    ship._inertia = np.diag([2_000_000, 2_000_000, 500_000])  # Larger stack
    ship._rcs_propellant_mass = 5_000  # More RCS for larger vehicle

    return ship


def saturn_v() -> StagedSpacecraft:
    """
    NASA Saturn V (three-stage) - Apollo program.

    Stage 1 (S-IC): 5 F-1 engines
    Stage 2 (S-II): 5 J-2 engines
    Stage 3 (S-IVB): 1 J-2 engine

    Total delta-V: ~15 km/s (with payload)
    """
    # S-IC (Stage 1) - 5 F-1 engines
    f1 = Thruster(thrust_max=6.77e6, isp=263.0)  # F-1 at sea level
    stage1 = Stage(
        name="S-IC",
        dry_mass=130_000,
        propellant_mass=2_160_000,
        thrusters=[f1] * 5,
    )

    # S-II (Stage 2) - 5 J-2 engines
    j2 = Thruster(thrust_max=1.033e6, isp=421.0)  # J-2 vacuum
    stage2 = Stage(
        name="S-II",
        dry_mass=40_000,
        propellant_mass=450_000,
        thrusters=[j2] * 5,
    )

    # S-IVB (Stage 3) - 1 J-2 engine (restartable for TLI)
    stage3 = Stage(
        name="S-IVB",
        dry_mass=13_000,
        propellant_mass=107_000,
        thrusters=[j2],
    )

    ship = StagedSpacecraft(
        name="Saturn V",
        stages=[stage1, stage2, stage3],
        payload_mass=45_000,  # Apollo CSM + LM
    )

    ship._inertia = np.diag([5_000_000, 5_000_000, 1_000_000])
    ship._rcs_propellant_mass = 500  # SM RCS

    return ship