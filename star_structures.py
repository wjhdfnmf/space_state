# star_structures.py

from typing import NamedTuple, Sequence, Dict, Any, List
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from star_logger import logger


# Type aliases for unit clarity (no runtime cost)
Length = float        # automical units (AU)
Velocity = float      # AU/day
Time = float          # days
Mass = float          # kg
Vector = NDArray[np.float64]     # shape (3,) unit vector
State = NDArray[np.float64]      # shape (7,) [r_x, r_y, r_z, v_x, v_y, v_z, m]

# Generic controller class - any guidance can inherit
@dataclass
class ControllerCycle:
    """Base class for any guidance cycle."""
    vgo: float                      # Velocity To Go
    thrust_dir: Vector = field(default_factory=lambda: np.zeros(3))  # Thrust Direction Unit Vector
    throttle: float = 0.0           # Throttle Setting (0 to 1)
    active: bool = False            # Guidance active or coasting

# Specific ZEM/ZEV cycle
@dataclass
class ZEMZEVCycle(ControllerCycle):
    primer_magnitude: float = 0.0  # |λ|
    zem: float = 0.0               # Zero-Effort-Miss
    zev: float = 0.0               # Zero-Effort-Velocity
    Tgo_sec: float = 0.0           # Time-to-go in seconds

# Generic guidance cycle - can hold any controller type
@dataclass
class GuidanceCycle:
    t: Time                         # Time of guidance update
    controller: ControllerCycle

# Immutable state vector
@dataclass(frozen=True)
class StateVector:
    r: Vector     # shape (3,)
    v: Vector     # shape (3,)
    m: Mass = 0.0   # mass in kg
    t: Time = 0.0   # time in days past epoch

    def to_array(self) -> NDArray[np.float64]: # shape (7,)
        return np.concatenate([self.r, self.v, [self.m]])
    
    @classmethod
    def from_array(cls, y: State, t: Time) -> 'StateVector':
        return cls(r=y[0:3], v=y[3:6], m=y[6], t=t)

# MAIN CLASS FOR PROPAGATION RESULTS
@dataclass
class PropagationResult:
    """
    Complete propagation result over one or more phases.
    Includes time, state vectors, guidance cycles, events, and conservation data.
    """
    t: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # shape (N,) → time in days past epoch
    y: NDArray[np.float64] = field(default_factory=lambda: np.empty((7, 0)))  # shape (7, N) → [r_x, r_y, r_z, v_x, v_y, v_z, m]
    guidance: Sequence[GuidanceCycle] = field(default_factory=list)  # Guidance cycles during propagation
    events: List[dict[str, Any]] = field(default_factory=list)  # List of event dictionaries
    conservation: Dict[str, Any]= field(default_factory=dict) # e.g., energy, momentum

    @property
    def state(self) -> StateVector:
        """Return current state as StateVector."""
        if self.t.size == 0:
            raise ValueError("PropagationResult has no data points")
        return StateVector.from_array(self.y[:, -1], self.t[-1])

    def append(self, phase: 'PropagationResult') -> None:
        """Append another propagation phase to this result."""
        if not np.isclose(self.t[-1], phase.t[0]):
            logger.warning("append propagation: time discontinuity detected between phases.")
        self.t = np.concatenate([self.t, phase.t])
        self.y = np.concatenate([self.y, phase.y], axis=1)
        self.guidance.extend(phase.guidance)
        self.events.extend(phase.events)
        self.conservation.update(phase.conservation)

    def log_cycle(self, t: Time, cycle: ControllerCycle) -> None:
        """Log a guidance cycle at time t."""
        self.guidance.append(GuidanceCycle(t=t, controller=cycle))