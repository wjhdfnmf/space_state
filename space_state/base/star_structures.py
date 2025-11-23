# star_structures.py

from typing import NamedTuple, Sequence, Dict, Any, List
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .star_logger import logger


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

    def initialize(self, state: NDArray[np.float64], time: Time) -> None:
        """Initialize propagation result with initial state."""
        if state.shape != (7,):
            raise ValueError("initalize: state must have shape (7,)")
        self.clear()
        self.t = np.array([time])
        self.y = state.reshape(7, 1)

    @property
    def state(self) -> NDArray[np.float64]:
        """Return current state as StateVector."""
        return self.y[:, -1]
    
    @property
    def time(self) -> Time:
        """Return current time in days."""
        return self.t[-1]

    def append(self, phase: 'PropagationResult') -> None:
        """Append another propagation phase to this result."""
        if not np.isclose(self.t[-1], phase.t[0]):
            logger.warning("append propagation: time discontinuity detected between phases.")
        if phase.y.shape[0] != 7:
            raise ValueError("append propagation: phase.y must have shape (7, N)")
        if phase.t.size != phase.y.shape[1]:
            raise ValueError("append propagation: phase.t and phase.y size mismatch")
        if phase.t[0] == self.t[-1]:
            phase.t = phase.t[1:]
            phase.y = phase.y[:, 1:]
        
        self.t = np.concatenate([self.t, phase.t])
        self.y = np.concatenate([self.y, phase.y], axis=1)
        self.guidance.extend(phase.guidance)
        self.events.extend(phase.events)
        self.conservation.update(phase.conservation)

    def append_raw(self, t_new: NDArray[np.float64], y_new: NDArray[np.float64]) -> None:
        """Append raw time and state arrays to the result."""
        if not np.isclose(self.t[-1], t_new[0]):
            logger.warning("append_raw: time discontinuity detected between phases.")
        if y_new.shape[0] != 7:
            raise ValueError("append_raw: y_new must have shape (7, N)")
        if t_new.size != y_new.shape[1]:
            raise ValueError("append_raw: t_new and y_new size mismatch")
        if t_new[0] == self.t[-1]:
            t_new = t_new[1:]
            y_new = y_new[:, 1:]
        
        self.t = np.concatenate([self.t, t_new])
        self.y = np.concatenate([self.y, y_new], axis=1)
    
    def clear(self) -> None:
        """Clear all stored data."""
        self.t = np.array([])
        self.y = np.empty((7, 0))
        self.guidance.clear()
        self.events.clear()
        self.conservation.clear()

    def log_cycle(self, t: Time, cycle: ControllerCycle) -> None:
        """Log a guidance cycle at time t."""
        self.guidance.append(GuidanceCycle(t=t, controller=cycle))

    def describe_state(self) -> None:
        """Log a summary of the last state."""
        logger.debug(f"Last state at t={self.time:.3f} days: r={self.state[:3]}, v={self.state[3:6]}, m={self.state[6]:.3f} kg")
        logger.debug(f"Normal position: {np.linalg.norm(self.state[:3]):.6f} AU")
        logger.debug(f"Normal velocity: {np.linalg.norm(self.state[3:6]):.6f} AU/day")