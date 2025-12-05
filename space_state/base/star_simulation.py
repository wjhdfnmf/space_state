# base/star_simulation.py
"""
Multi-spacecraft simulation manager.

Supports:
- Multiple independent spacecraft with own guidance
- Synchronized time advancement
- Relative state queries (range, velocity)
- Docking (merge entities) and separation (split entities)
"""

from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from .star_propagator import Propagator
from .star_structures import PropagationResult
from .star_ship import Spacecraft, StagedSpacecraft
from .star_constants import IDX_R, IDX_V, IDX_M, IDX_Q, IDX_OMEGA
from .star_logger import logger


@dataclass
class SpacecraftEntity:
    """
    A spacecraft entity in the simulation.

    Encapsulates a spacecraft with its propagator, guidance, and history.
    """
    id: str
    spacecraft: Spacecraft
    propagator: Propagator
    guidance: object = None
    active: bool = True
    parent_id: Optional[str] = None  # For tracking lineage after docking/separation

    @property
    def state(self) -> np.ndarray:
        """Current 14-element state vector."""
        return self.propagator.result.state

    @property
    def time(self) -> float:
        """Current simulation time (days)."""
        return self.propagator.result.time

    @property
    def position(self) -> np.ndarray:
        """Current position (AU, SSB frame)."""
        return self.state[IDX_R]

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity (AU/day, SSB frame)."""
        return self.state[IDX_V]

    @property
    def mass(self) -> float:
        """Current mass (kg)."""
        return self.state[IDX_M]

    @property
    def quaternion(self) -> np.ndarray:
        """Current attitude quaternion."""
        return self.state[IDX_Q]

    @property
    def angular_velocity(self) -> np.ndarray:
        """Current angular velocity (rad/day)."""
        return self.state[IDX_OMEGA]

    def __repr__(self):
        status = "active" if self.active else "inactive"
        return f"SpacecraftEntity(id='{self.id}', status={status}, t={self.time:.4f}d)"


class Simulation:
    """
    Multi-spacecraft simulation manager.

    Manages multiple spacecraft entities with synchronized time.
    Supports docking, separation, and relative navigation.
    """

    def __init__(self, name: str = "Simulation"):
        """
        Initialize an empty simulation.

        Args:
            name: Simulation identifier
        """
        self.name = name
        self.entities: Dict[str, SpacecraftEntity] = {}
        self._t_current: float = 0.0
        self._events: List[Dict[str, Any]] = []
        self._next_id: int = 1

    @property
    def current_time(self) -> float:
        """Current simulation time (days)."""
        return self._t_current

    @property
    def spacecraft_ids(self) -> List[str]:
        """List of all spacecraft IDs."""
        return list(self.entities.keys())

    @property
    def active_ids(self) -> List[str]:
        """List of active spacecraft IDs."""
        return [id for id, e in self.entities.items() if e.active]

    def add_spacecraft(
        self,
        spacecraft: Spacecraft,
        initial_state: np.ndarray,
        guidance: object = None,
        id: Optional[str] = None,
        t0: float = 0.0,
    ) -> str:
        """
        Add a spacecraft to the simulation.

        Args:
            spacecraft: Spacecraft object
            initial_state: 14-element state vector [r, v, m, q, ω]
            guidance: Guidance controller (optional)
            id: Unique identifier (auto-generated if None)
            t0: Initial time (days)

        Returns:
            Spacecraft ID
        """
        if id is None:
            id = f"SC-{self._next_id}"
            self._next_id += 1

        if id in self.entities:
            raise ValueError(f"Spacecraft ID '{id}' already exists")

        # Create propagator for this spacecraft
        propagator = Propagator(ship=spacecraft, guidance=guidance)
        propagator.result.initialize(initial_state, t0)

        # Create entity
        entity = SpacecraftEntity(
            id=id,
            spacecraft=spacecraft,
            propagator=propagator,
            guidance=guidance,
            active=True,
        )

        self.entities[id] = entity
        self._t_current = max(self._t_current, t0)

        logger.info(f"Added spacecraft '{id}' to simulation at t={t0:.4f} days")
        logger.info(f"  Position: {initial_state[:3]} AU")
        logger.info(f"  Mass: {initial_state[6]:,.0f} kg")

        return id

    def remove_spacecraft(self, id: str) -> bool:
        """
        Remove a spacecraft from the simulation.

        Args:
            id: Spacecraft ID to remove

        Returns:
            True if removed, False if not found
        """
        if id not in self.entities:
            logger.warning(f"Spacecraft '{id}' not found")
            return False

        entity = self.entities.pop(id)
        logger.info(f"Removed spacecraft '{id}' from simulation")
        return True

    def deactivate_spacecraft(self, id: str) -> bool:
        """
        Deactivate a spacecraft (stop propagating but keep history).

        Args:
            id: Spacecraft ID

        Returns:
            True if deactivated
        """
        if id not in self.entities:
            return False

        self.entities[id].active = False
        logger.info(f"Deactivated spacecraft '{id}'")
        return True

    def get_entity(self, id: str) -> Optional[SpacecraftEntity]:
        """Get spacecraft entity by ID."""
        return self.entities.get(id)

    def get_state(self, id: str) -> Optional[np.ndarray]:
        """Get current state of spacecraft."""
        entity = self.entities.get(id)
        return entity.state if entity else None

    def propagate_all(
        self,
        t_end: float,
        max_step: float = None,
    ) -> Dict[str, PropagationResult]:
        """
        Propagate all active spacecraft to the specified time.

        Args:
            t_end: Target time (days)
            max_step: Maximum integration step (days)

        Returns:
            Dictionary of propagation results by spacecraft ID
        """
        results = {}

        for id, entity in self.entities.items():
            if not entity.active:
                continue

            if entity.time >= t_end:
                logger.debug(f"Spacecraft '{id}' already at t={entity.time:.4f} >= {t_end:.4f}")
                results[id] = entity.propagator.result
                continue

            logger.debug(f"Propagating '{id}' from t={entity.time:.4f} to t={t_end:.4f}")

            if max_step:
                entity.propagator.propagate(t_end, max_step=max_step)
            else:
                entity.propagator.propagate(t_end)

            results[id] = entity.propagator.result

        self._t_current = t_end
        return results

    def propagate_steps(
        self,
        dt: float,
        n_steps: int = 1,
        max_step: float = None,
    ) -> Dict[str, PropagationResult]:
        """
        Propagate all spacecraft by time step(s).

        Args:
            dt: Time step (days)
            n_steps: Number of steps

        Returns:
            Dictionary of propagation results
        """
        results = {}
        for _ in range(n_steps):
            t_end = self._t_current + dt
            results = self.propagate_all(t_end, max_step=max_step)
        return results

    def get_relative_state(
        self,
        id1: str,
        id2: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get relative state between two spacecraft.

        Args:
            id1: First spacecraft ID (target)
            id2: Second spacecraft ID (chaser)

        Returns:
            Dictionary with relative position, velocity, range, closing rate
        """
        e1 = self.entities.get(id1)
        e2 = self.entities.get(id2)

        if e1 is None or e2 is None:
            return None

        # Relative position and velocity
        r_rel = e1.position - e2.position  # AU
        v_rel = e1.velocity - e2.velocity  # AU/day

        # Range and closing rate
        range_au = np.linalg.norm(r_rel)
        range_km = range_au * 1.496e8  # AU to km

        if range_au > 1e-12:
            r_hat = r_rel / range_au
            closing_rate_au_day = -np.dot(v_rel, r_hat)
            closing_rate_mps = closing_rate_au_day * 1.731e6  # AU/day to m/s
        else:
            closing_rate_mps = 0.0

        return {
            'r_rel_au': r_rel,
            'v_rel_au_day': v_rel,
            'range_km': range_km,
            'range_au': range_au,
            'closing_rate_mps': closing_rate_mps,
            't': self._t_current,
        }

    def dock(
        self,
        id_target: str,
        id_chaser: str,
        new_id: Optional[str] = None,
        docking_delta_v: np.ndarray = None,
    ) -> Optional[str]:
        """
        Dock two spacecraft together.

        The chaser docks to the target. A new combined entity is created.
        Both original entities are deactivated.

        Args:
            id_target: Target spacecraft ID (stays at position)
            id_chaser: Chaser spacecraft ID (docks to target)
            new_id: ID for combined spacecraft (auto-generated if None)
            docking_delta_v: Delta-V applied during docking (AU/day)

        Returns:
            New spacecraft ID, or None if docking failed
        """
        target = self.entities.get(id_target)
        chaser = self.entities.get(id_chaser)

        if target is None or chaser is None:
            logger.error(f"Docking failed: spacecraft not found")
            return None

        if not target.active or not chaser.active:
            logger.error(f"Docking failed: one or both spacecraft inactive")
            return None

        # Check proximity (must be within 1 km for docking)
        rel_state = self.get_relative_state(id_target, id_chaser)
        if rel_state['range_km'] > 1.0:
            logger.warning(f"Docking warning: range={rel_state['range_km']:.1f} km > 1 km")

        # Combine masses
        combined_mass = target.mass + chaser.mass

        # Use target position/velocity (with optional delta-V)
        r_combined = target.position.copy()
        v_combined = target.velocity.copy()
        if docking_delta_v is not None:
            v_combined += docking_delta_v

        # Use target attitude
        q_combined = target.quaternion.copy()
        omega_combined = target.angular_velocity.copy()

        # Create combined spacecraft (simplified - just mass addition)
        # In reality, would combine dry masses, propellants, etc.
        from .star_ship import Spacecraft, Thruster

        combined_ship = Spacecraft(
            name=f"{target.spacecraft._name}+{chaser.spacecraft._name}",
            dry_mass=target.spacecraft.dry_mass + chaser.spacecraft.dry_mass,
            propellant_mass=target.spacecraft.propellant_mass + chaser.spacecraft.propellant_mass,
            thrusters=target.spacecraft.thrusters + chaser.spacecraft.thrusters,
        )

        # Copy attitude control properties
        if hasattr(target.spacecraft, '_inertia'):
            combined_ship._inertia = target.spacecraft._inertia + chaser.spacecraft._inertia
        if hasattr(target.spacecraft, '_rcs_propellant_mass'):
            rcs1 = getattr(target.spacecraft, '_rcs_propellant_mass', 0)
            rcs2 = getattr(chaser.spacecraft, '_rcs_propellant_mass', 0)
            combined_ship._rcs_propellant_mass = rcs1 + rcs2

        # Create combined state
        combined_state = np.concatenate([
            r_combined,
            v_combined,
            [combined_mass],
            q_combined,
            omega_combined,
        ])

        # Deactivate original entities
        self.deactivate_spacecraft(id_target)
        self.deactivate_spacecraft(id_chaser)

        # Add combined entity
        if new_id is None:
            new_id = f"{id_target}+{id_chaser}"

        self.add_spacecraft(
            spacecraft=combined_ship,
            initial_state=combined_state,
            guidance=target.guidance,  # Use target's guidance
            id=new_id,
            t0=self._t_current,
        )

        # Log docking event
        event = {
            'type': 'docking',
            't': self._t_current,
            'target': id_target,
            'chaser': id_chaser,
            'combined': new_id,
            'combined_mass': combined_mass,
        }
        self._events.append(event)

        logger.info(f"DOCKING: {id_chaser} docked to {id_target} → {new_id}")
        logger.info(f"  Combined mass: {combined_mass:,.0f} kg")

        return new_id

    def separate(
        self,
        id: str,
        separation_mass: float,
        separation_delta_v: np.ndarray,
        new_id1: Optional[str] = None,
        new_id2: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Separate a spacecraft into two parts.

        Args:
            id: Spacecraft ID to separate
            separation_mass: Mass of separated part (kg)
            separation_delta_v: Relative velocity between parts (AU/day)
            new_id1: ID for remaining part
            new_id2: ID for separated part

        Returns:
            Tuple of (remaining_id, separated_id), or None if failed
        """
        entity = self.entities.get(id)
        if entity is None or not entity.active:
            logger.error(f"Separation failed: spacecraft '{id}' not found or inactive")
            return None

        if separation_mass >= entity.mass:
            logger.error(f"Separation failed: separation mass exceeds total mass")
            return None

        # Compute masses
        remaining_mass = entity.mass - separation_mass

        # Conservation of momentum: m1*v1 + m2*v2 = (m1+m2)*v0
        # With relative velocity: v2 - v1 = dv
        # Solving: v1 = v0 - (m2/(m1+m2))*dv
        #          v2 = v0 + (m1/(m1+m2))*dv
        total_mass = entity.mass
        v0 = entity.velocity
        v1 = v0 - (separation_mass / total_mass) * separation_delta_v
        v2 = v0 + (remaining_mass / total_mass) * separation_delta_v

        # Create two new spacecraft (simplified)
        ship1 = Spacecraft(
            name=f"{entity.spacecraft._name}_A",
            dry_mass=entity.spacecraft.dry_mass * (remaining_mass / total_mass),
            propellant_mass=entity.spacecraft.propellant_mass * (remaining_mass / total_mass),
            thrusters=entity.spacecraft.thrusters,
        )

        ship2 = Spacecraft(
            name=f"{entity.spacecraft._name}_B",
            dry_mass=entity.spacecraft.dry_mass * (separation_mass / total_mass),
            propellant_mass=entity.spacecraft.propellant_mass * (separation_mass / total_mass),
            thrusters=[],  # Separated part has no thrusters
        )

        # Create states
        state1 = np.concatenate([
            entity.position, v1, [remaining_mass],
            entity.quaternion, entity.angular_velocity,
        ])
        state2 = np.concatenate([
            entity.position, v2, [separation_mass],
            entity.quaternion, np.zeros(3),  # No rotation for separated part
        ])

        # Deactivate original
        self.deactivate_spacecraft(id)

        # Add new entities
        if new_id1 is None:
            new_id1 = f"{id}_A"
        if new_id2 is None:
            new_id2 = f"{id}_B"

        self.add_spacecraft(ship1, state1, entity.guidance, new_id1, self._t_current)
        self.add_spacecraft(ship2, state2, None, new_id2, self._t_current)

        # Log separation event
        event = {
            'type': 'separation',
            't': self._t_current,
            'original': id,
            'part1': new_id1,
            'part2': new_id2,
            'separation_mass': separation_mass,
        }
        self._events.append(event)

        logger.info(f"SEPARATION: {id} → {new_id1} + {new_id2}")
        logger.info(f"  {new_id1}: {remaining_mass:,.0f} kg")
        logger.info(f"  {new_id2}: {separation_mass:,.0f} kg")

        return (new_id1, new_id2)

    def get_all_states(self) -> Dict[str, np.ndarray]:
        """Get current states of all active spacecraft."""
        return {id: e.state for id, e in self.entities.items() if e.active}

    def get_summary(self) -> str:
        """Get simulation summary string."""
        lines = [
            f"Simulation: {self.name}",
            f"  Current time: {self._t_current:.4f} days",
            f"  Spacecraft: {len(self.active_ids)} active, {len(self.entities)} total",
            "",
        ]

        for id, entity in self.entities.items():
            status = "ACTIVE" if entity.active else "inactive"
            lines.append(f"  {id}: {status}")
            lines.append(f"    Position: {entity.position} AU")
            lines.append(f"    Mass: {entity.mass:,.0f} kg")

        if self._events:
            lines.append("")
            lines.append("  Events:")
            for event in self._events[-5:]:  # Last 5 events
                lines.append(f"    t={event['t']:.4f}: {event['type']}")

        return "\n".join(lines)

    def __repr__(self):
        return f"Simulation(name='{self.name}', t={self._t_current:.4f}, n_spacecraft={len(self.active_ids)})"
