# mission/__init__.py
"""
Mission Planning Framework

Provides high-level building blocks for mission design:
- Mission phases and plans
- Pre-built maneuvers
- Mission templates (Earth-Moon, Mars, LEO rendezvous)
- Delta-V budgets and fuel analysis
"""

from .mission_planner import (
    # Enums
    PhaseType,
    ManeuverType,
    # Data classes
    DeltaVBudget,
    MissionPhase,
    MissionPlan,
    # Maneuver calculators
    hohmann_transfer,
    escape_velocity,
    circular_velocity,
    plane_change_dv,
    # Mission templates
    create_earth_moon_mission,
    create_mars_transfer_mission,
    create_leo_rendezvous_mission,
    # Executor
    MissionExecutor,
    # Utilities
    quick_estimate,
)

__all__ = [
    # Enums
    "PhaseType",
    "ManeuverType",
    # Data classes
    "DeltaVBudget",
    "MissionPhase",
    "MissionPlan",
    # Maneuver calculators
    "hohmann_transfer",
    "escape_velocity",
    "circular_velocity",
    "plane_change_dv",
    # Mission templates
    "create_earth_moon_mission",
    "create_mars_transfer_mission",
    "create_leo_rendezvous_mission",
    # Executor
    "MissionExecutor",
    # Utilities
    "quick_estimate",
]
