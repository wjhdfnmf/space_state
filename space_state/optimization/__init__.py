# optimization/__init__.py
"""
Trajectory Optimization Module

Provides fuel-optimal trajectory planning capabilities:
- Impulsive maneuver optimization
- Multi-burn sequence optimization
- Rendezvous timing optimization
- Gravity assist trajectory design
"""

from .trajectory_opt import (
    # Data classes
    ManeuverNode,
    OptimizationResult,
    TrajectoryLeg,
    # Single maneuver optimization
    optimize_single_impulse,
    optimize_plane_change,
    optimize_phasing,
    # Multi-burn optimization
    optimize_hohmann,
    optimize_bielliptic,
    optimize_three_burn,
    # Transfer optimization
    optimize_lambert_departure,
    optimize_rendezvous,
    # Gravity assists
    GravityAssist,
    compute_flyby_geometry,
    optimize_powered_flyby,
    # Fuel estimation
    estimate_propellant,
    compute_mass_ratio,
    # Utilities
    delta_v_budget_margin,
)

__all__ = [
    # Data classes
    "ManeuverNode",
    "OptimizationResult",
    "TrajectoryLeg",
    # Single maneuver optimization
    "optimize_single_impulse",
    "optimize_plane_change",
    "optimize_phasing",
    # Multi-burn optimization
    "optimize_hohmann",
    "optimize_bielliptic",
    "optimize_three_burn",
    # Transfer optimization
    "optimize_lambert_departure",
    "optimize_rendezvous",
    # Gravity assists
    "GravityAssist",
    "compute_flyby_geometry",
    "optimize_powered_flyby",
    # Fuel estimation
    "estimate_propellant",
    "compute_mass_ratio",
    # Utilities
    "delta_v_budget_margin",
]
