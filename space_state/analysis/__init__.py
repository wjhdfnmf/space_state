# analysis/__init__.py
"""
Analysis Module

Provides statistical analysis and uncertainty quantification:
- Monte Carlo trajectory dispersion analysis
- Sensitivity analysis
- Statistical trajectory bounds
- Mission success probability
"""

from .monte_carlo import (
    # Data classes
    DispersionSource,
    DispersionConfig,
    MonteCarloResult,
    TrajectoryStatistics,
    # Core Monte Carlo
    run_monte_carlo,
    # Dispersion sources
    create_position_dispersion,
    create_velocity_dispersion,
    create_mass_dispersion,
    create_thrust_dispersion,
    apply_dispersions,
    # Statistics
    compute_trajectory_statistics,
    compute_percentile_bounds,
    compute_covariance,
    mahalanobis_distance,
    # Sensitivity analysis
    sensitivity_analysis,
    compute_state_transition_matrix,
    # Success criteria
    SuccessCriterion,
    evaluate_success,
    compute_success_probability,
)

__all__ = [
    # Data classes
    "DispersionSource",
    "DispersionConfig",
    "MonteCarloResult",
    "TrajectoryStatistics",
    # Core Monte Carlo
    "run_monte_carlo",
    # Dispersion sources
    "create_position_dispersion",
    "create_velocity_dispersion",
    "create_mass_dispersion",
    "create_thrust_dispersion",
    "apply_dispersions",
    # Statistics
    "compute_trajectory_statistics",
    "compute_percentile_bounds",
    "compute_covariance",
    "mahalanobis_distance",
    # Sensitivity analysis
    "sensitivity_analysis",
    "compute_state_transition_matrix",
    # Success criteria
    "SuccessCriterion",
    "evaluate_success",
    "compute_success_probability",
]
