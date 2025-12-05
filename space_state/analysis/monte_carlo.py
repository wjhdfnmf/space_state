# analysis/monte_carlo.py
"""
Monte Carlo Trajectory Analysis

Implements uncertainty quantification and dispersion analysis:
- Initial state dispersions
- Thrust magnitude/direction errors
- Navigation uncertainties
- Statistical trajectory bounds
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable, Any
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base.star_constants import AU_TO_KM, DAY_TO_SEC


# ==================== DATA CLASSES ====================

class DispersionType(Enum):
    """Types of dispersions."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    TRUNCATED_GAUSSIAN = "truncated_gaussian"


@dataclass
class DispersionSource:
    """Definition of a single dispersion source."""
    name: str
    state_indices: List[int]  # Which state elements are affected
    sigma: np.ndarray  # Standard deviation (1-sigma) for each element
    distribution: DispersionType = DispersionType.GAUSSIAN
    correlation_matrix: Optional[np.ndarray] = None  # For correlated errors
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (lower, upper) for truncation

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a random sample from this dispersion."""
        n = len(self.state_indices)

        if self.distribution == DispersionType.GAUSSIAN:
            if self.correlation_matrix is not None:
                # Correlated multivariate normal
                cov = np.outer(self.sigma, self.sigma) * self.correlation_matrix
                return rng.multivariate_normal(np.zeros(n), cov)
            else:
                return rng.normal(0, self.sigma)

        elif self.distribution == DispersionType.UNIFORM:
            # Uniform within ±3-sigma bounds
            return rng.uniform(-3 * self.sigma, 3 * self.sigma)

        elif self.distribution == DispersionType.TRUNCATED_GAUSSIAN:
            if self.bounds is None:
                self.bounds = (-3 * self.sigma, 3 * self.sigma)
            # Rejection sampling for truncated normal
            samples = np.zeros(n)
            for i in range(n):
                while True:
                    s = rng.normal(0, self.sigma[i])
                    if self.bounds[0][i] <= s <= self.bounds[1][i]:
                        samples[i] = s
                        break
            return samples

        return np.zeros(n)


@dataclass
class DispersionConfig:
    """Configuration for Monte Carlo dispersions."""
    sources: List[DispersionSource] = field(default_factory=list)
    seed: Optional[int] = None  # Random seed for reproducibility
    n_samples: int = 100  # Number of Monte Carlo samples

    def add_source(self, source: DispersionSource):
        """Add a dispersion source."""
        self.sources.append(source)


@dataclass
class TrajectoryStatistics:
    """Statistical summary of trajectory ensemble."""
    times: np.ndarray  # Time array
    mean: np.ndarray  # Mean trajectory (n_times, n_states)
    std: np.ndarray  # Standard deviation (n_times, n_states)
    percentiles: Dict[int, np.ndarray]  # Percentile trajectories {p: trajectory}
    covariance: Optional[np.ndarray] = None  # Covariance at final time (n_states, n_states)
    min_values: Optional[np.ndarray] = None  # Minimum values per state
    max_values: Optional[np.ndarray] = None  # Maximum values per state


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo analysis."""
    n_samples: int
    n_successful: int
    trajectories: List[np.ndarray]  # List of trajectory arrays
    times: List[np.ndarray]  # List of time arrays
    dispersions_applied: List[Dict[str, np.ndarray]]  # Dispersions for each run
    statistics: Optional[TrajectoryStatistics] = None
    success_probability: float = 0.0
    seed: Optional[int] = None

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return (self.n_successful / self.n_samples) * 100 if self.n_samples > 0 else 0


# ==================== DISPERSION SOURCES ====================

def create_position_dispersion(
    sigma_km: float = 1.0,
    correlation: float = 0.0,
    name: str = "position"
) -> DispersionSource:
    """
    Create position dispersion source.

    Args:
        sigma_km: Position uncertainty (km, 1-sigma)
        correlation: Correlation between x, y, z
        name: Source name

    Returns:
        DispersionSource for position
    """
    sigma_au = sigma_km / AU_TO_KM
    sigma = np.array([sigma_au, sigma_au, sigma_au])

    if correlation != 0:
        corr = np.array([
            [1.0, correlation, correlation],
            [correlation, 1.0, correlation],
            [correlation, correlation, 1.0]
        ])
    else:
        corr = None

    return DispersionSource(
        name=name,
        state_indices=[0, 1, 2],
        sigma=sigma,
        correlation_matrix=corr,
    )


def create_velocity_dispersion(
    sigma_mps: float = 1.0,
    correlation: float = 0.0,
    name: str = "velocity"
) -> DispersionSource:
    """
    Create velocity dispersion source.

    Args:
        sigma_mps: Velocity uncertainty (m/s, 1-sigma)
        correlation: Correlation between components
        name: Source name

    Returns:
        DispersionSource for velocity
    """
    # Convert m/s to AU/day
    sigma_au_day = (sigma_mps / 1000) * DAY_TO_SEC / AU_TO_KM
    sigma = np.array([sigma_au_day, sigma_au_day, sigma_au_day])

    if correlation != 0:
        corr = np.array([
            [1.0, correlation, correlation],
            [correlation, 1.0, correlation],
            [correlation, correlation, 1.0]
        ])
    else:
        corr = None

    return DispersionSource(
        name=name,
        state_indices=[3, 4, 5],
        sigma=sigma,
        correlation_matrix=corr,
    )


def create_mass_dispersion(
    sigma_kg: float = 10.0,
    name: str = "mass"
) -> DispersionSource:
    """
    Create mass dispersion source.

    Args:
        sigma_kg: Mass uncertainty (kg, 1-sigma)
        name: Source name

    Returns:
        DispersionSource for mass
    """
    return DispersionSource(
        name=name,
        state_indices=[6],
        sigma=np.array([sigma_kg]),
        distribution=DispersionType.TRUNCATED_GAUSSIAN,
        bounds=(np.array([-3 * sigma_kg]), np.array([3 * sigma_kg])),
    )


def create_thrust_dispersion(
    magnitude_percent: float = 3.0,
    pointing_deg: float = 0.5,
    name: str = "thrust"
) -> Dict[str, Any]:
    """
    Create thrust dispersion parameters.

    Note: Thrust dispersions are applied during propagation,
    not to initial state.

    Args:
        magnitude_percent: Thrust magnitude error (%, 1-sigma)
        pointing_deg: Thrust pointing error (degrees, 1-sigma)
        name: Source name

    Returns:
        Dictionary with thrust dispersion parameters
    """
    return {
        'name': name,
        'magnitude_sigma_percent': magnitude_percent,
        'pointing_sigma_deg': pointing_deg,
    }


def apply_dispersions(
    y0: np.ndarray,
    config: DispersionConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Apply dispersions to initial state.

    Args:
        y0: Nominal initial state
        config: Dispersion configuration
        rng: Random number generator

    Returns:
        Tuple of (dispersed state, dict of applied dispersions)
    """
    y_dispersed = y0.copy()
    dispersions = {}

    for source in config.sources:
        sample = source.sample(rng)
        dispersions[source.name] = sample

        for i, idx in enumerate(source.state_indices):
            if idx < len(y_dispersed):
                y_dispersed[idx] += sample[i]

    return y_dispersed, dispersions


# ==================== MONTE CARLO CORE ====================

def run_monte_carlo(
    propagate_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    y0: np.ndarray,
    config: DispersionConfig,
    success_criterion: Optional['SuccessCriterion'] = None,
    parallel: bool = False,
    n_workers: int = 4,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo trajectory analysis.

    Args:
        propagate_func: Function that takes initial state, returns (times, states)
        y0: Nominal initial state
        config: Dispersion configuration
        success_criterion: Optional success criterion
        parallel: Whether to run in parallel
        n_workers: Number of parallel workers
        progress_callback: Optional callback(completed, total)

    Returns:
        MonteCarloResult with all trajectories and statistics
    """
    # Initialize RNG
    if config.seed is not None:
        rng = np.random.default_rng(config.seed)
    else:
        rng = np.random.default_rng()

    trajectories = []
    times_list = []
    dispersions_list = []
    n_successful = 0

    def run_single(sample_idx: int, seed: int) -> Tuple[int, np.ndarray, np.ndarray, Dict, bool]:
        """Run a single Monte Carlo sample."""
        local_rng = np.random.default_rng(seed)
        y_dispersed, dispersions = apply_dispersions(y0, config, local_rng)

        try:
            t, y = propagate_func(y_dispersed)
            success = True
            if success_criterion is not None:
                success = evaluate_success(t, y, success_criterion)
            return sample_idx, t, y, dispersions, success
        except Exception as e:
            # Failed propagation
            return sample_idx, np.array([0]), y_dispersed.reshape(1, -1).T, dispersions, False

    if parallel:
        # Generate seeds for each sample
        seeds = rng.integers(0, 2**31, size=config.n_samples)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_single, i, seeds[i]): i
                for i in range(config.n_samples)
            }

            completed = 0
            for future in as_completed(futures):
                idx, t, y, disp, success = future.result()
                trajectories.append(y)
                times_list.append(t)
                dispersions_list.append(disp)
                if success:
                    n_successful += 1

                completed += 1
                if progress_callback:
                    progress_callback(completed, config.n_samples)
    else:
        # Sequential execution
        for i in range(config.n_samples):
            seed = rng.integers(0, 2**31)
            _, t, y, disp, success = run_single(i, seed)
            trajectories.append(y)
            times_list.append(t)
            dispersions_list.append(disp)
            if success:
                n_successful += 1

            if progress_callback:
                progress_callback(i + 1, config.n_samples)

    # Compute statistics
    statistics = compute_trajectory_statistics(times_list, trajectories)

    return MonteCarloResult(
        n_samples=config.n_samples,
        n_successful=n_successful,
        trajectories=trajectories,
        times=times_list,
        dispersions_applied=dispersions_list,
        statistics=statistics,
        success_probability=n_successful / config.n_samples if config.n_samples > 0 else 0,
        seed=config.seed,
    )


# ==================== STATISTICS ====================

def compute_trajectory_statistics(
    times_list: List[np.ndarray],
    trajectories: List[np.ndarray],
    percentiles: List[int] = [5, 25, 50, 75, 95],
) -> TrajectoryStatistics:
    """
    Compute statistical summary of trajectory ensemble.

    Args:
        times_list: List of time arrays
        trajectories: List of trajectory arrays (n_states, n_times)
        percentiles: Percentiles to compute

    Returns:
        TrajectoryStatistics summary
    """
    if not trajectories:
        return TrajectoryStatistics(
            times=np.array([]),
            mean=np.array([]),
            std=np.array([]),
            percentiles={},
        )

    # Find common time grid (use first trajectory's times as reference)
    t_ref = times_list[0]
    n_times = len(t_ref)
    n_states = trajectories[0].shape[0]
    n_samples = len(trajectories)

    # Interpolate all trajectories to common time grid
    ensemble = np.zeros((n_samples, n_states, n_times))
    for i, (t, y) in enumerate(zip(times_list, trajectories)):
        if len(t) == n_times:
            ensemble[i] = y
        else:
            # Interpolate to reference times
            for j in range(n_states):
                ensemble[i, j, :] = np.interp(t_ref, t, y[j, :])

    # Compute statistics
    mean = np.mean(ensemble, axis=0)
    std = np.std(ensemble, axis=0)

    # Percentiles
    pct_dict = {}
    for p in percentiles:
        pct_dict[p] = np.percentile(ensemble, p, axis=0)

    # Covariance at final time
    final_states = ensemble[:, :, -1]  # (n_samples, n_states)
    covariance = np.cov(final_states.T)

    # Min/max
    min_vals = np.min(ensemble, axis=0)
    max_vals = np.max(ensemble, axis=0)

    return TrajectoryStatistics(
        times=t_ref,
        mean=mean,
        std=std,
        percentiles=pct_dict,
        covariance=covariance,
        min_values=min_vals,
        max_values=max_vals,
    )


def compute_percentile_bounds(
    times: np.ndarray,
    trajectories: List[np.ndarray],
    state_index: int,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute percentile bounds for a single state variable.

    Args:
        times: Reference time array
        trajectories: List of trajectory arrays
        state_index: Which state to analyze
        lower_pct: Lower percentile
        upper_pct: Upper percentile

    Returns:
        Tuple of (lower_bound, median, upper_bound)
    """
    n_times = len(times)
    n_samples = len(trajectories)

    # Extract state variable
    data = np.zeros((n_samples, n_times))
    for i, y in enumerate(trajectories):
        if y.shape[1] == n_times:
            data[i] = y[state_index, :]
        else:
            data[i] = np.interp(times, np.linspace(0, 1, y.shape[1]), y[state_index, :])

    lower = np.percentile(data, lower_pct, axis=0)
    median = np.percentile(data, 50, axis=0)
    upper = np.percentile(data, upper_pct, axis=0)

    return lower, median, upper


def compute_covariance(
    trajectories: List[np.ndarray],
    time_index: int = -1,
) -> np.ndarray:
    """
    Compute state covariance matrix at a specific time.

    Args:
        trajectories: List of trajectory arrays
        time_index: Time index for covariance (-1 = final)

    Returns:
        Covariance matrix (n_states, n_states)
    """
    states = np.array([y[:, time_index] for y in trajectories])
    return np.cov(states.T)


def mahalanobis_distance(
    y: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
) -> float:
    """
    Compute Mahalanobis distance from mean.

    Args:
        y: State vector
        mean: Mean state
        cov: Covariance matrix

    Returns:
        Mahalanobis distance
    """
    diff = y - mean
    try:
        cov_inv = np.linalg.inv(cov)
        return np.sqrt(diff.T @ cov_inv @ diff)
    except np.linalg.LinAlgError:
        # Singular covariance - use Euclidean distance
        return np.linalg.norm(diff)


# ==================== SENSITIVITY ANALYSIS ====================

def sensitivity_analysis(
    propagate_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    y0: np.ndarray,
    perturbation_scales: np.ndarray,
    output_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Perform sensitivity analysis via finite differences.

    Args:
        propagate_func: Propagation function
        y0: Nominal initial state
        perturbation_scales: Scale for each state perturbation
        output_func: Function to compute scalar output from (t, y)

    Returns:
        Dictionary with sensitivity results
    """
    n_states = len(y0)

    # Nominal trajectory
    t_nom, y_nom = propagate_func(y0)

    if output_func is None:
        # Default: final position magnitude
        def output_func(t, y):
            return np.linalg.norm(y[0:3, -1])

    output_nom = output_func(t_nom, y_nom)

    # Compute sensitivities
    sensitivities = np.zeros(n_states)

    for i in range(n_states):
        if perturbation_scales[i] == 0:
            continue

        # Positive perturbation
        y_plus = y0.copy()
        y_plus[i] += perturbation_scales[i]
        t_plus, y_pert_plus = propagate_func(y_plus)
        output_plus = output_func(t_plus, y_pert_plus)

        # Negative perturbation
        y_minus = y0.copy()
        y_minus[i] -= perturbation_scales[i]
        t_minus, y_pert_minus = propagate_func(y_minus)
        output_minus = output_func(t_minus, y_pert_minus)

        # Central difference
        sensitivities[i] = (output_plus - output_minus) / (2 * perturbation_scales[i])

    # Normalize by nominal output
    normalized = sensitivities * perturbation_scales / output_nom if output_nom != 0 else sensitivities

    return {
        'sensitivities': sensitivities,
        'normalized_sensitivities': normalized,
        'perturbation_scales': perturbation_scales,
        'nominal_output': output_nom,
    }


def compute_state_transition_matrix(
    propagate_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    y0: np.ndarray,
    dt: float = 1e-6,
) -> np.ndarray:
    """
    Compute state transition matrix via finite differences.

    Args:
        propagate_func: Propagation function
        y0: Initial state
        dt: Perturbation size

    Returns:
        State transition matrix (n_states, n_states)
    """
    n = len(y0)
    _, y_nom = propagate_func(y0)
    y_final_nom = y_nom[:, -1]

    stm = np.zeros((n, n))

    for i in range(n):
        y_pert = y0.copy()
        y_pert[i] += dt
        _, y_pert_result = propagate_func(y_pert)
        y_final_pert = y_pert_result[:, -1]

        stm[:, i] = (y_final_pert - y_final_nom) / dt

    return stm


# ==================== SUCCESS CRITERIA ====================

@dataclass
class SuccessCriterion:
    """Definition of mission success criteria."""
    name: str
    check_func: Callable[[np.ndarray, np.ndarray], bool]
    description: str = ""


def create_altitude_criterion(
    min_altitude_km: float,
    max_altitude_km: float,
    body_radius_km: float,
    body_position_func: Optional[Callable[[float], np.ndarray]] = None,
) -> SuccessCriterion:
    """
    Create altitude-based success criterion.

    Args:
        min_altitude_km: Minimum acceptable altitude
        max_altitude_km: Maximum acceptable altitude
        body_radius_km: Body radius
        body_position_func: Function to get body position at time t

    Returns:
        SuccessCriterion for altitude
    """
    def check(t: np.ndarray, y: np.ndarray) -> bool:
        for i in range(len(t)):
            pos = y[0:3, i] * AU_TO_KM
            if body_position_func is not None:
                body_pos = body_position_func(t[i]) * AU_TO_KM
                rel_pos = pos - body_pos
                r = np.linalg.norm(rel_pos)
            else:
                r = np.linalg.norm(pos)

            altitude = r - body_radius_km
            if altitude < min_altitude_km or altitude > max_altitude_km:
                return False
        return True

    return SuccessCriterion(
        name="altitude",
        check_func=check,
        description=f"Altitude within {min_altitude_km}-{max_altitude_km} km",
    )


def create_final_state_criterion(
    target_state: np.ndarray,
    tolerance: np.ndarray,
    state_indices: Optional[List[int]] = None,
) -> SuccessCriterion:
    """
    Create final state matching criterion.

    Args:
        target_state: Target final state
        tolerance: Tolerance for each state element
        state_indices: Which state elements to check (None = all)

    Returns:
        SuccessCriterion for final state
    """
    def check(t: np.ndarray, y: np.ndarray) -> bool:
        final_state = y[:, -1]
        indices = state_indices if state_indices is not None else range(len(target_state))

        for i in indices:
            if abs(final_state[i] - target_state[i]) > tolerance[i]:
                return False
        return True

    return SuccessCriterion(
        name="final_state",
        check_func=check,
        description="Final state within tolerance",
    )


def evaluate_success(
    t: np.ndarray,
    y: np.ndarray,
    criterion: SuccessCriterion,
) -> bool:
    """
    Evaluate if trajectory meets success criterion.

    Args:
        t: Time array
        y: State array
        criterion: Success criterion

    Returns:
        True if successful
    """
    try:
        return criterion.check_func(t, y)
    except Exception:
        return False


def compute_success_probability(
    result: MonteCarloResult,
    criterion: SuccessCriterion,
) -> float:
    """
    Compute probability of success from Monte Carlo results.

    Args:
        result: Monte Carlo result
        criterion: Success criterion to evaluate

    Returns:
        Success probability (0 to 1)
    """
    n_success = 0
    for t, y in zip(result.times, result.trajectories):
        if evaluate_success(t, y, criterion):
            n_success += 1

    return n_success / result.n_samples if result.n_samples > 0 else 0
