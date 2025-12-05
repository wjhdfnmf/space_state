"""
Tests for Monte Carlo Analysis Module
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from analysis import (
    DispersionSource,
    DispersionConfig,
    MonteCarloResult,
    TrajectoryStatistics,
    run_monte_carlo,
    create_position_dispersion,
    create_velocity_dispersion,
    create_mass_dispersion,
    create_thrust_dispersion,
    apply_dispersions,
    compute_trajectory_statistics,
    compute_percentile_bounds,
    compute_covariance,
    mahalanobis_distance,
    sensitivity_analysis,
    compute_state_transition_matrix,
    SuccessCriterion,
    evaluate_success,
    compute_success_probability,
)
from analysis.monte_carlo import DispersionType


class TestDispersionSource:
    """Test DispersionSource data class."""

    def test_gaussian_dispersion(self):
        """Test Gaussian dispersion sampling."""
        source = DispersionSource(
            name="test",
            state_indices=[0, 1, 2],
            sigma=np.array([1.0, 1.0, 1.0]),
            distribution=DispersionType.GAUSSIAN,
        )

        rng = np.random.default_rng(42)
        samples = [source.sample(rng) for _ in range(1000)]
        samples = np.array(samples)

        # Check mean is near zero
        assert_allclose(np.mean(samples, axis=0), np.zeros(3), atol=0.1)

        # Check std is near sigma
        assert_allclose(np.std(samples, axis=0), np.ones(3), atol=0.1)

    def test_uniform_dispersion(self):
        """Test uniform dispersion sampling."""
        sigma = np.array([1.0, 2.0])
        source = DispersionSource(
            name="test",
            state_indices=[0, 1],
            sigma=sigma,
            distribution=DispersionType.UNIFORM,
        )

        rng = np.random.default_rng(42)
        samples = [source.sample(rng) for _ in range(1000)]
        samples = np.array(samples)

        # Uniform should be within ±3σ
        assert np.all(samples[:, 0] >= -3)
        assert np.all(samples[:, 0] <= 3)
        assert np.all(samples[:, 1] >= -6)
        assert np.all(samples[:, 1] <= 6)

    def test_correlated_dispersion(self):
        """Test correlated dispersion sampling."""
        corr = np.array([
            [1.0, 0.9],
            [0.9, 1.0]
        ])
        source = DispersionSource(
            name="test",
            state_indices=[0, 1],
            sigma=np.array([1.0, 1.0]),
            correlation_matrix=corr,
        )

        rng = np.random.default_rng(42)
        samples = [source.sample(rng) for _ in range(1000)]
        samples = np.array(samples)

        # Check correlation is high
        measured_corr = np.corrcoef(samples.T)[0, 1]
        assert measured_corr > 0.7  # Should be close to 0.9


class TestDispersionConfig:
    """Test DispersionConfig class."""

    def test_add_sources(self):
        """Test adding dispersion sources."""
        config = DispersionConfig(n_samples=100)

        config.add_source(create_position_dispersion(sigma_km=10.0))
        config.add_source(create_velocity_dispersion(sigma_mps=1.0))

        assert len(config.sources) == 2

    def test_config_with_seed(self):
        """Test reproducibility with seed."""
        config = DispersionConfig(n_samples=10, seed=12345)
        assert config.seed == 12345


class TestDispersionHelpers:
    """Test dispersion helper functions."""

    def test_create_position_dispersion(self):
        """Test position dispersion creation."""
        disp = create_position_dispersion(sigma_km=100.0)

        assert disp.name == "position"
        assert disp.state_indices == [0, 1, 2]
        assert len(disp.sigma) == 3

    def test_create_velocity_dispersion(self):
        """Test velocity dispersion creation."""
        disp = create_velocity_dispersion(sigma_mps=10.0)

        assert disp.name == "velocity"
        assert disp.state_indices == [3, 4, 5]

    def test_create_mass_dispersion(self):
        """Test mass dispersion creation."""
        disp = create_mass_dispersion(sigma_kg=50.0)

        assert disp.name == "mass"
        assert disp.state_indices == [6]

    def test_create_thrust_dispersion(self):
        """Test thrust dispersion creation."""
        disp = create_thrust_dispersion(magnitude_percent=5.0, pointing_deg=1.0)

        assert 'magnitude_sigma_percent' in disp
        assert 'pointing_sigma_deg' in disp


class TestApplyDispersions:
    """Test dispersion application."""

    def test_apply_single_dispersion(self):
        """Test applying single dispersion."""
        y0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1000.0])

        config = DispersionConfig()
        config.add_source(DispersionSource(
            name="test",
            state_indices=[0],
            sigma=np.array([0.1]),
        ))

        rng = np.random.default_rng(42)
        y_disp, dispersions = apply_dispersions(y0, config, rng)

        # First element should be different
        assert y_disp[0] != y0[0]
        # Others should be same
        assert_allclose(y_disp[1:], y0[1:])
        # Dispersions recorded
        assert 'test' in dispersions

    def test_apply_multiple_dispersions(self):
        """Test applying multiple dispersions."""
        y0 = np.zeros(7)
        y0[6] = 1000.0

        config = DispersionConfig()
        config.add_source(create_position_dispersion(sigma_km=10.0))
        config.add_source(create_velocity_dispersion(sigma_mps=1.0))

        rng = np.random.default_rng(42)
        y_disp, dispersions = apply_dispersions(y0, config, rng)

        assert 'position' in dispersions
        assert 'velocity' in dispersions


class TestMonteCarloRun:
    """Test Monte Carlo execution."""

    def test_simple_propagation(self):
        """Test Monte Carlo with simple propagation function."""
        def simple_propagate(y0):
            """Simple linear propagation."""
            t = np.linspace(0, 1, 10)
            y = np.zeros((len(y0), len(t)))
            for i in range(len(t)):
                y[:, i] = y0 * (1 + t[i])
            return t, y

        y0 = np.array([1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 100.0])

        config = DispersionConfig(n_samples=20, seed=42)
        config.add_source(create_position_dispersion(sigma_km=1.0))

        result = run_monte_carlo(simple_propagate, y0, config)

        assert result.n_samples == 20
        assert len(result.trajectories) == 20
        assert result.statistics is not None

    def test_success_criterion(self):
        """Test Monte Carlo with success criterion."""
        def propagate_with_random_fail(y0):
            t = np.linspace(0, 1, 10)
            y = np.zeros((len(y0), len(t)))
            for i in range(len(t)):
                y[:, i] = y0
            return t, y

        def always_succeed(t, y):
            return True

        criterion = SuccessCriterion(
            name="test",
            check_func=always_succeed,
        )

        y0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
        config = DispersionConfig(n_samples=10, seed=42)

        result = run_monte_carlo(propagate_with_random_fail, y0, config, criterion)

        assert result.n_successful == 10
        assert result.success_probability == 1.0


class TestTrajectoryStatistics:
    """Test trajectory statistics computation."""

    def test_statistics_computation(self):
        """Test basic statistics computation."""
        times = [np.linspace(0, 1, 10) for _ in range(5)]
        trajectories = [np.random.randn(3, 10) + i for i in range(5)]

        stats = compute_trajectory_statistics(times, trajectories)

        assert stats.times is not None
        assert stats.mean.shape == (3, 10)
        assert stats.std.shape == (3, 10)
        assert 50 in stats.percentiles

    def test_empty_trajectories(self):
        """Test with empty trajectory list."""
        stats = compute_trajectory_statistics([], [])

        assert len(stats.times) == 0


class TestPercentileBounds:
    """Test percentile bound computation."""

    def test_percentile_bounds(self):
        """Test percentile bound calculation."""
        times = np.linspace(0, 1, 10)
        trajectories = [np.random.randn(3, 10) for _ in range(50)]

        lower, median, upper = compute_percentile_bounds(
            times, trajectories,
            state_index=0,
            lower_pct=10,
            upper_pct=90,
        )

        assert len(lower) == 10
        assert len(median) == 10
        assert len(upper) == 10
        # Lower should be less than upper
        assert np.all(lower <= upper)


class TestCovariance:
    """Test covariance computation."""

    def test_covariance_matrix(self):
        """Test covariance matrix computation."""
        trajectories = [np.random.randn(3, 10) for _ in range(100)]

        cov = compute_covariance(trajectories, time_index=-1)

        assert cov.shape == (3, 3)
        # Covariance should be symmetric
        assert_allclose(cov, cov.T)
        # Diagonal should be positive
        assert np.all(np.diag(cov) > 0)


class TestMahalanobisDistance:
    """Test Mahalanobis distance calculation."""

    def test_mahalanobis_at_mean(self):
        """Test distance is zero at mean."""
        mean = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)

        dist = mahalanobis_distance(mean, mean, cov)
        assert_allclose(dist, 0.0, atol=1e-10)

    def test_mahalanobis_one_sigma(self):
        """Test distance is 1 at one sigma."""
        mean = np.zeros(2)
        cov = np.eye(2)
        y = np.array([1.0, 0.0])  # One sigma away in x

        dist = mahalanobis_distance(y, mean, cov)
        assert_allclose(dist, 1.0, atol=1e-10)


class TestSensitivityAnalysis:
    """Test sensitivity analysis."""

    def test_linear_sensitivity(self):
        """Test sensitivity of linear system."""
        def linear_propagate(y0):
            t = np.array([0, 1])
            y = np.zeros((len(y0), 2))
            y[:, 0] = y0
            y[:, 1] = 2 * y0  # Double the initial state
            return t, y

        y0 = np.array([1.0, 2.0, 3.0])
        scales = np.array([0.01, 0.01, 0.01])

        result = sensitivity_analysis(linear_propagate, y0, scales)

        assert 'sensitivities' in result
        assert 'normalized_sensitivities' in result


class TestStateTransitionMatrix:
    """Test state transition matrix computation."""

    def test_identity_stm(self):
        """Test STM for identity propagation."""
        def identity_propagate(y0):
            t = np.array([0, 1])
            y = np.column_stack([y0, y0])
            return t, y

        y0 = np.array([1.0, 2.0])
        stm = compute_state_transition_matrix(identity_propagate, y0)

        assert stm.shape == (2, 2)
        # For identity propagation, STM should be identity
        assert_allclose(stm, np.eye(2), atol=1e-4)


class TestSuccessCriterion:
    """Test success criteria."""

    def test_success_criterion_evaluation(self):
        """Test success criterion evaluation."""
        def always_true(t, y):
            return True

        def always_false(t, y):
            return False

        criterion_true = SuccessCriterion(name="true", check_func=always_true)
        criterion_false = SuccessCriterion(name="false", check_func=always_false)

        t = np.array([0, 1])
        y = np.zeros((3, 2))

        assert evaluate_success(t, y, criterion_true)
        assert not evaluate_success(t, y, criterion_false)

    def test_success_probability(self):
        """Test success probability computation."""
        def propagate(y0):
            t = np.array([0, 1])
            y = np.column_stack([y0, y0])
            return t, y

        # Create results with 50% success
        def check_positive(t, y):
            return y[0, -1] > 0

        criterion = SuccessCriterion(name="positive", check_func=check_positive)

        y0 = np.array([1.0, 0.0, 0.0])
        config = DispersionConfig(n_samples=20, seed=42)
        config.add_source(DispersionSource(
            name="test",
            state_indices=[0],
            sigma=np.array([2.0]),  # Large sigma to get some negatives
        ))

        result = run_monte_carlo(propagate, y0, config)

        prob = compute_success_probability(result, criterion)
        assert 0 <= prob <= 1


class TestMonteCarloResult:
    """Test MonteCarloResult class."""

    def test_success_rate(self):
        """Test success rate calculation."""
        result = MonteCarloResult(
            n_samples=100,
            n_successful=75,
            trajectories=[],
            times=[],
            dispersions_applied=[],
        )

        assert result.success_rate == 75.0

    def test_empty_result(self):
        """Test empty result."""
        result = MonteCarloResult(
            n_samples=0,
            n_successful=0,
            trajectories=[],
            times=[],
            dispersions_applied=[],
        )

        assert result.success_rate == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
