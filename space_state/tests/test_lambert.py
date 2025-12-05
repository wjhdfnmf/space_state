"""
Tests for Lambert Problem Solver
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from guidance.lambert import (
    lambert_solve, lambert_universal, stumpff_c, stumpff_s, LambertSolution
)
from base.star_constants import BODIES


class TestStumpffFunctions:
    """Test Stumpff functions C(z) and S(z)."""

    def test_stumpff_c_zero(self):
        """C(0) ≈ 0.5."""
        assert np.isclose(stumpff_c(0.0), 0.5, rtol=1e-6)

    def test_stumpff_s_zero(self):
        """S(0) ≈ 1/6."""
        assert np.isclose(stumpff_s(0.0), 1/6, rtol=1e-6)

    def test_stumpff_c_positive(self):
        """Test C(z) for elliptic case (z > 0)."""
        z = 1.0
        c = stumpff_c(z)
        expected = (1 - np.cos(1.0)) / 1.0
        assert np.isclose(c, expected, rtol=1e-10)

    def test_stumpff_s_positive(self):
        """Test S(z) for elliptic case (z > 0)."""
        z = 1.0
        s = stumpff_s(z)
        expected = (1.0 - np.sin(1.0)) / 1.0
        assert np.isclose(s, expected, rtol=1e-10)

    def test_stumpff_c_negative(self):
        """Test C(z) for hyperbolic case (z < 0)."""
        z = -1.0
        c = stumpff_c(z)
        expected = (1 - np.cosh(1.0)) / z
        assert np.isclose(c, expected, rtol=1e-10)

    def test_stumpff_s_negative(self):
        """Test S(z) for hyperbolic case (z < 0)."""
        z = -1.0
        s = stumpff_s(z)
        expected = (np.sinh(1.0) - 1.0) / 1.0
        assert np.isclose(s, expected, rtol=1e-10)


class TestLambertBasic:
    """Basic Lambert solver tests."""

    def test_simple_transfer(self):
        """Test simple circular orbit transfer."""
        mu = BODIES['earth']['mu']

        # Simple transfer between two points on a circular orbit
        r = 0.00005  # ~7500 km in AU
        r1 = np.array([r, 0.0, 0.0])
        r2 = np.array([0.0, r, 0.0])  # 90° transfer

        # Period of circular orbit at this radius
        T = 2 * np.pi * np.sqrt(r**3 / mu)
        tof = T / 4  # Quarter period for 90° transfer

        solution = lambert_universal(r1, r2, tof, mu, prograde=True)

        assert solution is not None
        assert solution.converged

    def test_hohmann_transfer(self):
        """Test approximate Hohmann transfer."""
        mu = BODIES['sun']['mu']

        # Earth to Mars approximate - slightly off 180° to avoid degeneracy
        # True Hohmann would be exactly 180° but that's numerically singular
        r1 = np.array([1.0, 0.0, 0.0])   # Earth at 1 AU
        # Mars at ~170° rather than exactly 180° (realistic encounter geometry)
        r2 = np.array([-1.5, 0.3, 0.0])  # ~168° transfer, ~1.53 AU

        r2_mag = np.linalg.norm(r2)
        a_transfer = (1.0 + r2_mag) / 2

        # TOF based on approximate Hohmann
        tof = np.pi * np.sqrt(a_transfer**3 / mu)

        solution = lambert_universal(r1, r2, tof, mu, prograde=True)

        assert solution is not None
        assert solution.converged
        # Semi-major axis should be reasonably close to expected
        # (not exact Hohmann due to off-180° geometry)
        assert 0.8 < solution.a < 2.0  # Reasonable elliptic transfer

    def test_retrograde_transfer(self):
        """Test retrograde transfer option."""
        mu = BODIES['sun']['mu']

        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([0.0, 1.5, 0.0])
        tof = 100.0  # days

        sol_pro = lambert_universal(r1, r2, tof, mu, prograde=True)
        sol_ret = lambert_universal(r1, r2, tof, mu, prograde=False)

        assert sol_pro is not None
        assert sol_ret is not None
        # Velocities should be different
        assert not np.allclose(sol_pro.v1, sol_ret.v1)


class TestLambertDeltaV:
    """Test delta-V calculations."""

    def test_deltav_with_initial_velocity(self):
        """Test delta-V when initial velocity is provided."""
        mu = BODIES['sun']['mu']

        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([0.0, 1.5, 0.0])
        tof = 150.0

        # Earth's circular velocity at 1 AU
        v_circ = np.sqrt(mu / 1.0)
        v1_initial = np.array([0.0, v_circ, 0.0])

        solution = lambert_solve(r1, r2, tof, mu, v1_initial=v1_initial)

        assert solution is not None
        assert solution.dv1 > 0  # Should require some delta-V

    def test_deltav_symmetric(self):
        """Delta-V for symmetric transfer should be similar at both ends."""
        mu = BODIES['sun']['mu']

        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([-1.0, 0.0, 0.0])  # 180° transfer

        # For 180° transfer at same radius, a = r
        tof = np.pi * np.sqrt(1.0**3 / mu)

        solution = lambert_solve(r1, r2, tof, mu)

        assert solution is not None
        # For this symmetric case, velocities should have same magnitude
        assert np.isclose(np.linalg.norm(solution.v1),
                         np.linalg.norm(solution.v2), rtol=0.05)


class TestLambertEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_tof_fails(self):
        """Very short TOF should fail to converge."""
        mu = BODIES['sun']['mu']

        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([1.5, 0.0, 0.0])
        tof = 0.001  # Way too short

        solution = lambert_universal(r1, r2, tof, mu)

        # Should return None or not converge
        assert solution is None or not solution.converged

    def test_same_position_fails(self):
        """Same start and end position is degenerate."""
        mu = BODIES['sun']['mu']

        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([1.0, 0.0, 0.0])  # Same position
        tof = 365.0

        # This is a degenerate case (cos_dnu = 1)
        # The solver may fail or return unusual results
        solution = lambert_universal(r1, r2, tof, mu)
        # Just check it doesn't crash - result may be None or invalid


class TestLambertMultirev:
    """Test multi-revolution solutions."""

    def test_multirev_returns_list(self):
        """lambert_multirev should return a list."""
        mu = BODIES['sun']['mu']

        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([0.0, 1.5, 0.0])
        tof = 200.0

        from guidance.lambert import lambert_multirev
        solutions = lambert_multirev(r1, r2, tof, mu)

        assert isinstance(solutions, list)
        assert len(solutions) >= 1  # At least direct transfer


class TestLambertPhysical:
    """Physical validation tests."""

    def test_energy_conservation(self):
        """Verify energy is consistent along transfer."""
        mu = BODIES['sun']['mu']

        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([0.0, 1.524, 0.0])
        tof = 200.0

        solution = lambert_universal(r1, r2, tof, mu)

        assert solution is not None

        # Energy at r1 with v1
        r1_mag = np.linalg.norm(r1)
        v1_mag = np.linalg.norm(solution.v1)
        E1 = 0.5 * v1_mag**2 - mu / r1_mag

        # Energy at r2 with v2
        r2_mag = np.linalg.norm(r2)
        v2_mag = np.linalg.norm(solution.v2)
        E2 = 0.5 * v2_mag**2 - mu / r2_mag

        # Energy should be conserved
        assert np.isclose(E1, E2, rtol=1e-6)

    def test_angular_momentum_direction(self):
        """Angular momentum direction should be consistent."""
        mu = BODIES['sun']['mu']

        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([0.0, 1.5, 0.1])  # Slight inclination
        tof = 150.0

        solution = lambert_universal(r1, r2, tof, mu, prograde=True)

        assert solution is not None

        # Angular momentum at both points
        h1 = np.cross(r1, solution.v1)
        h2 = np.cross(r2, solution.v2)

        # Should be parallel (same orbit)
        h1_norm = h1 / np.linalg.norm(h1)
        h2_norm = h2 / np.linalg.norm(h2)

        assert np.allclose(h1_norm, h2_norm, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
