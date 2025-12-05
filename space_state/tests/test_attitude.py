# tests/test_attitude.py
"""
Comprehensive Test Suite for Attitude Control System

Tests cover:
1. Quaternion mathematics
2. Reaction wheel allocation
3. RCS thruster allocation
4. Full attitude control loop
5. Fuel consumption tracking
6. Control stability
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from base.star_attitude import (
    quaternion_normalize,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_dcm,
    quaternion_error,
    compute_desired_quaternion,
    quaternion_derivative,
    euler_rotation_equation,
    attitude_pd_control,
    allocate_wheels,
    allocate_rcs,
    allocate_control,
    attitude_control,
    WHEEL_SPIN_AXES,
    RCS_TORQUE_MATRIX,
    RCS_GEOMETRY,
)
from base.star_ship import starship_raptor_vac, Spacecraft, Thruster
from base.star_constants import IDX_Q, IDX_OMEGA, RW_MAX_MOMENTUM, RW_MAX_TORQUE, RCS_THRUST
from base.star_logger import logger


class TestQuaternionMath:
    """Test quaternion mathematics utilities."""

    def test_quaternion_normalize(self):
        """Test quaternion normalization."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_norm = quaternion_normalize(q)
        assert_allclose(np.linalg.norm(q_norm), 1.0, rtol=1e-10)

    def test_quaternion_normalize_identity(self):
        """Test identity quaternion normalization."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        q_norm = quaternion_normalize(q)
        assert_allclose(q_norm, q, rtol=1e-10)

    def test_quaternion_normalize_near_zero(self):
        """Test normalization of near-zero quaternion returns identity."""
        q = np.array([1e-15, 0.0, 0.0, 0.0])
        q_norm = quaternion_normalize(q)
        assert_allclose(q_norm, [1.0, 0.0, 0.0, 0.0], rtol=1e-10)

    def test_quaternion_conjugate(self):
        """Test quaternion conjugate."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_conj = quaternion_conjugate(q)
        assert_allclose(q_conj, [0.5, -0.5, -0.5, -0.5], rtol=1e-10)

    def test_quaternion_multiply_identity(self):
        """Test multiplication with identity quaternion."""
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.5, 0.5, 0.5, 0.5])
        result = quaternion_multiply(q_id, q)
        assert_allclose(result, q, rtol=1e-10)

    def test_quaternion_multiply_inverse(self):
        """Test q * q^(-1) = identity."""
        q = quaternion_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        q_conj = quaternion_conjugate(q)
        result = quaternion_multiply(q, q_conj)
        assert_allclose(result, [1.0, 0.0, 0.0, 0.0], rtol=1e-10, atol=1e-14)

    def test_quaternion_to_dcm_identity(self):
        """Test identity quaternion gives identity rotation matrix."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        dcm = quaternion_to_dcm(q)
        assert_allclose(dcm, np.eye(3), rtol=1e-10, atol=1e-14)

    def test_quaternion_to_dcm_orthogonality(self):
        """Test DCM is orthogonal (R^T @ R = I)."""
        q = quaternion_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        dcm = quaternion_to_dcm(q)
        assert_allclose(dcm.T @ dcm, np.eye(3), rtol=1e-10, atol=1e-14)
        assert_allclose(np.linalg.det(dcm), 1.0, rtol=1e-10, atol=1e-14)

    def test_quaternion_error_zero(self):
        """Test error between identical quaternions is identity."""
        q = quaternion_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        q_err = quaternion_error(q, q)
        assert_allclose(q_err[0], 1.0, rtol=1e-6)  # Scalar part ≈ 1

    def test_compute_desired_quaternion_z_axis(self):
        """Test quaternion for +Z direction is identity."""
        thrust_dir = np.array([0.0, 0.0, 1.0])
        q = compute_desired_quaternion(thrust_dir)
        dcm = quaternion_to_dcm(q)
        body_z = dcm @ np.array([0.0, 0.0, 1.0])
        assert_allclose(body_z, thrust_dir, rtol=1e-10)

    def test_compute_desired_quaternion_arbitrary(self):
        """Test quaternion aligns body +Z with arbitrary direction."""
        thrust_dir = np.array([1.0, 1.0, 1.0])
        thrust_dir /= np.linalg.norm(thrust_dir)
        q = compute_desired_quaternion(thrust_dir)
        dcm = quaternion_to_dcm(q)
        body_z = dcm @ np.array([0.0, 0.0, 1.0])
        assert_allclose(body_z, thrust_dir, rtol=1e-10)


class TestQuaternionDynamics:
    """Test quaternion kinematics and dynamics."""

    def test_quaternion_derivative_zero_omega(self):
        """Test q_dot = 0 when omega = 0."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.zeros(3)
        q_dot = quaternion_derivative(q, omega)
        assert_allclose(q_dot, np.zeros(4), rtol=1e-10)

    def test_quaternion_derivative_orthogonal(self):
        """Test q_dot is orthogonal to q (q^T @ q_dot = 0)."""
        q = quaternion_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        omega = np.array([0.1, 0.2, 0.3])  # rad/day
        q_dot = quaternion_derivative(q, omega)
        dot_product = np.dot(q, q_dot)
        assert_allclose(dot_product, 0.0, atol=1e-10)

    def test_euler_rotation_zero_omega(self):
        """Test omega_dot with no rotation and no torque."""
        omega = np.zeros(3)
        I = np.diag([500_000, 500_000, 100_000])
        tau = np.zeros(3)
        omega_dot = euler_rotation_equation(omega, I, tau)
        assert_allclose(omega_dot, np.zeros(3), rtol=1e-10)

    def test_euler_rotation_with_torque(self):
        """Test omega_dot responds to applied torque."""
        omega = np.zeros(3)
        I = np.diag([500_000, 500_000, 100_000])
        tau = np.array([1000.0, 0.0, 0.0])  # 1000 N·m about X axis
        omega_dot = euler_rotation_equation(omega, I, tau)
        # omega_dot should be positive about X axis
        assert omega_dot[0] > 0


class TestReactionWheels:
    """Test reaction wheel allocation."""

    def test_wheel_axes_normalized(self):
        """Test wheel spin axes are unit vectors."""
        for i in range(4):
            axis = WHEEL_SPIN_AXES[i]
            assert_allclose(np.linalg.norm(axis), 1.0, rtol=1e-10)

    def test_wheel_axes_tetrahedral(self):
        """Test wheel axes have tetrahedral symmetry (equal angles)."""
        # Each pair should have the same dot product magnitude
        dot_products = []
        for i in range(4):
            for j in range(i + 1, 4):
                dot = np.dot(WHEEL_SPIN_AXES[i], WHEEL_SPIN_AXES[j])
                dot_products.append(abs(dot))
        # All should be equal (approximately -1/3 for tetrahedral)
        assert_allclose(dot_products, dot_products[0] * np.ones(len(dot_products)), rtol=1e-6)

    def test_allocate_wheels_zero_torque(self):
        """Test zero torque gives zero wheel commands."""
        tau = np.zeros(3)
        wheel_momentum = np.zeros(4)
        wheel_torques = allocate_wheels(tau, wheel_momentum)
        assert_allclose(wheel_torques, np.zeros(4), rtol=1e-10)

    def test_allocate_wheels_single_axis(self):
        """Test wheel allocation produces correct torque direction."""
        tau_desired = np.array([100.0, 0.0, 0.0])  # X-axis torque
        wheel_momentum = np.zeros(4)
        wheel_torques = allocate_wheels(tau_desired, wheel_momentum)

        # Compute resulting torque
        A = WHEEL_SPIN_AXES.T
        tau_actual = A @ wheel_torques
        # Direction should match
        tau_dir = tau_actual / np.linalg.norm(tau_actual)
        tau_desired_dir = tau_desired / np.linalg.norm(tau_desired)
        assert_allclose(tau_dir, tau_desired_dir, rtol=0.1)

    def test_allocate_wheels_saturation(self):
        """Test wheel torque saturation."""
        tau_desired = np.array([1e6, 0.0, 0.0])  # Very large torque
        wheel_momentum = np.zeros(4)
        wheel_torques = allocate_wheels(tau_desired, wheel_momentum)

        # All torques should be within limits
        assert np.all(np.abs(wheel_torques) <= RW_MAX_TORQUE)

    def test_allocate_wheels_momentum_saturation(self):
        """Test wheel torque reduced when momentum saturated."""
        tau_desired = np.array([100.0, 0.0, 0.0])
        # Nearly saturated wheels
        wheel_momentum = np.array([0.96, 0.96, 0.96, 0.96]) * RW_MAX_MOMENTUM
        wheel_torques = allocate_wheels(tau_desired, wheel_momentum)

        # Torques should be reduced
        assert np.all(np.abs(wheel_torques) < RW_MAX_TORQUE)


class TestRCSThruster:
    """Test RCS thruster allocation."""

    def test_rcs_geometry_size(self):
        """Test RCS geometry has 16 thrusters."""
        assert RCS_GEOMETRY.shape == (16, 6)

    def test_rcs_torque_matrix_shape(self):
        """Test RCS torque matrix shape."""
        assert RCS_TORQUE_MATRIX.shape == (3, 16)

    def test_rcs_torque_matrix_nonzero(self):
        """Test RCS thrusters can produce torque on all axes."""
        # Each axis should have thrusters that can produce positive and negative torque
        for axis in range(3):
            positive_torque = np.any(RCS_TORQUE_MATRIX[axis, :] > 0)
            negative_torque = np.any(RCS_TORQUE_MATRIX[axis, :] < 0)
            assert positive_torque, f"No positive torque capability on axis {axis}"
            assert negative_torque, f"No negative torque capability on axis {axis}"

    def test_allocate_rcs_zero_torque(self):
        """Test zero torque (within deadband) gives no thruster firings."""
        tau = np.array([0.1, 0.1, 0.1])  # Within default 0.5 N·m deadband
        result = allocate_rcs(tau)
        assert not np.any(result['thrusters'])
        assert result['mdot'] == 0.0

    def test_allocate_rcs_positive_x(self):
        """Test positive X torque fires correct thrusters."""
        tau = np.array([1000.0, 0.0, 0.0])  # Large positive X torque
        result = allocate_rcs(tau)

        # Some thrusters should be on
        assert np.sum(result['thrusters']) > 0

        # Resulting torque should be positive X
        assert result['torque_actual'][0] > 0

        # Mass flow should be positive
        assert result['mdot'] > 0

    def test_allocate_rcs_mass_flow(self):
        """Test RCS mass flow rate calculation."""
        tau = np.array([10000.0, 0.0, 0.0])  # Large torque
        result = allocate_rcs(tau)

        if result['mdot'] > 0:
            # Verify mass flow is reasonable
            # mdot = F / (Isp * g0), for 2 thrusters at 1200N each, Isp=290s
            expected_mdot_per_thruster = RCS_THRUST / (290.0 * 9.80665)
            assert result['mdot'] > 0
            assert result['mdot'] < 16 * expected_mdot_per_thruster  # Max all thrusters


class TestControlAllocation:
    """Test combined control allocation logic."""

    def test_allocate_control_uses_wheels_nominal(self):
        """Test wheels used for small torques with unsaturated momentum."""
        tau = np.array([50.0, 0.0, 0.0])  # Small torque
        wheel_momentum = np.zeros(4)
        result = allocate_control(tau, wheel_momentum)
        assert result['mode'] == 'wheels'

    def test_allocate_control_uses_wheels_large_torque_unsaturated(self):
        """Test wheels used for large torques when not saturated.

        New logic: RCS requires BOTH saturation AND high torque to avoid
        over-aggressive thruster firing. Large torque alone uses wheels.
        """
        tau = np.array([2000.0, 0.0, 0.0])  # Large torque
        wheel_momentum = np.zeros(4)  # Not saturated
        result = allocate_control(tau, wheel_momentum)
        assert result['mode'] == 'wheels'  # Wheels handle it when not saturated

    def test_allocate_control_uses_rcs_saturated_and_high_torque(self):
        """Test RCS used when wheels are saturated AND torque is high.

        New logic: RCS requires BOTH conditions:
        1. Wheels >90% saturated
        2. Torque demand >1500 N·m
        """
        tau = np.array([2000.0, 0.0, 0.0])  # High torque (>1500 N·m)
        wheel_momentum = np.ones(4) * 0.95 * RW_MAX_MOMENTUM  # 95% saturated
        result = allocate_control(tau, wheel_momentum)
        assert result['mode'] == 'rcs'


class TestPDController:
    """Test PD attitude controller."""

    def test_pd_control_zero_error(self):
        """Test zero error gives zero (or near-zero) torque."""
        q_err = np.array([1.0, 0.0, 0.0, 0.0])  # Identity = no error
        omega = np.zeros(3)
        tau = attitude_pd_control(q_err, omega)
        assert_allclose(tau, np.zeros(3), atol=1e-6)

    def test_pd_control_proportional(self):
        """Test proportional response to attitude error."""
        # Create small rotation error about X axis
        angle = np.radians(10)  # 10 degrees
        q_err = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        omega = np.zeros(3)

        tau = attitude_pd_control(q_err, omega, Kp=100.0, Kd=0.0)

        # Torque should be negative X (restoring)
        assert tau[0] < 0
        assert abs(tau[1]) < abs(tau[0])
        assert abs(tau[2]) < abs(tau[0])

    def test_pd_control_derivative(self):
        """Test derivative response to angular velocity."""
        q_err = np.array([1.0, 0.0, 0.0, 0.0])  # No attitude error
        omega = np.array([86400.0, 0.0, 0.0])  # 1 rad/day = 1 rad/s * 86400

        tau = attitude_pd_control(q_err, omega, Kp=0.0, Kd=100.0)

        # Torque should oppose rotation (negative X)
        assert tau[0] < 0


class TestSpacecraftIntegration:
    """Test integration with Spacecraft class."""

    def test_starship_has_inertia(self):
        """Test Starship has proper inertia tensor."""
        ship = starship_raptor_vac()
        assert ship.inertia.shape == (3, 3)
        assert np.allclose(ship.inertia, ship.inertia.T)  # Symmetric
        assert np.all(np.diag(ship.inertia) > 0)  # Positive diagonal

    def test_starship_has_wheel_momentum(self):
        """Test Starship has wheel momentum array."""
        ship = starship_raptor_vac()
        assert ship.wheel_momentum.shape == (4,)

    def test_starship_has_rcs_propellant(self):
        """Test Starship has RCS propellant."""
        ship = starship_raptor_vac()
        assert ship.rcs_propellant_mass > 0

    def test_rcs_propellant_consumption(self):
        """Test RCS propellant consumption tracking."""
        ship = starship_raptor_vac()
        initial_rcs = ship.rcs_propellant_mass

        # Consume some propellant
        consumed = ship.consume_rcs_propellant(100.0)

        assert consumed == 100.0
        assert ship.rcs_propellant_mass == initial_rcs - 100.0

    def test_rcs_propellant_depletion(self):
        """Test RCS propellant depletion handling."""
        ship = starship_raptor_vac()

        # Try to consume more than available
        initial_rcs = ship.rcs_propellant_mass
        consumed = ship.consume_rcs_propellant(initial_rcs + 1000.0)

        assert consumed == initial_rcs
        assert ship.rcs_propellant_mass == 0.0

    def test_total_mass_includes_rcs(self):
        """Test total mass includes RCS propellant."""
        ship = starship_raptor_vac()
        expected = ship.dry_mass + ship.propellant_mass + ship.rcs_propellant_mass
        assert ship.mass == expected


class TestAttitudeControlLoop:
    """Test full attitude control function."""

    def test_attitude_control_returns_tuple(self):
        """Test attitude_control returns (torque, dm_rcs)."""
        ship = starship_raptor_vac()

        # Create mock guidance
        class MockGuidance:
            def update(self, t, y):
                return None, np.array([0.0, 0.0, 1.0]), 1.0  # Thrust along +Z

        guidance = MockGuidance()

        # Create state vector
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega0 = np.zeros(3)
        y = np.concatenate([np.zeros(3), np.zeros(3), [ship.mass], q0, omega0])

        tau, dm_rcs = attitude_control(0.0, y, ship, guidance)

        assert tau.shape == (3,)
        assert isinstance(dm_rcs, float)

    def test_attitude_control_no_ship_returns_zero(self):
        """Test attitude_control with no ship returns zeros."""
        class MockGuidance:
            def update(self, t, y):
                return None, np.array([0.0, 0.0, 1.0]), 1.0

        y = np.zeros(14)
        y[7] = 1.0  # Identity quaternion

        tau, dm_rcs = attitude_control(0.0, y, None, MockGuidance())

        assert_allclose(tau, np.zeros(3))
        assert dm_rcs == 0.0


class TestAttitudePropagation:
    """Test attitude behavior during propagation."""

    def test_coast_mode_preserves_attitude(self):
        """Test that coast mode maintains attitude with rate damping only."""
        from base.star_propagator import Propagator
        from base.star_states import get_circular_state
        from base.star_constants import BODIES
        from guidance import BurnController

        ship = starship_raptor_vac()
        body = BODIES['moon']

        r0, v0, q0, omega0 = get_circular_state(body, 'lunar_50km')
        y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

        guid = BurnController(ship, body)
        prop = Propagator(ship, guid)
        prop.result.initialize(y0, 0.0)

        # Propagate for 0.01 days (coast mode - no burn scheduled)
        prop.propagate(0.01)

        # Check quaternion norm preserved
        q_final = prop.q
        q_norm = np.linalg.norm(q_final)
        assert 0.999 < q_norm < 1.001, f"Quaternion norm drifted: {q_norm}"

        # Check angular velocity is low (coast damping)
        omega_final = prop.omega
        omega_mag = np.linalg.norm(omega_final)
        assert omega_mag < 100.0, f"Angular velocity too high: {omega_mag} rad/day"

    def test_quaternion_norm_stability(self):
        """Test quaternion norm stays near 1.0 during propagation."""
        from base.star_propagator import Propagator
        from base.star_states import get_circular_state
        from base.star_constants import BODIES
        from guidance import BurnController

        ship = starship_raptor_vac()
        body = BODIES['moon']

        r0, v0, q0, omega0 = get_circular_state(body, 'lunar_50km')
        y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

        guid = BurnController(ship, body)
        prop = Propagator(ship, guid)
        prop.result.initialize(y0, 0.0)

        # Propagate for 0.05 days
        prop.propagate(0.05)

        # Check all quaternion norms in history
        q_history = prop.result.y[IDX_Q, :]
        q_norms = np.linalg.norm(q_history, axis=0)

        assert np.all(q_norms > 0.99), f"Min quaternion norm: {q_norms.min()}"
        assert np.all(q_norms < 1.01), f"Max quaternion norm: {q_norms.max()}"

    def test_rcs_not_consumed_during_coast(self):
        """Test that RCS propellant is not consumed during coast."""
        from base.star_propagator import Propagator
        from base.star_states import get_circular_state
        from base.star_constants import BODIES
        from guidance import BurnController

        ship = starship_raptor_vac()
        body = BODIES['moon']
        initial_rcs = ship.rcs_propellant_mass

        r0, v0, q0, omega0 = get_circular_state(body, 'lunar_50km')
        y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

        guid = BurnController(ship, body)
        prop = Propagator(ship, guid)
        prop.result.initialize(y0, 0.0)

        # Propagate for coast phase
        prop.propagate(0.01)

        # RCS should be unchanged
        assert ship.rcs_propellant_mass == initial_rcs, \
            f"RCS consumed during coast: {initial_rcs - ship.rcs_propellant_mass} kg"


def test_attitude_visualization():
    """Test that attitude data can be extracted from propagation results."""
    from base.star_propagator import Propagator
    from base.star_states import get_circular_state
    from base.star_constants import BODIES
    from guidance import BurnController

    ship = starship_raptor_vac()
    body = BODIES['moon']

    r0, v0, q0, omega0 = get_circular_state(body, 'lunar_50km')
    y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

    guid = BurnController(ship, body)
    prop = Propagator(ship, guid)
    prop.result.initialize(y0, 0.0)

    # Short propagation
    prop.propagate(0.001)

    # Extract attitude data
    q_history = prop.result.y[IDX_Q, :]
    omega_history = prop.result.y[IDX_OMEGA, :]

    assert q_history.shape[0] == 4
    assert omega_history.shape[0] == 3
    assert q_history.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
