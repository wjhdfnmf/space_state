# base/star_attitude.py
"""
Attitude Control Module for Spacecraft Dynamics

This module implements:
- Quaternion mathematics and kinematics
- Euler's rotation equations
- PD attitude controller
- Reaction wheel and RCS thruster allocation
- Main attitude control interface
"""

import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R

from .star_logger import logger
from .star_constants import IDX_Q, IDX_OMEGA, RW_MAX_MOMENTUM, RW_MAX_TORQUE, RCS_THRUST, RCS_ISP
from .star_config import CONFIG
from .star_utils import day_to_sec, sec_to_days, kgps_kgpd


# ==================== QUATERNION MATHEMATICS ====================

def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        logger.warning("Quaternion near zero, resetting to identity")
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion conjugate.

    Args:
        q: [q0, q1, q2, q3] scalar-first

    Returns:
        q* = [q0, -q1, -q2, -q3]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product: q1 ⊗ q2

    Args:
        q1, q2: [q0, q1, q2, q3] scalar-first

    Returns:
        q1 ⊗ q2 (scalar-first)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to direction cosine matrix (rotation matrix).

    Args:
        q: [q0, q1, q2, q3] scalar-first

    Returns:
        R: 3×3 rotation matrix (body → inertial frame)
    """
    q0, q1, q2, q3 = quaternion_normalize(q)

    return np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])


def quaternion_error(q_desired: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """
    Compute quaternion error: q_err = q_desired ⊗ q_current*

    Args:
        q_desired: Desired quaternion
        q_current: Current quaternion

    Returns:
        q_err: Error quaternion (shortest rotation)
    """
    q_err = quaternion_multiply(q_desired, quaternion_conjugate(q_current))

    # Ensure shortest rotation (q and -q represent same rotation)
    if q_err[0] < 0:
        q_err = -q_err

    q_err = quaternion_normalize(q_err)

    # Handle 180° singularity: when q_err[0] ≈ 0 (angle ≈ 180°),
    # the rotation axis becomes ambiguous. Add a substantial perturbation
    # to break the symmetry and give the controller a preferred direction.
    # The perturbation scales with how close we are to 180° for smooth transition.
    if abs(q_err[0]) < 0.15:  # Within ~163-180° of target
        # Perturbation strength scales from 0 at 163° to full at 180°
        proximity_to_180 = (0.15 - abs(q_err[0])) / 0.15  # 0 to 1
        base_perturbation = 0.15 * proximity_to_180  # Up to 0.15 perturbation

        # Add perturbation in body Z-axis direction (roll about thrust axis)
        # This is generally benign for thrust-pointing maneuvers
        # Pick a consistent direction to avoid oscillating between ±180°
        q_err[3] += base_perturbation  # Perturb q3 (z-component)
        q_err = quaternion_normalize(q_err)

    return q_err


def compute_desired_quaternion(thrust_dir_inertial: np.ndarray) -> np.ndarray:
    """
    Compute quaternion that aligns spacecraft body +Z axis with desired thrust direction.

    Args:
        thrust_dir_inertial: Desired thrust direction (unit vector in inertial frame)

    Returns:
        q: [q0, q1, q2, q3] quaternion (scalar-first)
    """
    # Normalize direction
    z_new = thrust_dir_inertial / np.linalg.norm(thrust_dir_inertial)

    # Choose perpendicular x-axis (avoid singularity when z_new is parallel to [0,0,1])
    if abs(z_new[2]) < 0.9:
        x_new = np.cross([0, 0, 1], z_new)
    else:
        x_new = np.cross([1, 0, 0], z_new)
    x_new /= np.linalg.norm(x_new)

    # Complete right-handed frame
    y_new = np.cross(z_new, x_new)

    # Build direction cosine matrix [x, y, z] columns
    dcm = np.column_stack([x_new, y_new, z_new])

    # Convert to quaternion using scipy (returns [x,y,z,w])
    q_scipy = R.from_matrix(dcm).as_quat()

    # Convert to scalar-first [w,x,y,z]
    return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])


# ==================== ROTATIONAL DYNAMICS ====================

def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Quaternion kinematic differential equation: q̇ = 0.5 * Ω(ω) * q

    Args:
        q: [q0, q1, q2, q3] scalar-first
        omega: [ωx, ωy, ωz] angular velocity in rad/day (body frame)

    Returns:
        q_dot: Time derivative of quaternion
    """
    q0, q1, q2, q3 = q
    wx, wy, wz = omega

    return 0.5 * np.array([
        -wx*q1 - wy*q2 - wz*q3,
         wx*q0 + wz*q2 - wy*q3,
         wy*q0 - wz*q1 + wx*q3,
         wz*q0 + wy*q1 - wx*q2
    ])


def euler_rotation_equation(omega: np.ndarray, I: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    Euler's rotation equations: I·ω̇ + ω×(I·ω) = τ

    Args:
        omega: [ωx, ωy, ωz] angular velocity (rad/day) in body frame
        I: 3×3 inertia tensor (kg·m²) in body frame
        tau: [τx, τy, τz] applied torque (N·m) in body frame

    Returns:
        omega_dot: Angular acceleration (rad/day²)
    """
    # Convert torque units from N·m to match angular units
    # τ is in N·m, ω is in rad/day
    # We need consistent units for integration

    I_omega = I @ omega  # kg·m² · rad/day
    gyroscopic_torque = np.cross(omega, I_omega)  # (rad/day) × (kg·m²·rad/day)

    # Solve for omega_dot: ω̇ = I^(-1) · (τ - ω×(I·ω))
    # Convert tau from N·m to (kg·m²·rad/day²) for consistency
    # N·m = kg·m²/s², and 1 day = 86400 s
    # So tau in (kg·m²·rad/day²) = tau_Nm * (86400)²
    tau_consistent = tau * (86400.0 ** 2)  # Convert N·m to kg·m²·rad/day²

    omega_dot = np.linalg.solve(I, tau_consistent - gyroscopic_torque)

    return omega_dot


# ==================== ATTITUDE CONTROL ====================

def attitude_pd_control(q_err: np.ndarray, omega: np.ndarray,
                       omega_desired: np.ndarray = None,
                       Kp: float = 50_000.0,
                       Kd: float = 200_000.0) -> np.ndarray:
    """
    PD control law on quaternion error.

    Args:
        q_err: Error quaternion [q0, q1, q2, q3]
        omega: Current angular velocity (rad/day)
        omega_desired: Desired angular velocity (rad/day), defaults to zero
        Kp: Proportional gain (N·m/rad)
        Kd: Derivative gain (N·m·s/rad)

    Returns:
        tau: Control torque (N·m) in body frame
    """
    if omega_desired is None:
        omega_desired = np.zeros(3)

    # Extract axis-angle from quaternion error
    q0, q1, q2, q3 = q_err
    angle = 2 * np.arccos(np.clip(q0, -1, 1))

    if abs(angle) < 1e-6:
        axis_error = np.zeros(3)
    else:
        # Error axis (unit vector)
        sin_half_angle = np.sin(angle / 2)
        if abs(sin_half_angle) < 1e-10:
            axis_error = np.zeros(3)
        else:
            axis_error = np.array([q1, q2, q3]) / sin_half_angle

    # Convert omega from rad/day to rad/s for damping term
    omega_rad_s = omega / 86400.0
    omega_desired_rad_s = omega_desired / 86400.0

    # PD control torque
    tau = -Kp * axis_error * angle - Kd * (omega_rad_s - omega_desired_rad_s)

    return tau


# ==================== ACTUATOR ALLOCATION ====================

# Reaction wheel configuration (4-wheel tetrahedral pyramid)
WHEEL_SPIN_AXES = np.array([
    [+1, +1, +1],  # Wheel 1
    [+1, -1, -1],  # Wheel 2
    [-1, +1, -1],  # Wheel 3
    [-1, -1, +1],  # Wheel 4
]) / np.sqrt(3)  # Normalized (54.74° cone angle)


def allocate_wheels(tau_desired: np.ndarray, wheel_momentum: np.ndarray) -> np.ndarray:
    """
    Allocate torque to 4 reaction wheels using pseudo-inverse.

    Args:
        tau_desired: [τx, τy, τz] desired torque (N·m)
        wheel_momentum: [h1, h2, h3, h4] current wheel momentum (N·m·s)

    Returns:
        wheel_torques: [τ1, τ2, τ3, τ4] commanded wheel torques (N·m)
    """
    # Mapping matrix: τ = A · τ_wheels, where A is (3, 4)
    A = WHEEL_SPIN_AXES.T  # Shape: (3, 4)

    # Pseudo-inverse allocation (minimum norm solution)
    wheel_torques = np.linalg.pinv(A) @ tau_desired  # Shape: (4,)

    # Saturate at max torque
    wheel_torques = np.clip(wheel_torques, -RW_MAX_TORQUE, RW_MAX_TORQUE)

    # Check for momentum saturation
    for i in range(4):
        if abs(wheel_momentum[i]) > 0.95 * RW_MAX_MOMENTUM:
            # Reduce torque on saturated wheels
            wheel_torques[i] *= 0.1

    return wheel_torques


"""
RCS Thruster Configuration (16 thrusters in 4 quads)

Starship RCS layout (simplified model):
- 4 thruster quads: Forward-Port, Forward-Starboard, Aft-Port, Aft-Starboard
- Each quad has 4 thrusters: +X, -X, +Y, -Y directions (body frame)
- Moment arm from CG to thruster location

Quad positions (body frame, meters from CG):
  Forward quads: z = +20m (toward nose)
  Aft quads: z = -20m (toward engines)
  Port quads: y = +4m
  Starboard quads: y = -4m
"""

# RCS thruster geometry (16 thrusters)
# Each row: [position_x, position_y, position_z, thrust_dir_x, thrust_dir_y, thrust_dir_z]
# Positions in meters from CG, thrust directions are unit vectors
RCS_GEOMETRY = np.array([
    # Forward-Port quad (thrusters 0-3)
    [0.0, +4.0, +20.0, +1.0, 0.0, 0.0],  # +X thrust
    [0.0, +4.0, +20.0, -1.0, 0.0, 0.0],  # -X thrust
    [0.0, +4.0, +20.0, 0.0, +1.0, 0.0],  # +Y thrust
    [0.0, +4.0, +20.0, 0.0, -1.0, 0.0],  # -Y thrust

    # Forward-Starboard quad (thrusters 4-7)
    [0.0, -4.0, +20.0, +1.0, 0.0, 0.0],  # +X thrust
    [0.0, -4.0, +20.0, -1.0, 0.0, 0.0],  # -X thrust
    [0.0, -4.0, +20.0, 0.0, +1.0, 0.0],  # +Y thrust
    [0.0, -4.0, +20.0, 0.0, -1.0, 0.0],  # -Y thrust

    # Aft-Port quad (thrusters 8-11)
    [0.0, +4.0, -20.0, +1.0, 0.0, 0.0],  # +X thrust
    [0.0, +4.0, -20.0, -1.0, 0.0, 0.0],  # -X thrust
    [0.0, +4.0, -20.0, 0.0, +1.0, 0.0],  # +Y thrust
    [0.0, +4.0, -20.0, 0.0, -1.0, 0.0],  # -Y thrust

    # Aft-Starboard quad (thrusters 12-15)
    [0.0, -4.0, -20.0, +1.0, 0.0, 0.0],  # +X thrust
    [0.0, -4.0, -20.0, -1.0, 0.0, 0.0],  # -X thrust
    [0.0, -4.0, -20.0, 0.0, +1.0, 0.0],  # +Y thrust
    [0.0, -4.0, -20.0, 0.0, -1.0, 0.0],  # -Y thrust
])

# Standard gravity for Isp calculation
G0 = 9.80665  # m/s²


def compute_rcs_torque_matrix() -> np.ndarray:
    """
    Compute the torque influence matrix for all 16 RCS thrusters.

    Returns:
        B: (3, 16) matrix where B @ thrust_vector = torque
           Each column is the torque produced by unit thrust from that thruster
    """
    B = np.zeros((3, 16))

    for i in range(16):
        position = RCS_GEOMETRY[i, 0:3]  # [x, y, z] meters
        thrust_dir = RCS_GEOMETRY[i, 3:6]  # [fx, fy, fz] unit vector

        # Torque = r × F = position × (thrust_dir * thrust_magnitude)
        # For unit thrust: τ = r × thrust_dir
        torque = np.cross(position, thrust_dir)
        B[:, i] = torque * RCS_THRUST  # Scale by actual thrust per thruster

    return B


# Precompute the torque matrix
RCS_TORQUE_MATRIX = compute_rcs_torque_matrix()


def allocate_rcs(tau_desired: np.ndarray, deadband: float = 50.0) -> dict:
    """
    Allocate torque to RCS thrusters using simplified bang-bang control.

    This uses a direct axis-by-axis selection strategy:
    - For +X torque: fire thrusters that produce positive X torque
    - For -X torque: fire thrusters that produce negative X torque
    - Same for Y and Z axes

    Args:
        tau_desired: [τx, τy, τz] desired torque (N·m)
        deadband: Minimum torque magnitude to activate thrusters (N·m)

    Returns:
        dict with 'thrusters' (16-element bool array), 'torque_actual', 'mdot'
    """
    thrusters_on = np.zeros(16, dtype=bool)
    torque_actual = np.zeros(3)

    # Check if any torque is needed (outside deadband)
    tau_mag = np.linalg.norm(tau_desired)
    if tau_mag < deadband:
        return {
            'thrusters': thrusters_on,
            'torque_actual': torque_actual,
            'mdot': 0.0
        }

    # For each axis, find thrusters that produce torque in the desired direction
    for axis in range(3):
        if abs(tau_desired[axis]) < deadband:
            continue

        # Find thrusters that produce torque in the right direction
        torque_contributions = RCS_TORQUE_MATRIX[axis, :]  # Torque per thruster on this axis

        if tau_desired[axis] > 0:
            # Need positive torque on this axis
            candidate_thrusters = torque_contributions > 0
        else:
            # Need negative torque on this axis
            candidate_thrusters = torque_contributions < 0

        # Select the 2 best thrusters (opposing corners for pure rotation)
        candidate_indices = np.where(candidate_thrusters)[0]
        if len(candidate_indices) > 0:
            # Sort by absolute torque contribution and take top 2
            abs_contributions = np.abs(torque_contributions[candidate_indices])
            sorted_indices = candidate_indices[np.argsort(abs_contributions)[::-1]]
            selected = sorted_indices[:2]  # Take top 2 thrusters
            thrusters_on[selected] = True

    # Compute actual torque from selected thrusters
    thrust_vector = thrusters_on.astype(float)  # 1.0 for on, 0.0 for off
    torque_actual = RCS_TORQUE_MATRIX @ thrust_vector

    # Compute mass flow rate
    num_thrusters_on = np.sum(thrusters_on)
    mdot_per_thruster = RCS_THRUST / (RCS_ISP * G0)  # kg/s per thruster
    mdot_total = num_thrusters_on * mdot_per_thruster  # kg/s

    # Log RCS activity (reduce spam - only log occasionally)
    # Use hash of torque to log only on changes
    global _last_rcs_log_torque
    if not hasattr(attitude_control, '_last_rcs_log_torque'):
        _last_rcs_log_torque = None
    torque_key = tuple(np.round(torque_actual, 0))
    if num_thrusters_on > 0 and torque_key != _last_rcs_log_torque:
        logger.debug(f"RCS: {num_thrusters_on} thrusters ON, τ_actual={torque_actual}, mdot={mdot_total:.4f} kg/s")
        _last_rcs_log_torque = torque_key

    return {
        'thrusters': thrusters_on,
        'torque_actual': torque_actual,
        'mdot': mdot_total  # kg/s
    }


def allocate_control(tau_desired: np.ndarray, wheel_momentum: np.ndarray,
                     error_angle_deg: float = 0.0) -> dict:
    """
    Decide whether to use reaction wheels or RCS based on saturation and attitude error.

    Args:
        tau_desired: [τx, τy, τz] desired torque (N·m)
        wheel_momentum: [h1, h2, h3, h4] current wheel momentum (N·m·s)
        error_angle_deg: Current attitude error in degrees (for RCS decision)

    Returns:
        dict with allocation results
    """
    # Check wheel saturation
    max_saturation = np.max(np.abs(wheel_momentum) / RW_MAX_MOMENTUM)
    tau_magnitude = np.linalg.norm(tau_desired)

    # Decision logic for when to use RCS:
    # Only use RCS in emergencies - it's too aggressive for fine control
    # RCS produces ~48,000 N·m per thruster config, wheels max at ~2000 N·m
    # RCS should only be used for very large torque demands (e.g., emergency spin recovery)
    use_rcs = (max_saturation > 0.90) and (tau_magnitude > 5000.0)

    if use_rcs:
        rcs_result = allocate_rcs(tau_desired)
        return {
            'mode': 'rcs',
            'torque': rcs_result['torque_actual'],
            'wheel_torques': np.zeros(4),
            'rcs_thrusters': rcs_result['thrusters'],
            'mdot': rcs_result['mdot']
        }
    else:
        wheel_torques = allocate_wheels(tau_desired, wheel_momentum)
        torque_actual = WHEEL_SPIN_AXES.T @ wheel_torques
        return {
            'mode': 'wheels',
            'torque': torque_actual,
            'wheel_torques': wheel_torques,
            'rcs_thrusters': np.zeros(16, dtype=bool),
            'mdot': 0.0
        }


# ==================== MAIN ATTITUDE CONTROL INTERFACE ====================

# Global counters for logging
_attitude_log_counter = 0
_last_mode = None
_mode_switch_count = 0


def attitude_control(t: float, y: np.ndarray, ship: object, guidance: object) -> Tuple[np.ndarray, float]:
    """
    Main attitude control function called by propagator.

    Computes control torque to point spacecraft toward desired thrust direction.
    Uses reaction wheels for fine control and RCS thrusters for large slews
    or when wheels are saturated.

    Args:
        t: Current time (days)
        y: Current state vector (14 elements)
        ship: Spacecraft object
        guidance: Guidance controller object

    Returns:
        tau_control: Control torque (N·m) in body frame
        dm_rcs: RCS propellant mass flow rate (kg/day, negative = consuming)
    """
    global _attitude_log_counter, _last_mode, _mode_switch_count

    if ship is None or guidance is None:
        return np.zeros(3), 0.0

    # Check if ship has attitude control capability
    if not hasattr(ship, 'inertia'):
        return np.zeros(3), 0.0

    # Get current attitude and angular velocity
    q_current = y[IDX_Q]
    omega_current = y[IDX_OMEGA]

    # Validate quaternion
    q_norm = np.linalg.norm(q_current)
    if not (0.999 < q_norm < 1.001):
        logger.warning(f"Quaternion norm drift: ||q|| = {q_norm:.6f} at t={t:.6f} days")

    # Get desired thrust direction from guidance
    try:
        _, thrust_dir_desired, throttle = guidance.update(t, y)
    except Exception as e:
        logger.debug(f"Guidance update failed at t={t:.6f}: {e}")
        return np.zeros(3), 0.0

    # Check if we're thrusting or should be tracking a specific direction (pre-slew)
    is_thrusting = throttle >= 0.01

    # Check if guidance is requesting attitude tracking even without thrust (e.g., pre-slew phase)
    # Default coast direction is [0,0,1] - any other direction means "track this"
    default_dir = np.array([0.0, 0.0, 1.0])
    is_tracking_direction = np.linalg.norm(thrust_dir_desired - default_dir) > 0.01

    # Attitude control modes:
    # 1. THRUST: throttle > 0 -> track direction and thrust
    # 2. PRE-SLEW: throttle = 0 but non-default direction -> track direction (no thrust)
    # 3. COAST: throttle = 0 and default direction -> rate damping only

    # COAST MODE: During coast (no thrust and no specific direction), rate damping only
    if not is_thrusting and not is_tracking_direction:
        # Pure rate damping - just prevent tumbling, don't track anything
        omega_rad_s = omega_current / 86400.0
        omega_magnitude_rad_s = np.linalg.norm(omega_rad_s)

        # Light damping torque proportional to angular velocity
        # Kd = 5000 N·m·s/rad for gentle stabilization
        tau_desired = -5_000.0 * omega_rad_s

        # For logging, report error as 0 (we're not tracking anything)
        error_angle_deg = 0.0
        q_err = np.array([1.0, 0.0, 0.0, 0.0])  # Identity (no error in coast mode)
    else:
        # THRUST MODE: Active pointing control to align with thrust direction
        # Compute desired quaternion (align body +Z with thrust direction)
        q_desired = compute_desired_quaternion(thrust_dir_desired)

        # Compute quaternion error
        q_err = quaternion_error(q_desired, q_current)

        # Extract error angle for logging
        error_angle_deg = 2.0 * np.arccos(np.clip(q_err[0], -1, 1)) * (180.0 / np.pi)

        # Angular velocity rate limiting to prevent runaway rotation
        MAX_OMEGA_RAD_DAY = 750.0  # ~0.5 deg/s max rotation rate
        omega_magnitude = np.linalg.norm(omega_current)
        omega_magnitude_rad_s = omega_magnitude / 86400.0

        if omega_magnitude > MAX_OMEGA_RAD_DAY * 1.5:
            # Emergency rate damping - only slow down, ignore position error
            logger.debug(f"Rate limiting active: ω={omega_magnitude:.0f} rad/day ({omega_magnitude_rad_s * 180/np.pi:.2f}°/s)")
            omega_rad_s = omega_current / 86400.0
            tau_desired = -50_000.0 * omega_rad_s
        elif omega_magnitude > MAX_OMEGA_RAD_DAY:
            # Blended mode: reduce position control, increase damping
            blend = (omega_magnitude - MAX_OMEGA_RAD_DAY) / (MAX_OMEGA_RAD_DAY * 0.5)
            blend = np.clip(blend, 0, 1)

            tau_pd = attitude_pd_control(q_err, omega_current, Kp=100.0, Kd=10_000.0)
            omega_rad_s = omega_current / 86400.0
            tau_damp = -50_000.0 * omega_rad_s
            tau_desired = (1 - blend) * tau_pd + blend * tau_damp
        else:
            # Normal PD control during thrust
            tau_desired = attitude_pd_control(q_err, omega_current, Kp=100.0, Kd=10_000.0)

    # Allocate to actuators (pass error angle for RCS decision)
    allocation = allocate_control(tau_desired, ship.wheel_momentum, error_angle_deg)

    # Track mode switches
    if _last_mode is not None and allocation['mode'] != _last_mode:
        _mode_switch_count += 1
        logger.debug(f"Mode switch: {_last_mode} -> {allocation['mode']} at t={t:.6f} days (switch #{_mode_switch_count})")
    _last_mode = allocation['mode']

    # NOTE: Wheel momentum is NOT updated here during dynamics function calls.
    # The ODE solver calls this function many times per timestep (for adaptive stepping),
    # so updating wheel_momentum here would cause incorrect accumulation.
    # Wheel momentum is updated by the propagator after each integration step.
    # Store the wheel torques for the propagator to use:
    if allocation['mode'] == 'wheels':
        ship._wheel_torques = allocation['wheel_torques']
    else:
        ship._wheel_torques = np.zeros(4)

    # Handle RCS fuel consumption
    # NOTE: RCS propellant is NOT consumed directly here - that caused a bug where
    # fuel was consumed on every ODE function evaluation (100s of times per second).
    # Instead, we return the mass flow rate and let the propagator handle it properly
    # after each integration step.
    dm_rcs_day = 0.0
    if allocation['mode'] == 'rcs' and allocation['mdot'] > 0:
        # Check if we have RCS propellant available
        if ship.rcs_propellant_mass <= 0:
            # No RCS available, fall back to wheels only
            logger.debug(f"RCS propellant depleted - using wheels only at t={t:.6f} days")
            allocation = allocate_control(tau_desired * 0.1, ship.wheel_momentum)  # Reduce demand for wheels
            allocation['mode'] = 'wheels_only'
        else:
            # Convert to mass flow rate in kg/day (negative = losing mass)
            dm_rcs_day = -kgps_kgpd(allocation['mdot'])
            # Store the RCS mass flow rate on ship for propagator to use
            ship._rcs_mdot_kgpd = -dm_rcs_day  # Positive = consuming

    # Enhanced logging every ~50 calls during burns/pre-slew, ~500 during coast
    _attitude_log_counter += 1
    is_active = is_thrusting or is_tracking_direction
    log_interval = 50 if is_active else 500

    if _attitude_log_counter % log_interval == 0:
        omega_deg_s = np.linalg.norm(omega_current) * (180.0 / np.pi) / 86400.0
        wheel_saturation = np.max(np.abs(ship.wheel_momentum) / RW_MAX_MOMENTUM) * 100.0

        # Determine phase label for logging
        if is_thrusting:
            phase = "BURN"
        elif is_tracking_direction:
            phase = "PRESLEW"
        else:
            phase = "COAST"

        logger.debug(
            f"Attitude: t={t:.5f}d | {phase:7s} | err={error_angle_deg:6.2f}° | ω={omega_deg_s:.4f}°/s | "
            f"mode={allocation['mode']:6s} | τ={np.linalg.norm(allocation['torque']):8.1f} N·m | "
            f"wheel_sat={wheel_saturation:5.1f}% | RCS={ship.rcs_propellant_mass:.0f}kg"
        )

    return allocation['torque'], dm_rcs_day
