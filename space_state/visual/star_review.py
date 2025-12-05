# visual/star_review.py
import numpy as np
import matplotlib.pyplot as plt

from base.star_logger import logger
from base.star_utils import au_to_km
from base.star_constants import BODIES, ts, epoch_jd, IDX_Q, IDX_OMEGA
from base.star_structures import PropagationResult


def review(result: PropagationResult, ship, body_name: str = "earth"):
    if result.t.size < 2:
        logger.warning("log_orbit_summary: result has fewer than 2 points")
        return

    body = BODIES[body_name]
    radius_km = au_to_km(body["radius"])

    # Earth-centered position in km
    earth_pos_au = np.array([
        body["skyfield"].at(ts.tt_jd(epoch_jd + t)).position.au
        for t in result.t
    ]).T  # (3, N)

    r_ssb_au = result.y[:3, :]                       # (3, N)
    r_rel_au = r_ssb_au - earth_pos_au
    r_norm_km = au_to_km(np.linalg.norm(r_rel_au, axis=0))

    altitude_km = r_norm_km - radius_km

    tof_hours = (result.t[-1] - result.t[0]) * 24.0
    tof_days = tof_hours / 24.0

    logger.info("------------------------------------------")
    logger.info(f"       ORBIT SUMMARY — {body_name.upper()}")
    logger.info("------------------------------------------")
    logger.info(f"Time of Flight       : {tof_days:.6f} days  ({tof_hours:.2f} hours)")
    logger.info(f"Initial Altitude     : {altitude_km[0]:8.1f} km")
    logger.info(f"Final Altitude       : {altitude_km[-1]:8.1f} km")
    logger.info(f"Minimum Altitude     : {altitude_km.min():8.1f} km")
    logger.info(f"Maximum Altitude     : {altitude_km.max():8.1f} km")
    logger.info(f"Mean Altitude        : {altitude_km.mean():8.1f} km")
    logger.info(f"Perigee Safety       : {altitude_km.min() - 100:.1f} km above 100 km limit")
    logger.info(f"Total Points         : {len(result.t)}")
    logger.info("------------------------------------------")
    logger.info(f"Final Mass           : {ship.mass:.1f} kg")
    logger.info(f"Main Propellant Used : {ship.initial_propellant_mass - ship.propellant_mass:.1f} kg")
    logger.info(f"Main Propellant Left : {ship.propellant_mass:.1f} kg")

    # RCS propellant (if available)
    if hasattr(ship, 'rcs_propellant_mass') and hasattr(ship, 'initial_rcs_propellant_mass'):
        rcs_used = ship.initial_rcs_propellant_mass - ship.rcs_propellant_mass
        logger.info(f"RCS Propellant Used  : {rcs_used:.1f} kg")
        logger.info(f"RCS Propellant Left  : {ship.rcs_propellant_mass:.1f} kg")
    elif hasattr(ship, 'rcs_propellant_mass'):
        logger.info(f"RCS Propellant Left  : {ship.rcs_propellant_mass:.1f} kg")

    # Wheel momentum (if available)
    if hasattr(ship, 'wheel_momentum'):
        wheel_mom = ship.wheel_momentum
        logger.info(f"Wheel Momentum       : [{wheel_mom[0]:.2f}, {wheel_mom[1]:.2f}, {wheel_mom[2]:.2f}, {wheel_mom[3]:.2f}] N·m·s")

    logger.info("------------------------------------------")

    # Bonus: detect if re-entry risk
    if altitude_km.min() < 80:
        logger.warning(f"PERIGEE BELOW 80 km → Atmospheric re-entry likely!")
    elif altitude_km.min() < 120:
        logger.warning(f"Low perigee ({altitude_km.min():.1f} km) → significant drag expected")


def plot_attitude_metrics(result: PropagationResult, title: str = "Attitude Control Metrics"):
    """
    Plot quaternion components, angular velocity, and pointing error over time.

    Args:
        result: PropagationResult containing 14-element state history
        title: Plot title
    """
    if result.t.size < 2:
        logger.warning("plot_attitude_metrics: result has fewer than 2 points")
        return

    t = result.t
    t_hours = (t - t[0]) * 24.0  # Convert to hours from start

    # Extract quaternion and angular velocity
    q = result.y[IDX_Q, :]       # (4, N)
    omega = result.y[IDX_OMEGA, :]  # (3, N)

    # Convert angular velocity to deg/s
    omega_deg_s = omega * (180.0 / np.pi) / 86400.0  # rad/day -> deg/s

    # Compute quaternion norm (should stay near 1.0)
    q_norm = np.linalg.norm(q, axis=0)

    # Compute pointing angle from body +Z to velocity direction
    # (This is approximate - uses velocity from state)
    v = result.y[3:6, :]  # Velocity (AU/day)
    pointing_error = np.zeros(len(t))

    for i in range(len(t)):
        # Desired direction (velocity)
        v_unit = v[:, i] / np.linalg.norm(v[:, i])

        # Actual body +Z from quaternion
        from base.star_attitude import quaternion_to_dcm
        R = quaternion_to_dcm(q[:, i])
        body_z = R @ np.array([0.0, 0.0, 1.0])

        # Angle between them
        cos_angle = np.clip(np.dot(body_z, v_unit), -1.0, 1.0)
        pointing_error[i] = np.arccos(cos_angle) * (180.0 / np.pi)

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Plot 1: Quaternion components
    ax1 = axes[0]
    ax1.plot(t_hours, q[0, :], 'b-', label='q₀ (scalar)', linewidth=1)
    ax1.plot(t_hours, q[1, :], 'r-', label='q₁', linewidth=1)
    ax1.plot(t_hours, q[2, :], 'g-', label='q₂', linewidth=1)
    ax1.plot(t_hours, q[3, :], 'm-', label='q₃', linewidth=1)
    ax1.set_ylabel('Quaternion')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    # Plot 2: Quaternion norm (should be 1.0)
    ax2 = axes[1]
    ax2.plot(t_hours, q_norm, 'k-', linewidth=1)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylabel('||q||')
    ax2.set_ylim([0.999, 1.001])
    ax2.grid(True, alpha=0.3)

    # Plot 3: Angular velocity
    ax3 = axes[2]
    ax3.plot(t_hours, omega_deg_s[0, :], 'r-', label='ωₓ', linewidth=1)
    ax3.plot(t_hours, omega_deg_s[1, :], 'g-', label='ωᵧ', linewidth=1)
    ax3.plot(t_hours, omega_deg_s[2, :], 'b-', label='ω_z', linewidth=1)
    ax3.set_ylabel('Angular Velocity (deg/s)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Pointing error
    ax4 = axes[3]
    ax4.plot(t_hours, pointing_error, 'k-', linewidth=1)
    ax4.set_ylabel('Pointing Error (deg)')
    ax4.set_xlabel('Time (hours)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('attitude_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Log summary statistics
    logger.info("------------------------------------------")
    logger.info("       ATTITUDE CONTROL SUMMARY")
    logger.info("------------------------------------------")
    logger.info(f"Initial pointing error : {pointing_error[0]:.2f}°")
    logger.info(f"Final pointing error   : {pointing_error[-1]:.2f}°")
    logger.info(f"Max pointing error     : {np.max(pointing_error):.2f}°")
    logger.info(f"Mean pointing error    : {np.mean(pointing_error):.2f}°")
    logger.info(f"Quaternion norm range  : [{q_norm.min():.6f}, {q_norm.max():.6f}]")
    logger.info(f"Max angular rate       : {np.max(np.abs(omega_deg_s)):.4f}°/s")
    logger.info("------------------------------------------")