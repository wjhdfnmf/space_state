# guidance/test_burn.py
"""
Test Burn Guidance Controller

Provides guidance for orbital maneuvers with proper attitude pre-slew.
The controller ensures the spacecraft is aligned with the burn direction
BEFORE the burn starts, preventing thrust losses from attitude error.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from base.star_utils import sec_to_days, ssb_to_body_centered
from base.star_constants import world_time
from base.star_logger import logger
from guidance.star_tools import estimate_burn_duration

# Default pre-slew time before burn (allow time for attitude alignment)
# For large slews (180°), need ~600s at 0.5°/s max rate, plus settling time
DEFAULT_PRESLEW_SECONDS = 900.0  # 15 minutes for large slews


# ————————————————————————
# Test Burn Controller
# ————————————————————————
class BurnController:
    """
    Burn guidance controller with attitude pre-slew support.

    The controller has three phases:
    1. COAST: Before pre-slew window - returns default direction
    2. PRE-SLEW: Before burn - returns burn direction with throttle=0
    3. BURN: During burn - returns burn direction with throttle=1.0

    This ensures the spacecraft aligns with the burn attitude before
    the burn starts, preventing thrust efficiency losses.
    """

    def __init__(
        self,
        ship: object,
        central_body: str,
        preslew_time_sec: float = DEFAULT_PRESLEW_SECONDS,
    ):
        self.central_body = central_body
        self.ship = ship
        self.accelerate = True
        self.t_start = 0.0
        self.t_end = 0.0
        self.t_preslew = 0.0  # Time to start pre-slew (before burn)
        self.preslew_time_sec = preslew_time_sec
        self.thrust_dir = np.zeros(3)
        self.target = None  # Store full target dict
        self.zemzev = None  # For compatibility

    #
    # Set Target
    #
    def set_target(self, target: dict):
        """
        Set the burn target parameters.

        Args:
            target: Dict with keys:
                - delta_v: Required delta-V (m/s)
                - t_center: Center time of burn (days)
                - dir: Thrust direction unit vector (optional)
        """
        self.target = target

        # Extract target parameters
        delta_v = target['delta_v']
        target_time = target['t_center']
        thrust_dir = target.get('dir', np.zeros(3))

        # Validate inputs
        if delta_v == 0:
            raise ValueError("Delta-V cannot be zero for BurnController.")

        # Determine if accelerating or decelerating
        if delta_v > 0:
            self.accelerate = True
        else:
            delta_v = -delta_v
            self.accelerate = False

        # Estimate burn duration
        burn_time_sec = estimate_burn_duration(
            delta_v,
            self.ship,
            verbose=True
        )

        logger.info(f"Estimated burn duration: {burn_time_sec:.2f} seconds.")

        # Check for invalid burn time
        if not np.isfinite(burn_time_sec) or burn_time_sec <= 0:
            error_msg = f"Invalid burn time: {burn_time_sec} seconds. "
            if burn_time_sec == np.inf:
                error_msg += "Insufficient propellant for required delta-V."
            elif burn_time_sec <= 0:
                error_msg += "Zero or negative burn time calculated."
            else:
                error_msg += "Non-finite burn time calculated."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Set burn start and end times
        duration = sec_to_days(burn_time_sec)
        self.t_start = target_time - duration / 2
        self.t_end = self.t_start + duration

        # Set pre-slew start time (before burn start)
        preslew_duration = sec_to_days(self.preslew_time_sec)
        self.t_preslew = self.t_start - preslew_duration

        logger.debug(f"Burn window: t_start={self.t_start:.6f} days, t_end={self.t_end:.6f} days, duration={duration:.6f} days")
        logger.debug(f"Pre-slew starts at t={self.t_preslew:.6f} days ({self.preslew_time_sec:.0f}s before burn)")

        if np.linalg.norm(thrust_dir) > 0:
            self.thrust_dir = thrust_dir / np.linalg.norm(thrust_dir)
        else:
            self.thrust_dir = np.zeros(3)

    #
    # Check if in pre-slew phase (before burn, but should be aligning)
    #
    def in_preslew(self, t: float) -> bool:
        """Return True if in pre-slew phase (attitude alignment before burn)."""
        if self.t_preslew == 0.0:
            return False
        return self.t_preslew <= t < self.t_start

    #
    # Check if the burn is currently active
    #
    def active(self, t: float, y=None) -> bool:
        """Return True if burn is active (thrusting)."""
        if self.t_end == 0.0:
            return False
        return self.t_start <= t < self.t_end

    #
    # Get current burn direction (even during pre-slew)
    #
    def get_burn_direction(self, t: float, y_ssb: np.ndarray) -> np.ndarray:
        """
        Get the burn direction vector for attitude alignment.

        Uses stored thrust_dir if set, otherwise computes from velocity.
        """
        if np.linalg.norm(self.thrust_dir) > 0:
            thrust_dir = self.thrust_dir.copy()
        else:
            # Compute from current velocity
            state = ssb_to_body_centered(world_time(t), y_ssb, self.central_body)
            thrust_dir = state['v_kms'] / np.linalg.norm(state['v_kms'])

        if not self.accelerate:
            thrust_dir = -thrust_dir

        return thrust_dir

    #
    # Update method to provide VGO, direction, and throttle
    #
    def update(self, t: float, y_ssb: np.ndarray) -> tuple:
        """
        Guidance update called every integration step.

        Returns:
            tuple: (vgo, thrust_dir, throttle)
                - vgo: Remaining delta-V estimate (placeholder)
                - thrust_dir: Unit vector of desired thrust direction
                - throttle: 0.0 (coast/pre-slew) or 1.0 (burning)
        """
        # Phase 1: Before pre-slew - coast, no specific attitude target
        if not self.in_preslew(t) and not self.active(t):
            return 0.0, np.array([0.0, 0.0, 1.0]), 0.0

        # Get burn direction (used for both pre-slew and burn phases)
        thrust_dir = self.get_burn_direction(t, y_ssb)

        # Phase 2: Pre-slew - provide direction but NO throttle
        if self.in_preslew(t):
            return 0.0, thrust_dir, 0.0  # Direction for attitude, no thrust

        # Phase 3: Burn - full throttle
        return 1.0, thrust_dir, 1.0