# guidance/test_burn.py
"""
Test Burn Guidance Controller
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from base.star_utils import sec_to_days, ssb_to_body_centered
from base.star_constants import world_time
from base.star_logger import logger
from guidance.star_tools import estimate_burn_duration

# ————————————————————————
# Test Burn Controller
# ————————————————————————
class BurnController:
    def __init__(
        self,
        ship: object,
        central_body: str,
    ):
        
        self.central_body = central_body
        self.ship = ship
        self.accelerate = True
        self.t_start = 0.0
        self.t_end = 0.0
        self.thrust_dir = np.zeros(3)

    #
    # Set Target
    #
    def set_target(self, target: dict):

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
        
        # Set burn start and end times
        duration = sec_to_days(burn_time_sec)
        self.t_start = target_time - duration / 2
        self.t_end = self.t_start + duration

        if np.linalg.norm(thrust_dir) > 0:
            self.thrust_dir = thrust_dir / np.linalg.norm(thrust_dir)
        else:
            self.thrust_dir = np.zeros(3)

    
    #
    # Check if the burn is currently active
    #
    def active(self, t: float) -> bool:
        #if self.t_end == 0.0:
        #    raise ValueError("BurnController: Burn time not set. Call set_target() first.")
        # Active when t in [t_start, t_end]
        return self.t_start <= t < self.t_end

    #
    # Update method to provide VGO, direction, and throttle
    #
    def update(self, t: float, y_ssb: np.ndarray) -> None:
        if not self.active(t):
            return 0.0, np.array([0.0, 0.0, 1.0]), 0.0  # No thrust
        
        # Compute thrust direction in SSB frame based on current velocity
        state = ssb_to_body_centered(world_time(t), y_ssb, self.central_body)
        thrust_dir = state['v_kms'] / np.linalg.norm(state['v_kms'])

        if np.linalg.norm(self.thrust_dir) > 0:
            thrust_dir = self.thrust_dir

        if not self.accelerate:
            thrust_dir = -thrust_dir  # Decelerate

        return 1.0, thrust_dir, 1.0