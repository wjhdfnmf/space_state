# guidance/proximity_ops.py
"""
Proximity Operations and Rendezvous Guidance

Implements Hill/Clohessy-Wiltshire relative motion dynamics and
guidance strategies for spacecraft rendezvous and docking.

Reference Frames:
- LVLH (Local Vertical Local Horizontal): x=radial, y=along-track, z=cross-track
- Hill frame: Rotating frame centered on target spacecraft

Key Equations (Clohessy-Wiltshire):
    ẍ - 2nẏ - 3n²x = aₓ
    ÿ + 2nẋ = aᵧ
    z̈ + n²z = aᵤ

where n = √(μ/a³) is the mean motion of the target orbit.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from base.star_constants import BODIES, AU_TO_KM, DAY_TO_SEC
from base.star_soi import get_body_position, get_body_velocity
from base.star_logger import logger


class ApproachMode(Enum):
    """Approach strategy modes."""
    V_BAR = "v_bar"          # Along velocity vector (standard safe approach)
    R_BAR = "r_bar"          # Along radial vector (direct, more delta-V)
    STATION_KEEP = "hold"    # Hold position relative to target
    INTERCEPT = "intercept"  # Direct intercept trajectory
    COELLIPTIC = "coelliptic"  # Coelliptic approach (same altitude)


@dataclass
class RelativeState:
    """Relative state in LVLH/Hill frame."""
    r_lvlh: np.ndarray      # Position [x, y, z] in LVLH (km)
    v_lvlh: np.ndarray      # Velocity [vx, vy, vz] in LVLH (m/s)
    range_km: float         # Range magnitude
    range_rate_mps: float   # Range rate (negative = closing)
    t: float                # Time (days)

    @property
    def x(self) -> float:
        """Radial position (km) - positive = above target."""
        return self.r_lvlh[0]

    @property
    def y(self) -> float:
        """Along-track position (km) - positive = ahead of target."""
        return self.r_lvlh[1]

    @property
    def z(self) -> float:
        """Cross-track position (km) - positive = right of target."""
        return self.r_lvlh[2]


@dataclass
class ManeuverPlan:
    """Planned proximity maneuver."""
    delta_v_lvlh: np.ndarray  # Delta-V in LVLH frame (m/s)
    t_execute: float          # Execution time (days)
    duration_s: float         # Burn duration (seconds)
    description: str = ""


# ------------------- FRAME TRANSFORMS -------------------

def inertial_to_lvlh(
    r_target: np.ndarray,
    v_target: np.ndarray,
    r_chaser: np.ndarray,
    v_chaser: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform chaser state from inertial to LVLH frame.

    LVLH Frame Definition:
    - x: Radial (away from central body)
    - y: Along-track (velocity direction)
    - z: Cross-track (completes right-hand system)

    Args:
        r_target: Target position (AU, inertial)
        v_target: Target velocity (AU/day, inertial)
        r_chaser: Chaser position (AU, inertial)
        v_chaser: Chaser velocity (AU/day, inertial)

    Returns:
        (r_lvlh_km, v_lvlh_mps): Relative position (km) and velocity (m/s) in LVLH
    """
    # Relative position and velocity in inertial frame
    r_rel = r_chaser - r_target  # AU
    v_rel = v_chaser - v_target  # AU/day

    # LVLH basis vectors
    r_hat = r_target / np.linalg.norm(r_target)  # Radial
    h = np.cross(r_target, v_target)  # Angular momentum
    h_hat = h / np.linalg.norm(h)  # Cross-track (orbit normal)
    y_hat = np.cross(h_hat, r_hat)  # Along-track

    # Rotation matrix from inertial to LVLH
    R = np.array([r_hat, y_hat, h_hat])

    # Transform relative state
    r_lvlh_au = R @ r_rel
    v_lvlh_au_day = R @ v_rel

    # Convert units
    r_lvlh_km = r_lvlh_au * AU_TO_KM
    v_lvlh_mps = v_lvlh_au_day * AU_TO_KM / DAY_TO_SEC * 1000  # AU/day to m/s

    return r_lvlh_km, v_lvlh_mps


def lvlh_to_inertial(
    r_target: np.ndarray,
    v_target: np.ndarray,
    r_lvlh_km: np.ndarray,
    v_lvlh_mps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform relative state from LVLH to inertial frame.

    Args:
        r_target: Target position (AU, inertial)
        v_target: Target velocity (AU/day, inertial)
        r_lvlh_km: Relative position in LVLH (km)
        v_lvlh_mps: Relative velocity in LVLH (m/s)

    Returns:
        (r_chaser_au, v_chaser_au_day): Chaser state in inertial frame
    """
    # LVLH basis vectors
    r_hat = r_target / np.linalg.norm(r_target)
    h = np.cross(r_target, v_target)
    h_hat = h / np.linalg.norm(h)
    y_hat = np.cross(h_hat, r_hat)

    # Rotation matrix from LVLH to inertial (transpose of inertial to LVLH)
    R_inv = np.array([r_hat, y_hat, h_hat]).T

    # Convert units
    r_lvlh_au = r_lvlh_km / AU_TO_KM
    v_lvlh_au_day = v_lvlh_mps / 1000 * DAY_TO_SEC / AU_TO_KM

    # Transform to inertial
    r_rel_au = R_inv @ r_lvlh_au
    v_rel_au_day = R_inv @ v_lvlh_au_day

    # Add target state
    r_chaser = r_target + r_rel_au
    v_chaser = v_target + v_rel_au_day

    return r_chaser, v_chaser


def get_relative_state(
    r_target: np.ndarray,
    v_target: np.ndarray,
    r_chaser: np.ndarray,
    v_chaser: np.ndarray,
    t: float = 0.0,
) -> RelativeState:
    """
    Compute relative state in LVLH frame.

    Args:
        r_target, v_target: Target state (AU, AU/day)
        r_chaser, v_chaser: Chaser state (AU, AU/day)
        t: Current time (days)

    Returns:
        RelativeState in LVLH frame
    """
    r_lvlh, v_lvlh = inertial_to_lvlh(r_target, v_target, r_chaser, v_chaser)

    range_km = np.linalg.norm(r_lvlh)
    if range_km > 1e-6:
        r_hat = r_lvlh / range_km
        range_rate_mps = np.dot(v_lvlh, r_hat)
    else:
        range_rate_mps = 0.0

    return RelativeState(
        r_lvlh=r_lvlh,
        v_lvlh=v_lvlh,
        range_km=range_km,
        range_rate_mps=range_rate_mps,
        t=t,
    )


# ------------------- CLOHESSY-WILTSHIRE DYNAMICS -------------------

def cw_state_transition(
    r0_km: np.ndarray,
    v0_mps: np.ndarray,
    n: float,
    dt_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate relative state using Clohessy-Wiltshire equations.

    Args:
        r0_km: Initial relative position [x, y, z] (km)
        v0_mps: Initial relative velocity [vx, vy, vz] (m/s)
        n: Target orbit mean motion (rad/s)
        dt_s: Time step (seconds)

    Returns:
        (r_km, v_mps): Propagated relative state
    """
    x0, y0, z0 = r0_km * 1000  # Convert to meters
    vx0, vy0, vz0 = v0_mps

    nt = n * dt_s
    c = np.cos(nt)
    s = np.sin(nt)

    # CW state transition matrix (6x6)
    # Position
    x = (4 - 3*c) * x0 + s/n * vx0 + 2*(1-c)/n * vy0
    y = 6*(s - nt) * x0 + y0 + 2*(c-1)/n * vx0 + (4*s - 3*nt)/n * vy0
    z = c * z0 + s/n * vz0

    # Velocity
    vx = 3*n*s * x0 + c * vx0 + 2*s * vy0
    vy = 6*n*(c - 1) * x0 - 2*s * vx0 + (4*c - 3) * vy0
    vz = -n*s * z0 + c * vz0

    r_km = np.array([x, y, z]) / 1000
    v_mps = np.array([vx, vy, vz])

    return r_km, v_mps


def cw_delta_v_to_target(
    r0_km: np.ndarray,
    v0_mps: np.ndarray,
    rf_km: np.ndarray,
    n: float,
    dt_s: float,
) -> np.ndarray:
    """
    Compute delta-V to reach target relative position using CW equations.

    Two-impulse transfer: burn at t=0 to enter transfer, burn at t=dt to arrive.

    Args:
        r0_km: Initial relative position (km)
        v0_mps: Initial relative velocity (m/s)
        rf_km: Target relative position (km)
        n: Mean motion (rad/s)
        dt_s: Transfer time (seconds)

    Returns:
        delta_v_mps: Required delta-V at t=0 (m/s)
    """
    x0, y0, z0 = r0_km * 1000  # meters
    xf, yf, zf = rf_km * 1000  # meters

    nt = n * dt_s
    c = np.cos(nt)
    s = np.sin(nt)

    # Solve for required initial velocity
    # Using CW equations inverted

    # For x-y coupled motion
    det = s * (4*s - 3*nt) - 2*(1-c) * 2*(c-1)
    if abs(det) < 1e-10:
        det = 1e-10  # Avoid division by zero

    # Target position minus free-drift contribution from position
    x_target = xf - (4 - 3*c) * x0
    y_target = yf - 6*(s - nt) * x0 - y0

    # Required velocity
    vx_req = ((4*s - 3*nt) * x_target - 2*(1-c) * y_target) / det * n
    vy_req = (-2*(c-1) * x_target + s * y_target) / det * n

    # Z is decoupled
    if abs(s) > 1e-10:
        vz_req = (zf - c * z0) * n / s
    else:
        vz_req = 0.0

    v_required = np.array([vx_req, vy_req, vz_req])
    delta_v = v_required - v0_mps

    return delta_v


# ------------------- APPROACH STRATEGIES -------------------

def compute_vbar_approach(
    rel_state: RelativeState,
    n: float,
    approach_rate_mps: float = 0.1,
    standoff_km: float = 0.2,
) -> ManeuverPlan:
    """
    Compute V-bar approach maneuver (along velocity vector).

    V-bar approach is the standard safe approach method:
    - Approach along the velocity vector (y-axis in LVLH)
    - Natural trajectory if slightly below target

    Args:
        rel_state: Current relative state
        n: Target orbit mean motion (rad/s)
        approach_rate_mps: Desired closing rate (m/s)
        standoff_km: Standoff distance on V-bar (km)

    Returns:
        ManeuverPlan for V-bar approach
    """
    # Target position: on V-bar behind target
    target_pos_km = np.array([0.0, -standoff_km, 0.0])

    # Current position
    r0 = rel_state.r_lvlh
    v0 = rel_state.v_lvlh

    # Estimate transfer time based on y-distance and approach rate
    dy = abs(r0[1] - target_pos_km[1])
    dt_s = max(dy * 1000 / approach_rate_mps, 60.0)  # At least 1 minute

    # Compute required delta-V
    dv = cw_delta_v_to_target(r0, v0, target_pos_km, n, dt_s)

    return ManeuverPlan(
        delta_v_lvlh=dv,
        t_execute=rel_state.t,
        duration_s=dt_s,
        description=f"V-bar approach to {standoff_km:.0f}m standoff",
    )


def compute_rbar_approach(
    rel_state: RelativeState,
    n: float,
    approach_rate_mps: float = 0.05,
    standoff_km: float = 0.1,
) -> ManeuverPlan:
    """
    Compute R-bar approach maneuver (along radial vector).

    R-bar approach:
    - Approach from below (or above) along radial
    - More direct but requires more delta-V

    Args:
        rel_state: Current relative state
        n: Target orbit mean motion (rad/s)
        approach_rate_mps: Desired closing rate (m/s)
        standoff_km: Standoff distance on R-bar (km)

    Returns:
        ManeuverPlan for R-bar approach
    """
    # Target position: on R-bar below target (negative x)
    target_pos_km = np.array([-standoff_km, 0.0, 0.0])

    r0 = rel_state.r_lvlh
    v0 = rel_state.v_lvlh

    # Estimate transfer time
    dx = abs(r0[0] - target_pos_km[0])
    dt_s = max(dx * 1000 / approach_rate_mps, 60.0)

    dv = cw_delta_v_to_target(r0, v0, target_pos_km, n, dt_s)

    return ManeuverPlan(
        delta_v_lvlh=dv,
        t_execute=rel_state.t,
        duration_s=dt_s,
        description=f"R-bar approach to {standoff_km*1000:.0f}m standoff",
    )


def compute_station_keeping(
    rel_state: RelativeState,
    n: float,
    hold_position_km: Optional[np.ndarray] = None,
) -> ManeuverPlan:
    """
    Compute station-keeping maneuver to hold position.

    Args:
        rel_state: Current relative state
        n: Target orbit mean motion (rad/s)
        hold_position_km: Desired hold position (default: current position)

    Returns:
        ManeuverPlan for station-keeping
    """
    if hold_position_km is None:
        hold_position_km = rel_state.r_lvlh.copy()

    # For station-keeping, we want zero relative velocity
    # This requires continuous delta-V to counteract CW drift

    # Compute delta-V to null velocity
    dv = -rel_state.v_lvlh.copy()

    # Add correction for position error
    pos_error = hold_position_km - rel_state.r_lvlh
    dv += pos_error * 1000 * n * 0.1  # Proportional position control

    return ManeuverPlan(
        delta_v_lvlh=dv,
        t_execute=rel_state.t,
        duration_s=1.0,
        description=f"Station-keeping at {np.linalg.norm(hold_position_km)*1000:.0f}m",
    )


def compute_hohmann_transfer_relative(
    rel_state: RelativeState,
    n: float,
    delta_altitude_km: float,
) -> List[ManeuverPlan]:
    """
    Compute two-impulse Hohmann-like transfer to change relative altitude.

    Used for coelliptic approaches and phasing.

    Args:
        rel_state: Current relative state
        n: Mean motion (rad/s)
        delta_altitude_km: Altitude change (positive = raise, negative = lower)

    Returns:
        List of two ManeuverPlans (burn 1 and burn 2)
    """
    # Convert delta altitude to delta-V (simplified)
    # For small changes: Δv ≈ n * Δr / 2 for each burn
    dv_magnitude = n * abs(delta_altitude_km) * 1000 / 2

    # First burn: tangential to change apogee/perigee
    burn1_dv = np.array([0.0, np.sign(delta_altitude_km) * dv_magnitude, 0.0])

    # Second burn: half orbit later to circularize
    # Time for half orbit
    T_half = np.pi / n

    burn2_dv = burn1_dv.copy()  # Same magnitude, same direction

    plan1 = ManeuverPlan(
        delta_v_lvlh=burn1_dv,
        t_execute=rel_state.t,
        duration_s=1.0,
        description=f"Hohmann burn 1: {'+' if delta_altitude_km > 0 else ''}{delta_altitude_km:.1f} km altitude change",
    )

    plan2 = ManeuverPlan(
        delta_v_lvlh=burn2_dv,
        t_execute=rel_state.t + T_half / DAY_TO_SEC,
        duration_s=1.0,
        description=f"Hohmann burn 2: circularize",
    )

    return [plan1, plan2]


# ------------------- PROXIMITY GUIDANCE CONTROLLER -------------------

@dataclass
class ProximityGuidanceConfig:
    """Configuration for proximity guidance controller."""
    mode: ApproachMode = ApproachMode.V_BAR
    target_standoff_km: float = 0.2          # 200 m default standoff
    approach_rate_mps: float = 0.1           # 0.1 m/s approach rate
    station_keep_deadband_m: float = 1.0     # 1 m position deadband
    max_delta_v_mps: float = 5.0             # Maximum single maneuver delta-V


class ProximityGuidance:
    """
    Proximity operations guidance controller.

    Provides automated guidance for rendezvous and docking operations.
    """

    def __init__(
        self,
        target_body: str = 'earth',
        target_altitude_km: float = 400.0,
        config: Optional[ProximityGuidanceConfig] = None,
    ):
        """
        Initialize proximity guidance.

        Args:
            target_body: Central body for orbit
            target_altitude_km: Target orbit altitude
            config: Guidance configuration
        """
        self.target_body = target_body
        self.target_altitude_km = target_altitude_km
        self.config = config or ProximityGuidanceConfig()

        # Compute mean motion
        mu = BODIES[target_body]['mu_km3_s2']
        body_radius = BODIES[target_body].get('radius', 0) * AU_TO_KM
        a_km = body_radius + target_altitude_km
        self.n = np.sqrt(mu / a_km**3)  # rad/s

        # State tracking
        self._last_rel_state: Optional[RelativeState] = None
        self._maneuver_queue: List[ManeuverPlan] = []
        self._total_delta_v_mps = 0.0

    @property
    def mean_motion(self) -> float:
        """Target orbit mean motion (rad/s)."""
        return self.n

    @property
    def orbital_period_s(self) -> float:
        """Target orbital period (seconds)."""
        return 2 * np.pi / self.n

    def update(
        self,
        r_target: np.ndarray,
        v_target: np.ndarray,
        r_chaser: np.ndarray,
        v_chaser: np.ndarray,
        t: float,
    ) -> Optional[ManeuverPlan]:
        """
        Update guidance and compute next maneuver if needed.

        Args:
            r_target, v_target: Target state (AU, AU/day)
            r_chaser, v_chaser: Chaser state (AU, AU/day)
            t: Current time (days)

        Returns:
            ManeuverPlan if a maneuver is needed, None otherwise
        """
        # Compute relative state
        rel_state = get_relative_state(r_target, v_target, r_chaser, v_chaser, t)
        self._last_rel_state = rel_state

        # Check if we have queued maneuvers
        if self._maneuver_queue:
            next_maneuver = self._maneuver_queue[0]
            if t >= next_maneuver.t_execute:
                self._maneuver_queue.pop(0)
                self._total_delta_v_mps += np.linalg.norm(next_maneuver.delta_v_lvlh)
                return next_maneuver

        # Generate new maneuver based on mode
        if self.config.mode == ApproachMode.V_BAR:
            return self._update_vbar(rel_state)
        elif self.config.mode == ApproachMode.R_BAR:
            return self._update_rbar(rel_state)
        elif self.config.mode == ApproachMode.STATION_KEEP:
            return self._update_station_keep(rel_state)
        else:
            return None

    def _update_vbar(self, rel_state: RelativeState) -> Optional[ManeuverPlan]:
        """Update V-bar approach guidance."""
        # Check if we're on V-bar (y < 0, x ≈ 0, z ≈ 0)
        on_vbar = (
            abs(rel_state.x) < 0.1 and
            abs(rel_state.z) < 0.1 and
            rel_state.y < 0
        )

        if not on_vbar and rel_state.range_km > self.config.target_standoff_km:
            # Need to get to V-bar first
            return compute_vbar_approach(
                rel_state, self.n,
                self.config.approach_rate_mps,
                self.config.target_standoff_km,
            )

        # Already on V-bar or close enough
        return None

    def _update_rbar(self, rel_state: RelativeState) -> Optional[ManeuverPlan]:
        """Update R-bar approach guidance."""
        if rel_state.range_km > self.config.target_standoff_km:
            return compute_rbar_approach(
                rel_state, self.n,
                self.config.approach_rate_mps,
                self.config.target_standoff_km,
            )
        return None

    def _update_station_keep(self, rel_state: RelativeState) -> Optional[ManeuverPlan]:
        """Update station-keeping guidance."""
        # Check if velocity is too high
        v_mag = np.linalg.norm(rel_state.v_lvlh)
        if v_mag > 0.01:  # > 1 cm/s
            return compute_station_keeping(rel_state, self.n)
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get current guidance status."""
        if self._last_rel_state is None:
            return {'initialized': False}

        return {
            'initialized': True,
            'mode': self.config.mode.value,
            'range_km': self._last_rel_state.range_km,
            'range_rate_mps': self._last_rel_state.range_rate_mps,
            'position_lvlh_km': self._last_rel_state.r_lvlh.tolist(),
            'velocity_lvlh_mps': self._last_rel_state.v_lvlh.tolist(),
            'total_delta_v_mps': self._total_delta_v_mps,
            'queued_maneuvers': len(self._maneuver_queue),
        }


# ------------------- UTILITY FUNCTIONS -------------------

def predict_closest_approach(
    rel_state: RelativeState,
    n: float,
    t_max_s: float = 7200,  # 2 hours
) -> Tuple[float, float, np.ndarray]:
    """
    Predict closest approach time and distance.

    Args:
        rel_state: Current relative state
        n: Mean motion (rad/s)
        t_max_s: Maximum prediction time (seconds)

    Returns:
        (t_closest_s, range_closest_km, position_closest_km)
    """
    # Propagate in small steps and find minimum range
    dt = 10.0  # 10 second steps
    n_steps = int(t_max_s / dt)

    r = rel_state.r_lvlh.copy()
    v = rel_state.v_lvlh.copy()

    min_range = rel_state.range_km
    min_t = 0.0
    min_pos = r.copy()

    for i in range(n_steps):
        r, v = cw_state_transition(r, v, n, dt)
        range_km = np.linalg.norm(r)

        if range_km < min_range:
            min_range = range_km
            min_t = (i + 1) * dt
            min_pos = r.copy()

    return min_t, min_range, min_pos


def compute_safe_trajectory(
    rel_state: RelativeState,
    n: float,
    keep_out_radius_km: float = 0.2,
) -> bool:
    """
    Check if current trajectory is safe (misses keep-out zone).

    Args:
        rel_state: Current relative state
        n: Mean motion (rad/s)
        keep_out_radius_km: Keep-out zone radius (km)

    Returns:
        True if trajectory is safe
    """
    t_closest, range_closest, _ = predict_closest_approach(rel_state, n)
    return range_closest > keep_out_radius_km
