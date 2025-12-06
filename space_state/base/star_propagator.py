# base/star_propagator.py
import time
from typing import Callable, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from scipy.integrate import solve_ivp

from .star_dynamics import n_body_acceleration, thrust_acceleration
from .star_structures import PropagationResult, GuidanceCycle, ControllerCycle
from .star_utils import log_guidance
from .star_config import CONFIG
from .star_ship import Spacecraft
from .star_logger import logger

if TYPE_CHECKING:
    from .star_fdr import FlightDataRecorder
    from .star_console import MissionController
    from .star_disturbance import DisturbanceTorqueModel


# ------------------- PROPAGATOR CLASS -------------------
class Propagator:
    def __init__(
        self,
        ship: Spacecraft = None,
        guidance: object = None,
        result: PropagationResult = None,
        bodies: dict = None,
        drag_enabled: bool = False,
        drag_area: float = 100.0,
        drag_coefficient: float = 2.2,
        disturbance_model: 'DisturbanceTorqueModel' = None,
    ):
        self.ship = ship
        self.guidance = guidance
        self.result = result if result is not None else PropagationResult()
        self.bodies = bodies if bodies is not None else CONFIG.all_bodies

        # Atmospheric drag settings
        self.drag_enabled = drag_enabled
        self.drag_area = drag_area
        self.drag_coefficient = drag_coefficient

        # Disturbance torque model (optional)
        self._disturbance_model = disturbance_model
        self._current_soi_body = 'earth'  # Track current SOI body for disturbance torques

        # Coast mode - skip attitude control for fast interplanetary cruise
        self._coast_mode = False

        # Flight Data Recorder and Mission Controller (optional)
        self._fdr: Optional['FlightDataRecorder'] = None
        self._controller: Optional['MissionController'] = None
        self._fdr_enabled = False
    
    @property
    def y(self) -> np.ndarray:
        """Current state vector."""
        return self.result.state

    @property
    def t(self) -> float:
        """Current time."""
        return self.result.time
    
    @property
    def r(self) -> np.ndarray:
        """Current position vector."""
        return self.result.state[:3]
        
    @property
    def v(self) -> np.ndarray:
        """Current velocity vector."""
        return self.result.state[3:6]
    
    @property
    def m(self) -> float:
        """Current mass."""
        return self.result.state[6]

    @property
    def q(self) -> np.ndarray:
        """Current quaternion [q0, q1, q2, q3]."""
        from .star_constants import IDX_Q
        return self.result.state[IDX_Q]

    @property
    def omega(self) -> np.ndarray:
        """Current angular velocity (rad/day)."""
        from .star_constants import IDX_OMEGA
        return self.result.state[IDX_OMEGA]

    @property
    def coast_mode(self) -> bool:
        """Whether coast mode is enabled (skips attitude control)."""
        return self._coast_mode

    @coast_mode.setter
    def coast_mode(self, value: bool):
        """Enable/disable coast mode."""
        self._coast_mode = value

    @property
    def disturbance_model(self) -> Optional['DisturbanceTorqueModel']:
        """Disturbance torque model (if enabled)."""
        return self._disturbance_model

    @disturbance_model.setter
    def disturbance_model(self, model: 'DisturbanceTorqueModel'):
        """Set disturbance torque model."""
        self._disturbance_model = model

    def set_soi_body(self, body_name: str) -> None:
        """
        Set the current SOI body for disturbance torque calculations.

        Args:
            body_name: Name of the body whose SOI spacecraft is in
        """
        self._current_soi_body = body_name

    def attach_fdr(self, fdr: 'FlightDataRecorder') -> None:
        """
        Attach a Flight Data Recorder for telemetry logging.

        Args:
            fdr: FlightDataRecorder instance
        """
        self._fdr = fdr
        self._fdr_enabled = True
        if self.ship is not None:
            fdr.set_spacecraft(self.ship)
        logger.info("FDR attached to propagator")

    def attach_controller(self, controller: 'MissionController') -> None:
        """
        Attach a MissionController for display and logging.

        Args:
            controller: MissionController instance
        """
        self._controller = controller
        self._fdr_enabled = True
        if self.ship is not None:
            controller.set_spacecraft(self.ship)
        if self.bodies is not None:
            controller.set_bodies(self.bodies)
        logger.info("MissionController attached to propagator")

    def _record_telemetry(self, force: bool = False) -> None:
        """Record current state to FDR/controller if attached."""
        if self._fdr is not None:
            self._fdr.record_frame(
                self.t, self.y, self.ship, self.bodies, force=force
            )
        if self._controller is not None:
            self._controller.update(
                self.t, self.y, force=force
            )

    def coast(
        self,
        t_end: float,
        max_step: float = 5.0,
        rtol: float = 1e-8,
        atol: float = 1e-8,
    ) -> 'PropagationResult':
        """
        Fast coast propagation for interplanetary cruise.

        Temporarily enables coast mode which:
        - Skips attitude control computations
        - Uses larger integration steps
        - Maintains translational accuracy

        Args:
            t_end: End time (days)
            max_step: Maximum integration step (days), default 5 days
            rtol: Relative tolerance (can be looser for coast)
            atol: Absolute tolerance

        Returns:
            PropagationResult
        """
        # Enable coast mode
        self._coast_mode = True

        # Propagate with larger steps
        result = self.propagate(t_end, max_step=max_step, rtol=rtol, atol=atol)

        # Disable coast mode
        self._coast_mode = False

        return result

    #
    # DYNAMICS FUNCTION
    #

    def dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        from .star_constants import IDX_R, IDX_V, IDX_M, IDX_Q, IDX_OMEGA
        from .star_attitude import quaternion_derivative, euler_rotation_equation, attitude_control

        # Extract state components
        r, v, m = y[IDX_R], y[IDX_V], y[IDX_M]
        q, omega = y[IDX_Q], y[IDX_OMEGA]

        # Force quaternion normalization at every dynamics call to prevent drift
        # This is critical because ODE solvers can take large steps where q drifts
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-6:
            q = q / q_norm
        else:
            q = np.array([1.0, 0.0, 0.0, 0.0])  # Reset to identity if degenerate

        if self.ship is not None:
            self.ship.update_propellant(m)

        # Translational dynamics
        # In coast mode, use simplified Sun-only gravity for speed (skip ephemeris queries)
        if self._coast_mode:
            from .star_constants import BODIES
            GM_SUN = BODIES['sun']['mu']  # AU³/day²
            r_mag = np.linalg.norm(r)
            a_grav = -GM_SUN * r / (r_mag ** 3)  # Heliocentric gravity only
            a_thrust = np.zeros(3)
            dm_main = 0.0
        else:
            a_grav = n_body_acceleration(t, y, self.bodies)  # AU/day²
            a_thrust, dm_main = thrust_acceleration(t, y, self.ship, self.guidance) # AU/day², kg/day

        # Atmospheric drag (with automatic SOI-based body detection)
        # Skip in coast mode - no drag in deep space
        if self.drag_enabled and not self._coast_mode:
            from .star_environment import compute_drag_acceleration
            a_drag = compute_drag_acceleration(
                r, v, m, t,
                area=self.drag_area,
                drag_coefficient=self.drag_coefficient,
            )
        else:
            a_drag = np.zeros(3)

        a_total = a_grav + a_thrust + a_drag

        # Rotational dynamics
        # Skip attitude control in coast mode for fast interplanetary cruise
        if self._coast_mode:
            # Coast mode: no attitude control, no rotation
            tau_control = np.zeros(3)
            q_dot = np.zeros(4)  # Keep attitude fixed
            omega_dot = np.zeros(3)
        else:
            tau_control, dm_rcs = attitude_control(t, y, self.ship, self.guidance)
            q_dot = quaternion_derivative(q, omega)

            # Compute disturbance torques (if model is enabled)
            if self._disturbance_model is not None:
                tau_disturbance = self._disturbance_model.total_disturbance(
                    t, r, v, q, self.ship, self._current_soi_body, self.bodies
                )
            else:
                tau_disturbance = np.zeros(3)

            # Total torque = control + disturbance
            tau_total = tau_control + tau_disturbance

            # Only compute rotational dynamics if ship has inertia defined
            if self.ship is not None and hasattr(self.ship, 'inertia'):
                omega_dot = euler_rotation_equation(omega, self.ship.inertia, tau_total)
            else:
                omega_dot = np.zeros(3)  # No rotation if no ship or inertia

        # Baumgarte stabilization for quaternion constraint (||q|| = 1)
        # Skip in coast mode since quaternion is fixed
        if not self._coast_mode:
            # This adds a corrective term to q_dot that drives the quaternion toward unit length
            # Without this, numerical integration causes ||q|| to drift from 1.0
            # Use aggressive stabilization (gain=1.0) and also project q_dot to be orthogonal to q
            q_norm_sq = np.dot(q, q)

            # Project q_dot to be tangent to the unit quaternion manifold
            # This ensures q_dot is orthogonal to q, preventing radial drift
            q_dot -= q * (np.dot(q, q_dot) / q_norm_sq)

            # Add Baumgarte term to drive back to unit sphere if drifted
            stabilization_gain = 1.0  # Conservative gain (per-timestep correction)
            q_dot += stabilization_gain * (1.0 - q_norm_sq) * q

        # Total mass rate of change
        # NOTE: RCS propellant is tracked SEPARATELY and NOT included in state derivative.
        # This is because RCS mass is tracked via ship._rcs_propellant_mass, not in the
        # state vector. Including it here would cause update_propellant() to incorrectly
        # attribute RCS consumption to main propellant.
        dm_dt = dm_main  # Only main engine affects state mass

        return np.concatenate([v, a_total, [dm_dt], q_dot, omega_dot])

    #
    # PROPAGATION METHOD
    #

    def propagate(
            self,
            t_end: float,
            events: Optional[Callable] = None,
            max_step: float = CONFIG.max_step,
            rtol: float = CONFIG.rtol,
            atol: float = CONFIG.atol
        ) -> PropagationResult:
        """Propagate the spacecraft state over the given time span in days."""

        #logger.debug(f"Propagating from t={self.result.time:.3f} days to t={t_end:.3f} days")

        sol = solve_ivp(
            fun=self.dynamics,
            t_span=[self.result.time, t_end],
            y0=self.result.state,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            events=events,
            dense_output=False,
        )

        # Normalize quaternion in the final state to prevent drift accumulation
        from .star_constants import IDX_Q
        final_q = sol.y[IDX_Q, -1]
        q_norm = np.linalg.norm(final_q)
        if q_norm > 1e-6:  # Sanity check
            sol.y[IDX_Q, -1] = final_q / q_norm

        # Append results
        self.result.append_raw(sol.t, sol.y)

        # Update attitude control actuator states after integration step
        # These are tracked separately from state vector to allow independent bookkeeping
        if self.ship is not None:
            dt_days = sol.t[-1] - sol.t[0] if len(sol.t) > 1 else 0.0
            dt_sec = dt_days * 86400.0  # Convert to seconds

            # Update wheel momentum based on torques applied during this step
            if hasattr(self.ship, '_wheel_torques'):
                from .star_constants import RW_MAX_MOMENTUM
                self.ship._wheel_momentum += self.ship._wheel_torques * dt_sec
                self.ship._wheel_momentum = np.clip(
                    self.ship._wheel_momentum, -RW_MAX_MOMENTUM, RW_MAX_MOMENTUM
                )
                self.ship._wheel_torques = np.zeros(4)  # Reset for next step

            # Update RCS propellant consumption
            if hasattr(self.ship, '_rcs_mdot_kgpd'):
                rcs_consumed_kg = self.ship._rcs_mdot_kgpd * dt_days
                if rcs_consumed_kg > 0:
                    self.ship.consume_rcs_propellant(rcs_consumed_kg)
                self.ship._rcs_mdot_kgpd = 0.0  # Reset for next step

            # Check for staging events (multi-stage vehicles)
            if hasattr(self.ship, 'check_staging'):
                t_current = sol.t[-1]
                if self.ship.check_staging(t_current):
                    # Stage was jettisoned - update mass in state vector
                    from .star_constants import IDX_M
                    new_mass = self.ship.mass
                    self.result._state[IDX_M] = new_mass
                    # Log the staging event
                    logger.debug(f"Staging at t={t_current:.4f} days, new mass={new_mass:,.0f} kg")

        # Log guidance cycles
        log_guidance(sol, self.guidance, self.result)

        # Record telemetry to FDR/controller if attached
        if self._fdr_enabled:
            self._record_telemetry()

        return self.result