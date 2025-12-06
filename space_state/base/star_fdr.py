# base/star_fdr.py
"""
Flight Data Recorder (FDR) - Comprehensive Mission Telemetry Logging

Records all flight metrics during spacecraft simulation for:
- Post-mission review and validation
- Mission documentation
- Data export for external analysis

Logged metrics include:
- Time and mission elapsed time (MET)
- Position and velocity (SSB and body-relative)
- Orbital elements (a, e, i, Ω, ω, θ)
- Attitude (quaternion, Euler angles, angular rates)
- Propulsion (thrust, throttle, propellant, mass flow)
- Distances to celestial bodies
- Environmental data (eclipse, thermal, radiation)
- Events (burns, staging, SOI transitions)
- Communication windows
"""

import json
import csv
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import numpy as np

from .star_logger import logger


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TelemetryFrame:
    """Single frame of telemetry data at a specific time."""

    # Time
    t: float                           # Simulation time (days from epoch)
    met_sec: float                     # Mission Elapsed Time (seconds)
    met_str: str                       # MET formatted string (HH:MM:SS.mmm)

    # Position (SSB frame, AU)
    r_x: float = 0.0
    r_y: float = 0.0
    r_z: float = 0.0
    r_mag_au: float = 0.0
    r_mag_km: float = 0.0

    # Velocity (SSB frame, AU/day)
    v_x: float = 0.0
    v_y: float = 0.0
    v_z: float = 0.0
    v_mag_aud: float = 0.0
    v_mag_kms: float = 0.0

    # Mass (kg)
    mass: float = 0.0
    dry_mass: float = 0.0
    propellant_mass: float = 0.0
    rcs_propellant_mass: float = 0.0

    # Attitude (quaternion, scalar-first)
    q0: float = 1.0
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0
    q_norm: float = 1.0

    # Attitude (Euler angles, degrees)
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    # Angular velocity (deg/s)
    omega_x: float = 0.0
    omega_y: float = 0.0
    omega_z: float = 0.0
    omega_mag: float = 0.0

    # Thrust and propulsion
    thrust_n: float = 0.0
    throttle: float = 0.0
    isp_s: float = 0.0
    mdot_kgs: float = 0.0

    # Thrust direction (unit vector)
    thrust_dir_x: float = 0.0
    thrust_dir_y: float = 0.0
    thrust_dir_z: float = 0.0

    # Acceleration (m/s²)
    accel_thrust: float = 0.0
    accel_grav: float = 0.0
    accel_total: float = 0.0

    # Body-relative data
    primary_body: str = ""
    altitude_km: float = 0.0
    speed_rel_kms: float = 0.0
    distance_km: float = 0.0

    # Orbital elements (if in orbit)
    sma_km: float = 0.0
    eccentricity: float = 0.0
    inclination_deg: float = 0.0
    raan_deg: float = 0.0
    arg_periapsis_deg: float = 0.0
    true_anomaly_deg: float = 0.0
    orbital_period_hr: float = 0.0

    # Distances to major bodies (km)
    dist_sun_km: float = 0.0
    dist_earth_km: float = 0.0
    dist_moon_km: float = 0.0
    dist_mars_km: float = 0.0
    dist_jupiter_km: float = 0.0

    # Environmental
    eclipse: bool = False
    solar_flux_w_m2: float = 0.0
    temperature_k: float = 0.0

    # Reaction wheels
    wheel_momentum_1: float = 0.0
    wheel_momentum_2: float = 0.0
    wheel_momentum_3: float = 0.0
    wheel_momentum_4: float = 0.0
    wheel_total_momentum: float = 0.0

    # Communication
    comm_earth_possible: bool = True
    comm_delay_sec: float = 0.0

    # Event flags
    is_burning: bool = False
    is_coasting: bool = True
    event_name: str = ""


@dataclass
class MissionEvent:
    """Record of a mission event (burn, staging, etc.)."""
    t: float                           # Time (days)
    met_sec: float                     # MET (seconds)
    event_type: str                    # burn_start, burn_end, staging, soi_entry, etc.
    description: str
    delta_v_mps: float = 0.0
    mass_before_kg: float = 0.0
    mass_after_kg: float = 0.0
    body_name: str = ""
    altitude_km: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionSummary:
    """Summary statistics for the complete mission."""
    mission_name: str
    spacecraft_name: str
    start_time: float
    end_time: float
    duration_days: float
    duration_str: str

    # Mass budget
    initial_mass_kg: float = 0.0
    final_mass_kg: float = 0.0
    propellant_used_kg: float = 0.0
    rcs_used_kg: float = 0.0

    # Delta-V budget
    total_delta_v_mps: float = 0.0
    burns: List[Dict[str, Any]] = field(default_factory=list)

    # Trajectory stats
    max_altitude_km: float = 0.0
    min_altitude_km: float = 0.0
    max_velocity_kms: float = 0.0
    max_distance_sun_km: float = 0.0
    min_distance_sun_km: float = 0.0

    # Bodies encountered
    bodies_visited: List[str] = field(default_factory=list)
    soi_transitions: List[Dict[str, Any]] = field(default_factory=list)

    # Status
    mission_success: bool = True
    objectives_met: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# FLIGHT DATA RECORDER
# ============================================================================

class FlightDataRecorder:
    """
    Flight Data Recorder for comprehensive mission telemetry logging.

    Usage:
        fdr = FlightDataRecorder(mission_name="Mars Transfer")
        fdr.set_spacecraft(ship)

        # During propagation (automatic or manual):
        fdr.record_frame(t, state, ship, bodies)

        # At mission end:
        fdr.finalize()
        fdr.save()
    """

    def __init__(
        self,
        mission_name: str = "Unnamed Mission",
        log_dir: Optional[str] = None,
        record_interval: float = 0.001,  # days (~86 seconds)
        verbose: bool = False,
    ):
        """
        Initialize the Flight Data Recorder.

        Args:
            mission_name: Name for this mission
            log_dir: Directory for log files (default: space_state/logs/missions/)
            record_interval: Minimum time between telemetry frames (days)
            verbose: Enable verbose logging
        """
        self.mission_name = mission_name
        self.verbose = verbose
        self.record_interval = record_interval

        # Set up log directory
        if log_dir is None:
            base_dir = Path(__file__).parent.parent / 'logs' / 'missions'
        else:
            base_dir = Path(log_dir)

        # Create mission-specific folder with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.mission_id = f"{mission_name.lower().replace(' ', '_')}_{timestamp}"
        self.log_dir = base_dir / self.mission_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.telemetry: List[TelemetryFrame] = []
        self.events: List[MissionEvent] = []
        self.summary: Optional[MissionSummary] = None

        # State tracking
        self._spacecraft = None
        self._start_time: Optional[float] = None
        self._last_record_time: float = -999.0
        self._initial_mass: float = 0.0
        self._initial_rcs_mass: float = 0.0
        self._total_delta_v: float = 0.0
        self._burn_start_mass: float = 0.0
        self._burn_start_time: float = 0.0
        self._is_burning: bool = False
        self._bodies_visited: set = set()

        # Constants (loaded lazily)
        self._au_to_km: float = 1.495978707e8
        self._day_to_sec: float = 86400.0

        logger.info(f"FDR initialized: {self.mission_id}")
        logger.info(f"Log directory: {self.log_dir}")

    def set_spacecraft(self, ship) -> None:
        """Set the spacecraft being tracked."""
        self._spacecraft = ship
        self._initial_mass = ship.mass
        if hasattr(ship, '_rcs_propellant_mass'):
            self._initial_rcs_mass = ship._rcs_propellant_mass

    def set_start_time(self, t: float) -> None:
        """Set the mission start time for MET calculation."""
        self._start_time = t

    def record_frame(
        self,
        t: float,
        state: np.ndarray,
        ship = None,
        bodies: Optional[dict] = None,
        force: bool = False,
    ) -> None:
        """
        Record a telemetry frame.

        Args:
            t: Current time (days from epoch)
            state: State vector [r, v, m, q, ω]
            ship: Spacecraft object (optional, uses stored reference)
            bodies: Dict of celestial bodies for distance calculations
            force: Force recording even if interval not met
        """
        if ship is None:
            ship = self._spacecraft

        # Adaptive recording: record 10x more frequently during burns
        is_burning = ship is not None and ship.thrust > 0 and ship.throttle > 0
        effective_interval = self.record_interval / 10 if is_burning else self.record_interval

        # Check record interval
        if not force and (t - self._last_record_time) < effective_interval:
            return

        self._last_record_time = t

        if self._start_time is None:
            self._start_time = t

        # Calculate MET
        met_sec = (t - self._start_time) * self._day_to_sec
        met_str = self._format_met(met_sec)

        frame = TelemetryFrame(
            t=t,
            met_sec=met_sec,
            met_str=met_str,
        )

        # Position
        frame.r_x = state[0]
        frame.r_y = state[1]
        frame.r_z = state[2]
        frame.r_mag_au = np.linalg.norm(state[:3])
        frame.r_mag_km = frame.r_mag_au * self._au_to_km

        # Velocity
        frame.v_x = state[3]
        frame.v_y = state[4]
        frame.v_z = state[5]
        frame.v_mag_aud = np.linalg.norm(state[3:6])
        frame.v_mag_kms = frame.v_mag_aud * self._au_to_km / self._day_to_sec

        # Mass
        frame.mass = state[6]
        if ship is not None:
            frame.dry_mass = ship.dry_mass
            frame.propellant_mass = ship.propellant_mass
            if hasattr(ship, '_rcs_propellant_mass'):
                frame.rcs_propellant_mass = ship._rcs_propellant_mass

        # Attitude
        q = state[7:11]
        frame.q0, frame.q1, frame.q2, frame.q3 = q
        frame.q_norm = np.linalg.norm(q)

        # Convert to Euler angles
        roll, pitch, yaw = self._quaternion_to_euler(q)
        frame.roll_deg = np.degrees(roll)
        frame.pitch_deg = np.degrees(pitch)
        frame.yaw_deg = np.degrees(yaw)

        # Angular velocity
        omega = state[11:14]
        omega_degs = omega * 180.0 / (np.pi * self._day_to_sec)  # rad/day to deg/s
        frame.omega_x = omega_degs[0]
        frame.omega_y = omega_degs[1]
        frame.omega_z = omega_degs[2]
        frame.omega_mag = np.linalg.norm(omega_degs)

        # Propulsion
        if ship is not None:
            frame.thrust_n = ship.thrust
            frame.throttle = ship.throttle
            frame.isp_s = ship.isp
            frame.mdot_kgs = ship.mdot if hasattr(ship, 'mdot') else 0.0

            # Thrust direction
            if hasattr(ship, '_thrust_direction'):
                td = ship._thrust_direction
                frame.thrust_dir_x = td[0]
                frame.thrust_dir_y = td[1]
                frame.thrust_dir_z = td[2]

            # Acceleration
            if frame.mass > 0:
                frame.accel_thrust = frame.thrust_n / frame.mass

            # Wheel momentum
            if hasattr(ship, '_wheel_momentum'):
                wm = ship._wheel_momentum
                frame.wheel_momentum_1 = wm[0]
                frame.wheel_momentum_2 = wm[1]
                frame.wheel_momentum_3 = wm[2]
                frame.wheel_momentum_4 = wm[3]
                frame.wheel_total_momentum = np.linalg.norm(wm)

            # Burning status
            frame.is_burning = ship.thrust > 0 and ship.throttle > 0
            frame.is_coasting = not frame.is_burning

        # Body distances
        if bodies is not None:
            self._compute_body_distances(frame, state, t, bodies)

        # Communication delay to Earth
        frame.comm_delay_sec = frame.dist_earth_km / 299792.458 if frame.dist_earth_km > 0 else 0.0

        self.telemetry.append(frame)

        if self.verbose:
            logger.debug(f"FDR: {met_str} | Alt={frame.altitude_km:.0f}km | V={frame.v_mag_kms:.2f}km/s | M={frame.mass:.0f}kg")

    def record_trajectory(
        self,
        result,
        ship = None,
        bodies: Optional[dict] = None,
        downsample: int = 1,
        replace: bool = True,
    ) -> int:
        """
        Record telemetry from a complete propagation result trajectory.

        This extracts and records multiple frames from an already-completed
        propagation, useful for post-propagation data capture.

        Args:
            result: PropagationResult object with t and y arrays
            ship: Spacecraft object
            bodies: Dict of celestial bodies
            downsample: Record every Nth point (1=all, 10=every 10th)
            replace: If True, clear existing telemetry before recording (default: True)

        Returns:
            Number of frames recorded
        """
        if ship is None:
            ship = self._spacecraft

        if not hasattr(result, 't') or result.t is None or len(result.t) == 0:
            return 0

        # Clear existing telemetry to avoid duplicates and ordering issues
        if replace:
            self.telemetry.clear()
            self._last_record_time = -999.0

        frames_recorded = 0
        times = result.t
        states = result.y

        for i in range(0, len(times), downsample):
            t = times[i]
            state = states[:, i] if states.ndim > 1 else states

            # Force recording (skip interval check)
            self.record_frame(t, state, ship, bodies, force=True)
            frames_recorded += 1

        # Sort telemetry by time to ensure chronological order
        self._sort_telemetry()

        logger.debug(f"FDR: Recorded {frames_recorded} frames from trajectory")
        return frames_recorded

    def _sort_telemetry(self) -> None:
        """Sort telemetry frames by time and remove duplicates."""
        if not self.telemetry:
            return

        # Sort by time
        self.telemetry.sort(key=lambda f: f.t)

        # Remove near-duplicates (frames with same time within tolerance)
        unique_frames = []
        last_t = -999.0
        tolerance = 1e-10  # ~0.86 microseconds

        for frame in self.telemetry:
            if abs(frame.t - last_t) > tolerance:
                unique_frames.append(frame)
                last_t = frame.t

        self.telemetry = unique_frames

    def record_event(
        self,
        t: float,
        event_type: str,
        description: str,
        delta_v_mps: float = 0.0,
        mass_before_kg: float = 0.0,
        mass_after_kg: float = 0.0,
        body_name: str = "",
        altitude_km: float = 0.0,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a mission event."""
        if self._start_time is None:
            self._start_time = t

        met_sec = (t - self._start_time) * self._day_to_sec

        event = MissionEvent(
            t=t,
            met_sec=met_sec,
            event_type=event_type,
            description=description,
            delta_v_mps=delta_v_mps,
            mass_before_kg=mass_before_kg,
            mass_after_kg=mass_after_kg,
            body_name=body_name,
            altitude_km=altitude_km,
            data=data or {},
        )

        self.events.append(event)

        # Track delta-V
        if delta_v_mps > 0:
            self._total_delta_v += delta_v_mps

        # Track bodies
        if body_name:
            self._bodies_visited.add(body_name)

        logger.info(f"FDR EVENT [{event_type}]: {description}")

    def start_burn(self, t: float, burn_name: str = "burn") -> None:
        """Mark the start of a burn for delta-V tracking."""
        if self._spacecraft is not None:
            self._burn_start_mass = self._spacecraft.mass
            self._burn_start_time = t
            self._is_burning = True
            self.record_event(t, "burn_start", f"Started: {burn_name}")

    def end_burn(self, t: float, burn_name: str = "burn") -> None:
        """Mark the end of a burn and calculate delta-V."""
        if self._spacecraft is not None and self._is_burning:
            mass_after = self._spacecraft.mass
            if mass_after < self._burn_start_mass:
                # Calculate actual delta-V using rocket equation
                isp = self._spacecraft.max_isp
                ve = isp * 9.80665
                dv = ve * np.log(self._burn_start_mass / mass_after)

                self.record_event(
                    t, "burn_end", f"Completed: {burn_name}",
                    delta_v_mps=dv,
                    mass_before_kg=self._burn_start_mass,
                    mass_after_kg=mass_after,
                )

            self._is_burning = False

    def finalize(self, success: bool = True, objectives: Optional[List[str]] = None) -> MissionSummary:
        """
        Finalize the mission and generate summary.

        Args:
            success: Whether mission objectives were met
            objectives: List of objectives that were met
        """
        if not self.telemetry:
            logger.warning("FDR: No telemetry data recorded")
            return None

        first = self.telemetry[0]
        last = self.telemetry[-1]

        duration_days = last.t - first.t
        duration_sec = duration_days * self._day_to_sec

        # Collect trajectory statistics
        altitudes = [f.altitude_km for f in self.telemetry if f.altitude_km > 0]
        velocities = [f.v_mag_kms for f in self.telemetry]
        sun_distances = [f.dist_sun_km for f in self.telemetry if f.dist_sun_km > 0]

        # Collect burn information
        burns = []
        for event in self.events:
            if event.event_type == "burn_end" and event.delta_v_mps > 0:
                burns.append({
                    'name': event.description.replace("Completed: ", ""),
                    'time_days': event.t,
                    'met_sec': event.met_sec,
                    'delta_v_mps': event.delta_v_mps,
                    'mass_before_kg': event.mass_before_kg,
                    'mass_after_kg': event.mass_after_kg,
                    'propellant_kg': event.mass_before_kg - event.mass_after_kg,
                })

        # SOI transitions
        soi_transitions = [
            {'time_days': e.t, 'body': e.body_name, 'description': e.description}
            for e in self.events if 'soi' in e.event_type.lower()
        ]

        self.summary = MissionSummary(
            mission_name=self.mission_name,
            spacecraft_name=self._spacecraft._name if self._spacecraft else "Unknown",
            start_time=first.t,
            end_time=last.t,
            duration_days=duration_days,
            duration_str=self._format_duration(duration_sec),
            initial_mass_kg=self._initial_mass,
            final_mass_kg=last.mass,
            propellant_used_kg=self._initial_mass - last.mass,
            rcs_used_kg=self._initial_rcs_mass - last.rcs_propellant_mass if last.rcs_propellant_mass > 0 else 0,
            total_delta_v_mps=self._total_delta_v,
            burns=burns,
            max_altitude_km=max(altitudes) if altitudes else 0,
            min_altitude_km=min(altitudes) if altitudes else 0,
            max_velocity_kms=max(velocities) if velocities else 0,
            max_distance_sun_km=max(sun_distances) if sun_distances else 0,
            min_distance_sun_km=min(sun_distances) if sun_distances else 0,
            bodies_visited=list(self._bodies_visited),
            soi_transitions=soi_transitions,
            mission_success=success,
            objectives_met=objectives or [],
        )

        return self.summary

    def save(self, formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Save mission data to files.

        Args:
            formats: List of formats to save ('json', 'csv', 'summary')
                    Default: all formats

        Returns:
            Dict mapping format to file path
        """
        if formats is None:
            formats = ['json', 'csv', 'summary']

        saved_files = {}

        # Save telemetry as CSV
        if 'csv' in formats:
            csv_path = self.log_dir / 'telemetry.csv'
            self._save_telemetry_csv(csv_path)
            saved_files['csv'] = str(csv_path)

        # Save events as JSON
        if 'json' in formats:
            # Events
            events_path = self.log_dir / 'events.json'
            self._save_events_json(events_path)
            saved_files['events'] = str(events_path)

            # Summary
            if self.summary:
                summary_path = self.log_dir / 'summary.json'
                self._save_summary_json(summary_path)
                saved_files['summary_json'] = str(summary_path)

        # Save human-readable summary
        if 'summary' in formats:
            summary_path = self.log_dir / 'mission_report.txt'
            self._save_summary_text(summary_path)
            saved_files['report'] = str(summary_path)

        logger.info(f"FDR: Saved mission data to {self.log_dir}")
        return saved_files

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _format_met(self, met_sec: float) -> str:
        """Format MET in HH:MM:SS.mmm"""
        hours = int(met_sec // 3600)
        minutes = int((met_sec % 3600) // 60)
        seconds = met_sec % 60
        return f"{hours:03d}:{minutes:02d}:{seconds:06.3f}"

    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if days > 0:
            return f"{days}d {hours:02d}h {minutes:02d}m {secs:02d}s"
        elif hours > 0:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        elif minutes > 0:
            return f"{minutes}m {secs:02d}s"
        else:
            return f"{secs}s"

    def _quaternion_to_euler(self, q: np.ndarray) -> tuple:
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        q0, q1, q2, q3 = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q0 * q2 - q3 * q1)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def _compute_body_distances(
        self,
        frame: TelemetryFrame,
        state: np.ndarray,
        t: float,
        bodies: dict,
    ) -> None:
        """Compute distances to celestial bodies, relative velocity, and orbital elements."""
        from .star_constants import epoch_jd, ts, BODIES

        # Handle case where bodies is a list of names instead of a dict
        if isinstance(bodies, list):
            bodies = {name: BODIES[name] for name in bodies if name in BODIES}
        elif bodies is None:
            bodies = BODIES

        r_sc = state[:3]  # Spacecraft position (AU)
        v_sc = state[3:6]  # Spacecraft velocity (AU/day)
        time_obj = ts.tt_jd(epoch_jd + t)

        # Distance to Sun
        frame.dist_sun_km = np.linalg.norm(r_sc) * self._au_to_km

        # Solar flux at current distance (inverse square law from 1 AU)
        r_sun_au = np.linalg.norm(r_sc)
        if r_sun_au > 0:
            frame.solar_flux_w_m2 = 1361.0 / (r_sun_au ** 2)  # Solar constant at 1 AU is ~1361 W/m²

        # Distance to major bodies
        body_names = ['earth', 'moon', 'mars', 'jupiter']
        for body_name in body_names:
            if body_name in bodies:
                body = bodies[body_name]
                r_body = body['skyfield'].at(time_obj).position.au
                v_body = body['skyfield'].at(time_obj).velocity.au_per_d
                dist_au = np.linalg.norm(r_sc - r_body)
                dist_km = dist_au * self._au_to_km

                setattr(frame, f'dist_{body_name}_km', dist_km)

                # For Earth: compute detailed data
                if body_name == 'earth':
                    r_body_km = body['radius'] * self._au_to_km
                    frame.altitude_km = dist_km - r_body_km
                    frame.distance_km = dist_km
                    frame.primary_body = 'earth'

                    # Relative velocity
                    v_rel_aud = v_sc - v_body
                    frame.speed_rel_kms = np.linalg.norm(v_rel_aud) * self._au_to_km / self._day_to_sec

                    # Compute orbital elements relative to Earth
                    try:
                        r_rel_au = r_sc - r_body
                        r_rel_km = r_rel_au * self._au_to_km
                        v_rel_kms = v_rel_aud * self._au_to_km / self._day_to_sec

                        mu_earth_km3s2 = body['mu_km3_s2']

                        # Compute orbital elements
                        elements = self._compute_orbital_elements(r_rel_km, v_rel_kms, mu_earth_km3s2)
                        if elements:
                            frame.sma_km = elements['a']
                            frame.eccentricity = elements['e']
                            frame.inclination_deg = elements['i']
                            frame.raan_deg = elements['raan']
                            frame.arg_periapsis_deg = elements['aop']
                            frame.true_anomaly_deg = elements['ta']
                            if elements['a'] > 0:
                                frame.orbital_period_hr = 2 * np.pi * np.sqrt(elements['a']**3 / mu_earth_km3s2) / 3600

                        # Compute gravitational acceleration
                        r_mag_km = np.linalg.norm(r_rel_km)
                        if r_mag_km > 0:
                            frame.accel_grav = mu_earth_km3s2 / (r_mag_km ** 2) * 1000  # m/s²
                            frame.accel_total = np.sqrt(frame.accel_thrust**2 + frame.accel_grav**2) if frame.accel_thrust else frame.accel_grav

                    except Exception:
                        pass  # Orbital elements computation failed

    def _compute_orbital_elements(self, r_km: np.ndarray, v_kms: np.ndarray, mu: float) -> Optional[Dict]:
        """Compute Keplerian orbital elements from state vectors."""
        r = np.asarray(r_km)
        v = np.asarray(v_kms)

        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)

        if r_mag == 0:
            return None

        # Specific angular momentum
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)

        if h_mag == 0:
            return None

        # Node vector
        n = np.cross([0, 0, 1], h)
        n_mag = np.linalg.norm(n)

        # Eccentricity vector
        e_vec = ((v_mag**2 - mu/r_mag) * r - np.dot(r, v) * v) / mu
        e = np.linalg.norm(e_vec)

        # Semi-major axis
        energy = v_mag**2 / 2 - mu / r_mag
        if abs(energy) < 1e-10:
            a = float('inf')  # Parabolic
        else:
            a = -mu / (2 * energy)

        # Inclination
        i = np.arccos(np.clip(h[2] / h_mag, -1, 1))

        # RAAN
        if n_mag > 1e-10:
            raan = np.arccos(np.clip(n[0] / n_mag, -1, 1))
            if n[1] < 0:
                raan = 2 * np.pi - raan
        else:
            raan = 0

        # Argument of periapsis
        if n_mag > 1e-10 and e > 1e-10:
            aop = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * e), -1, 1))
            if e_vec[2] < 0:
                aop = 2 * np.pi - aop
        else:
            aop = 0

        # True anomaly
        if e > 1e-10:
            ta = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1, 1))
            if np.dot(r, v) < 0:
                ta = 2 * np.pi - ta
        else:
            ta = 0

        return {
            'a': a,
            'e': e,
            'i': np.degrees(i),
            'raan': np.degrees(raan),
            'aop': np.degrees(aop),
            'ta': np.degrees(ta),
        }

    def _save_telemetry_csv(self, path: Path) -> None:
        """Save telemetry to CSV file."""
        if not self.telemetry:
            return

        fieldnames = list(asdict(self.telemetry[0]).keys())

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for frame in self.telemetry:
                writer.writerow(asdict(frame))

    def _save_events_json(self, path: Path) -> None:
        """Save events to JSON file."""
        events_data = [asdict(e) for e in self.events]
        with open(path, 'w') as f:
            json.dump(events_data, f, indent=2)

    def _save_summary_json(self, path: Path) -> None:
        """Save summary to JSON file."""
        if self.summary:
            with open(path, 'w') as f:
                json.dump(asdict(self.summary), f, indent=2)

    def _save_summary_text(self, path: Path) -> None:
        """Save human-readable mission report."""
        lines = []
        lines.append("=" * 78)
        lines.append(f"  MISSION REPORT: {self.mission_name.upper()}")
        lines.append("=" * 78)
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Mission ID: {self.mission_id}")
        lines.append("")

        if self.summary:
            s = self.summary

            lines.append("-" * 78)
            lines.append("  MISSION OVERVIEW")
            lines.append("-" * 78)
            lines.append(f"  Spacecraft:        {s.spacecraft_name}")
            lines.append(f"  Duration:          {s.duration_str} ({s.duration_days:.2f} days)")
            lines.append(f"  Status:            {'SUCCESS' if s.mission_success else 'INCOMPLETE'}")
            lines.append("")

            lines.append("-" * 78)
            lines.append("  MASS BUDGET")
            lines.append("-" * 78)
            lines.append(f"  Initial Mass:      {s.initial_mass_kg:,.0f} kg")
            lines.append(f"  Final Mass:        {s.final_mass_kg:,.0f} kg")
            lines.append(f"  Propellant Used:   {s.propellant_used_kg:,.0f} kg")
            if s.rcs_used_kg > 0:
                lines.append(f"  RCS Used:          {s.rcs_used_kg:,.0f} kg")
            lines.append("")

            lines.append("-" * 78)
            lines.append("  DELTA-V BUDGET")
            lines.append("-" * 78)
            lines.append(f"  Total Delta-V:     {s.total_delta_v_mps:,.1f} m/s ({s.total_delta_v_mps/1000:.3f} km/s)")
            lines.append("")
            if s.burns:
                lines.append("  Burns:")
                for burn in s.burns:
                    lines.append(f"    - {burn['name']}: {burn['delta_v_mps']:,.1f} m/s (prop: {burn['propellant_kg']:,.0f} kg)")
            lines.append("")

            lines.append("-" * 78)
            lines.append("  TRAJECTORY STATISTICS")
            lines.append("-" * 78)
            lines.append(f"  Max Velocity:      {s.max_velocity_kms:.2f} km/s")
            if s.max_altitude_km > 0:
                lines.append(f"  Max Altitude:      {s.max_altitude_km:,.0f} km")
                lines.append(f"  Min Altitude:      {s.min_altitude_km:,.0f} km")
            lines.append(f"  Sun Distance Range: {s.min_distance_sun_km/1e6:.2f} - {s.max_distance_sun_km/1e6:.2f} million km")
            lines.append("")

            if s.bodies_visited:
                lines.append("-" * 78)
                lines.append("  BODIES ENCOUNTERED")
                lines.append("-" * 78)
                for body in s.bodies_visited:
                    lines.append(f"    - {body}")
                lines.append("")

        lines.append("-" * 78)
        lines.append("  EVENT LOG")
        lines.append("-" * 78)
        for event in self.events:
            met_str = self._format_met(event.met_sec)
            lines.append(f"  [{met_str}] {event.event_type}: {event.description}")
        lines.append("")

        lines.append("=" * 78)
        lines.append("  END OF REPORT")
        lines.append("=" * 78)

        with open(path, 'w') as f:
            f.write('\n'.join(lines))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_mission_plots(
    fdr: 'FlightDataRecorder',
    save_dir: Optional[Path] = None,
    show: bool = False,
) -> Dict[str, str]:
    """
    Generate comprehensive mission plots from FDR telemetry data.

    Creates publication-quality matplotlib figures for mission review:
    - Mission overview (multi-panel)
    - Altitude profile
    - Velocity profile
    - Mass budget
    - Orbital elements
    - Burn profile

    Args:
        fdr: FlightDataRecorder with telemetry data
        save_dir: Directory to save plots (default: fdr.log_dir)
        show: Whether to display plots interactively

    Returns:
        Dict mapping plot name to file path
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator

    if not fdr.telemetry:
        logger.warning("No telemetry data for plot generation")
        return {}

    if save_dir is None:
        save_dir = fdr.log_dir

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_plots = {}

    # Extract data arrays from telemetry
    times_sec = np.array([f.met_sec for f in fdr.telemetry])
    times_min = times_sec / 60.0
    times_hr = times_sec / 3600.0

    altitudes = np.array([f.altitude_km for f in fdr.telemetry])
    velocities = np.array([f.speed_rel_kms for f in fdr.telemetry])
    masses = np.array([f.mass for f in fdr.telemetry])
    propellant = np.array([f.propellant_mass for f in fdr.telemetry])
    sma = np.array([f.sma_km for f in fdr.telemetry])
    eccentricity = np.array([f.eccentricity for f in fdr.telemetry])
    inclination = np.array([f.inclination_deg for f in fdr.telemetry])
    throttle = np.array([f.throttle for f in fdr.telemetry])
    thrust = np.array([f.thrust_n for f in fdr.telemetry])
    accel_grav = np.array([f.accel_grav for f in fdr.telemetry])
    solar_flux = np.array([f.solar_flux_w_m2 for f in fdr.telemetry])

    # Use hours or minutes based on mission duration
    max_time_hr = times_hr[-1] if len(times_hr) > 0 else 0
    if max_time_hr > 2:
        time_label = "Time [hours]"
        times = times_hr
    else:
        time_label = "Time [minutes]"
        times = times_min

    # Style configuration
    plt.style.use('dark_background')
    colors = {
        'altitude': '#00CED1',      # Cyan
        'velocity': '#FF6B6B',      # Coral
        'mass': '#4ECDC4',          # Teal
        'propellant': '#FFE66D',    # Yellow
        'sma': '#95E1D3',           # Mint
        'eccentricity': '#F38181',  # Pink
        'inclination': '#AA96DA',   # Purple
        'thrust': '#FF4757',        # Red
        'accel': '#2ED573',         # Green
        'solar': '#FFA502',         # Orange
    }

    # =========================================================================
    # 1. MISSION OVERVIEW (Multi-panel)
    # =========================================================================
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Mission Overview: {fdr.mission_name}", fontsize=14, fontweight='bold', color='white')

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Altitude
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, altitudes, color=colors['altitude'], linewidth=1.5)
    ax1.fill_between(times, 0, altitudes, alpha=0.3, color=colors['altitude'])
    ax1.set_ylabel('Altitude [km]', color='white')
    ax1.set_title('Altitude Profile', fontsize=10, color='white')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(times[0], times[-1])

    # Velocity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, velocities, color=colors['velocity'], linewidth=1.5)
    ax2.set_ylabel('Velocity [km/s]', color='white')
    ax2.set_title('Relative Velocity', fontsize=10, color='white')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(times[0], times[-1])

    # Mass
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(times, masses/1000, color=colors['mass'], linewidth=1.5, label='Total')
    ax3.plot(times, propellant/1000, color=colors['propellant'], linewidth=1.5, linestyle='--', label='Propellant')
    ax3.set_ylabel('Mass [tons]', color='white')
    ax3.set_title('Mass Budget', fontsize=10, color='white')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(times[0], times[-1])

    # Orbital Elements
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(times, sma/1000, color=colors['sma'], linewidth=1.5, label='SMA')
    ax4.set_ylabel('SMA [1000 km]', color=colors['sma'])
    ax4.tick_params(axis='y', colors=colors['sma'])
    ax4.set_title('Orbital Elements', fontsize=10, color='white')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(times[0], times[-1])

    ax4b = ax4.twinx()
    ax4b.plot(times, eccentricity, color=colors['eccentricity'], linewidth=1.5, linestyle='--', label='Eccentricity')
    ax4b.set_ylabel('Eccentricity', color=colors['eccentricity'])
    ax4b.tick_params(axis='y', colors=colors['eccentricity'])

    # Thrust Profile
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.fill_between(times, 0, throttle * 100, alpha=0.7, color=colors['thrust'])
    ax5.set_ylabel('Throttle [%]', color='white')
    ax5.set_xlabel(time_label)
    ax5.set_title('Thrust Profile', fontsize=10, color='white')
    ax5.set_ylim(0, 105)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(times[0], times[-1])

    # Acceleration
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(times, accel_grav, color=colors['accel'], linewidth=1.5)
    ax6.set_ylabel('Gravitational Accel [m/s²]', color='white')
    ax6.set_xlabel(time_label)
    ax6.set_title('Gravitational Acceleration', fontsize=10, color='white')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(times[0], times[-1])

    plt.tight_layout()
    overview_path = save_dir / 'mission_overview.png'
    plt.savefig(overview_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
    saved_plots['overview'] = str(overview_path)
    if show:
        plt.show()
    plt.close()

    # =========================================================================
    # 2. DETAILED ALTITUDE PROFILE
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, altitudes, color=colors['altitude'], linewidth=2)
    ax.fill_between(times, 0, altitudes, alpha=0.3, color=colors['altitude'])

    # Mark burn regions
    burn_mask = throttle > 0.1
    if np.any(burn_mask):
        burn_times = times[burn_mask]
        burn_alts = altitudes[burn_mask]
        ax.scatter(burn_times, burn_alts, c=colors['thrust'], s=10, alpha=0.5, label='Burn')

    ax.set_xlabel(time_label, fontsize=12)
    ax.set_ylabel('Altitude [km]', fontsize=12)
    ax.set_title(f'Altitude Profile - {fdr.mission_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(times[0], times[-1])

    # Add statistics annotation
    stats_text = f"Min: {np.min(altitudes):,.0f} km\nMax: {np.max(altitudes):,.0f} km"
    ax.annotate(stats_text, xy=(0.98, 0.95), xycoords='axes fraction',
                fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.tight_layout()
    alt_path = save_dir / 'altitude_profile.png'
    plt.savefig(alt_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
    saved_plots['altitude'] = str(alt_path)
    if show:
        plt.show()
    plt.close()

    # =========================================================================
    # 3. VELOCITY PROFILE
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, velocities, color=colors['velocity'], linewidth=2)

    # Mark burn regions
    if np.any(burn_mask):
        ax.scatter(burn_times, velocities[burn_mask], c=colors['thrust'], s=10, alpha=0.5, label='Burn')

    ax.set_xlabel(time_label, fontsize=12)
    ax.set_ylabel('Relative Velocity [km/s]', fontsize=12)
    ax.set_title(f'Velocity Profile - {fdr.mission_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(times[0], times[-1])

    # Add delta-V annotation if available
    if fdr.summary and fdr.summary.total_delta_v_mps > 0:
        dv_text = f"Total ΔV: {fdr.summary.total_delta_v_mps:,.0f} m/s"
        ax.annotate(dv_text, xy=(0.98, 0.95), xycoords='axes fraction',
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.tight_layout()
    vel_path = save_dir / 'velocity_profile.png'
    plt.savefig(vel_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
    saved_plots['velocity'] = str(vel_path)
    if show:
        plt.show()
    plt.close()

    # =========================================================================
    # 4. MASS BUDGET
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Total mass
    ax1.plot(times, masses/1000, color=colors['mass'], linewidth=2, label='Total Mass')
    ax1.fill_between(times, fdr.telemetry[0].dry_mass/1000, masses/1000,
                     alpha=0.3, color=colors['propellant'], label='Propellant')
    ax1.axhline(y=fdr.telemetry[0].dry_mass/1000, color='white', linestyle='--',
                alpha=0.5, label='Dry Mass')
    ax1.set_ylabel('Mass [tons]', fontsize=12)
    ax1.set_title(f'Mass Budget - {fdr.mission_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(times[0], times[-1])

    # Propellant consumption rate
    if len(times) > 1:
        dt = np.diff(times) * 3600  # hours to seconds
        dm = -np.diff(masses)
        mdot = dm / dt  # kg/s
        mdot = np.append(mdot, mdot[-1])  # Pad to match length

        ax2.plot(times, mdot, color=colors['thrust'], linewidth=1.5)
        ax2.fill_between(times, 0, mdot, alpha=0.3, color=colors['thrust'])
        ax2.set_ylabel('Mass Flow Rate [kg/s]', fontsize=12)
        ax2.set_xlabel(time_label, fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(times[0], times[-1])

    # Add summary annotation
    if fdr.summary:
        prop_text = f"Propellant Used: {fdr.summary.propellant_used_kg:,.0f} kg"
        ax1.annotate(prop_text, xy=(0.98, 0.05), xycoords='axes fraction',
                     fontsize=10, ha='right', va='bottom',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.tight_layout()
    mass_path = save_dir / 'mass_budget.png'
    plt.savefig(mass_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
    saved_plots['mass'] = str(mass_path)
    if show:
        plt.show()
    plt.close()

    # =========================================================================
    # 5. ORBITAL ELEMENTS
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # SMA
    axes[0].plot(times, sma/1000, color=colors['sma'], linewidth=2)
    axes[0].set_ylabel('Semi-Major Axis [1000 km]', fontsize=11)
    axes[0].set_title(f'Orbital Elements - {fdr.mission_name}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(times[0], times[-1])

    # Eccentricity
    axes[1].plot(times, eccentricity, color=colors['eccentricity'], linewidth=2)
    axes[1].set_ylabel('Eccentricity', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(times[0], times[-1])

    # Inclination
    axes[2].plot(times, inclination, color=colors['inclination'], linewidth=2)
    axes[2].set_ylabel('Inclination [deg]', fontsize=11)
    axes[2].set_xlabel(time_label, fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(times[0], times[-1])

    plt.tight_layout()
    orbit_path = save_dir / 'orbital_elements.png'
    plt.savefig(orbit_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
    saved_plots['orbital_elements'] = str(orbit_path)
    if show:
        plt.show()
    plt.close()

    # =========================================================================
    # 6. BURN PROFILE (if burns occurred)
    # =========================================================================
    if np.any(burn_mask):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Throttle
        ax1.fill_between(times, 0, throttle * 100, color=colors['thrust'], alpha=0.7)
        ax1.set_ylabel('Throttle [%]', fontsize=12)
        ax1.set_title(f'Burn Profile - {fdr.mission_name}', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(times[0], times[-1])

        # Thrust
        ax2.plot(times, thrust/1000, color=colors['thrust'], linewidth=2)
        ax2.fill_between(times, 0, thrust/1000, alpha=0.3, color=colors['thrust'])
        ax2.set_ylabel('Thrust [kN]', fontsize=12)
        ax2.set_xlabel(time_label, fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(times[0], times[-1])

        plt.tight_layout()
        burn_path = save_dir / 'burn_profile.png'
        plt.savefig(burn_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
        saved_plots['burn'] = str(burn_path)
        if show:
            plt.show()
        plt.close()

    logger.info(f"FDR: Generated {len(saved_plots)} mission plots in {save_dir}")
    return saved_plots


def create_fdr(
    mission_name: str,
    ship = None,
    start_time: float = 0.0,
    record_interval: float = 0.001,
) -> FlightDataRecorder:
    """
    Create and configure a Flight Data Recorder.

    Args:
        mission_name: Name for this mission
        ship: Spacecraft object to track
        start_time: Mission start time (days from epoch)
        record_interval: Minimum time between frames (days)

    Returns:
        Configured FlightDataRecorder
    """
    fdr = FlightDataRecorder(
        mission_name=mission_name,
        record_interval=record_interval,
    )

    if ship is not None:
        fdr.set_spacecraft(ship)

    fdr.set_start_time(start_time)

    return fdr
