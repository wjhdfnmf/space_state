# base/star_console.py
"""
Console Display - Real-time Flight Metrics Display

Provides a clean, forward-facing console display for spacecraft telemetry
during mission simulation. Designed to look like a spacecraft flight computer
display with key metrics updated in real-time.

Features:
- Clean ASCII-based display
- Configurable update rate
- Multiple display modes (full, compact, minimal)
- Color support (when available)
- Mission phase tracking
"""

import sys
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import numpy as np

from .star_logger import logger


# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

class DisplayMode(Enum):
    """Display mode options."""
    FULL = "full"           # All metrics
    COMPACT = "compact"     # Key metrics only
    MINIMAL = "minimal"     # Single line
    SILENT = "silent"       # No display


class DisplayTheme(Enum):
    """Color themes for display."""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    NASA = "nasa"


# ANSI color codes (for terminals that support them)
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Foreground
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright foreground
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_CYAN = '\033[96m'

    # Background
    BG_BLACK = '\033[40m'
    BG_BLUE = '\033[44m'


def supports_color() -> bool:
    """Check if terminal supports ANSI colors."""
    if os.getenv('NO_COLOR'):
        return False
    if os.getenv('FORCE_COLOR'):
        return True
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


# ============================================================================
# DISPLAY DATA STRUCTURES
# ============================================================================

@dataclass
class DisplayState:
    """Current state to display."""
    # Time
    met_sec: float = 0.0
    t_days: float = 0.0

    # Position
    altitude_km: float = 0.0
    distance_km: float = 0.0
    r_mag_km: float = 0.0

    # Velocity
    velocity_kms: float = 0.0
    velocity_rel_kms: float = 0.0

    # Orientation
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    omega_degs: float = 0.0

    # Mass
    mass_kg: float = 0.0
    propellant_kg: float = 0.0
    rcs_kg: float = 0.0

    # Propulsion
    thrust_n: float = 0.0
    throttle_pct: float = 0.0
    isp_s: float = 0.0
    accel_mps2: float = 0.0

    # Guidance
    delta_v_mps: float = 0.0
    burn_time_sec: float = 0.0
    vgo_mps: float = 0.0
    tgo_sec: float = 0.0

    # Body distances
    dist_sun_km: float = 0.0
    dist_earth_km: float = 0.0
    dist_moon_km: float = 0.0
    dist_mars_km: float = 0.0
    dist_target_km: float = 0.0

    # Environment
    eclipse: bool = False
    solar_flux: float = 0.0
    comm_delay_sec: float = 0.0

    # Status
    primary_body: str = "earth"
    phase: str = ""
    status: str = "NOMINAL"
    event: str = ""


# ============================================================================
# CONSOLE DISPLAY CLASS
# ============================================================================

class ConsoleDisplay:
    """
    Real-time console display for spacecraft flight metrics.

    Usage:
        display = ConsoleDisplay(mode=DisplayMode.FULL)

        # During simulation:
        display.update(state, ship, phase="TMI Burn")

        # Or use the flight computer style:
        display.show_flight_computer(state)
    """

    def __init__(
        self,
        mode: DisplayMode = DisplayMode.FULL,
        use_color: bool = True,
        update_interval: float = 0.1,  # seconds
        mission_name: str = "",
        spacecraft_name: str = "",
    ):
        """
        Initialize the console display.

        Args:
            mode: Display mode (FULL, COMPACT, MINIMAL, SILENT)
            use_color: Enable ANSI colors
            update_interval: Minimum time between display updates (seconds)
            mission_name: Mission name to show
            spacecraft_name: Spacecraft name to show
        """
        self.mode = mode
        self.use_color = use_color and supports_color()
        self.update_interval = update_interval
        self.mission_name = mission_name
        self.spacecraft_name = spacecraft_name

        self._last_update = 0.0
        self._frame_count = 0
        self._start_time = time.time()

        # Constants
        self._au_to_km = 1.495978707e8
        self._day_to_sec = 86400.0

        # Display dimensions
        self.width = 78

    def _get_colors(self):
        """Get color object (real or dummy)."""
        if self.use_color:
            return Colors
        else:
            # Return a dummy object with empty strings for all color attributes
            class NoColors:
                RESET = ''
                BOLD = ''
                DIM = ''
                BLACK = ''
                RED = ''
                GREEN = ''
                YELLOW = ''
                BLUE = ''
                MAGENTA = ''
                CYAN = ''
                WHITE = ''
                BRIGHT_RED = ''
                BRIGHT_GREEN = ''
                BRIGHT_YELLOW = ''
                BRIGHT_BLUE = ''
                BRIGHT_CYAN = ''
                BG_BLACK = ''
                BG_BLUE = ''
            return NoColors

    def update(
        self,
        t: float,
        state: np.ndarray,
        ship = None,
        phase: str = "",
        status: str = "NOMINAL",
        event: str = "",
        bodies: Optional[dict] = None,
        force: bool = False,
    ) -> None:
        """
        Update the display with current telemetry.

        Args:
            t: Current time (days from epoch)
            state: State vector [r, v, m, q, omega]
            ship: Spacecraft object
            phase: Current mission phase name
            status: Status string (NOMINAL, WARNING, CRITICAL)
            event: Current event name
            bodies: Dict of celestial bodies
            force: Force update regardless of interval
        """
        if self.mode == DisplayMode.SILENT:
            return

        # Check update interval
        now = time.time()
        if not force and (now - self._last_update) < self.update_interval:
            return

        self._last_update = now
        self._frame_count += 1

        # Build display state
        ds = self._build_display_state(t, state, ship, phase, status, event, bodies)

        # Render based on mode
        if self.mode == DisplayMode.MINIMAL:
            self._render_minimal(ds)
        elif self.mode == DisplayMode.COMPACT:
            self._render_compact(ds)
        else:
            self._render_full(ds)

    def show_header(self) -> None:
        """Display mission header."""
        if self.mode == DisplayMode.SILENT:
            return

        c = self._get_colors()

        print()
        print(f"{c.BOLD}{c.CYAN}{'=' * self.width}{c.RESET}")
        if self.mission_name:
            title = f"  MISSION: {self.mission_name.upper()}"
            if self.spacecraft_name:
                title += f"  |  SPACECRAFT: {self.spacecraft_name.upper()}"
            print(f"{c.BOLD}{c.WHITE}{title}{c.RESET}")
        print(f"{c.BOLD}{c.CYAN}{'=' * self.width}{c.RESET}")
        print()

    def show_phase_banner(self, phase: str, description: str = "") -> None:
        """Display a phase transition banner."""
        if self.mode == DisplayMode.SILENT:
            return

        c = self._get_colors()

        print()
        print(f"{c.BOLD}{c.YELLOW}{'-' * self.width}{c.RESET}")
        print(f"{c.BOLD}{c.YELLOW}  >>> {phase.upper()}{c.RESET}")
        if description:
            print(f"{c.DIM}  {description}{c.RESET}")
        print(f"{c.BOLD}{c.YELLOW}{'-' * self.width}{c.RESET}")
        print()

    def show_event(self, event_name: str, details: str = "") -> None:
        """Display an event notification."""
        if self.mode == DisplayMode.SILENT:
            return

        c = self._get_colors()

        print(f"{c.BRIGHT_GREEN}[EVENT] {event_name}{c.RESET}", end='')
        if details:
            print(f" - {details}")
        else:
            print()

    def show_warning(self, message: str) -> None:
        """Display a warning message."""
        if self.mode == DisplayMode.SILENT:
            return

        c = self._get_colors()
        print(f"{c.BRIGHT_YELLOW}[WARNING] {message}{c.RESET}")

    def show_critical(self, message: str) -> None:
        """Display a critical alert."""
        if self.mode == DisplayMode.SILENT:
            return

        c = self._get_colors()
        print(f"{c.BRIGHT_RED}[CRITICAL] {message}{c.RESET}")

    def show_summary(self, summary_dict: Dict[str, Any]) -> None:
        """Display a summary table."""
        if self.mode == DisplayMode.SILENT:
            return

        c = self._get_colors()

        print()
        print(f"{c.BOLD}{'-' * self.width}{c.RESET}")

        for key, value in summary_dict.items():
            if isinstance(value, float):
                print(f"  {key:<35} {value:>12.2f}")
            else:
                print(f"  {key:<35} {str(value):>12}")

        print(f"{c.BOLD}{'-' * self.width}{c.RESET}")
        print()

    # =========================================================================
    # FLIGHT COMPUTER STYLE DISPLAY
    # =========================================================================

    def show_flight_computer(self, ds: DisplayState) -> None:
        """
        Display flight computer style telemetry panel.

        This creates a visually appealing, aerospace-style display.
        """
        if self.mode == DisplayMode.SILENT:
            return

        c = self._get_colors()

        # Clear previous lines (simple approach - print blank lines)
        # In a real terminal app, we'd use cursor control

        # Format MET
        met = self._format_met(ds.met_sec)

        # Status color
        status_color = c.GREEN
        if ds.status == "WARNING":
            status_color = c.YELLOW
        elif ds.status == "CRITICAL":
            status_color = c.RED

        # Build display
        lines = []

        # Header line
        header = f" MET {met}  |  {ds.phase.upper():<20}  |  {status_color}{ds.status:>10}{c.RESET} "
        lines.append(f"{c.BOLD}{c.CYAN}|{header:^74}|{c.RESET}")
        lines.append(f"{c.DIM}+{'-' * 76}+{c.RESET}")

        # Main telemetry grid
        col1 = f" ALT  {self._fmt_dist(ds.altitude_km):>12}"
        col2 = f" VEL  {ds.velocity_kms:>9.2f} km/s"
        col3 = f" MASS {ds.mass_kg:>10,.0f} kg"
        lines.append(f"{c.CYAN}|{col1}  |{col2}  |{col3}  |{c.RESET}")

        # Attitude and propulsion
        col1 = f" R/P/Y {ds.roll_deg:>5.1f}/{ds.pitch_deg:>5.1f}/{ds.yaw_deg:>5.1f}"
        col2 = f" THR  {ds.throttle_pct:>6.1f}% {ds.thrust_n/1000:>6.0f}kN"
        col3 = f" PROP {ds.propellant_kg:>10,.0f} kg"
        lines.append(f"{c.CYAN}|{col1:>15}  |{col2}  |{col3}  |{c.RESET}")

        lines.append(f"{c.DIM}+{'-' * 76}+{c.RESET}")

        # Distances
        dist_line = f" SUN {ds.dist_sun_km/1e6:>6.2f}Mkm | EARTH {ds.dist_earth_km/1e6:>6.2f}Mkm"
        if ds.dist_mars_km > 0:
            dist_line += f" | MARS {ds.dist_mars_km/1e6:>6.2f}Mkm"
        lines.append(f"{c.DIM}|{dist_line:<74}|{c.RESET}")

        # Comm delay
        if ds.comm_delay_sec > 0:
            comm_line = f" COMM DELAY: {ds.comm_delay_sec:>6.1f}s ({ds.comm_delay_sec/60:.1f} min)"
            if ds.eclipse:
                comm_line += f"  |  ECLIPSE"
            lines.append(f"{c.DIM}|{comm_line:<74}|{c.RESET}")

        lines.append(f"{c.DIM}+{'-' * 76}+{c.RESET}")

        # Print all lines
        for line in lines:
            print(line)

    # =========================================================================
    # INTERNAL RENDERING METHODS
    # =========================================================================

    def _build_display_state(
        self,
        t: float,
        state: np.ndarray,
        ship,
        phase: str,
        status: str,
        event: str,
        bodies: Optional[dict],
    ) -> DisplayState:
        """Build DisplayState from raw data."""
        ds = DisplayState()

        # Time
        ds.t_days = t
        ds.met_sec = t * self._day_to_sec  # Simple MET (assumes t=0 start)

        # Position
        r_au = state[:3]
        ds.r_mag_km = np.linalg.norm(r_au) * self._au_to_km

        # Velocity
        v_aud = state[3:6]
        ds.velocity_kms = np.linalg.norm(v_aud) * self._au_to_km / self._day_to_sec

        # Mass
        ds.mass_kg = state[6]

        # Attitude
        q = state[7:11]
        roll, pitch, yaw = self._quaternion_to_euler(q)
        ds.roll_deg = np.degrees(roll)
        ds.pitch_deg = np.degrees(pitch)
        ds.yaw_deg = np.degrees(yaw)

        # Angular velocity
        omega = state[11:14]
        ds.omega_degs = np.linalg.norm(omega) * 180.0 / (np.pi * self._day_to_sec)

        # Ship data
        if ship is not None:
            ds.propellant_kg = ship.propellant_mass
            if hasattr(ship, '_rcs_propellant_mass'):
                ds.rcs_kg = ship._rcs_propellant_mass
            ds.thrust_n = ship.thrust
            ds.throttle_pct = ship.throttle * 100
            ds.isp_s = ship.isp
            if ds.mass_kg > 0:
                ds.accel_mps2 = ds.thrust_n / ds.mass_kg

        # Body distances
        if bodies is not None:
            ds = self._compute_body_distances(ds, state, t, bodies)

        # Communication delay
        ds.comm_delay_sec = ds.dist_earth_km / 299792.458

        # Status
        ds.phase = phase
        ds.status = status
        ds.event = event

        return ds

    def _compute_body_distances(
        self,
        ds: DisplayState,
        state: np.ndarray,
        t: float,
        bodies: dict,
    ) -> DisplayState:
        """Compute distances to celestial bodies."""
        try:
            from .star_constants import epoch_jd, ts

            r_sc = state[:3]
            time_obj = ts.tt_jd(epoch_jd + t)

            # Sun distance
            ds.dist_sun_km = np.linalg.norm(r_sc) * self._au_to_km

            # Earth
            if 'earth' in bodies:
                r_earth = bodies['earth']['skyfield'].at(time_obj).position.au
                ds.dist_earth_km = np.linalg.norm(r_sc - r_earth) * self._au_to_km
                r_earth_km = bodies['earth']['radius'] * self._au_to_km
                ds.altitude_km = ds.dist_earth_km - r_earth_km
                ds.distance_km = ds.dist_earth_km
                ds.primary_body = "earth"

            # Moon
            if 'moon' in bodies:
                r_moon = bodies['moon']['skyfield'].at(time_obj).position.au
                ds.dist_moon_km = np.linalg.norm(r_sc - r_moon) * self._au_to_km

            # Mars
            if 'mars' in bodies:
                r_mars = bodies['mars']['skyfield'].at(time_obj).position.au
                ds.dist_mars_km = np.linalg.norm(r_sc - r_mars) * self._au_to_km

        except Exception as e:
            logger.debug(f"Could not compute body distances: {e}")

        return ds

    def _render_full(self, ds: DisplayState) -> None:
        """Render full display mode."""
        self.show_flight_computer(ds)

    def _render_compact(self, ds: DisplayState) -> None:
        """Render compact display mode."""
        c = self._get_colors()
        met = self._format_met(ds.met_sec)

        line = (
            f"{c.CYAN}[{met}]{c.RESET} "
            f"ALT:{self._fmt_dist(ds.altitude_km):>10} "
            f"V:{ds.velocity_kms:>7.2f}km/s "
            f"M:{ds.mass_kg:>8,.0f}kg "
            f"THR:{ds.throttle_pct:>5.1f}%"
        )

        if ds.phase:
            line += f" | {c.YELLOW}{ds.phase}{c.RESET}"

        print(line)

    def _render_minimal(self, ds: DisplayState) -> None:
        """Render minimal (single line) display mode."""
        c = self._get_colors()
        met = self._format_met(ds.met_sec)

        # Single line with key metrics
        print(
            f"\r{c.CYAN}T+{met}{c.RESET} "
            f"Alt:{self._fmt_dist_short(ds.altitude_km)} "
            f"V:{ds.velocity_kms:.1f}km/s "
            f"M:{ds.mass_kg:.0f}kg",
            end='', flush=True
        )

    # =========================================================================
    # FORMATTING HELPERS
    # =========================================================================

    def _format_met(self, met_sec: float) -> str:
        """Format MET in HH:MM:SS."""
        hours = int(abs(met_sec) // 3600)
        minutes = int((abs(met_sec) % 3600) // 60)
        seconds = int(abs(met_sec) % 60)
        sign = '-' if met_sec < 0 else ''
        return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _fmt_dist(self, km: float) -> str:
        """Format distance with appropriate units."""
        if abs(km) >= 1e9:
            return f"{km/1e9:.2f} Gkm"
        elif abs(km) >= 1e6:
            return f"{km/1e6:.2f} Mkm"
        elif abs(km) >= 1e3:
            return f"{km/1e3:.1f} kkm"
        else:
            return f"{km:.0f} km"

    def _fmt_dist_short(self, km: float) -> str:
        """Format distance with short units."""
        if abs(km) >= 1e6:
            return f"{km/1e6:.1f}M"
        elif abs(km) >= 1e3:
            return f"{km/1e3:.0f}k"
        else:
            return f"{km:.0f}"

    def _quaternion_to_euler(self, q: np.ndarray) -> tuple:
        """Convert quaternion to Euler angles."""
        q0, q1, q2, q3 = q

        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q0 * q2 - q3 * q1)
        pitch = np.arcsin(np.clip(sinp, -1, 1))

        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


# ============================================================================
# MISSION CONTROLLER - COMBINES DISPLAY AND LOGGING
# ============================================================================

class MissionController:
    """
    Combined controller for mission display and data logging.

    Coordinates between:
    - ConsoleDisplay: Real-time visual feedback
    - FlightDataRecorder: Data logging and persistence

    Usage:
        controller = MissionController(
            mission_name="Mars Transfer",
            spacecraft_name="Starship"
        )
        controller.set_spacecraft(ship)

        # During propagation:
        controller.update(t, state, phase="TMI Burn")

        # At mission end:
        controller.finalize()
    """

    def __init__(
        self,
        mission_name: str = "Unnamed Mission",
        spacecraft_name: str = "",
        display_mode: DisplayMode = DisplayMode.COMPACT,
        enable_fdr: bool = True,
        display_interval: float = 1.0,  # seconds
        fdr_interval: float = 0.001,    # days (~86 seconds)
    ):
        """Initialize the mission controller."""
        from .star_fdr import FlightDataRecorder

        self.mission_name = mission_name
        self.spacecraft_name = spacecraft_name
        self._spacecraft = None
        self._bodies = None
        self._start_time: Optional[float] = None

        # Initialize display
        self.display = ConsoleDisplay(
            mode=display_mode,
            update_interval=display_interval,
            mission_name=mission_name,
            spacecraft_name=spacecraft_name,
        )

        # Initialize FDR if enabled
        self.fdr: Optional[FlightDataRecorder] = None
        if enable_fdr:
            self.fdr = FlightDataRecorder(
                mission_name=mission_name,
                record_interval=fdr_interval,
            )

        self._current_phase = ""

    def set_spacecraft(self, ship) -> None:
        """Set the spacecraft being tracked."""
        self._spacecraft = ship
        self.spacecraft_name = ship._name if hasattr(ship, '_name') else "Unknown"
        self.display.spacecraft_name = self.spacecraft_name
        if self.fdr:
            self.fdr.set_spacecraft(ship)

    def set_bodies(self, bodies: dict) -> None:
        """Set celestial bodies for distance calculations."""
        self._bodies = bodies

    def set_start_time(self, t: float) -> None:
        """Set mission start time."""
        self._start_time = t
        if self.fdr:
            self.fdr.set_start_time(t)

    def begin_mission(self) -> None:
        """Display mission start header."""
        self.display.show_header()
        if self.fdr:
            self.fdr.record_event(
                self._start_time or 0.0,
                "mission_start",
                f"Mission '{self.mission_name}' started"
            )

    def begin_phase(self, phase: str, description: str = "") -> None:
        """Begin a new mission phase."""
        self._current_phase = phase
        self.display.show_phase_banner(phase, description)
        if self.fdr:
            self.fdr.record_event(
                self._start_time or 0.0,
                "phase_start",
                f"{phase}: {description}" if description else phase
            )

    def update(
        self,
        t: float,
        state: np.ndarray,
        phase: str = "",
        status: str = "NOMINAL",
        event: str = "",
        force: bool = False,
    ) -> None:
        """
        Update display and recording with current state.

        Args:
            t: Current time (days from epoch)
            state: State vector
            phase: Current phase name (optional, uses stored)
            status: Status string
            event: Current event name
            force: Force update regardless of interval
        """
        if phase:
            self._current_phase = phase

        # Update display
        self.display.update(
            t=t,
            state=state,
            ship=self._spacecraft,
            phase=self._current_phase,
            status=status,
            event=event,
            bodies=self._bodies,
            force=force,
        )

        # Record to FDR
        if self.fdr:
            self.fdr.record_frame(
                t=t,
                state=state,
                ship=self._spacecraft,
                bodies=self._bodies,
                force=force,
            )

    def start_burn(self, t: float, burn_name: str) -> None:
        """Mark the start of a burn."""
        self.display.show_event("BURN START", burn_name)
        if self.fdr:
            self.fdr.start_burn(t, burn_name)

    def end_burn(self, t: float, burn_name: str) -> None:
        """Mark the end of a burn."""
        self.display.show_event("BURN COMPLETE", burn_name)
        if self.fdr:
            self.fdr.end_burn(t, burn_name)

    def log_event(
        self,
        t: float,
        event_type: str,
        description: str,
        **kwargs
    ) -> None:
        """Log a mission event."""
        self.display.show_event(event_type, description)
        if self.fdr:
            self.fdr.record_event(t, event_type, description, **kwargs)

    def finalize(self, success: bool = True, objectives: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Finalize the mission and save data.

        Returns:
            Dict with summary and saved file paths, or None if no FDR
        """
        result = {}

        if self.fdr:
            summary = self.fdr.finalize(success, objectives)
            saved_files = self.fdr.save()

            result['summary'] = summary
            result['files'] = saved_files

            # Display summary
            self.display.show_summary({
                'Duration': f"{summary.duration_str}",
                'Total Delta-V': f"{summary.total_delta_v_mps:.1f} m/s",
                'Propellant Used': f"{summary.propellant_used_kg:,.0f} kg",
                'Final Mass': f"{summary.final_mass_kg:,.0f} kg",
            })

            logger.info(f"Mission data saved to: {self.fdr.log_dir}")

        return result if result else None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_mission_controller(
    mission_name: str,
    ship = None,
    bodies: dict = None,
    display_mode: DisplayMode = DisplayMode.COMPACT,
    enable_fdr: bool = True,
) -> MissionController:
    """
    Create and configure a mission controller.

    Args:
        mission_name: Name for this mission
        ship: Spacecraft object
        bodies: Dict of celestial bodies
        display_mode: Display mode
        enable_fdr: Enable flight data recording

    Returns:
        Configured MissionController
    """
    controller = MissionController(
        mission_name=mission_name,
        display_mode=display_mode,
        enable_fdr=enable_fdr,
    )

    if ship:
        controller.set_spacecraft(ship)
    if bodies:
        controller.set_bodies(bodies)

    return controller
