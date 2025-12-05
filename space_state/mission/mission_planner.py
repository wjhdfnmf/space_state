# mission/mission_planner.py
"""
Mission Planning Framework

Provides plug-and-play building blocks for mission design:
- Mission phases (launch, transfer, orbit insertion, landing)
- Pre-built maneuvers (Hohmann, bi-elliptic, plane change)
- Mission templates (Earth-Moon, Mars transfer, LEO ops)
- Delta-V budgets and mass analysis
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple
from enum import Enum
import numpy as np

from base.star_constants import BODIES, AU_TO_KM, DAY_TO_SEC
from base.star_soi import get_body_position, get_body_velocity
from base.star_ship import Spacecraft, Thruster
from base.star_propagator import Propagator
from base.star_structures import PropagationResult
from base.star_states import get_circular_state
from guidance.lambert import lambert_solve
from base.star_logger import logger


# ------------------- ENUMS -------------------

class PhaseType(Enum):
    """Types of mission phases."""
    LAUNCH = "launch"
    ORBIT_INSERTION = "orbit_insertion"
    COAST = "coast"
    TRANSFER = "transfer"
    LANDING = "landing"
    RENDEZVOUS = "rendezvous"
    STATION_KEEPING = "station_keeping"
    DEORBIT = "deorbit"


class ManeuverType(Enum):
    """Types of orbital maneuvers."""
    HOHMANN = "hohmann"
    BI_ELLIPTIC = "bi_elliptic"
    PLANE_CHANGE = "plane_change"
    PHASING = "phasing"
    CIRCULARIZE = "circularize"
    ESCAPE = "escape"
    CAPTURE = "capture"


# ------------------- DATA CLASSES -------------------

@dataclass
class DeltaVBudget:
    """Delta-V budget for a mission or maneuver."""
    name: str
    delta_v_kmps: float
    margin_percent: float = 10.0  # Contingency margin

    @property
    def total_kmps(self) -> float:
        """Total delta-V including margin."""
        return self.delta_v_kmps * (1 + self.margin_percent / 100)

    def fuel_mass(self, dry_mass: float, isp_s: float) -> float:
        """Calculate required fuel mass using Tsiolkovsky equation."""
        g0 = 9.80665
        dv_mps = self.total_kmps * 1000
        mass_ratio = np.exp(dv_mps / (isp_s * g0))
        return dry_mass * (mass_ratio - 1)


@dataclass
class MissionPhase:
    """A phase of a mission."""
    name: str
    phase_type: PhaseType
    duration_days: Optional[float] = None
    delta_v_budget: Optional[DeltaVBudget] = None
    start_body: Optional[str] = None
    end_body: Optional[str] = None
    start_altitude_km: Optional[float] = None
    end_altitude_km: Optional[float] = None
    notes: str = ""

    def __repr__(self):
        dv = f", Δv={self.delta_v_budget.total_kmps:.2f} km/s" if self.delta_v_budget else ""
        dur = f", {self.duration_days:.1f}d" if self.duration_days else ""
        return f"Phase({self.name}{dv}{dur})"


@dataclass
class MissionPlan:
    """Complete mission plan with multiple phases."""
    name: str
    phases: List[MissionPhase] = field(default_factory=list)
    spacecraft: Optional[Spacecraft] = None

    @property
    def total_delta_v(self) -> float:
        """Total mission delta-V in km/s."""
        return sum(p.delta_v_budget.total_kmps for p in self.phases if p.delta_v_budget)

    @property
    def total_duration_days(self) -> float:
        """Total mission duration in days."""
        return sum(p.duration_days for p in self.phases if p.duration_days)

    def add_phase(self, phase: MissionPhase) -> 'MissionPlan':
        """Add a phase to the mission."""
        self.phases.append(phase)
        return self

    def fuel_requirement(self, payload_mass: float, isp_s: float) -> Dict[str, float]:
        """Calculate fuel requirements for entire mission."""
        # Work backwards from payload
        current_mass = payload_mass
        fuel_per_phase = {}

        for phase in reversed(self.phases):
            if phase.delta_v_budget:
                fuel = phase.delta_v_budget.fuel_mass(current_mass, isp_s)
                fuel_per_phase[phase.name] = fuel
                current_mass += fuel

        return {
            'phase_fuel': fuel_per_phase,
            'total_fuel': sum(fuel_per_phase.values()),
            'initial_mass': current_mass,
            'payload_mass': payload_mass,
            'mass_ratio': current_mass / payload_mass,
        }

    def summary(self) -> str:
        """Generate mission summary."""
        lines = [
            f"Mission: {self.name}",
            "=" * 50,
            "",
            "Phases:",
        ]

        for i, phase in enumerate(self.phases, 1):
            lines.append(f"  {i}. {phase}")

        lines.extend([
            "",
            f"Total Delta-V: {self.total_delta_v:.2f} km/s",
            f"Total Duration: {self.total_duration_days:.1f} days",
        ])

        return "\n".join(lines)


# ------------------- MANEUVER CALCULATORS -------------------

def hohmann_transfer(
    r1_km: float,
    r2_km: float,
    mu_km3_s2: float,
) -> Tuple[float, float, float]:
    """
    Calculate Hohmann transfer delta-Vs.

    Args:
        r1_km: Initial orbit radius (km)
        r2_km: Final orbit radius (km)
        mu_km3_s2: Gravitational parameter (km³/s²)

    Returns:
        (dv1_kmps, dv2_kmps, tof_days): Delta-Vs and transfer time
    """
    a_transfer = (r1_km + r2_km) / 2

    # Circular velocities
    v1_circ = np.sqrt(mu_km3_s2 / r1_km)
    v2_circ = np.sqrt(mu_km3_s2 / r2_km)

    # Transfer orbit velocities
    v1_transfer = np.sqrt(mu_km3_s2 * (2/r1_km - 1/a_transfer))
    v2_transfer = np.sqrt(mu_km3_s2 * (2/r2_km - 1/a_transfer))

    # Delta-Vs
    dv1 = abs(v1_transfer - v1_circ)
    dv2 = abs(v2_circ - v2_transfer)

    # Transfer time (half period)
    tof_s = np.pi * np.sqrt(a_transfer**3 / mu_km3_s2)
    tof_days = tof_s / 86400

    return dv1, dv2, tof_days


def escape_velocity(r_km: float, mu_km3_s2: float) -> float:
    """Calculate escape velocity at given radius."""
    return np.sqrt(2 * mu_km3_s2 / r_km)


def circular_velocity(r_km: float, mu_km3_s2: float) -> float:
    """Calculate circular orbit velocity at given radius."""
    return np.sqrt(mu_km3_s2 / r_km)


def plane_change_dv(v_kmps: float, angle_deg: float) -> float:
    """Calculate delta-V for plane change maneuver."""
    angle_rad = np.radians(angle_deg)
    return 2 * v_kmps * np.sin(angle_rad / 2)


# ------------------- MISSION TEMPLATES -------------------

def create_earth_moon_mission(
    leo_altitude_km: float = 400,
    lunar_orbit_km: float = 100,
    margin_percent: float = 10.0,
) -> MissionPlan:
    """
    Create Earth-Moon mission plan.

    Standard Apollo-style free-return trajectory.
    """
    plan = MissionPlan(name="Earth-Moon Transfer")

    mu_earth = BODIES['earth']['mu_km3_s2']
    mu_moon = BODIES['moon']['mu_km3_s2']

    earth_radius = 6371
    moon_radius = 1737
    moon_distance = 384400

    r_leo = earth_radius + leo_altitude_km
    r_lunar = moon_radius + lunar_orbit_km

    # Phase 1: LEO parking orbit
    plan.add_phase(MissionPhase(
        name="LEO Parking Orbit",
        phase_type=PhaseType.ORBIT_INSERTION,
        duration_days=0.1,
        start_body='earth',
        start_altitude_km=leo_altitude_km,
        notes="Circular parking orbit",
    ))

    # Phase 2: TLI (Trans-Lunar Injection)
    # Approximate as escape from Earth plus trajectory adjustment
    v_leo = circular_velocity(r_leo, mu_earth)
    v_tli = np.sqrt(mu_earth * (2/r_leo - 2/(r_leo + moon_distance)))
    dv_tli = v_tli - v_leo

    plan.add_phase(MissionPhase(
        name="Trans-Lunar Injection",
        phase_type=PhaseType.TRANSFER,
        delta_v_budget=DeltaVBudget("TLI", dv_tli, margin_percent),
        duration_days=0.01,
        start_body='earth',
        end_body='moon',
        notes="Single impulse to lunar transfer orbit",
    ))

    # Phase 3: Lunar coast
    # Transfer time approximately 3 days
    plan.add_phase(MissionPhase(
        name="Lunar Transit",
        phase_type=PhaseType.COAST,
        duration_days=3.0,
        notes="Coast to Moon",
    ))

    # Phase 4: LOI (Lunar Orbit Insertion)
    # Arrive at Moon with ~0.8-1.0 km/s excess velocity
    v_lunar = circular_velocity(r_lunar, mu_moon)
    dv_loi = 0.9  # Approximate capture delta-V

    plan.add_phase(MissionPhase(
        name="Lunar Orbit Insertion",
        phase_type=PhaseType.ORBIT_INSERTION,
        delta_v_budget=DeltaVBudget("LOI", dv_loi, margin_percent),
        duration_days=0.01,
        end_body='moon',
        end_altitude_km=lunar_orbit_km,
        notes="Capture into low lunar orbit",
    ))

    return plan


def create_mars_transfer_mission(
    leo_altitude_km: float = 400,
    mars_orbit_km: float = 400,
    margin_percent: float = 15.0,
) -> MissionPlan:
    """
    Create Earth-Mars transfer mission plan.

    Uses Hohmann-like transfer trajectory.
    """
    plan = MissionPlan(name="Earth-Mars Transfer")

    mu_sun = BODIES['sun']['mu_km3_s2']
    mu_earth = BODIES['earth']['mu_km3_s2']
    mu_mars = BODIES['mars barycenter']['mu_km3_s2']

    earth_radius = 6371
    mars_radius = 3390

    r_earth_au = 1.0
    r_mars_au = 1.524

    r_leo = earth_radius + leo_altitude_km
    r_mars_orbit = mars_radius + mars_orbit_km

    # Phase 1: LEO parking orbit
    plan.add_phase(MissionPhase(
        name="LEO Parking Orbit",
        phase_type=PhaseType.ORBIT_INSERTION,
        duration_days=0.1,
        start_body='earth',
        start_altitude_km=leo_altitude_km,
    ))

    # Phase 2: TMI (Trans-Mars Injection)
    # Heliocentric Hohmann transfer
    r1_km = r_earth_au * AU_TO_KM
    r2_km = r_mars_au * AU_TO_KM
    dv1_helio, dv2_helio, tof_days = hohmann_transfer(r1_km, r2_km, mu_sun)

    # Add Earth escape energy
    v_inf_dep = dv1_helio  # Hyperbolic excess at Earth
    v_leo = circular_velocity(r_leo, mu_earth)
    v_escape = np.sqrt(v_leo**2 + v_inf_dep**2)  # Vis-viva for hyperbolic
    dv_tmi = v_escape - v_leo

    plan.add_phase(MissionPhase(
        name="Trans-Mars Injection",
        phase_type=PhaseType.TRANSFER,
        delta_v_budget=DeltaVBudget("TMI", dv_tmi, margin_percent),
        duration_days=0.01,
        start_body='earth',
        notes="Earth departure burn",
    ))

    # Phase 3: Interplanetary cruise
    plan.add_phase(MissionPhase(
        name="Interplanetary Cruise",
        phase_type=PhaseType.COAST,
        duration_days=tof_days,
        notes=f"Heliocentric coast (~{tof_days:.0f} days)",
    ))

    # Phase 4: MOI (Mars Orbit Insertion)
    v_mars_orbit = circular_velocity(r_mars_orbit, mu_mars)
    v_inf_arr = dv2_helio  # Hyperbolic excess at Mars
    # Capture from hyperbolic to circular
    dv_moi = np.sqrt(v_mars_orbit**2 + v_inf_arr**2) - v_mars_orbit

    plan.add_phase(MissionPhase(
        name="Mars Orbit Insertion",
        phase_type=PhaseType.ORBIT_INSERTION,
        delta_v_budget=DeltaVBudget("MOI", dv_moi, margin_percent),
        end_body='mars barycenter',
        end_altitude_km=mars_orbit_km,
        notes="Mars capture burn",
    ))

    return plan


def create_leo_rendezvous_mission(
    chaser_altitude_km: float = 400,
    target_altitude_km: float = 420,
    phase_angle_deg: float = 10.0,
    margin_percent: float = 20.0,
) -> MissionPlan:
    """
    Create LEO rendezvous mission plan.

    Phasing maneuver to catch up with target.
    """
    plan = MissionPlan(name="LEO Rendezvous")

    mu_earth = BODIES['earth']['mu_km3_s2']
    earth_radius = 6371

    r_chaser = earth_radius + chaser_altitude_km
    r_target = earth_radius + target_altitude_km

    # Phase 1: Initial orbit
    plan.add_phase(MissionPhase(
        name="Chaser Parking Orbit",
        phase_type=PhaseType.ORBIT_INSERTION,
        start_body='earth',
        start_altitude_km=chaser_altitude_km,
    ))

    # Phase 2: Phasing maneuver
    # Lower orbit to catch up, then raise back
    phase_angle_rad = np.radians(phase_angle_deg)
    n_target = np.sqrt(mu_earth / r_target**3)  # Mean motion

    # Time to close phase angle
    t_phase = phase_angle_rad / n_target / 86400  # days

    # Phasing orbit: slightly lower to have higher angular velocity
    # Δn = phase_angle / t_phase
    # n_phasing = n_target + Δn
    # a_phasing = (mu / n_phasing²)^(1/3)
    delta_n = phase_angle_rad / (t_phase * 86400)
    n_phasing = n_target + delta_n
    a_phasing_km = (mu_earth / n_phasing**2)**(1/3)

    # Hohmann to phasing orbit and back
    dv1, dv2, _ = hohmann_transfer(r_target, a_phasing_km, mu_earth)
    dv_phase = 2 * (dv1 + dv2)  # Down and back up

    plan.add_phase(MissionPhase(
        name="Phasing Maneuver",
        phase_type=PhaseType.TRANSFER,
        delta_v_budget=DeltaVBudget("Phasing", dv_phase, margin_percent),
        duration_days=t_phase,
        notes="Catch-up maneuver",
    ))

    # Phase 3: Proximity operations
    plan.add_phase(MissionPhase(
        name="Proximity Operations",
        phase_type=PhaseType.RENDEZVOUS,
        delta_v_budget=DeltaVBudget("ProxOps", 0.05, margin_percent),
        duration_days=0.25,  # 6 hours
        notes="Final approach and docking",
    ))

    return plan


# ------------------- MISSION EXECUTOR -------------------

class MissionExecutor:
    """
    Executes a mission plan using the propagator.

    Converts high-level mission phases into actual simulations.
    """

    def __init__(
        self,
        plan: MissionPlan,
        spacecraft: Spacecraft,
        initial_state: np.ndarray,
        t0: float = 0.0,
    ):
        self.plan = plan
        self.spacecraft = spacecraft
        self.initial_state = initial_state.copy()
        self.t0 = t0

        self.result = PropagationResult()
        self.result.initialize(initial_state, t0)

        self.phase_results: List[PropagationResult] = []
        self.current_phase_idx = 0

    def execute_phase(self, phase: MissionPhase) -> PropagationResult:
        """Execute a single mission phase."""
        logger.info(f"Executing phase: {phase.name}")

        if phase.phase_type == PhaseType.COAST:
            return self._execute_coast(phase)
        elif phase.phase_type in (PhaseType.TRANSFER, PhaseType.ORBIT_INSERTION):
            return self._execute_burn(phase)
        else:
            return self._execute_coast(phase)

    def _execute_coast(self, phase: MissionPhase) -> PropagationResult:
        """Execute coasting phase."""
        if phase.duration_days is None:
            return self.result

        propagator = Propagator(
            ship=self.spacecraft,
            guidance=None,
            result=self.result,
        )

        t_end = self.result.time + phase.duration_days
        propagator.propagate(t_end)

        return self.result

    def _execute_burn(self, phase: MissionPhase) -> PropagationResult:
        """Execute a burn phase."""
        if phase.delta_v_budget is None:
            return self._execute_coast(phase)

        from guidance.star_tools import BurnGuidance

        # Get current velocity direction for prograde burn
        v = self.result.state[3:6]
        v_hat = v / np.linalg.norm(v)

        # Convert delta-V to AU/day
        dv_kmps = phase.delta_v_budget.total_kmps
        dv_au_day = dv_kmps / 1731.46

        # Create burn guidance
        guidance = BurnGuidance(
            t_start=self.result.time,
            delta_v=dv_au_day * v_hat,
        )

        propagator = Propagator(
            ship=self.spacecraft,
            guidance=guidance,
            result=self.result,
        )

        # Propagate through burn (short time)
        t_end = self.result.time + 0.01  # days
        propagator.propagate(t_end)

        return self.result

    def execute_all(self) -> PropagationResult:
        """Execute all mission phases."""
        for phase in self.plan.phases:
            self.execute_phase(phase)
            self.phase_results.append(self.result)
            self.current_phase_idx += 1

        return self.result


# ------------------- UTILITY FUNCTIONS -------------------

def quick_estimate(
    departure_body: str,
    arrival_body: str,
    departure_altitude_km: float = 400,
    arrival_altitude_km: float = 400,
) -> Dict[str, float]:
    """
    Quick delta-V estimate for interplanetary transfer.

    Returns approximate delta-V requirements.
    """
    # Get body parameters
    dep_mu = BODIES[departure_body]['mu_km3_s2']
    arr_mu = BODIES.get(arrival_body, {}).get('mu_km3_s2', 0)
    sun_mu = BODIES['sun']['mu_km3_s2']

    # Get orbital radii (approximate)
    body_radii = {
        'earth': (6371, 1.0),
        'mars barycenter': (3390, 1.524),
        'venus barycenter': (6052, 0.723),
        'moon': (1737, 0.00257),  # Earth-Moon distance in AU
    }

    if departure_body not in body_radii or arrival_body not in body_radii:
        return {'error': 'Unknown body'}

    dep_radius, dep_sun_au = body_radii[departure_body]
    arr_radius, arr_sun_au = body_radii[arrival_body]

    r_dep_orbit = dep_radius + departure_altitude_km
    r_arr_orbit = arr_radius + arrival_altitude_km

    # Heliocentric transfer
    r1_km = dep_sun_au * AU_TO_KM
    r2_km = arr_sun_au * AU_TO_KM

    dv1_helio, dv2_helio, tof_days = hohmann_transfer(r1_km, r2_km, sun_mu)

    # Departure burn (from parking orbit)
    v_park = circular_velocity(r_dep_orbit, dep_mu)
    v_escape = np.sqrt(v_park**2 + dv1_helio**2)
    dv_departure = v_escape - v_park

    # Arrival burn (to parking orbit)
    if arr_mu > 0:
        v_arr_park = circular_velocity(r_arr_orbit, arr_mu)
        v_arr_hyp = np.sqrt(v_arr_park**2 + dv2_helio**2)
        dv_arrival = v_arr_hyp - v_arr_park
    else:
        dv_arrival = dv2_helio

    return {
        'dv_departure_kmps': dv_departure,
        'dv_arrival_kmps': dv_arrival,
        'dv_total_kmps': dv_departure + dv_arrival,
        'transfer_time_days': tof_days,
        'helio_dv1_kmps': dv1_helio,
        'helio_dv2_kmps': dv2_helio,
    }
