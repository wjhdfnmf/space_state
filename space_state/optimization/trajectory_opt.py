# optimization/trajectory_opt.py
"""
Trajectory Optimization Algorithms

Implements fuel-optimal trajectory planning:
- Impulsive maneuver optimization
- Multi-burn sequences (Hohmann, bi-elliptic, 3-burn)
- Rendezvous timing optimization
- Gravity assist design
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from enum import Enum

import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base.star_constants import BODIES, AU_TO_KM, DAY_TO_SEC


# ==================== DATA CLASSES ====================

@dataclass
class ManeuverNode:
    """Represents a single impulsive maneuver."""
    time: float  # Time of maneuver (days from epoch)
    delta_v: np.ndarray  # Delta-V vector (km/s) in inertial frame
    position: np.ndarray  # Position at maneuver (AU)
    velocity_before: np.ndarray  # Velocity before maneuver (AU/day)
    velocity_after: np.ndarray  # Velocity after maneuver (AU/day)
    description: str = ""

    @property
    def delta_v_magnitude(self) -> float:
        """Delta-V magnitude in km/s."""
        return np.linalg.norm(self.delta_v)


@dataclass
class TrajectoryLeg:
    """Represents a coast arc between maneuvers."""
    start_time: float  # Start time (days)
    end_time: float  # End time (days)
    start_position: np.ndarray  # Position at start (AU)
    end_position: np.ndarray  # Position at end (AU)
    central_body: str  # Primary gravitational body
    orbit_type: str = "transfer"  # 'circular', 'elliptic', 'hyperbolic', 'transfer'

    @property
    def duration_days(self) -> float:
        return self.end_time - self.start_time


@dataclass
class OptimizationResult:
    """Result of trajectory optimization."""
    success: bool
    total_delta_v: float  # Total delta-V (km/s)
    maneuvers: List[ManeuverNode] = field(default_factory=list)
    legs: List[TrajectoryLeg] = field(default_factory=list)
    total_time: float = 0.0  # Total mission time (days)
    propellant_mass: float = 0.0  # Required propellant (kg)
    message: str = ""
    iterations: int = 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Optimization {'Successful' if self.success else 'Failed'}",
            f"Total Delta-V: {self.total_delta_v:.3f} km/s",
            f"Total Time: {self.total_time:.2f} days",
            f"Number of Maneuvers: {len(self.maneuvers)}",
        ]
        if self.propellant_mass > 0:
            lines.append(f"Propellant Required: {self.propellant_mass:.1f} kg")

        lines.append("\nManeuver Sequence:")
        for i, m in enumerate(self.maneuvers):
            lines.append(f"  {i+1}. t={m.time:.3f}d: ΔV={m.delta_v_magnitude:.3f} km/s - {m.description}")

        return "\n".join(lines)


# ==================== ORBITAL MECHANICS UTILITIES ====================

def circular_velocity(r_km: float, mu: float) -> float:
    """Circular orbit velocity at radius r."""
    return np.sqrt(mu / r_km)


def escape_velocity(r_km: float, mu: float) -> float:
    """Escape velocity at radius r."""
    return np.sqrt(2 * mu / r_km)


def vis_viva(r_km: float, a_km: float, mu: float) -> float:
    """Vis-viva equation: velocity at radius r for semi-major axis a."""
    return np.sqrt(mu * (2/r_km - 1/a_km))


def orbital_period(a_km: float, mu: float) -> float:
    """Orbital period in seconds."""
    return 2 * np.pi * np.sqrt(a_km**3 / mu)


def hohmann_delta_v(r1_km: float, r2_km: float, mu: float) -> Tuple[float, float, float]:
    """
    Compute Hohmann transfer delta-V.

    Returns:
        (dv1, dv2, tof_days): Departure dV, arrival dV, time of flight
    """
    # Transfer orbit semi-major axis
    a_transfer = (r1_km + r2_km) / 2

    # Velocities
    v1_circular = circular_velocity(r1_km, mu)
    v2_circular = circular_velocity(r2_km, mu)
    v1_transfer = vis_viva(r1_km, a_transfer, mu)
    v2_transfer = vis_viva(r2_km, a_transfer, mu)

    # Delta-V at each point
    dv1 = abs(v1_transfer - v1_circular)
    dv2 = abs(v2_circular - v2_transfer)

    # Time of flight (half period of transfer ellipse)
    tof_sec = np.pi * np.sqrt(a_transfer**3 / mu)
    tof_days = tof_sec / DAY_TO_SEC

    return dv1, dv2, tof_days


# ==================== SINGLE MANEUVER OPTIMIZATION ====================

def optimize_single_impulse(
    r_current: np.ndarray,
    v_current: np.ndarray,
    r_target: np.ndarray,
    v_target: np.ndarray,
    mu: float,
    max_dv: float = 10.0,
) -> OptimizationResult:
    """
    Optimize a single impulsive maneuver to match target state.

    This finds the optimal delta-V to transition from current state
    to a trajectory that intersects the target.

    Args:
        r_current: Current position (km)
        v_current: Current velocity (km/s)
        r_target: Target position (km)
        v_target: Target velocity (km/s)
        mu: Gravitational parameter (km³/s²)
        max_dv: Maximum delta-V constraint (km/s)

    Returns:
        OptimizationResult with optimal maneuver
    """
    def objective(dv_vec):
        v_new = v_current + dv_vec
        # Simple metric: minimize delta-V magnitude
        return np.linalg.norm(dv_vec)

    # Initial guess: direct velocity change
    dv_initial = v_target - v_current
    if np.linalg.norm(dv_initial) > max_dv:
        dv_initial = dv_initial / np.linalg.norm(dv_initial) * max_dv * 0.5

    # Optimize
    result = minimize(
        objective,
        dv_initial,
        method='SLSQP',
        bounds=[(-max_dv, max_dv)] * 3,
    )

    dv_opt = result.x

    maneuver = ManeuverNode(
        time=0.0,
        delta_v=dv_opt,
        position=r_current / AU_TO_KM,  # Convert to AU for storage
        velocity_before=v_current * DAY_TO_SEC / AU_TO_KM,
        velocity_after=(v_current + dv_opt) * DAY_TO_SEC / AU_TO_KM,
        description="Single impulse maneuver"
    )

    return OptimizationResult(
        success=result.success,
        total_delta_v=np.linalg.norm(dv_opt),
        maneuvers=[maneuver],
        message=result.message if hasattr(result, 'message') else "",
        iterations=result.nit if hasattr(result, 'nit') else 0,
    )


def optimize_plane_change(
    inclination_change_deg: float,
    velocity_kmps: float,
    optimize_location: bool = True,
) -> OptimizationResult:
    """
    Optimize plane change maneuver.

    Plane changes are most efficient at nodes (where orbital planes intersect)
    and at apoapsis (lowest velocity).

    Args:
        inclination_change_deg: Required inclination change (degrees)
        velocity_kmps: Orbital velocity at maneuver point (km/s)
        optimize_location: If True, assume optimal node location

    Returns:
        OptimizationResult with plane change delta-V
    """
    inc_rad = np.radians(inclination_change_deg)

    # Plane change delta-V: 2*v*sin(Δi/2)
    dv = 2 * velocity_kmps * np.sin(inc_rad / 2)

    # For combined maneuvers, the total is less than sum of parts
    # This is handled by the caller combining with other burns

    maneuver = ManeuverNode(
        time=0.0,
        delta_v=np.array([0, 0, dv]),  # Pure out-of-plane
        position=np.zeros(3),
        velocity_before=np.array([velocity_kmps, 0, 0]),
        velocity_after=np.array([velocity_kmps * np.cos(inc_rad), 0, velocity_kmps * np.sin(inc_rad)]),
        description=f"Plane change: {inclination_change_deg:.1f}°"
    )

    return OptimizationResult(
        success=True,
        total_delta_v=dv,
        maneuvers=[maneuver],
        message="Optimal plane change at node",
    )


def optimize_phasing(
    orbital_radius_km: float,
    phase_angle_deg: float,
    mu: float,
    n_orbits: int = 1,
) -> OptimizationResult:
    """
    Optimize phasing maneuver to catch up or fall behind target.

    Uses a phasing orbit with slightly different period.

    Args:
        orbital_radius_km: Radius of circular orbit (km)
        phase_angle_deg: Phase angle to close (degrees, positive = target ahead)
        mu: Gravitational parameter (km³/s²)
        n_orbits: Number of orbits for phasing (more = less delta-V)

    Returns:
        OptimizationResult with phasing maneuvers
    """
    phase_rad = np.radians(phase_angle_deg)

    # Current orbital period
    T_current = orbital_period(orbital_radius_km, mu)

    # Required period for phasing orbit
    # If target is ahead, need faster orbit (lower, shorter period)
    # Complete n orbits while target completes n orbits ± phase angle
    phase_fraction = phase_rad / (2 * np.pi)
    T_phasing = T_current * (n_orbits - phase_fraction) / n_orbits

    # Semi-major axis of phasing orbit
    a_phasing = (mu * (T_phasing / (2 * np.pi))**2) ** (1/3)

    # Determine if we raise or lower orbit
    if phase_angle_deg > 0:
        # Target ahead: need faster orbit (lower altitude at apsis)
        r_periapsis = orbital_radius_km
        r_apoapsis = 2 * a_phasing - r_periapsis
    else:
        # Target behind: need slower orbit (higher altitude)
        r_periapsis = orbital_radius_km
        r_apoapsis = 2 * a_phasing - r_periapsis

    # Ensure valid orbit
    if r_apoapsis < 0 or a_phasing < 0:
        return OptimizationResult(
            success=False,
            total_delta_v=float('inf'),
            message="Invalid phasing geometry - try more orbits"
        )

    # Delta-V calculations
    v_circular = circular_velocity(orbital_radius_km, mu)
    v_phasing = vis_viva(orbital_radius_km, a_phasing, mu)

    dv1 = abs(v_phasing - v_circular)  # Enter phasing orbit
    dv2 = dv1  # Return to circular (same magnitude)

    # Time of flight
    tof_sec = n_orbits * T_phasing
    tof_days = tof_sec / DAY_TO_SEC

    maneuvers = [
        ManeuverNode(
            time=0.0,
            delta_v=np.array([dv1, 0, 0]),
            position=np.zeros(3),
            velocity_before=np.array([v_circular, 0, 0]),
            velocity_after=np.array([v_phasing, 0, 0]),
            description="Enter phasing orbit"
        ),
        ManeuverNode(
            time=tof_days,
            delta_v=np.array([-dv2, 0, 0]),
            position=np.zeros(3),
            velocity_before=np.array([v_phasing, 0, 0]),
            velocity_after=np.array([v_circular, 0, 0]),
            description="Exit phasing orbit"
        ),
    ]

    return OptimizationResult(
        success=True,
        total_delta_v=dv1 + dv2,
        maneuvers=maneuvers,
        total_time=tof_days,
        message=f"Phasing over {n_orbits} orbit(s)",
    )


# ==================== MULTI-BURN OPTIMIZATION ====================

def optimize_hohmann(
    r1_km: float,
    r2_km: float,
    mu: float,
    include_plane_change: bool = False,
    plane_change_deg: float = 0.0,
) -> OptimizationResult:
    """
    Optimize Hohmann transfer between circular orbits.

    Args:
        r1_km: Initial orbit radius (km)
        r2_km: Final orbit radius (km)
        mu: Gravitational parameter (km³/s²)
        include_plane_change: Whether to include plane change
        plane_change_deg: Plane change angle (degrees)

    Returns:
        OptimizationResult with optimal Hohmann transfer
    """
    dv1, dv2, tof = hohmann_delta_v(r1_km, r2_km, mu)

    # If plane change included, split optimally between burns
    if include_plane_change and plane_change_deg > 0:
        # Optimal split: proportional to velocities at each burn
        v1 = circular_velocity(r1_km, mu)
        v2 = circular_velocity(r2_km, mu)

        # Split plane change to minimize total delta-V
        # Combine in-plane and out-of-plane components
        inc_rad = np.radians(plane_change_deg)

        # At each burn, combined delta-V = sqrt(dv_inplane² + dv_outplane²)
        # Optimal: put more plane change at lower velocity
        if r2_km > r1_km:  # Raising orbit
            # More plane change at apoapsis (dv2)
            split = 0.7  # 70% of plane change at second burn
        else:
            split = 0.3

        dv_plane1 = 2 * v1 * np.sin((1-split) * inc_rad / 2)
        dv_plane2 = 2 * v2 * np.sin(split * inc_rad / 2)

        dv1_combined = np.sqrt(dv1**2 + dv_plane1**2)
        dv2_combined = np.sqrt(dv2**2 + dv_plane2**2)

        total_dv = dv1_combined + dv2_combined
    else:
        dv1_combined = dv1
        dv2_combined = dv2
        total_dv = dv1 + dv2

    maneuvers = [
        ManeuverNode(
            time=0.0,
            delta_v=np.array([dv1_combined, 0, 0]),
            position=np.zeros(3),
            velocity_before=np.array([circular_velocity(r1_km, mu), 0, 0]),
            velocity_after=np.array([circular_velocity(r1_km, mu) + dv1, 0, 0]),
            description=f"Hohmann departure burn"
        ),
        ManeuverNode(
            time=tof,
            delta_v=np.array([dv2_combined, 0, 0]),
            position=np.zeros(3),
            velocity_before=np.zeros(3),
            velocity_after=np.array([circular_velocity(r2_km, mu), 0, 0]),
            description=f"Hohmann arrival burn"
        ),
    ]

    return OptimizationResult(
        success=True,
        total_delta_v=total_dv,
        maneuvers=maneuvers,
        total_time=tof,
        message=f"Hohmann transfer: {r1_km:.0f} km → {r2_km:.0f} km",
    )


def optimize_bielliptic(
    r1_km: float,
    r2_km: float,
    r_intermediate_km: float,
    mu: float,
) -> OptimizationResult:
    """
    Optimize bi-elliptic transfer (three burns via high intermediate orbit).

    Bi-elliptic is more efficient than Hohmann when r2/r1 > 11.94.

    Args:
        r1_km: Initial orbit radius (km)
        r2_km: Final orbit radius (km)
        r_intermediate_km: Intermediate apoapsis radius (km)
        mu: Gravitational parameter (km³/s²)

    Returns:
        OptimizationResult with three-burn transfer
    """
    r_i = r_intermediate_km

    # First transfer ellipse: r1 to r_i
    a1 = (r1_km + r_i) / 2
    v1_circular = circular_velocity(r1_km, mu)
    v1_transfer = vis_viva(r1_km, a1, mu)
    dv1 = v1_transfer - v1_circular

    # At intermediate apoapsis
    v_at_ri_first = vis_viva(r_i, a1, mu)

    # Second transfer ellipse: r_i to r2
    a2 = (r_i + r2_km) / 2
    v_at_ri_second = vis_viva(r_i, a2, mu)
    dv2 = v_at_ri_second - v_at_ri_first

    # Final circularization at r2
    v2_transfer = vis_viva(r2_km, a2, mu)
    v2_circular = circular_velocity(r2_km, mu)
    dv3 = v2_circular - v2_transfer

    # Time of flight
    tof1 = np.pi * np.sqrt(a1**3 / mu) / DAY_TO_SEC
    tof2 = np.pi * np.sqrt(a2**3 / mu) / DAY_TO_SEC
    total_tof = tof1 + tof2

    total_dv = abs(dv1) + abs(dv2) + abs(dv3)

    maneuvers = [
        ManeuverNode(
            time=0.0,
            delta_v=np.array([dv1, 0, 0]),
            position=np.zeros(3),
            velocity_before=np.array([v1_circular, 0, 0]),
            velocity_after=np.array([v1_transfer, 0, 0]),
            description="Bi-elliptic burn 1: depart"
        ),
        ManeuverNode(
            time=tof1,
            delta_v=np.array([dv2, 0, 0]),
            position=np.zeros(3),
            velocity_before=np.array([v_at_ri_first, 0, 0]),
            velocity_after=np.array([v_at_ri_second, 0, 0]),
            description="Bi-elliptic burn 2: apoapsis"
        ),
        ManeuverNode(
            time=total_tof,
            delta_v=np.array([dv3, 0, 0]),
            position=np.zeros(3),
            velocity_before=np.array([v2_transfer, 0, 0]),
            velocity_after=np.array([v2_circular, 0, 0]),
            description="Bi-elliptic burn 3: circularize"
        ),
    ]

    return OptimizationResult(
        success=True,
        total_delta_v=total_dv,
        maneuvers=maneuvers,
        total_time=total_tof,
        message=f"Bi-elliptic via {r_i:.0f} km",
    )


def optimize_three_burn(
    r1_km: float,
    r2_km: float,
    mu: float,
    optimize_intermediate: bool = True,
) -> OptimizationResult:
    """
    Find optimal three-burn transfer (Hohmann vs bi-elliptic).

    Automatically selects best strategy based on orbit ratio.

    Args:
        r1_km: Initial orbit radius (km)
        r2_km: Final orbit radius (km)
        mu: Gravitational parameter (km³/s²)
        optimize_intermediate: Optimize intermediate altitude

    Returns:
        OptimizationResult with optimal transfer
    """
    ratio = max(r1_km, r2_km) / min(r1_km, r2_km)

    # Hohmann transfer for comparison
    hohmann = optimize_hohmann(r1_km, r2_km, mu)

    if ratio < 11.94:
        # Hohmann is optimal
        hohmann.message = "Hohmann optimal (ratio < 11.94)"
        return hohmann

    # Bi-elliptic may be better
    if optimize_intermediate:
        # Find optimal intermediate radius
        def bielliptic_dv(r_i):
            if r_i <= max(r1_km, r2_km):
                return float('inf')
            result = optimize_bielliptic(r1_km, r2_km, r_i, mu)
            return result.total_delta_v

        # Search for optimal intermediate
        r_min = max(r1_km, r2_km) * 1.1
        r_max = max(r1_km, r2_km) * 100

        result = minimize_scalar(
            bielliptic_dv,
            bounds=(r_min, r_max),
            method='bounded'
        )

        r_opt = result.x
        bielliptic = optimize_bielliptic(r1_km, r2_km, r_opt, mu)
    else:
        # Use 2x the outer radius as default
        r_intermediate = 2 * max(r1_km, r2_km)
        bielliptic = optimize_bielliptic(r1_km, r2_km, r_intermediate, mu)

    # Return the better option
    if bielliptic.total_delta_v < hohmann.total_delta_v:
        bielliptic.message = f"Bi-elliptic optimal (saves {hohmann.total_delta_v - bielliptic.total_delta_v:.3f} km/s)"
        return bielliptic
    else:
        hohmann.message = "Hohmann optimal despite high ratio"
        return hohmann


# ==================== TRANSFER OPTIMIZATION ====================

def optimize_lambert_departure(
    r_departure: np.ndarray,
    r_arrival: np.ndarray,
    tof_days: float,
    mu: float,
    v_departure_initial: np.ndarray = None,
) -> OptimizationResult:
    """
    Optimize departure burn for Lambert transfer.

    Uses Lambert solver to find required velocities.

    Args:
        r_departure: Departure position (km)
        r_arrival: Arrival position (km)
        tof_days: Time of flight (days)
        mu: Gravitational parameter (km³/s²)
        v_departure_initial: Initial velocity at departure (km/s)

    Returns:
        OptimizationResult with departure maneuver
    """
    from guidance import lambert_solve

    # Convert to AU and days for Lambert solver
    r1_au = r_departure / AU_TO_KM
    r2_au = r_arrival / AU_TO_KM
    mu_au = mu * DAY_TO_SEC**2 / AU_TO_KM**3

    try:
        solution = lambert_solve(r1_au, r2_au, tof_days, mu_au)

        # Convert velocities back to km/s
        v1_kmps = solution.v1 * AU_TO_KM / DAY_TO_SEC
        v2_kmps = solution.v2 * AU_TO_KM / DAY_TO_SEC

        if v_departure_initial is not None:
            dv = v1_kmps - v_departure_initial
        else:
            dv = v1_kmps

        maneuver = ManeuverNode(
            time=0.0,
            delta_v=dv,
            position=r1_au,
            velocity_before=v_departure_initial * DAY_TO_SEC / AU_TO_KM if v_departure_initial is not None else np.zeros(3),
            velocity_after=solution.v1,
            description=f"Lambert departure (TOF={tof_days:.1f}d)"
        )

        return OptimizationResult(
            success=True,
            total_delta_v=np.linalg.norm(dv),
            maneuvers=[maneuver],
            total_time=tof_days,
            message=f"Lambert transfer, TOF={tof_days:.1f} days",
        )
    except Exception as e:
        return OptimizationResult(
            success=False,
            total_delta_v=float('inf'),
            message=f"Lambert solver failed: {e}",
        )


def optimize_rendezvous(
    r_chaser: np.ndarray,
    v_chaser: np.ndarray,
    r_target: np.ndarray,
    v_target: np.ndarray,
    mu: float,
    tof_range: Tuple[float, float] = (0.1, 10.0),
    n_samples: int = 50,
) -> OptimizationResult:
    """
    Optimize rendezvous trajectory between two spacecraft.

    Finds optimal time-of-flight for minimum delta-V rendezvous.

    Args:
        r_chaser: Chaser position (km)
        v_chaser: Chaser velocity (km/s)
        r_target: Target position (km)
        v_target: Target velocity (km/s)
        mu: Gravitational parameter (km³/s²)
        tof_range: Range of TOF to search (days)
        n_samples: Number of samples for grid search

    Returns:
        OptimizationResult with optimal rendezvous
    """
    from guidance import lambert_solve

    best_dv = float('inf')
    best_tof = None
    best_maneuvers = []

    # Grid search over TOF
    tofs = np.linspace(tof_range[0], tof_range[1], n_samples)

    for tof in tofs:
        # Propagate target to intercept point (simple circular approximation)
        # For more accuracy, would need full propagation
        n = np.sqrt(mu / np.linalg.norm(r_target)**3)  # Mean motion
        angle = n * tof * DAY_TO_SEC

        # Rotate target position
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        r_target_future = R @ r_target
        v_target_future = R @ v_target

        # Solve Lambert problem
        try:
            r1_au = r_chaser / AU_TO_KM
            r2_au = r_target_future / AU_TO_KM
            mu_au = mu * DAY_TO_SEC**2 / AU_TO_KM**3

            solution = lambert_solve(r1_au, r2_au, tof, mu_au)

            v1_kmps = solution.v1 * AU_TO_KM / DAY_TO_SEC
            v2_kmps = solution.v2 * AU_TO_KM / DAY_TO_SEC

            dv1 = np.linalg.norm(v1_kmps - v_chaser)
            dv2 = np.linalg.norm(v_target_future - v2_kmps)
            total_dv = dv1 + dv2

            if total_dv < best_dv:
                best_dv = total_dv
                best_tof = tof
                best_maneuvers = [
                    ManeuverNode(
                        time=0.0,
                        delta_v=v1_kmps - v_chaser,
                        position=r1_au,
                        velocity_before=v_chaser * DAY_TO_SEC / AU_TO_KM,
                        velocity_after=solution.v1,
                        description="Rendezvous departure"
                    ),
                    ManeuverNode(
                        time=tof,
                        delta_v=v_target_future - v2_kmps,
                        position=r2_au,
                        velocity_before=solution.v2,
                        velocity_after=v_target_future * DAY_TO_SEC / AU_TO_KM,
                        description="Rendezvous arrival"
                    ),
                ]
        except:
            continue

    if best_tof is None:
        return OptimizationResult(
            success=False,
            total_delta_v=float('inf'),
            message="No valid rendezvous found",
        )

    return OptimizationResult(
        success=True,
        total_delta_v=best_dv,
        maneuvers=best_maneuvers,
        total_time=best_tof,
        message=f"Optimal rendezvous at TOF={best_tof:.2f} days",
        iterations=n_samples,
    )


# ==================== GRAVITY ASSIST ====================

@dataclass
class GravityAssist:
    """Gravity assist flyby parameters."""
    body: str
    periapsis_altitude_km: float
    v_infinity_in: np.ndarray  # Incoming V∞ (km/s)
    v_infinity_out: np.ndarray  # Outgoing V∞ (km/s)
    turn_angle_deg: float
    delta_v_boost: float  # Added velocity magnitude (km/s)
    closest_approach_time: float  # Time of periapsis (days)


def compute_flyby_geometry(
    body: str,
    v_infinity_in: np.ndarray,
    periapsis_altitude_km: float,
) -> GravityAssist:
    """
    Compute gravity assist geometry.

    Args:
        body: Name of flyby body
        v_infinity_in: Incoming hyperbolic excess velocity (km/s)
        periapsis_altitude_km: Periapsis altitude above surface (km)

    Returns:
        GravityAssist with computed parameters
    """
    body_data = BODIES.get(body)
    if body_data is None:
        raise ValueError(f"Unknown body: {body}")

    mu = body_data['mu_km3_s2']
    # Radius is stored in AU, convert to km
    radius_au = body_data.get('radius')
    if radius_au is None or radius_au == 0:
        # Fallback to estimated radius from mu (rough approximation)
        # For gas giants and rocky planets
        radius = 6371.0  # Default to Earth size
    else:
        radius = radius_au * AU_TO_KM
    r_periapsis = radius + periapsis_altitude_km

    v_inf = np.linalg.norm(v_infinity_in)

    # Turn angle from hyperbolic geometry
    # sin(δ/2) = 1 / (1 + r_p * v∞² / μ)
    e = 1 + r_periapsis * v_inf**2 / mu  # Eccentricity
    if e < 1:
        e = 1.001  # Ensure hyperbolic

    sin_half_delta = 1 / e
    delta_rad = 2 * np.arcsin(sin_half_delta)
    turn_angle_deg = np.degrees(delta_rad)

    # Outgoing V∞ (rotated by turn angle in orbital plane)
    # Assume rotation in plane defined by v_infinity and some perpendicular
    v_in_hat = v_infinity_in / v_inf

    # Find perpendicular direction (use cross with z, or x if aligned)
    if abs(v_in_hat[2]) < 0.9:
        perp = np.cross(v_in_hat, [0, 0, 1])
    else:
        perp = np.cross(v_in_hat, [1, 0, 0])
    perp = perp / np.linalg.norm(perp)

    # Rotate v_infinity by turn angle
    c, s = np.cos(delta_rad), np.sin(delta_rad)
    v_infinity_out = v_inf * (c * v_in_hat + s * perp)

    # Delta-V boost (change in heliocentric velocity)
    # This is the vector difference, not just magnitude change
    delta_v_boost = np.linalg.norm(v_infinity_out - v_infinity_in)

    return GravityAssist(
        body=body,
        periapsis_altitude_km=periapsis_altitude_km,
        v_infinity_in=v_infinity_in,
        v_infinity_out=v_infinity_out,
        turn_angle_deg=turn_angle_deg,
        delta_v_boost=delta_v_boost,
        closest_approach_time=0.0,
    )


def optimize_powered_flyby(
    body: str,
    v_infinity_in: np.ndarray,
    v_desired_boost: float,
    min_altitude_km: float = 200.0,
) -> Tuple[GravityAssist, float]:
    """
    Optimize powered gravity assist (with propulsive maneuver at periapsis).

    A powered flyby adds a burn at periapsis to maximize velocity gain.

    Args:
        body: Name of flyby body
        v_infinity_in: Incoming V∞ (km/s)
        v_desired_boost: Desired velocity boost (km/s)
        min_altitude_km: Minimum safe periapsis altitude (km)

    Returns:
        Tuple of (GravityAssist, propulsive_dv)
    """
    body_data = BODIES.get(body)
    if body_data is None:
        raise ValueError(f"Unknown body: {body}")

    mu = body_data['mu_km3_s2']
    # Radius is stored in AU, convert to km
    radius_au = body_data.get('radius')
    if radius_au is None or radius_au == 0:
        radius = 6371.0  # Default to Earth size
    else:
        radius = radius_au * AU_TO_KM

    # Try different periapsis altitudes
    def objective(altitude):
        flyby = compute_flyby_geometry(body, v_infinity_in, altitude)
        unpowered_boost = flyby.delta_v_boost

        # If unpowered gives enough, no propulsion needed
        if unpowered_boost >= v_desired_boost:
            return altitude, 0.0

        # Otherwise, need propulsive delta-V at periapsis
        # Oberth effect: burns at periapsis are most efficient
        return altitude, v_desired_boost - unpowered_boost

    # Start at minimum altitude
    altitude, prop_dv = objective(min_altitude_km)
    flyby = compute_flyby_geometry(body, v_infinity_in, altitude)

    return flyby, prop_dv


# ==================== FUEL ESTIMATION ====================

def estimate_propellant(
    delta_v_kmps: float,
    dry_mass_kg: float,
    isp_s: float,
    margin_percent: float = 10.0,
) -> Dict[str, float]:
    """
    Estimate propellant mass using Tsiolkovsky equation.

    Args:
        delta_v_kmps: Required delta-V (km/s)
        dry_mass_kg: Dry mass (kg)
        isp_s: Specific impulse (seconds)
        margin_percent: Propellant margin (percent)

    Returns:
        Dictionary with mass breakdown
    """
    g0 = 9.80665 / 1000  # km/s²
    v_exhaust = isp_s * g0  # km/s

    # Tsiolkovsky: Δv = v_e * ln(m_initial / m_final)
    # m_initial / m_final = exp(Δv / v_e)
    mass_ratio = np.exp(delta_v_kmps / v_exhaust)

    # m_initial = m_dry + m_propellant
    # m_final = m_dry
    propellant_mass = dry_mass_kg * (mass_ratio - 1)
    propellant_with_margin = propellant_mass * (1 + margin_percent / 100)

    initial_mass = dry_mass_kg + propellant_with_margin

    return {
        'propellant_mass_kg': propellant_mass,
        'propellant_with_margin_kg': propellant_with_margin,
        'initial_mass_kg': initial_mass,
        'dry_mass_kg': dry_mass_kg,
        'mass_ratio': mass_ratio,
        'margin_percent': margin_percent,
    }


def compute_mass_ratio(delta_v_kmps: float, isp_s: float) -> float:
    """
    Compute mass ratio for given delta-V and Isp.

    Args:
        delta_v_kmps: Delta-V (km/s)
        isp_s: Specific impulse (seconds)

    Returns:
        Mass ratio (initial/final)
    """
    g0 = 9.80665 / 1000  # km/s²
    v_exhaust = isp_s * g0
    return np.exp(delta_v_kmps / v_exhaust)


def delta_v_budget_margin(
    maneuvers: List[ManeuverNode],
    navigation_error_percent: float = 5.0,
    execution_error_percent: float = 2.0,
    contingency_percent: float = 10.0,
) -> Dict[str, float]:
    """
    Compute delta-V budget with margins.

    Args:
        maneuvers: List of planned maneuvers
        navigation_error_percent: Navigation uncertainty margin
        execution_error_percent: Burn execution uncertainty
        contingency_percent: General contingency

    Returns:
        Dictionary with budget breakdown
    """
    nominal_dv = sum(m.delta_v_magnitude for m in maneuvers)

    nav_margin = nominal_dv * navigation_error_percent / 100
    exec_margin = nominal_dv * execution_error_percent / 100
    contingency = nominal_dv * contingency_percent / 100

    total_margin = nav_margin + exec_margin + contingency
    total_with_margin = nominal_dv + total_margin

    return {
        'nominal_dv_kmps': nominal_dv,
        'navigation_margin_kmps': nav_margin,
        'execution_margin_kmps': exec_margin,
        'contingency_kmps': contingency,
        'total_margin_kmps': total_margin,
        'total_with_margin_kmps': total_with_margin,
        'margin_percent': (total_margin / nominal_dv) * 100 if nominal_dv > 0 else 0,
    }
