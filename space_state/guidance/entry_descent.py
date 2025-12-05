# guidance/entry_descent.py
"""
Entry, Descent, and Landing (EDL) Module

Provides calculations for atmospheric entry including:
- Entry corridor analysis
- Ballistic coefficient and deceleration
- Heat rate and total heat load
- Skip entry trajectories
- Parachute deployment conditions

Applicable to:
- Earth reentry (LEO, lunar return, Mars return)
- Mars EDL
- Venus atmospheric probes
- Titan entry

Key Parameters:
- Entry Flight Path Angle (EFPA): Angle below local horizontal at entry interface
- Ballistic Coefficient: m/(Cd*A) - determines deceleration profile
- Entry Velocity: Speed at atmospheric interface
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum, auto

from base.star_logger import logger


# Physical Constants
R_EARTH = 6378.137  # km
R_MARS = 3389.5  # km
R_VENUS = 6051.8  # km
R_TITAN = 2574.7  # km

G0 = 9.80665  # m/s² (standard gravity)


class EntryType(Enum):
    """Type of atmospheric entry."""
    BALLISTIC = auto()      # Unguided ballistic entry
    LIFTING = auto()        # Guided lifting entry (capsule with L/D)
    SKIP = auto()           # Skip entry (aerobraking with exit)
    AEROBRAKING = auto()    # Multiple-pass aerobraking


@dataclass
class AtmosphereModel:
    """Exponential atmosphere model parameters."""
    name: str
    surface_density: float    # kg/m³
    scale_height: float       # km
    surface_pressure: float   # Pa
    gamma: float              # Specific heat ratio
    gas_constant: float       # J/(kg·K)
    surface_temp: float       # K
    entry_altitude: float     # km (atmospheric interface)


# Atmosphere models
ATMOSPHERES = {
    'earth': AtmosphereModel(
        name='Earth',
        surface_density=1.225,
        scale_height=8.5,
        surface_pressure=101325,
        gamma=1.4,
        gas_constant=287.0,
        surface_temp=288.15,
        entry_altitude=120.0
    ),
    'mars': AtmosphereModel(
        name='Mars',
        surface_density=0.020,
        scale_height=11.1,
        surface_pressure=636,
        gamma=1.3,
        gas_constant=188.9,
        surface_temp=210.0,
        entry_altitude=125.0
    ),
    'venus': AtmosphereModel(
        name='Venus',
        surface_density=65.0,
        scale_height=15.9,
        surface_pressure=9.2e6,
        gamma=1.3,
        gas_constant=188.9,
        surface_temp=737.0,
        entry_altitude=200.0
    ),
    'titan': AtmosphereModel(
        name='Titan',
        surface_density=5.3,
        scale_height=40.0,
        surface_pressure=146700,
        gamma=1.4,
        gas_constant=290.0,
        surface_temp=94.0,
        entry_altitude=1200.0
    ),
}


@dataclass
class EntryConditions:
    """Entry interface conditions."""
    velocity: float           # km/s
    flight_path_angle: float  # degrees (negative = descending)
    altitude: float           # km
    latitude: float           # degrees
    longitude: float          # degrees
    azimuth: float           # degrees (heading)


@dataclass
class EntryTrajectoryPoint:
    """Single point on entry trajectory."""
    time: float              # seconds
    altitude: float          # km
    velocity: float          # km/s
    flight_path_angle: float # degrees
    deceleration: float      # g's
    heat_rate: float         # W/cm²
    dynamic_pressure: float  # Pa
    mach_number: float       # dimensionless


@dataclass
class EntryResult:
    """Complete entry analysis results."""
    peak_deceleration_g: float
    peak_heat_rate_w_cm2: float
    total_heat_load_j_cm2: float
    time_to_peak_decel: float     # seconds
    altitude_peak_decel: float    # km
    altitude_peak_heating: float  # km
    max_dynamic_pressure: float   # Pa
    terminal_velocity: float      # km/s
    trajectory: List[EntryTrajectoryPoint]


def atmospheric_density(altitude_km: float, atmosphere: AtmosphereModel) -> float:
    """
    Calculate atmospheric density using exponential model.

    Parameters
    ----------
    altitude_km : float
        Altitude above surface (km)
    atmosphere : AtmosphereModel
        Atmosphere parameters

    Returns
    -------
    float
        Density (kg/m³)
    """
    return atmosphere.surface_density * np.exp(-altitude_km / atmosphere.scale_height)


def ballistic_coefficient(mass_kg: float, cd: float, area_m2: float) -> float:
    """
    Calculate ballistic coefficient.

    β = m / (Cd × A)

    Lower β = more deceleration, lower peak heating
    Higher β = less deceleration, higher terminal velocity

    Parameters
    ----------
    mass_kg : float
        Vehicle mass (kg)
    cd : float
        Drag coefficient
    area_m2 : float
        Reference area (m²)

    Returns
    -------
    float
        Ballistic coefficient (kg/m²)
    """
    return mass_kg / (cd * area_m2)


def entry_corridor_bounds(
    velocity_km_s: float,
    body: str = 'earth',
    max_decel_g: float = 10.0,
    skip_out_angle: float = -4.0
) -> Tuple[float, float]:
    """
    Calculate entry corridor bounds (steep and shallow limits).

    Parameters
    ----------
    velocity_km_s : float
        Entry velocity (km/s)
    body : str
        Target body ('earth', 'mars', etc.)
    max_decel_g : float
        Maximum allowable deceleration (g's)
    skip_out_angle : float
        Shallow limit for skip-out (degrees)

    Returns
    -------
    steep_limit, shallow_limit : tuple
        Flight path angle limits (degrees, negative)
    """
    # Shallow limit - skip-out boundary
    # Depends on velocity and L/D ratio
    shallow_limit = skip_out_angle

    # Steep limit - g-load constraint
    # Approximate: steeper angle = higher peak g's
    # For ballistic entry: g_peak ∝ sin²(γ) × V² × ρ / β

    # Empirical relationship for steep limit
    if body == 'earth':
        if velocity_km_s > 11.0:  # Lunar/interplanetary return
            steep_limit = -7.0 - (max_decel_g - 10) * 0.3
        else:  # LEO reentry
            steep_limit = -3.0 - (max_decel_g - 5) * 0.5
    elif body == 'mars':
        steep_limit = -15.0 - (max_decel_g - 10) * 0.5
    else:
        steep_limit = -10.0

    return steep_limit, shallow_limit


def peak_deceleration_estimate(
    velocity_km_s: float,
    flight_path_angle_deg: float,
    ballistic_coef: float,
    body: str = 'earth'
) -> float:
    """
    Estimate peak deceleration during entry.

    Parameters
    ----------
    velocity_km_s : float
        Entry velocity (km/s)
    flight_path_angle_deg : float
        Entry flight path angle (degrees, negative)
    ballistic_coef : float
        Ballistic coefficient (kg/m²)
    body : str
        Target body

    Returns
    -------
    float
        Peak deceleration (g's)
    """
    atm = ATMOSPHERES.get(body, ATMOSPHERES['earth'])

    v = velocity_km_s * 1000  # m/s
    gamma = np.radians(abs(flight_path_angle_deg))

    # Allen-Eggers approximation for peak deceleration
    # a_max = V² × sin(γ) / (2 × e × H)
    # where e = 2.718... and H is scale height

    H = atm.scale_height * 1000  # m
    a_max = v**2 * np.sin(gamma) / (2 * np.e * H)

    # Adjust for ballistic coefficient effect
    # Higher β = deeper penetration = slightly lower peak g
    beta_ref = 100  # Reference ballistic coefficient
    beta_factor = (beta_ref / ballistic_coef) ** 0.3

    g_peak = a_max * beta_factor / G0

    return g_peak


def peak_heating_rate(
    velocity_km_s: float,
    nose_radius_m: float,
    altitude_km: float,
    body: str = 'earth'
) -> float:
    """
    Estimate peak convective heating rate using Sutton-Graves correlation.

    q = k × sqrt(ρ/r_n) × V³

    Parameters
    ----------
    velocity_km_s : float
        Velocity (km/s)
    nose_radius_m : float
        Nose radius (m)
    altitude_km : float
        Altitude (km)
    body : str
        Target body

    Returns
    -------
    float
        Heat rate (W/cm²)
    """
    atm = ATMOSPHERES.get(body, ATMOSPHERES['earth'])
    rho = atmospheric_density(altitude_km, atm)

    v = velocity_km_s * 1000  # m/s

    # Sutton-Graves constant (depends on gas composition)
    if body == 'earth':
        k = 1.7415e-4  # For air
    elif body == 'mars':
        k = 1.9e-4  # For CO2
    else:
        k = 1.8e-4

    # Heat rate in W/m²
    q = k * np.sqrt(rho / nose_radius_m) * v**3

    # Convert to W/cm²
    return q / 1e4


def total_heat_load(
    velocity_km_s: float,
    ballistic_coef: float,
    nose_radius_m: float,
    body: str = 'earth'
) -> float:
    """
    Estimate total integrated heat load.

    Q ∝ V³ × sqrt(β × r_n) / sin(γ)

    Parameters
    ----------
    velocity_km_s : float
        Entry velocity (km/s)
    ballistic_coef : float
        Ballistic coefficient (kg/m²)
    nose_radius_m : float
        Nose radius (m)
    body : str
        Target body

    Returns
    -------
    float
        Total heat load (J/cm²)
    """
    v = velocity_km_s * 1000  # m/s

    # Chapman's approximation for total heat load
    # Q = k × V³ × sqrt(ρ₀ × H × r_n / β)

    atm = ATMOSPHERES.get(body, ATMOSPHERES['earth'])
    rho0 = atm.surface_density
    H = atm.scale_height * 1000  # m

    k_q = 1.0e-8  # Empirical constant

    Q = k_q * v**3 * np.sqrt(rho0 * H * nose_radius_m / ballistic_coef)

    # Convert to J/cm²
    return Q / 1e4


def simulate_entry(
    entry: EntryConditions,
    ballistic_coef: float,
    nose_radius_m: float,
    body: str = 'earth',
    lift_to_drag: float = 0.0,
    dt: float = 0.5
) -> EntryResult:
    """
    Simulate atmospheric entry trajectory.

    Parameters
    ----------
    entry : EntryConditions
        Initial entry conditions
    ballistic_coef : float
        Ballistic coefficient (kg/m²)
    nose_radius_m : float
        Nose radius for heating (m)
    body : str
        Target body
    lift_to_drag : float
        Lift-to-drag ratio (0 for ballistic)
    dt : float
        Time step (seconds)

    Returns
    -------
    EntryResult
        Complete entry analysis
    """
    atm = ATMOSPHERES.get(body, ATMOSPHERES['earth'])

    # Get planet radius
    radii = {'earth': R_EARTH, 'mars': R_MARS, 'venus': R_VENUS, 'titan': R_TITAN}
    R = radii.get(body, R_EARTH) * 1000  # m

    # Initialize state
    h = entry.altitude * 1000  # m
    v = entry.velocity * 1000  # m/s
    gamma = np.radians(entry.flight_path_angle)

    trajectory = []
    total_heat = 0.0
    peak_decel = 0.0
    peak_heat_rate = 0.0
    peak_q = 0.0
    t = 0.0

    time_peak_decel = 0.0
    alt_peak_decel = 0.0
    alt_peak_heat = 0.0

    while h > 0 and v > 100:  # Stop at surface or low velocity
        # Atmospheric properties
        alt_km = h / 1000
        rho = atmospheric_density(alt_km, atm)

        # Dynamic pressure
        q_dyn = 0.5 * rho * v**2

        # Drag deceleration
        drag = q_dyn / ballistic_coef
        a_drag = drag  # m/s²

        # Lift (if any)
        lift = lift_to_drag * drag

        # Gravity
        g = G0 * (R / (R + h))**2

        # Equations of motion
        dv_dt = -drag - g * np.sin(gamma)
        dgamma_dt = (lift / v) + (v / (R + h) - g / v) * np.cos(gamma)
        dh_dt = v * np.sin(gamma)

        # Heat rate
        q_heat = peak_heating_rate(v / 1000, nose_radius_m, alt_km, body)

        # Mach number (approximate)
        T = atm.surface_temp * np.exp(-alt_km / atm.scale_height * 0.1)  # Rough
        a_sound = np.sqrt(atm.gamma * atm.gas_constant * max(T, 100))
        mach = v / a_sound

        # Record trajectory point
        point = EntryTrajectoryPoint(
            time=t,
            altitude=alt_km,
            velocity=v / 1000,
            flight_path_angle=np.degrees(gamma),
            deceleration=a_drag / G0,
            heat_rate=q_heat,
            dynamic_pressure=q_dyn,
            mach_number=mach
        )
        trajectory.append(point)

        # Track peaks
        if a_drag / G0 > peak_decel:
            peak_decel = a_drag / G0
            time_peak_decel = t
            alt_peak_decel = alt_km

        if q_heat > peak_heat_rate:
            peak_heat_rate = q_heat
            alt_peak_heat = alt_km

        if q_dyn > peak_q:
            peak_q = q_dyn

        # Integrate heat load
        total_heat += q_heat * dt

        # Update state
        v += dv_dt * dt
        gamma += dgamma_dt * dt
        h += dh_dt * dt
        t += dt

        # Safety limit
        if t > 1000:
            break

    return EntryResult(
        peak_deceleration_g=peak_decel,
        peak_heat_rate_w_cm2=peak_heat_rate,
        total_heat_load_j_cm2=total_heat,
        time_to_peak_decel=time_peak_decel,
        altitude_peak_decel=alt_peak_decel,
        altitude_peak_heating=alt_peak_heat,
        max_dynamic_pressure=peak_q,
        terminal_velocity=v / 1000,
        trajectory=trajectory
    )


def skip_entry_analysis(
    entry_velocity_km_s: float,
    entry_fpa_deg: float,
    ballistic_coef: float,
    lift_to_drag: float,
    body: str = 'earth'
) -> Dict[str, float]:
    """
    Analyze skip entry trajectory.

    Parameters
    ----------
    entry_velocity_km_s : float
        Entry velocity
    entry_fpa_deg : float
        Entry flight path angle (degrees)
    ballistic_coef : float
        Ballistic coefficient
    lift_to_drag : float
        Lift-to-drag ratio
    body : str
        Target body

    Returns
    -------
    dict
        Skip entry analysis results
    """
    atm = ATMOSPHERES.get(body, ATMOSPHERES['earth'])

    # Approximate skip-out velocity
    v_entry = entry_velocity_km_s
    gamma = np.radians(abs(entry_fpa_deg))

    # Velocity loss in atmosphere is roughly proportional to ∫ρ ds
    # For skip entry: Δv/v ≈ ρ₀ × H / (β × sin(γ)) for shallow entries

    rho0 = atm.surface_density
    H = atm.scale_height * 1000  # m

    # Energy loss factor
    delta_v_factor = rho0 * H / (ballistic_coef * np.sin(gamma)) * 0.1

    # Skip-out velocity
    v_skip_out = v_entry * (1 - delta_v_factor)

    # Apogee after skip (simplified)
    radii = {'earth': R_EARTH, 'mars': R_MARS, 'venus': R_VENUS, 'titan': R_TITAN}
    R = radii.get(body, R_EARTH)
    mu = {'earth': 398600.4, 'mars': 42828.0, 'venus': 324859.0, 'titan': 8978.0}
    mu_body = mu.get(body, 398600.4)

    # Approximate apogee altitude
    r_entry = R + atm.entry_altitude
    v_skip = v_skip_out
    gamma_out = abs(entry_fpa_deg) * 0.5  # Rough estimate

    # Vis-viva for apogee
    energy = v_skip**2 / 2 - mu_body / r_entry
    a = -mu_body / (2 * energy) if energy < 0 else float('inf')

    if a > 0 and a < 1e6:
        apogee = 2 * a - r_entry - R
    else:
        apogee = float('inf')  # Escape

    return {
        'entry_velocity_km_s': entry_velocity_km_s,
        'skip_out_velocity_km_s': v_skip_out,
        'velocity_loss_km_s': entry_velocity_km_s - v_skip_out,
        'apogee_altitude_km': apogee,
        'will_escape': apogee == float('inf'),
        'second_entry_velocity_km_s': v_skip_out if apogee < 1e5 else 0.0,
    }


def parachute_deployment_conditions(
    mach_target: float = 2.0,
    altitude_km: float = 10.0,
    body: str = 'earth'
) -> Dict[str, float]:
    """
    Calculate conditions for parachute deployment.

    Parameters
    ----------
    mach_target : float
        Target Mach number for deployment
    altitude_km : float
        Deployment altitude (km)
    body : str
        Target body

    Returns
    -------
    dict
        Deployment conditions
    """
    atm = ATMOSPHERES.get(body, ATMOSPHERES['earth'])

    # Temperature at altitude (simplified)
    T = atm.surface_temp * np.exp(-altitude_km / atm.scale_height * 0.1)
    T = max(T, 180)  # Minimum temperature

    # Speed of sound
    a = np.sqrt(atm.gamma * atm.gas_constant * T)

    # Deployment velocity
    v_deploy = mach_target * a

    # Dynamic pressure
    rho = atmospheric_density(altitude_km, atm)
    q = 0.5 * rho * v_deploy**2

    return {
        'altitude_km': altitude_km,
        'mach_number': mach_target,
        'velocity_m_s': v_deploy,
        'velocity_km_s': v_deploy / 1000,
        'dynamic_pressure_pa': q,
        'density_kg_m3': rho,
        'temperature_k': T,
        'speed_of_sound_m_s': a,
    }


def mars_edl_profile(
    entry_velocity_km_s: float = 5.5,
    entry_fpa_deg: float = -12.0,
    vehicle_mass_kg: float = 900.0,
    heatshield_diameter_m: float = 4.5
) -> Dict[str, Any]:
    """
    Generate Mars EDL profile similar to MSL/Perseverance.

    Parameters
    ----------
    entry_velocity_km_s : float
        Entry velocity
    entry_fpa_deg : float
        Entry flight path angle
    vehicle_mass_kg : float
        Entry vehicle mass
    heatshield_diameter_m : float
        Heatshield diameter

    Returns
    -------
    dict
        Complete Mars EDL profile
    """
    # Calculate parameters
    area = np.pi * (heatshield_diameter_m / 2)**2
    cd = 1.6  # Typical Mars entry vehicle
    beta = ballistic_coefficient(vehicle_mass_kg, cd, area)

    # Entry conditions
    entry = EntryConditions(
        velocity=entry_velocity_km_s,
        flight_path_angle=entry_fpa_deg,
        altitude=125.0,  # Mars entry interface
        latitude=0.0,
        longitude=0.0,
        azimuth=90.0
    )

    # Simulate entry
    result = simulate_entry(
        entry=entry,
        ballistic_coef=beta,
        nose_radius_m=heatshield_diameter_m / 4,  # Nose radius
        body='mars',
        lift_to_drag=0.24,  # Typical lifting entry
        dt=0.5
    )

    # Parachute deployment (Mach 2.0)
    chute = parachute_deployment_conditions(2.0, 10.0, 'mars')

    return {
        'entry_conditions': entry,
        'ballistic_coefficient_kg_m2': beta,
        'entry_result': result,
        'parachute_deployment': chute,
        'timeline': {
            'entry_interface_km': 125.0,
            'peak_heating_altitude_km': result.altitude_peak_heating,
            'peak_decel_altitude_km': result.altitude_peak_decel,
            'parachute_deploy_altitude_km': chute['altitude_km'],
        }
    }


def earth_reentry_corridor(
    velocity_km_s: float,
    vehicle_type: str = 'capsule'
) -> Dict[str, float]:
    """
    Calculate Earth reentry corridor for given velocity.

    Parameters
    ----------
    velocity_km_s : float
        Entry velocity (km/s)
    vehicle_type : str
        'capsule' (Apollo-like), 'shuttle', or 'ballistic'

    Returns
    -------
    dict
        Entry corridor parameters
    """
    # Corridor parameters depend on vehicle L/D
    ld_ratios = {
        'capsule': 0.3,    # Apollo: ~0.3
        'shuttle': 1.1,    # Space Shuttle: ~1.1
        'ballistic': 0.0,  # No lift
    }

    ld = ld_ratios.get(vehicle_type, 0.3)

    # Skip-out boundary (shallow limit)
    if velocity_km_s > 11.0:  # Lunar return
        shallow = -4.0 - ld * 2.0  # L/D helps avoid skip-out
    else:  # LEO
        shallow = -1.0 - ld * 1.0

    # Undershoot boundary (steep limit - g-load)
    if velocity_km_s > 11.0:
        steep = -7.5 + ld * 1.5  # L/D reduces peak g's
    else:
        steep = -4.0 + ld * 0.5

    # Corridor width
    width = abs(steep - shallow)

    return {
        'velocity_km_s': velocity_km_s,
        'vehicle_type': vehicle_type,
        'lift_to_drag': ld,
        'steep_limit_deg': steep,
        'shallow_limit_deg': shallow,
        'corridor_width_deg': width,
        'nominal_entry_deg': (steep + shallow) / 2,
    }
