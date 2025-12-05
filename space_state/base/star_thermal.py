# base/star_thermal.py
"""
Thermal Analysis Module

Provides spacecraft thermal environment calculations including:
- Solar flux at various distances
- Planetary albedo and IR flux
- Radiative equilibrium temperature
- Eclipse thermal transients

Heat Sources:
- Direct solar radiation
- Planetary reflected sunlight (albedo)
- Planetary infrared emission
- Internal heat dissipation

Heat Sinks:
- Radiation to space
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


# Physical Constants
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
SOLAR_CONSTANT_1AU = 1361.0  # W/m² at 1 AU
AU_TO_KM = 149597870.7  # km per AU


# Planetary thermal properties
PLANET_PROPERTIES = {
    'mercury': {
        'albedo': 0.12,
        'ir_emissivity': 0.90,
        'surface_temp_k': 440,  # Subsolar average
        'radius_km': 2439.7,
    },
    'venus': {
        'albedo': 0.76,
        'ir_emissivity': 0.99,
        'surface_temp_k': 737,
        'radius_km': 6051.8,
    },
    'earth': {
        'albedo': 0.30,
        'ir_emissivity': 0.95,
        'surface_temp_k': 288,
        'radius_km': 6378.137,
    },
    'moon': {
        'albedo': 0.12,
        'ir_emissivity': 0.90,
        'surface_temp_k': 270,  # Subsolar
        'radius_km': 1737.4,
    },
    'mars': {
        'albedo': 0.25,
        'ir_emissivity': 0.95,
        'surface_temp_k': 210,
        'radius_km': 3389.5,
    },
    'jupiter': {
        'albedo': 0.52,
        'ir_emissivity': 0.90,
        'surface_temp_k': 165,  # Cloud top
        'radius_km': 71492.0,
    },
    'saturn': {
        'albedo': 0.47,
        'ir_emissivity': 0.90,
        'surface_temp_k': 134,
        'radius_km': 58232.0,
    },
    'uranus': {
        'albedo': 0.51,
        'ir_emissivity': 0.90,
        'surface_temp_k': 76,
        'radius_km': 25362.0,
    },
    'neptune': {
        'albedo': 0.41,
        'ir_emissivity': 0.90,
        'surface_temp_k': 72,
        'radius_km': 24622.0,
    },
}


@dataclass
class ThermalEnvironment:
    """Thermal environment at a point in space."""
    solar_flux: float           # W/m² (direct solar)
    albedo_flux: float         # W/m² (reflected sunlight)
    planetary_ir: float        # W/m² (planetary infrared)
    total_incident: float      # W/m² (total heat input)
    sink_temperature: float    # K (deep space = 3K)
    equilibrium_temp: float    # K (for given absorptivity/emissivity)

    def __repr__(self) -> str:
        return (f"ThermalEnvironment(solar={self.solar_flux:.1f} W/m², "
                f"albedo={self.albedo_flux:.1f} W/m², "
                f"IR={self.planetary_ir:.1f} W/m², "
                f"T_eq={self.equilibrium_temp:.1f} K)")


@dataclass
class SpacecraftThermal:
    """Spacecraft thermal properties."""
    solar_absorptivity: float   # α_s (0-1)
    ir_emissivity: float        # ε_IR (0-1)
    area_sun_facing: float      # m² (projected area toward sun)
    area_radiator: float        # m² (radiating area)
    internal_power: float       # W (internal heat dissipation)
    mass: float                 # kg
    specific_heat: float        # J/(kg·K)


def solar_flux_at_distance(distance_au: float) -> float:
    """
    Calculate solar flux at a given distance from the Sun.

    Parameters
    ----------
    distance_au : float
        Distance from Sun (AU)

    Returns
    -------
    float
        Solar flux (W/m²)
    """
    return SOLAR_CONSTANT_1AU / (distance_au ** 2)


def solar_flux_at_distance_km(distance_km: float) -> float:
    """
    Calculate solar flux at a given distance from the Sun.

    Parameters
    ----------
    distance_km : float
        Distance from Sun (km)

    Returns
    -------
    float
        Solar flux (W/m²)
    """
    distance_au = distance_km / AU_TO_KM
    return solar_flux_at_distance(distance_au)


def planetary_view_factor(
    altitude_km: float,
    planet_radius_km: float
) -> float:
    """
    Calculate view factor from spacecraft to planet.

    For a point above a sphere:
    F = (r_p / (r_p + h))² = sin²(ρ)

    Parameters
    ----------
    altitude_km : float
        Altitude above planet surface (km)
    planet_radius_km : float
        Planet radius (km)

    Returns
    -------
    float
        View factor (0-1)
    """
    r = planet_radius_km + altitude_km
    sin_rho = planet_radius_km / r
    return sin_rho ** 2


def albedo_flux(
    solar_flux: float,
    albedo: float,
    view_factor: float
) -> float:
    """
    Calculate albedo (reflected sunlight) flux.

    Parameters
    ----------
    solar_flux : float
        Incident solar flux at planet (W/m²)
    albedo : float
        Planet albedo (0-1)
    view_factor : float
        View factor to planet (0-1)

    Returns
    -------
    float
        Albedo flux incident on spacecraft (W/m²)
    """
    return solar_flux * albedo * view_factor


def planetary_ir_flux(
    planet_temp: float,
    view_factor: float,
    emissivity: float = 1.0
) -> float:
    """
    Calculate planetary infrared flux.

    Parameters
    ----------
    planet_temp : float
        Effective planet temperature (K)
    view_factor : float
        View factor to planet (0-1)
    emissivity : float
        Planet effective emissivity

    Returns
    -------
    float
        IR flux incident on spacecraft (W/m²)
    """
    return STEFAN_BOLTZMANN * emissivity * (planet_temp ** 4) * view_factor


def equilibrium_temperature(
    absorbed_flux: float,
    emissivity: float,
    area_ratio: float = 1.0
) -> float:
    """
    Calculate radiative equilibrium temperature.

    T = (Q_absorbed / (ε × σ × A_radiator))^0.25

    Parameters
    ----------
    absorbed_flux : float
        Total absorbed heat per unit absorber area (W/m²)
    emissivity : float
        IR emissivity of radiating surface
    area_ratio : float
        Ratio of radiator area to absorber area

    Returns
    -------
    float
        Equilibrium temperature (K)
    """
    if absorbed_flux <= 0:
        return 3.0  # Deep space background

    return (absorbed_flux / (STEFAN_BOLTZMANN * emissivity * area_ratio)) ** 0.25


def compute_thermal_environment(
    distance_from_sun_au: float,
    altitude_km: Optional[float] = None,
    planet: Optional[str] = None,
    absorptivity: float = 0.3,
    emissivity: float = 0.9
) -> ThermalEnvironment:
    """
    Compute complete thermal environment.

    Parameters
    ----------
    distance_from_sun_au : float
        Distance from Sun (AU)
    altitude_km : float, optional
        Altitude above planet (km). If None, deep space.
    planet : str, optional
        Planet name for albedo/IR. If None, deep space.
    absorptivity : float
        Solar absorptivity (α)
    emissivity : float
        IR emissivity (ε)

    Returns
    -------
    ThermalEnvironment
        Complete thermal environment
    """
    # Direct solar flux
    solar = solar_flux_at_distance(distance_from_sun_au)

    # Planetary contributions
    albedo = 0.0
    ir = 0.0

    if planet and altitude_km is not None and planet in PLANET_PROPERTIES:
        props = PLANET_PROPERTIES[planet]
        vf = planetary_view_factor(altitude_km, props['radius_km'])
        albedo = albedo_flux(solar, props['albedo'], vf)
        ir = planetary_ir_flux(props['surface_temp_k'], vf, props['ir_emissivity'])

    # Total incident flux (absorbed)
    total = solar * absorptivity + albedo * absorptivity + ir * emissivity

    # Equilibrium temperature
    t_eq = equilibrium_temperature(total, emissivity)

    return ThermalEnvironment(
        solar_flux=solar,
        albedo_flux=albedo,
        planetary_ir=ir,
        total_incident=total,
        sink_temperature=3.0,  # Deep space
        equilibrium_temp=t_eq
    )


def hot_cold_case_temperatures(
    distance_from_sun_au: float,
    altitude_km: Optional[float] = None,
    planet: Optional[str] = None,
    alpha_hot: float = 0.9,
    epsilon_hot: float = 0.3,
    alpha_cold: float = 0.1,
    epsilon_cold: float = 0.9
) -> Tuple[float, float]:
    """
    Calculate hot and cold case equilibrium temperatures.

    Hot case: High absorptivity, low emissivity
    Cold case: Low absorptivity, high emissivity

    Parameters
    ----------
    distance_from_sun_au : float
        Distance from Sun
    altitude_km : float, optional
        Altitude above planet
    planet : str, optional
        Planet name
    alpha_hot, epsilon_hot : float
        Hot case optical properties
    alpha_cold, epsilon_cold : float
        Cold case optical properties

    Returns
    -------
    T_hot, T_cold : tuple
        Hot and cold case temperatures (K)
    """
    env_hot = compute_thermal_environment(
        distance_from_sun_au, altitude_km, planet,
        alpha_hot, epsilon_hot
    )
    env_cold = compute_thermal_environment(
        distance_from_sun_au, altitude_km, planet,
        alpha_cold, epsilon_cold
    )

    return env_hot.equilibrium_temp, env_cold.equilibrium_temp


def eclipse_cooldown_time(
    spacecraft: SpacecraftThermal,
    initial_temp: float,
    final_temp: float,
    environment: ThermalEnvironment
) -> float:
    """
    Estimate time to cool down during eclipse.

    Uses lumped capacitance model.

    Parameters
    ----------
    spacecraft : SpacecraftThermal
        Spacecraft thermal properties
    initial_temp : float
        Initial temperature (K)
    final_temp : float
        Target temperature (K)
    environment : ThermalEnvironment
        Thermal environment (eclipse = zero solar)

    Returns
    -------
    float
        Cooldown time (seconds)
    """
    if final_temp >= initial_temp:
        return 0.0

    # Thermal mass
    C = spacecraft.mass * spacecraft.specific_heat  # J/K

    # Radiative cooling rate at initial temperature
    q_rad = STEFAN_BOLTZMANN * spacecraft.ir_emissivity * spacecraft.area_radiator * initial_temp**4

    # Internal heat generation
    q_internal = spacecraft.internal_power

    # Net cooling rate
    q_net = q_rad - q_internal  # W

    if q_net <= 0:
        return np.inf  # Won't cool down

    # Approximate linear cooling for small temperature changes
    dT = initial_temp - final_temp
    dt = C * dT / q_net

    return dt


def eclipse_warmup_time(
    spacecraft: SpacecraftThermal,
    initial_temp: float,
    final_temp: float,
    environment: ThermalEnvironment
) -> float:
    """
    Estimate time to warm up after eclipse exit.

    Parameters
    ----------
    spacecraft : SpacecraftThermal
        Spacecraft thermal properties
    initial_temp : float
        Initial temperature (K)
    final_temp : float
        Target temperature (K)
    environment : ThermalEnvironment
        Thermal environment

    Returns
    -------
    float
        Warmup time (seconds)
    """
    if final_temp <= initial_temp:
        return 0.0

    # Thermal mass
    C = spacecraft.mass * spacecraft.specific_heat

    # Heat input
    q_solar = environment.solar_flux * spacecraft.solar_absorptivity * spacecraft.area_sun_facing
    q_internal = spacecraft.internal_power
    q_in = q_solar + q_internal

    # Radiative loss at initial temp
    q_out = STEFAN_BOLTZMANN * spacecraft.ir_emissivity * spacecraft.area_radiator * initial_temp**4

    # Net heating rate
    q_net = q_in - q_out

    if q_net <= 0:
        return np.inf  # Won't warm up

    dT = final_temp - initial_temp
    dt = C * dT / q_net

    return dt


def estimate_heater_power(
    spacecraft: SpacecraftThermal,
    min_temp: float,
    eclipse_environment: ThermalEnvironment
) -> float:
    """
    Estimate heater power needed to maintain minimum temperature during eclipse.

    Parameters
    ----------
    spacecraft : SpacecraftThermal
        Spacecraft thermal properties
    min_temp : float
        Minimum allowable temperature (K)
    eclipse_environment : ThermalEnvironment
        Eclipse thermal environment

    Returns
    -------
    float
        Required heater power (W)
    """
    # Heat loss during eclipse (radiation only)
    q_rad = STEFAN_BOLTZMANN * spacecraft.ir_emissivity * spacecraft.area_radiator * min_temp**4

    # Internal dissipation helps
    q_internal = spacecraft.internal_power

    # Net heater power needed
    heater_power = max(0, q_rad - q_internal)

    return heater_power


def battery_sizing_for_eclipse(
    spacecraft: SpacecraftThermal,
    eclipse_duration_min: float,
    heater_power: float,
    other_loads_w: float,
    depth_of_discharge: float = 0.3,
    battery_efficiency: float = 0.9
) -> Dict[str, float]:
    """
    Size battery for eclipse survival.

    Parameters
    ----------
    spacecraft : SpacecraftThermal
        Spacecraft properties
    eclipse_duration_min : float
        Maximum eclipse duration (minutes)
    heater_power : float
        Heater power requirement (W)
    other_loads_w : float
        Other electrical loads during eclipse (W)
    depth_of_discharge : float
        Maximum DoD (0-1)
    battery_efficiency : float
        Round-trip efficiency

    Returns
    -------
    dict
        Battery sizing results
    """
    eclipse_hours = eclipse_duration_min / 60
    total_power = heater_power + other_loads_w

    # Energy required
    energy_wh = total_power * eclipse_hours

    # Battery capacity (accounting for DoD and efficiency)
    capacity_wh = energy_wh / (depth_of_discharge * battery_efficiency)

    return {
        'eclipse_duration_min': eclipse_duration_min,
        'total_power_w': total_power,
        'heater_power_w': heater_power,
        'energy_required_wh': energy_wh,
        'battery_capacity_wh': capacity_wh,
        'depth_of_discharge': depth_of_discharge,
    }


def solar_panel_sizing(
    power_required_w: float,
    distance_au: float,
    panel_efficiency: float = 0.30,
    degradation_factor: float = 0.90,
    sun_incidence_factor: float = 0.85
) -> Dict[str, float]:
    """
    Size solar panels for given power requirement.

    Parameters
    ----------
    power_required_w : float
        End-of-life power requirement (W)
    distance_au : float
        Distance from Sun (AU)
    panel_efficiency : float
        Panel efficiency (0-1)
    degradation_factor : float
        End-of-life degradation factor
    sun_incidence_factor : float
        Average cosine of sun angle

    Returns
    -------
    dict
        Solar panel sizing results
    """
    # Solar flux at distance
    flux = solar_flux_at_distance(distance_au)

    # Effective power per unit area
    power_density = flux * panel_efficiency * degradation_factor * sun_incidence_factor

    # Required area
    area_m2 = power_required_w / power_density

    return {
        'power_required_w': power_required_w,
        'distance_au': distance_au,
        'solar_flux_w_m2': flux,
        'power_density_w_m2': power_density,
        'panel_area_m2': area_m2,
        'panel_efficiency': panel_efficiency,
    }
