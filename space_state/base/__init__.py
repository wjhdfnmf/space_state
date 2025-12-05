# base/__init__.py

from .star_config import CONFIG
from .star_constants import BODIES, world_time
from .star_orbits import ALL_ORBITS, EARTH_ORBITS, MOON_ORBITS
from .star_propagator import Propagator
from .star_dynamics import n_body_acceleration
from .star_states import get_circular_state, get_circular_state_at_apoapsis
from .star_ship import (
    Spacecraft, Thruster, Stage, StagedSpacecraft,
    centaur_rl10, starship_raptor_vac, agena_engine,
    starship_full_stack, saturn_v,
)
from .star_simulation import Simulation, SpacecraftEntity
from .star_io import (
    save_state, load_state,
    save_propagation_result, load_propagation_result,
    save_simulation, load_simulation_data,
    export_trajectory_csv, import_trajectory_csv,
)
from .star_environment import (
    SolarRadiationPressure, AtmosphericDrag, EnvironmentModel,
    compute_srp_acceleration, estimate_srp_delta_v,
    MultiBodyAtmosphere, compute_drag_acceleration, ATMOSPHERE_PARAMS,
)
from .star_power import (
    SolarPanel, Battery, PowerLoad, PowerSystem, PowerMode,
    detect_eclipse, detect_earth_eclipse, detect_moon_eclipse,
)
from .star_soi import (
    OrbitalRegime, SOIInfo, SOI_RADII,
    determine_soi, get_soi_info, get_body_position, get_body_velocity,
    create_soi_events, create_periapsis_event, create_apoapsis_event,
    create_atmosphere_entry_event, create_altitude_event, create_impact_event,
)
from .star_elements import (
    OrbitalElements,
    cartesian_to_keplerian, keplerian_to_cartesian,
    true_to_eccentric_anomaly, eccentric_to_true_anomaly,
    mean_to_eccentric_anomaly, eccentric_to_mean_anomaly,
    true_to_mean_anomaly, mean_to_true_anomaly,
    propagate_kepler, time_since_periapsis, time_to_true_anomaly,
    vis_viva, circular_velocity, escape_velocity, orbital_period,
    flight_path_angle, radius_at_true_anomaly,
)
from .star_eclipse import (
    EclipseType, EclipseState, EclipsePrediction,
    shadow_cone_geometry, cylindrical_shadow_check, conical_shadow_check,
    compute_eclipse_state, estimate_eclipse_duration_leo,
    estimate_eclipses_per_orbit_geo, predict_eclipses,
    communication_blackout_check,
)
from .star_thermal import (
    ThermalEnvironment, SpacecraftThermal, PLANET_PROPERTIES,
    STEFAN_BOLTZMANN, SOLAR_CONSTANT_1AU,
    solar_flux_at_distance, solar_flux_at_distance_km,
    planetary_view_factor, albedo_flux, planetary_ir_flux,
    equilibrium_temperature, compute_thermal_environment,
    hot_cold_case_temperatures, eclipse_cooldown_time, eclipse_warmup_time,
    estimate_heater_power, battery_sizing_for_eclipse, solar_panel_sizing,
)

__all__ = [
    "CONFIG",
    "BODIES", "world_time",
    "ALL_ORBITS", "EARTH_ORBITS", "MOON_ORBITS",
    "Propagator",
    "Simulation", "SpacecraftEntity",
    "n_body_acceleration",
    "get_circular_state", "get_circular_state_at_apoapsis",
    "Spacecraft", "Thruster", "Stage", "StagedSpacecraft",
    "centaur_rl10", "starship_raptor_vac", "agena_engine",
    "starship_full_stack", "saturn_v",
    "save_state", "load_state",
    "save_propagation_result", "load_propagation_result",
    "save_simulation", "load_simulation_data",
    "export_trajectory_csv", "import_trajectory_csv",
    "SolarRadiationPressure", "AtmosphericDrag", "EnvironmentModel",
    "compute_srp_acceleration", "estimate_srp_delta_v",
    "SolarPanel", "Battery", "PowerLoad", "PowerSystem", "PowerMode",
    "detect_eclipse", "detect_earth_eclipse", "detect_moon_eclipse",
    "MultiBodyAtmosphere", "compute_drag_acceleration", "ATMOSPHERE_PARAMS",
    "OrbitalRegime", "SOIInfo", "SOI_RADII",
    "determine_soi", "get_soi_info", "get_body_position", "get_body_velocity",
    "create_soi_events", "create_periapsis_event", "create_apoapsis_event",
    "create_atmosphere_entry_event", "create_altitude_event", "create_impact_event",
    # Orbital elements
    "OrbitalElements",
    "cartesian_to_keplerian", "keplerian_to_cartesian",
    "true_to_eccentric_anomaly", "eccentric_to_true_anomaly",
    "mean_to_eccentric_anomaly", "eccentric_to_mean_anomaly",
    "true_to_mean_anomaly", "mean_to_true_anomaly",
    "propagate_kepler", "time_since_periapsis", "time_to_true_anomaly",
    "vis_viva", "circular_velocity", "escape_velocity", "orbital_period",
    "flight_path_angle", "radius_at_true_anomaly",
    # Eclipse modeling
    "EclipseType", "EclipseState", "EclipsePrediction",
    "shadow_cone_geometry", "cylindrical_shadow_check", "conical_shadow_check",
    "compute_eclipse_state", "estimate_eclipse_duration_leo",
    "estimate_eclipses_per_orbit_geo", "predict_eclipses",
    "communication_blackout_check",
    # Thermal analysis
    "ThermalEnvironment", "SpacecraftThermal", "PLANET_PROPERTIES",
    "STEFAN_BOLTZMANN", "SOLAR_CONSTANT_1AU",
    "solar_flux_at_distance", "solar_flux_at_distance_km",
    "planetary_view_factor", "albedo_flux", "planetary_ir_flux",
    "equilibrium_temperature", "compute_thermal_environment",
    "hot_cold_case_temperatures", "eclipse_cooldown_time", "eclipse_warmup_time",
    "estimate_heater_power", "battery_sizing_for_eclipse", "solar_panel_sizing",
]

