# star_guidance/__init__.py

from .test_burn import BurnController
from .finite_burn_zemzev2 import FiniteBurnZEMZEV_V2
from .lambert import LambertSolution, lambert_solve, lambert_universal, lambert_multirev
from .porkchop import PorkchopResult, compute_porkchop, compute_lunar_porkchop, plot_porkchop
from .proximity_ops import (
    ApproachMode, RelativeState, ManeuverPlan,
    ProximityGuidanceConfig, ProximityGuidance,
    inertial_to_lvlh, lvlh_to_inertial, get_relative_state,
    cw_state_transition, cw_delta_v_to_target,
    compute_vbar_approach, compute_rbar_approach,
    compute_station_keeping, compute_hohmann_transfer_relative,
    predict_closest_approach, compute_safe_trajectory,
)
from .flyby import (
    FlybyGeometry, BPlaneTarget,
    compute_v_infinity, flyby_turn_angle, flyby_periapsis_from_turn_angle,
    b_plane_from_periapsis, periapsis_from_b_plane,
    rotate_v_infinity, compute_flyby, powered_flyby_delta_v,
    maximum_turn_angle, flyby_delta_v_vector, design_flyby_for_target,
    gravity_assist_sequence_delta_v,
)
from .patched_conics import (
    ReferenceFrame, TrajectorySegment, PatchedConicTrajectory,
    sphere_of_influence_radius, get_soi_radius, SOI_RADII_AU,
    heliocentric_to_planetocentric, planetocentric_to_heliocentric,
    check_soi_entry, compute_c3, v_infinity_from_c3,
    departure_v_infinity, arrival_conditions,
    interplanetary_transfer_type, estimate_transfer_delta_v,
)
from .station_keeping import (
    StationKeepingMode, GEOStationKeepingBudget, LagrangePoint, LowThrustBudget,
    geo_inclination_drift, geo_north_south_delta_v, geo_east_west_delta_v,
    geo_eccentricity_delta_v, estimate_geo_station_keeping_budget,
    compute_lagrange_points, sun_earth_lagrange_points, earth_moon_lagrange_points,
    lagrange_station_keeping_budget, low_thrust_delta_v, estimate_low_thrust_budget,
    hohmann_correction, plane_change_delta_v, combined_maneuver_delta_v,
)
from .entry_descent import (
    EntryType, AtmosphereModel, EntryConditions, EntryTrajectoryPoint, EntryResult,
    ATMOSPHERES, atmospheric_density, ballistic_coefficient,
    entry_corridor_bounds, peak_deceleration_estimate, peak_heating_rate,
    total_heat_load, simulate_entry, skip_entry_analysis,
    parachute_deployment_conditions, mars_edl_profile, earth_reentry_corridor,
)
from .ground_track import (
    GroundStation, VisibilityWindow, CoverageResult, GROUND_STATIONS,
    eci_to_ecef, ecef_to_geodetic, ground_track,
    elevation_angle, azimuth_angle, find_visibility_windows,
    compute_access_statistics, swath_width, revisit_time_estimate,
    sun_synchronous_inclination, walker_constellation_coverage,
    horizon_distance, line_of_sight_range, communication_window_geometry,
)
from .launch_window import (
    LaunchSite, LaunchWindow, InterplanetaryWindow, LAUNCH_SITES,
    launch_azimuth, launch_window_for_inclination, iss_launch_window,
    synodic_period, hohmann_phase_angle, interplanetary_launch_opportunities,
    earth_mars_window_dates, plane_change_penalty, gto_launch_parameters,
    daily_launch_windows,
)

__all__ = [
    "BurnController",
    "FiniteBurnZEMZEV_V2",
    "LambertSolution", "lambert_solve", "lambert_universal", "lambert_multirev",
    "PorkchopResult", "compute_porkchop", "compute_lunar_porkchop", "plot_porkchop",
    # Proximity operations
    "ApproachMode", "RelativeState", "ManeuverPlan",
    "ProximityGuidanceConfig", "ProximityGuidance",
    "inertial_to_lvlh", "lvlh_to_inertial", "get_relative_state",
    "cw_state_transition", "cw_delta_v_to_target",
    "compute_vbar_approach", "compute_rbar_approach",
    "compute_station_keeping", "compute_hohmann_transfer_relative",
    "predict_closest_approach", "compute_safe_trajectory",
    # Flyby / Gravity assist
    "FlybyGeometry", "BPlaneTarget",
    "compute_v_infinity", "flyby_turn_angle", "flyby_periapsis_from_turn_angle",
    "b_plane_from_periapsis", "periapsis_from_b_plane",
    "rotate_v_infinity", "compute_flyby", "powered_flyby_delta_v",
    "maximum_turn_angle", "flyby_delta_v_vector", "design_flyby_for_target",
    "gravity_assist_sequence_delta_v",
    # Patched conics
    "ReferenceFrame", "TrajectorySegment", "PatchedConicTrajectory",
    "sphere_of_influence_radius", "get_soi_radius", "SOI_RADII_AU",
    "heliocentric_to_planetocentric", "planetocentric_to_heliocentric",
    "check_soi_entry", "compute_c3", "v_infinity_from_c3",
    "departure_v_infinity", "arrival_conditions",
    "interplanetary_transfer_type", "estimate_transfer_delta_v",
    # Station keeping
    "StationKeepingMode", "GEOStationKeepingBudget", "LagrangePoint", "LowThrustBudget",
    "geo_inclination_drift", "geo_north_south_delta_v", "geo_east_west_delta_v",
    "geo_eccentricity_delta_v", "estimate_geo_station_keeping_budget",
    "compute_lagrange_points", "sun_earth_lagrange_points", "earth_moon_lagrange_points",
    "lagrange_station_keeping_budget", "low_thrust_delta_v", "estimate_low_thrust_budget",
    "hohmann_correction", "plane_change_delta_v", "combined_maneuver_delta_v",
    # Entry, Descent, Landing
    "EntryType", "AtmosphereModel", "EntryConditions", "EntryTrajectoryPoint", "EntryResult",
    "ATMOSPHERES", "atmospheric_density", "ballistic_coefficient",
    "entry_corridor_bounds", "peak_deceleration_estimate", "peak_heating_rate",
    "total_heat_load", "simulate_entry", "skip_entry_analysis",
    "parachute_deployment_conditions", "mars_edl_profile", "earth_reentry_corridor",
    # Ground track and coverage
    "GroundStation", "VisibilityWindow", "CoverageResult", "GROUND_STATIONS",
    "eci_to_ecef", "ecef_to_geodetic", "ground_track",
    "elevation_angle", "azimuth_angle", "find_visibility_windows",
    "compute_access_statistics", "swath_width", "revisit_time_estimate",
    "sun_synchronous_inclination", "walker_constellation_coverage",
    "horizon_distance", "line_of_sight_range", "communication_window_geometry",
    # Launch windows
    "LaunchSite", "LaunchWindow", "InterplanetaryWindow", "LAUNCH_SITES",
    "launch_azimuth", "launch_window_for_inclination", "iss_launch_window",
    "synodic_period", "hohmann_phase_angle", "interplanetary_launch_opportunities",
    "earth_mars_window_dates", "plane_change_penalty", "gto_launch_parameters",
    "daily_launch_windows",
]

