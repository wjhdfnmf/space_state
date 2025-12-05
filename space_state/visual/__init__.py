# visual/__init__.py
"""
Visualization Module

Provides 2D and 3D visualization capabilities including:
- Static orbit/trajectory plots
- Interactive web-based viewers with time sliders
- Mission dashboards
- Attitude visualization
- Ground track mapping
- Maneuver planning tools
"""

from .star_vis2D import plot_orbital_metrics
from .star_vis3D import Orbit3D
from .star_review import review, plot_attitude_metrics

from .orbit_3d import (
    # Classes
    OrbitPlotter3D,
    # Static plots
    plot_orbit_3d,
    plot_trajectory_3d,
    plot_body,
    plot_reference_frame,
    # Animation
    create_orbit_animation,
    create_trajectory_animation,
    # Ground tracks
    plot_ground_track,
    # Multi-body
    plot_solar_system,
    plot_earth_moon,
    # Interactive
    create_interactive_orbit,
    create_interactive_trajectory,
    # Utilities
    set_equal_aspect_3d,
    create_sphere,
    BODY_COLORS,
    BODY_RADII_KM,
)

from .maneuver_planner import (
    ManeuverNode,
    hohmann_delta_v,
    plane_change_delta_v,
    combined_maneuver_delta_v,
    preview_orbit,
    create_maneuver_figure,
    create_maneuver_planner_app,
    run_maneuver_planner,
)

# Interactive mission visualization (requires Plotly/Dash)
from .interactive_mission import (
    # Data structures
    TrajectoryData,
    BodyData,
    EventMarker,
    # Interactive viewers
    create_interactive_trajectory_viewer,
    create_mission_dashboard,
    create_attitude_visualizer,
    create_interactive_ground_track,
    create_mission_timeline,
    # Dash application
    create_mission_viewer_app,
    run_mission_viewer,
    # Utilities
    trajectory_from_propagation_result,
)

__all__ = [
    # Existing
    "plot_orbital_metrics",
    "Orbit3D",
    "review",
    "plot_attitude_metrics",
    # New 3D visualization
    "OrbitPlotter3D",
    "plot_orbit_3d",
    "plot_trajectory_3d",
    "plot_body",
    "plot_reference_frame",
    "create_orbit_animation",
    "create_trajectory_animation",
    "plot_ground_track",
    "plot_solar_system",
    "plot_earth_moon",
    "create_interactive_orbit",
    "create_interactive_trajectory",
    "set_equal_aspect_3d",
    "create_sphere",
    "BODY_COLORS",
    "BODY_RADII_KM",
    # Maneuver Planner
    "ManeuverNode",
    "hohmann_delta_v",
    "plane_change_delta_v",
    "combined_maneuver_delta_v",
    "preview_orbit",
    "create_maneuver_figure",
    "create_maneuver_planner_app",
    "run_maneuver_planner",
    # Interactive Mission Visualization
    "TrajectoryData",
    "BodyData",
    "EventMarker",
    "create_interactive_trajectory_viewer",
    "create_mission_dashboard",
    "create_attitude_visualizer",
    "create_interactive_ground_track",
    "create_mission_timeline",
    "create_mission_viewer_app",
    "run_mission_viewer",
    "trajectory_from_propagation_result",
]