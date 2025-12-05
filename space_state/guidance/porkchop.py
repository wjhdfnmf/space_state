# guidance/porkchop.py
"""
Porkchop Plot Generator

Generates delta-V contour plots for interplanetary/lunar transfers
as a function of departure and arrival dates.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from tqdm import tqdm

from .lambert import lambert_solve, LambertSolution
from base.star_constants import BODIES, ts, epoch_jd
from base.star_utils import body_position, body_velocity, apd_to_mps, au_to_km
from base.star_logger import logger


@dataclass
class PorkchopResult:
    """Results from porkchop plot computation."""
    departure_dates: np.ndarray  # Days from epoch
    arrival_dates: np.ndarray    # Days from epoch
    dv1_grid: np.ndarray         # Departure delta-V (m/s)
    dv2_grid: np.ndarray         # Arrival delta-V (m/s)
    dv_total_grid: np.ndarray    # Total delta-V (m/s)
    tof_grid: np.ndarray         # Time of flight (days)
    c3_grid: np.ndarray          # C3 energy (km²/s²)

    # Optimal points
    min_dv_total: float
    min_dv_departure: float
    min_dv_arrival: float
    optimal_departure: float
    optimal_arrival: float
    optimal_tof: float

    # Metadata
    origin_body: str
    target_body: str
    origin_orbit_alt_km: float
    target_orbit_alt_km: float


def compute_porkchop(
    origin_body: str = 'earth',
    target_body: str = 'mars barycenter',
    departure_start: float = 0.0,
    departure_end: float = 365.0,
    arrival_start: float = 100.0,
    arrival_end: float = 500.0,
    n_departure: int = 50,
    n_arrival: int = 50,
    origin_orbit_alt_km: float = 200.0,
    target_orbit_alt_km: float = 200.0,
    prograde: bool = True,
    show_progress: bool = True,
) -> PorkchopResult:
    """
    Compute porkchop plot data for interplanetary transfer.

    Args:
        origin_body: Starting body name (e.g., 'earth')
        target_body: Destination body name (e.g., 'mars barycenter')
        departure_start: First departure date (days from epoch)
        departure_end: Last departure date (days from epoch)
        arrival_start: First arrival date (days from epoch)
        arrival_end: Last arrival date (days from epoch)
        n_departure: Number of departure date samples
        n_arrival: Number of arrival date samples
        origin_orbit_alt_km: Parking orbit altitude at origin (km)
        target_orbit_alt_km: Target orbit altitude at destination (km)
        prograde: True for prograde transfers
        show_progress: Show progress bar

    Returns:
        PorkchopResult with all computed data
    """
    # Get body data
    origin = BODIES[origin_body]
    target = BODIES[target_body]
    mu_sun = BODIES['sun']['mu']  # Heliocentric transfer

    # Create date grids
    dep_dates = np.linspace(departure_start, departure_end, n_departure)
    arr_dates = np.linspace(arrival_start, arrival_end, n_arrival)

    # Initialize result grids
    dv1_grid = np.full((n_arrival, n_departure), np.nan)
    dv2_grid = np.full((n_arrival, n_departure), np.nan)
    dv_total_grid = np.full((n_arrival, n_departure), np.nan)
    tof_grid = np.full((n_arrival, n_departure), np.nan)
    c3_grid = np.full((n_arrival, n_departure), np.nan)

    # Compute escape velocity at origin and capture velocity at target
    origin_radius = origin['radius'] + au_to_km(origin_orbit_alt_km) / 1.496e8
    target_radius = target['radius'] + au_to_km(target_orbit_alt_km) / 1.496e8

    # Iterate over all date combinations
    iterator = tqdm(range(n_departure), desc="Computing porkchop", disable=not show_progress)

    for i in iterator:
        t_dep = dep_dates[i]

        # Get origin position and velocity at departure
        time_dep = ts.tt_jd(epoch_jd + t_dep)
        r1 = body_position(origin, time_dep)
        v1_body = body_velocity(origin, time_dep)

        for j in range(n_arrival):
            t_arr = arr_dates[j]
            tof = t_arr - t_dep

            # Skip if arrival before departure
            if tof <= 0:
                continue

            # Get target position and velocity at arrival
            time_arr = ts.tt_jd(epoch_jd + t_arr)
            r2 = body_position(target, time_arr)
            v2_body = body_velocity(target, time_arr)

            # Solve Lambert problem
            solution = lambert_solve(
                r1, r2, tof, mu_sun,
                v1_initial=v1_body,
                v2_target=v2_body,
                prograde=prograde,
            )

            if solution is None or not solution.converged:
                continue

            # Compute v_infinity at departure and arrival
            v_inf_dep = solution.v1 - v1_body  # Hyperbolic excess velocity at origin
            v_inf_arr = solution.v2 - v2_body  # Hyperbolic excess velocity at target

            v_inf_dep_mag = np.linalg.norm(v_inf_dep)  # AU/day
            v_inf_arr_mag = np.linalg.norm(v_inf_arr)  # AU/day

            # Convert to km/s for delta-V calculations
            v_inf_dep_kps = v_inf_dep_mag * 1731.46  # AU/day to km/s
            v_inf_arr_kps = v_inf_arr_mag * 1731.46

            # C3 (characteristic energy) at departure
            c3 = v_inf_dep_kps**2  # km²/s²

            # Delta-V to escape from parking orbit
            # v_escape = sqrt(v_inf² + 2*mu/r) - v_circular
            mu_origin_kps = origin['mu'] * (1.496e8)**3 / 86400**2  # Convert to km³/s²
            r_park_km = (origin['radius'] + au_to_km(origin_orbit_alt_km) / 1.496e8) * 1.496e8

            v_circular_origin = np.sqrt(mu_origin_kps / r_park_km)
            v_escape = np.sqrt(v_inf_dep_kps**2 + 2 * mu_origin_kps / r_park_km)
            dv_departure = (v_escape - v_circular_origin) * 1000  # m/s

            # Delta-V to capture into target orbit
            mu_target_kps = target['mu'] * (1.496e8)**3 / 86400**2
            r_target_km = (target['radius'] + au_to_km(target_orbit_alt_km) / 1.496e8) * 1.496e8

            v_circular_target = np.sqrt(mu_target_kps / r_target_km)
            v_arrival = np.sqrt(v_inf_arr_kps**2 + 2 * mu_target_kps / r_target_km)
            dv_arrival = (v_arrival - v_circular_target) * 1000  # m/s

            # Store results
            dv1_grid[j, i] = dv_departure
            dv2_grid[j, i] = dv_arrival
            dv_total_grid[j, i] = dv_departure + dv_arrival
            tof_grid[j, i] = tof
            c3_grid[j, i] = c3

    # Find optimal point
    valid_mask = ~np.isnan(dv_total_grid)
    if np.any(valid_mask):
        min_idx = np.nanargmin(dv_total_grid)
        j_opt, i_opt = np.unravel_index(min_idx, dv_total_grid.shape)
        min_dv_total = dv_total_grid[j_opt, i_opt]
        optimal_departure = dep_dates[i_opt]
        optimal_arrival = arr_dates[j_opt]
        optimal_tof = optimal_arrival - optimal_departure
        min_dv_departure = dv1_grid[j_opt, i_opt]
        min_dv_arrival = dv2_grid[j_opt, i_opt]
    else:
        min_dv_total = np.nan
        optimal_departure = np.nan
        optimal_arrival = np.nan
        optimal_tof = np.nan
        min_dv_departure = np.nan
        min_dv_arrival = np.nan

    logger.info(f"Porkchop computation complete: {origin_body} → {target_body}")
    logger.info(f"  Optimal departure: day {optimal_departure:.1f}")
    logger.info(f"  Optimal arrival: day {optimal_arrival:.1f}")
    logger.info(f"  Optimal TOF: {optimal_tof:.1f} days")
    logger.info(f"  Minimum Δv: {min_dv_total:.0f} m/s")

    return PorkchopResult(
        departure_dates=dep_dates,
        arrival_dates=arr_dates,
        dv1_grid=dv1_grid,
        dv2_grid=dv2_grid,
        dv_total_grid=dv_total_grid,
        tof_grid=tof_grid,
        c3_grid=c3_grid,
        min_dv_total=min_dv_total,
        min_dv_departure=min_dv_departure,
        min_dv_arrival=min_dv_arrival,
        optimal_departure=optimal_departure,
        optimal_arrival=optimal_arrival,
        optimal_tof=optimal_tof,
        origin_body=origin_body,
        target_body=target_body,
        origin_orbit_alt_km=origin_orbit_alt_km,
        target_orbit_alt_km=target_orbit_alt_km,
    )


def plot_porkchop(
    result: PorkchopResult,
    plot_type: str = 'total',
    levels: int = 20,
    max_dv: float = None,
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot porkchop contour.

    Args:
        result: PorkchopResult from compute_porkchop
        plot_type: 'total', 'departure', 'arrival', 'c3', or 'tof'
        levels: Number of contour levels
        max_dv: Maximum delta-V to display (clips higher values)
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Select data to plot
    if plot_type == 'total':
        data = result.dv_total_grid
        label = 'Total Δv (m/s)'
        title = f'Total Delta-V: {result.origin_body.title()} → {result.target_body.title()}'
    elif plot_type == 'departure':
        data = result.dv1_grid
        label = 'Departure Δv (m/s)'
        title = f'Departure Delta-V: {result.origin_body.title()} → {result.target_body.title()}'
    elif plot_type == 'arrival':
        data = result.dv2_grid
        label = 'Arrival Δv (m/s)'
        title = f'Arrival Delta-V: {result.origin_body.title()} → {result.target_body.title()}'
    elif plot_type == 'c3':
        data = result.c3_grid
        label = 'C3 (km²/s²)'
        title = f'Launch C3: {result.origin_body.title()} → {result.target_body.title()}'
    elif plot_type == 'tof':
        data = result.tof_grid
        label = 'Time of Flight (days)'
        title = f'Transfer Time: {result.origin_body.title()} → {result.target_body.title()}'
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    # Clip data if max_dv specified
    if max_dv is not None and plot_type in ['total', 'departure', 'arrival']:
        data = np.clip(data, None, max_dv)

    # Create meshgrid
    D, A = np.meshgrid(result.departure_dates, result.arrival_dates)

    # Plot contours
    contour = ax.contourf(D, A, data, levels=levels, cmap='viridis', extend='both')
    ax.contour(D, A, data, levels=levels//2, colors='white', alpha=0.3, linewidths=0.5)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, label=label)

    # Mark optimal point
    if not np.isnan(result.optimal_departure):
        ax.plot(result.optimal_departure, result.optimal_arrival, 'r*',
                markersize=15, markeredgecolor='white', markeredgewidth=1,
                label=f'Optimal: {result.min_dv_total:.0f} m/s')
        ax.legend(loc='upper left')

    # Add TOF contour lines
    if plot_type != 'tof':
        tof_levels = [100, 200, 300, 400, 500]
        tof_contours = ax.contour(D, A, result.tof_grid, levels=tof_levels,
                                   colors='red', linestyles='dashed', alpha=0.5)
        ax.clabel(tof_contours, inline=True, fontsize=8, fmt='%d days')

    ax.set_xlabel('Departure Date (days from epoch)')
    ax.set_ylabel('Arrival Date (days from epoch)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved porkchop plot to {save_path}")

    if show:
        plt.show()

    return fig


def compute_lunar_porkchop(
    departure_start: float = 0.0,
    departure_end: float = 30.0,
    tof_min: float = 2.0,
    tof_max: float = 5.0,
    n_departure: int = 50,
    n_tof: int = 50,
    parking_alt_km: float = 200.0,
    lunar_alt_km: float = 100.0,
    show_progress: bool = True,
) -> PorkchopResult:
    """
    Compute porkchop for Earth-Moon transfer.

    Uses TOF instead of arrival date for cleaner visualization.
    """
    earth = BODIES['earth']
    moon = BODIES['moon']
    mu_earth = earth['mu']

    dep_dates = np.linspace(departure_start, departure_end, n_departure)
    tof_values = np.linspace(tof_min, tof_max, n_tof)

    dv1_grid = np.full((n_tof, n_departure), np.nan)
    dv2_grid = np.full((n_tof, n_departure), np.nan)
    dv_total_grid = np.full((n_tof, n_departure), np.nan)
    tof_grid = np.full((n_tof, n_departure), np.nan)
    c3_grid = np.full((n_tof, n_departure), np.nan)

    iterator = tqdm(range(n_departure), desc="Lunar porkchop", disable=not show_progress)

    for i in iterator:
        t_dep = dep_dates[i]
        time_dep = ts.tt_jd(epoch_jd + t_dep)

        # Departure from circular Earth orbit
        r_earth = body_position(earth, time_dep)
        v_earth = body_velocity(earth, time_dep)

        for j, tof in enumerate(tof_values):
            t_arr = t_dep + tof
            time_arr = ts.tt_jd(epoch_jd + t_arr)

            # Moon position at arrival
            r_moon = body_position(moon, time_arr)
            v_moon = body_velocity(moon, time_arr)

            # Earth-centered positions
            r1_rel = np.array([earth['radius'] + parking_alt_km / 1.496e8, 0.0, 0.0])
            r2_rel = r_moon - r_earth

            # Solve Lambert in Earth-centered frame
            solution = lambert_solve(
                r1_rel, r2_rel, tof, mu_earth,
                prograde=True,
            )

            if solution is None or not solution.converged:
                continue

            # Circular velocity at parking orbit
            r_park = np.linalg.norm(r1_rel)
            v_circ = np.sqrt(mu_earth / r_park)

            # Departure delta-V
            v1_mag = np.linalg.norm(solution.v1)
            dv_dep = abs(v1_mag - v_circ) * 1731.46 * 1000  # m/s

            # Arrival delta-V (relative to Moon)
            v_arr_rel = solution.v2 - (v_moon - v_earth)
            v_inf = np.linalg.norm(v_arr_rel) * 1731.46  # km/s

            # Capture into lunar orbit
            mu_moon = moon['mu'] * (1.496e8)**3 / 86400**2
            r_lunar = (moon['radius'] + lunar_alt_km / 1.496e8) * 1.496e8
            v_circ_lunar = np.sqrt(mu_moon / r_lunar)
            v_capture = np.sqrt(v_inf**2 + 2 * mu_moon / r_lunar)
            dv_arr = (v_capture - v_circ_lunar) * 1000  # m/s

            dv1_grid[j, i] = dv_dep
            dv2_grid[j, i] = dv_arr
            dv_total_grid[j, i] = dv_dep + dv_arr
            tof_grid[j, i] = tof
            c3_grid[j, i] = (v1_mag - v_circ)**2 * 1731.46**2

    # Find optimal
    valid_mask = ~np.isnan(dv_total_grid)
    if np.any(valid_mask):
        min_idx = np.nanargmin(dv_total_grid)
        j_opt, i_opt = np.unravel_index(min_idx, dv_total_grid.shape)
        min_dv_total = dv_total_grid[j_opt, i_opt]
        optimal_departure = dep_dates[i_opt]
        optimal_tof = tof_values[j_opt]
        optimal_arrival = optimal_departure + optimal_tof
    else:
        min_dv_total = np.nan
        optimal_departure = optimal_tof = optimal_arrival = np.nan

    return PorkchopResult(
        departure_dates=dep_dates,
        arrival_dates=tof_values,  # Using TOF for lunar
        dv1_grid=dv1_grid,
        dv2_grid=dv2_grid,
        dv_total_grid=dv_total_grid,
        tof_grid=tof_grid,
        c3_grid=c3_grid,
        min_dv_total=min_dv_total,
        min_dv_departure=dv1_grid[j_opt, i_opt] if not np.isnan(min_dv_total) else np.nan,
        min_dv_arrival=dv2_grid[j_opt, i_opt] if not np.isnan(min_dv_total) else np.nan,
        optimal_departure=optimal_departure,
        optimal_arrival=optimal_arrival,
        optimal_tof=optimal_tof,
        origin_body='earth',
        target_body='moon',
        origin_orbit_alt_km=parking_alt_km,
        target_orbit_alt_km=lunar_alt_km,
    )
