# Space State API Reference

A comprehensive astrodynamics simulation framework for spacecraft trajectory design, mission planning, and analysis.

## Table of Contents

1. [Core Modules](#core-modules)
2. [Guidance & Control](#guidance--control)
3. [Mission Planning](#mission-planning)
4. [Trajectory Optimization](#trajectory-optimization)
5. [Monte Carlo Analysis](#monte-carlo-analysis)
6. [Visualization](#visualization)
7. [Quick Start Examples](#quick-start-examples)

---

## Core Modules

### `base.star_states` - Orbital States

```python
from base.star_states import get_circular_state, get_hohmann_state

# Get initial state for circular orbit
r0, v0, q0, omega0 = get_circular_state(BODIES['moon'], 'lunar_50km')

# Get Hohmann transfer state
state = get_hohmann_state(r_initial_km, r_final_km, mu)
```

### `base.star_propagator` - Trajectory Propagation

```python
from base.star_propagator import Propagator
from base.star_ship import starship_raptor_vac

ship = starship_raptor_vac()
guidance = BurnController(ship, body)
prop = Propagator(ship, guidance)

# Initialize and propagate
prop.result.initialize(y0, t_start)
prop.propagate(duration_days)

# Access results
positions = prop.result.y[:3, :]  # Position history
velocities = prop.result.y[3:6, :]  # Velocity history
```

### `base.star_soi` - Sphere of Influence

```python
from base.star_soi import (
    determine_soi, get_soi_info, compute_soi_radius,
    multi_body_atmosphere, compute_drag_acceleration
)

# Determine which body's SOI contains position
soi = determine_soi(position_au, time_days)

# Get atmospheric density
rho = multi_body_atmosphere(position_au, time_days)
```

### `base.star_comm` - Radio Communications

```python
from base.star_comm import (
    Antenna, Transmitter, Receiver, GroundStation,
    DSN_STATIONS, compute_link_budget, max_data_rate,
    find_contact_windows
)

# Create antenna and link budget
antenna = Antenna("HGA", diameter_m=2.0, frequency_hz=8.4e9)
tx = Transmitter(power_w=50, antenna=antenna)
rx = Receiver(antenna=DSN_STATIONS['goldstone'].antenna)

budget = compute_link_budget(tx, rx, distance_km=1e6)
rate = max_data_rate(tx, rx, distance_km=1e6)
```

---

## Guidance & Control

### `guidance.BurnController` - Burn Management

```python
from guidance import BurnController, Burn

controller = BurnController(ship, central_body)

# Schedule a burn
controller.schedule_burn(Burn(
    t_start=1.0,      # days
    duration=0.01,    # days
    delta_v=3.2,      # km/s
    direction='prograde'
))
```

### `guidance.lambert_solve` - Lambert Problem

```python
from guidance import lambert_solve

solution = lambert_solve(r1_au, r2_au, tof_days, mu_au)
# solution.v1 - departure velocity
# solution.v2 - arrival velocity
```

### `guidance.ProximityGuidance` - Rendezvous

```python
from guidance import (
    ProximityGuidance, ProximityGuidanceConfig,
    inertial_to_lvlh, compute_vbar_approach
)

# Configure proximity operations
config = ProximityGuidanceConfig(
    approach_mode=ApproachMode.VBAR,
    keep_out_zone_m=200.0
)
guidance = ProximityGuidance(config, target_state, mu)

# Get relative state in LVLH frame
rel_state = get_relative_state(chaser_state, target_state)
```

### `base.star_attitude` - Attitude Control

```python
from base.star_attitude import (
    quaternion_multiply, quaternion_error,
    attitude_pd_control, allocate_wheels
)

# Compute attitude error
q_err = quaternion_error(q_desired, q_current)

# PD control torque
tau = attitude_pd_control(q_err, omega, Kp=100.0, Kd=10000.0)

# Allocate to reaction wheels
wheel_torques = allocate_wheels(tau, wheel_momentum)
```

---

## Mission Planning

### `mission.MissionPlan` - Mission Structure

```python
from mission import (
    MissionPlan, MissionPhase, DeltaVBudget,
    PhaseType, create_earth_moon_mission
)

# Create custom mission
plan = MissionPlan(name="Lunar Transfer")

plan.add_phase(MissionPhase(
    name="TLI",
    phase_type=PhaseType.MANEUVER,
    delta_v_budget=DeltaVBudget("TLI", 3.2, margin_percent=5.0),
    duration_days=0.01
))

# Or use template
plan = create_earth_moon_mission()
print(plan.summary())
```

### `mission.quick_estimate` - Fast Delta-V

```python
from mission import quick_estimate

result = quick_estimate('earth', 'mars barycenter')
# result['dv_total_kmps'] - Total delta-V
# result['transfer_time_days'] - Hohmann transfer time
```

---

## Trajectory Optimization

### `optimization.optimize_hohmann` - Transfer Optimization

```python
from optimization import (
    optimize_hohmann, optimize_bielliptic, optimize_three_burn,
    optimize_phasing, optimize_plane_change
)

# Hohmann transfer
result = optimize_hohmann(r1_km=6771, r2_km=42164, mu=398600.4)
print(f"Delta-V: {result.total_delta_v:.3f} km/s")

# Three-burn (auto-selects Hohmann vs bi-elliptic)
result = optimize_three_burn(r1_km, r2_km, mu)

# Phasing maneuver
result = optimize_phasing(
    orbital_radius_km=6771,
    phase_angle_deg=30.0,
    mu=398600.4,
    n_orbits=2
)
```

### `optimization.GravityAssist` - Flyby Design

```python
from optimization import compute_flyby_geometry, optimize_powered_flyby

flyby = compute_flyby_geometry(
    body='earth',
    v_infinity_in=np.array([5.0, 0.0, 0.0]),
    periapsis_altitude_km=300
)
# flyby.turn_angle_deg - Turn angle
# flyby.delta_v_boost - Velocity boost
```

### `optimization.estimate_propellant` - Fuel Mass

```python
from optimization import estimate_propellant, compute_mass_ratio

fuel = estimate_propellant(
    delta_v_kmps=4.0,
    dry_mass_kg=5000,
    isp_s=450,
    margin_percent=10
)
print(f"Propellant: {fuel['propellant_mass_kg']:.0f} kg")
```

---

## Monte Carlo Analysis

### `analysis.run_monte_carlo` - Dispersion Analysis

```python
from analysis import (
    run_monte_carlo, DispersionConfig,
    create_position_dispersion, create_velocity_dispersion
)

# Configure dispersions
config = DispersionConfig(n_samples=1000, seed=42)
config.add_source(create_position_dispersion(sigma_km=10.0))
config.add_source(create_velocity_dispersion(sigma_mps=1.0))

# Run Monte Carlo
result = run_monte_carlo(propagate_func, y0, config)

# Analyze results
print(f"Success rate: {result.success_rate:.1f}%")
print(f"Final position 3σ: {result.statistics.std[:3, -1]}")
```

### `analysis.sensitivity_analysis` - Sensitivity

```python
from analysis import sensitivity_analysis

result = sensitivity_analysis(
    propagate_func,
    y0,
    perturbation_scales=np.array([1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1.0])
)
# result['normalized_sensitivities'] - Sensitivity coefficients
```

---

## Visualization

### `visual.OrbitPlotter3D` - 3D Orbit Plots

```python
from visual import OrbitPlotter3D, plot_orbit_3d

# Quick plot
plot_orbit_3d(positions_km, body='earth', title='LEO Orbit')

# Detailed plot
plotter = OrbitPlotter3D(title="Earth-Moon Transfer", dark_mode=True)
plotter.add_body('earth')
plotter.add_body('moon', position=np.array([384400, 0, 0]))
plotter.add_trajectory(positions_km, color='cyan', label='Transfer')
plotter.add_marker(injection_point, color='lime', label='TLI')
plotter.show(elev=30, azim=45)
```

### `visual.create_orbit_animation` - Animation

```python
from visual import create_orbit_animation

anim = create_orbit_animation(
    positions_km,
    body='earth',
    interval=50,
    trail_length=100,
    save_path='orbit.gif'
)
```

### `visual.create_interactive_orbit` - Plotly

```python
from visual import create_interactive_orbit

fig = create_interactive_orbit(positions_km, body='earth')
fig.show()  # Opens in browser
```

---

## Quick Start Examples

### Earth to Moon Transfer

```python
from base.star_ship import starship_raptor_vac
from base.star_states import get_circular_state
from base.star_propagator import Propagator
from guidance import BurnController, Burn
from visual import OrbitPlotter3D

# Setup
ship = starship_raptor_vac()
body = BODIES['earth']

# Initial LEO state
r0, v0, q0, omega0 = get_circular_state(body, 'earth_leo')
y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

# Guidance with TLI burn
guidance = BurnController(ship, body)
guidance.schedule_burn(Burn(t_start=0.0, duration=0.01, delta_v=3.2, direction='prograde'))

# Propagate
prop = Propagator(ship, guidance)
prop.result.initialize(y0, 0.0)
prop.propagate(5.0)  # 5 days

# Visualize
positions_km = prop.result.y[:3, :] * AU_TO_KM
plotter = OrbitPlotter3D(title="TLI")
plotter.add_body('earth')
plotter.add_trajectory(positions_km)
plotter.show()
```

### LEO Rendezvous

```python
from guidance import ProximityGuidance, compute_vbar_approach

# Compute approach maneuvers
maneuvers = compute_vbar_approach(
    rel_pos_lvlh=np.array([100, 0, 0]),  # 100m ahead
    approach_velocity=0.1,  # m/s
    n=0.0011,  # mean motion (rad/s)
    final_distance=10.0  # m
)
```

### Monte Carlo with Success Criteria

```python
from analysis import run_monte_carlo, DispersionConfig, SuccessCriterion

def check_orbit_achieved(t, y):
    final_r = np.linalg.norm(y[:3, -1]) * AU_TO_KM
    return 35000 < final_r < 45000  # GEO altitude

criterion = SuccessCriterion(name="GEO", check_func=check_orbit_achieved)

config = DispersionConfig(n_samples=500)
config.add_source(create_velocity_dispersion(sigma_mps=0.5))

result = run_monte_carlo(propagate, y0, config, criterion)
print(f"GEO insertion success: {result.success_probability*100:.1f}%")
```

---

## Test Suite

Run all tests:
```bash
python -m pytest space_state/tests/ -v
```

Current test count: **333 tests**

| Module | Tests |
|--------|-------|
| Attitude Control | 45 |
| Communications | 29 |
| Environment | 22 |
| Lambert Solver | 16 |
| Mission Planning | 18 |
| Monte Carlo | 25 |
| Optimization | 25 |
| Power Systems | 26 |
| Proximity Ops | 21 |
| SOI/Atmosphere | 26 |
| States | 18 |
| I/O | 14 |
| Validation | 27 |
| Visualization | 21 |
