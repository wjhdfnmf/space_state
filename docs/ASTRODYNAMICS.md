# Astrodynamics Module

Advanced orbital mechanics capabilities for interplanetary mission design.

## Table of Contents

1. [Orbital Elements](#orbital-elements)
2. [Gravity Assist / Flyby](#gravity-assist--flyby)
3. [Patched Conics](#patched-conics)
4. [Station Keeping](#station-keeping)
5. [Eclipse Modeling](#eclipse-modeling)
6. [Thermal Analysis](#thermal-analysis)
7. [Entry, Descent, Landing](#entry-descent-and-landing)
8. [Ground Track and Coverage](#ground-track-and-coverage)
9. [Launch Windows](#launch-windows)
10. [Example: Jupiter Gravity Assist Mission](#example-jupiter-gravity-assist-mission)

---

## Orbital Elements

Convert between Cartesian state vectors and classical Keplerian elements.

### Basic Usage

```python
from base import cartesian_to_keplerian, keplerian_to_cartesian, OrbitalElements

# Earth parameters
MU_EARTH = 398600.4418  # km³/s²

# Define state vector (position and velocity in km, km/s)
r = np.array([7000.0, 0.0, 0.0])      # Position
v = np.array([0.0, 7.546, 0.0])       # Velocity (circular orbit)

# Convert to Keplerian elements
elements = cartesian_to_keplerian(r, v, MU_EARTH)

print(f"Semi-major axis: {elements.a:.1f} km")
print(f"Eccentricity: {elements.e:.6f}")
print(f"Inclination: {np.degrees(elements.i):.2f}°")
print(f"RAAN: {np.degrees(elements.raan):.2f}°")
print(f"Arg of Periapsis: {np.degrees(elements.omega):.2f}°")
print(f"True Anomaly: {np.degrees(elements.nu):.2f}°")

# Convert back to Cartesian
r_back, v_back = keplerian_to_cartesian(elements, MU_EARTH)
```

### Orbital Elements Class

The `OrbitalElements` dataclass provides useful properties:

```python
elements = OrbitalElements(
    a=10000.0,      # Semi-major axis (km)
    e=0.2,          # Eccentricity
    i=np.radians(28.5),   # Inclination (rad)
    raan=np.radians(45),  # RAAN (rad)
    omega=np.radians(30), # Argument of periapsis (rad)
    nu=np.radians(60),    # True anomaly (rad)
    mu=MU_EARTH           # Gravitational parameter
)

# Derived properties
print(f"Orbital period: {elements.period:.1f} seconds")
print(f"Periapsis: {elements.periapsis:.1f} km")
print(f"Apoapsis: {elements.apoapsis:.1f} km")
print(f"Specific energy: {elements.energy:.3f} km²/s²")
```

### Anomaly Conversions

Convert between different anomaly representations:

```python
from base import (
    true_to_eccentric_anomaly, eccentric_to_true_anomaly,
    mean_to_eccentric_anomaly, eccentric_to_mean_anomaly,
    true_to_mean_anomaly, mean_to_true_anomaly
)

e = 0.3  # Eccentricity
nu = np.radians(45)  # True anomaly

# True → Eccentric → Mean
E = true_to_eccentric_anomaly(nu, e)
M = eccentric_to_mean_anomaly(E, e)

# Or directly
M = true_to_mean_anomaly(nu, e)

# Reverse: Mean → Eccentric → True
E = mean_to_eccentric_anomaly(M, e)  # Solves Kepler's equation
nu = eccentric_to_true_anomaly(E, e)
```

### Kepler Propagation

Propagate orbital elements forward in time (analytical 2-body):

```python
from base import propagate_kepler, time_to_true_anomaly

# Propagate by half an orbital period
T = elements.period
new_elements = propagate_kepler(elements, T/2, MU_EARTH)

# Find time to reach a specific true anomaly
t = time_to_true_anomaly(elements, np.radians(180), MU_EARTH)
```

### Orbital Velocities

```python
from base import vis_viva, circular_velocity, escape_velocity, orbital_period

r = 7000.0  # km
a = 10000.0  # km

# Circular orbit velocity
v_circ = circular_velocity(r, MU_EARTH)  # ~7.55 km/s at LEO

# Escape velocity
v_esc = escape_velocity(r, MU_EARTH)  # ~10.7 km/s at LEO

# Velocity on ellipse at radius r
v = vis_viva(r, a, MU_EARTH)

# Orbital period
T = orbital_period(a, MU_EARTH)  # seconds
```

---

## Gravity Assist / Flyby

Design gravity assist maneuvers for interplanetary missions.

### Basic Flyby Calculation

```python
from guidance import compute_flyby, FlybyGeometry

# Jupiter parameters
MU_JUPITER = 126686534.0  # km³/s²
R_JUPITER = 71492.0       # km

# Incoming V-infinity vector (km/s)
v_inf_in = np.array([6.0, 2.0, 0.0])

# Periapsis distance (km above Jupiter's cloud tops)
rp = R_JUPITER + 200000  # 200,000 km altitude

# Compute flyby
flyby = compute_flyby(v_inf_in, rp, MU_JUPITER)

print(f"V-infinity magnitude: {flyby.v_inf_mag:.2f} km/s")
print(f"Turn angle: {np.degrees(flyby.turn_angle):.1f}°")
print(f"Hyperbolic eccentricity: {flyby.e_hyperbolic:.3f}")
print(f"Delta-V gain: {flyby.delta_v_gain:.2f} km/s")
print(f"Outgoing V-infinity: {flyby.v_inf_out}")
```

### Flyby Turn Angle

Calculate turn angle for a given periapsis, or required periapsis for desired turn:

```python
from guidance import flyby_turn_angle, flyby_periapsis_from_turn_angle, maximum_turn_angle

# Turn angle from periapsis
v_inf = 5.0  # km/s
rp = 10000.0  # km
delta = flyby_turn_angle(v_inf, rp, MU_EARTH)

# Periapsis from desired turn angle
desired_turn = np.radians(60)  # 60 degrees
rp_needed = flyby_periapsis_from_turn_angle(v_inf, desired_turn, MU_EARTH)

# Maximum possible turn (limited by body surface)
rp_min = R_EARTH + 200  # 200 km minimum altitude
max_turn = maximum_turn_angle(v_inf, rp_min, MU_EARTH)
```

### B-Plane Targeting

The B-plane is perpendicular to the incoming V-infinity vector:

```python
from guidance import BPlaneTarget, b_plane_from_periapsis, periapsis_from_b_plane

# B-plane impact parameter from periapsis
v_inf = 5.0  # km/s
rp = 8000.0  # km
b = b_plane_from_periapsis(rp, v_inf, MU_EARTH)

# Periapsis from B-plane parameter
rp_back = periapsis_from_b_plane(b, v_inf, MU_EARTH)

# Create B-plane target for mission design
target = BPlaneTarget.from_periapsis(
    rp=8000.0,
    v_inf=5.0,
    mu=MU_EARTH,
    theta=np.radians(30)  # B-plane angle (controls inclination change)
)
print(f"B·T = {target.b_dot_t:.1f} km")
print(f"B·R = {target.b_dot_r:.1f} km")
```

### Powered Gravity Assist

Calculate delta-V for a powered flyby (thrust at periapsis):

```python
from guidance import powered_flyby_delta_v

# Want to increase V-infinity from 5 to 7 km/s
v_inf_in = 5.0
v_inf_out = 7.0
rp = 6700.0  # km (300 km altitude at Earth)

dv = powered_flyby_delta_v(v_inf_in, v_inf_out, rp, MU_EARTH)
print(f"Required delta-V at periapsis: {dv:.2f} km/s")

# Compare to direct delta-V (without Oberth effect)
dv_direct = v_inf_out - v_inf_in
print(f"Savings from Oberth effect: {dv_direct - dv:.2f} km/s")
```

### Design Flyby for Target Direction

```python
from guidance import design_flyby_for_target

v_inf_in = np.array([5.0, 0.0, 0.0])
target_direction = np.array([0.8, 0.6, 0.0])  # Desired outgoing direction
target_direction = target_direction / np.linalg.norm(target_direction)

flyby = design_flyby_for_target(
    v_inf_in=v_inf_in,
    target_direction=target_direction,
    mu=MU_EARTH,
    body_radius=R_EARTH,
    min_altitude=200.0  # km
)

if flyby is not None:
    print(f"Flyby possible with rp = {flyby.periapsis:.1f} km")
else:
    print("Turn angle too large - flyby not possible")
```

---

## Patched Conics

Model interplanetary trajectories as connected two-body problems.

### Reference Frame Transformations

```python
from guidance import heliocentric_to_planetocentric, planetocentric_to_heliocentric

# Spacecraft heliocentric state
r_helio = np.array([1.01, 0.0, 0.0])  # AU
v_helio = np.array([0.0, 30.5, 0.0])  # km/s

# Earth heliocentric state
r_earth = np.array([1.0, 0.0, 0.0])
v_earth = np.array([0.0, 29.78, 0.0])

# Convert to Earth-centered
r_ec, v_ec = heliocentric_to_planetocentric(r_helio, v_helio, r_earth, v_earth)

# Convert back
r_helio_back, v_helio_back = planetocentric_to_heliocentric(r_ec, v_ec, r_earth, v_earth)
```

### Sphere of Influence

```python
from guidance import SOI_RADII_AU, check_soi_entry, sphere_of_influence_radius

# Pre-computed SOI radii (AU)
print(f"Earth SOI: {SOI_RADII_AU['earth'] * 149597870.7:.0f} km")
print(f"Jupiter SOI: {SOI_RADII_AU['jupiter'] * 149597870.7:.0f} km")

# Check if inside a body's SOI
r_sc = np.array([1.005, 0.0, 0.0])  # Spacecraft position (AU)
r_earth = np.array([1.0, 0.0, 0.0])  # Earth position (AU)

inside = check_soi_entry(r_sc, r_earth, 'earth')
print(f"Inside Earth SOI: {inside}")

# Calculate SOI radius from first principles
r_soi = sphere_of_influence_radius(
    a_planet=1.0,      # AU (Earth's orbit)
    m_planet=1.0,      # Relative mass
    m_sun=332946.0     # Sun/Earth mass ratio
)
```

### Characteristic Energy (C3)

```python
from guidance import compute_c3, v_infinity_from_c3, departure_v_infinity

# C3 from V-infinity
v_inf = 3.0  # km/s
c3 = compute_c3(v_inf)  # 9.0 km²/s²

# V-infinity from C3
v_inf = v_infinity_from_c3(c3)

# Full departure analysis
v_sc_helio = np.array([32.0, 1.0, 0.0])  # km/s
v_earth = np.array([29.78, 0.0, 0.0])    # km/s

v_inf_vec, v_inf_mag, c3 = departure_v_infinity(v_sc_helio, v_earth)
print(f"Departure V∞: {v_inf_mag:.2f} km/s")
print(f"C3: {c3:.1f} km²/s²")
```

### Transfer Delta-V Estimates

```python
from guidance import estimate_transfer_delta_v, interplanetary_transfer_type

# Earth to Mars Hohmann-like transfer
result = estimate_transfer_delta_v('earth', 'mars')

print(f"Departure V∞: {result['v_inf_departure_km_s']:.2f} km/s")
print(f"Arrival V∞: {result['v_inf_arrival_km_s']:.2f} km/s")
print(f"Departure C3: {result['c3_departure']:.1f} km²/s²")
print(f"Transfer time: {result['transfer_time_days']:.0f} days")

# Classify transfer energy
transfer_type = interplanetary_transfer_type(
    result['c3_departure'],
    result['c3_arrival']
)
print(f"Transfer type: {transfer_type}")
```

---

## Station Keeping

Maintain spacecraft position in specific orbits.

### GEO Station Keeping

```python
from guidance import (
    estimate_geo_station_keeping_budget,
    geo_north_south_delta_v, geo_east_west_delta_v,
    geo_inclination_drift
)

# Estimate annual delta-V budget for GEO satellite
budget = estimate_geo_station_keeping_budget(
    inclination_deadband=0.05,  # degrees
    longitude_deadband=0.1,     # degrees
    area_to_mass=0.02           # m²/kg
)

print(f"North-South (inclination): {budget.north_south:.1f} m/s/year")
print(f"East-West (longitude): {budget.east_west:.1f} m/s/year")
print(f"Eccentricity control: {budget.eccentricity:.1f} m/s/year")
print(f"Total annual budget: {budget.total:.1f} m/s/year")
print(f"Maneuvers per year: {budget.maneuvers_per_year}")

# Calculate inclination drift due to lunar/solar perturbations
initial_inc = 0.0  # degrees
final_inc = geo_inclination_drift(initial_inc, years=5.0)
print(f"Inclination after 5 years: {final_inc:.2f}°")
```

### Lagrange Points

```python
from guidance import (
    sun_earth_lagrange_points, earth_moon_lagrange_points,
    lagrange_station_keeping_budget
)

# Sun-Earth Lagrange points
se_points = sun_earth_lagrange_points()
print(f"Sun-Earth L1 distance: {se_points['L1'].distance_km:.0f} km")
print(f"Sun-Earth L2 distance: {se_points['L2'].distance_km:.0f} km")
print(f"L4/L5 stable: {se_points['L4'].stable}")

# Earth-Moon Lagrange points
em_points = earth_moon_lagrange_points()

# Station keeping budgets
l2_budget = lagrange_station_keeping_budget('L2', 'sun-earth')
print(f"Sun-Earth L2 station keeping: {l2_budget:.1f} m/s/year")

gateway_budget = lagrange_station_keeping_budget('L2', 'earth-moon')
print(f"Earth-Moon L2 (Gateway): {gateway_budget:.1f} m/s/year")
```

### Low-Thrust Propulsion

```python
from guidance import low_thrust_delta_v, estimate_low_thrust_budget

# Calculate delta-V from continuous thrust
thrust = 0.1      # N (100 mN electric thruster)
mass = 1000       # kg
burn_time = 3600  # 1 hour

dv = low_thrust_delta_v(thrust, mass, burn_time)
print(f"Delta-V from 1-hour burn: {dv:.3f} m/s")

# Annual propulsion budget for electric propulsion
budget = estimate_low_thrust_budget(
    thrust_n=0.1,
    isp_s=3000,           # High-Isp ion thruster
    spacecraft_mass=1000,
    delta_v_requirement=50  # m/s/year
)
print(f"Propellant per year: {budget.propellant_per_year:.2f} kg")
```

---

## Eclipse Modeling

Detect and predict eclipses for power and thermal analysis.

### Eclipse Detection

```python
from base import (
    EclipseType, cylindrical_shadow_check, conical_shadow_check,
    compute_eclipse_state
)
import numpy as np

# Simple cylindrical shadow check
r_sc = np.array([-50000.0, 0.0, 0.0])      # Spacecraft behind Earth
r_body = np.array([0.0, 0.0, 0.0])          # Earth at origin
r_sun = np.array([1.496e8, 0.0, 0.0])       # Sun position (km)

eclipse = cylindrical_shadow_check(r_sc, r_body, r_sun, body_radius=6378)
print(f"Eclipse type: {eclipse.name}")

# More accurate conical shadow model
eclipse_type, illumination = conical_shadow_check(
    r_sc, r_body, r_sun,
    body_radius=6378,    # Earth radius (km)
    sun_radius=695700    # Sun radius (km)
)
print(f"Eclipse: {eclipse_type.name}, Illumination: {illumination:.0%}")
```

### Shadow Cone Geometry

```python
from base import shadow_cone_geometry

# Earth's shadow geometry
R_SUN = 695700      # km
R_EARTH = 6378.137  # km
D_SUN_EARTH = 1.496e8  # km

umbra_length, umbra_angle, penumbra_angle = shadow_cone_geometry(
    R_SUN, R_EARTH, D_SUN_EARTH
)

print(f"Umbra extends: {umbra_length:.0f} km")
print(f"Umbra half-angle: {np.degrees(umbra_angle):.3f}°")
print(f"Penumbra half-angle: {np.degrees(penumbra_angle):.3f}°")
```

### Eclipse Duration Estimates

```python
from base import estimate_eclipse_duration_leo, estimate_eclipses_per_orbit_geo

# LEO eclipse duration
altitude = 400  # km (ISS)
inclination = 51.6  # degrees
beta_angle = 0  # Sun angle relative to orbit plane

duration = estimate_eclipse_duration_leo(altitude, inclination, beta_angle)
print(f"LEO eclipse duration: {duration:.1f} minutes")

# Dawn-dusk orbit (minimal eclipses)
duration_dusk = estimate_eclipse_duration_leo(800, 98, 75)
print(f"Sun-sync dawn/dusk eclipse: {duration_dusk:.1f} minutes")

# GEO eclipse statistics
geo_stats = estimate_eclipses_per_orbit_geo()
print(f"GEO eclipse seasons per year: {geo_stats['eclipse_seasons_per_year']}")
print(f"Max eclipse duration: {geo_stats['max_eclipse_duration_minutes']} minutes")
print(f"Total eclipse hours per year: {geo_stats['total_eclipse_hours_per_year']}")
```

### Eclipse Prediction

```python
from base import predict_eclipses, EclipsePrediction
import numpy as np

# Predict eclipses along a trajectory
# (trajectory_positions shape: (3, N), times in days since epoch)
positions = np.zeros((3, 1000))
times = np.linspace(0, 1, 1000)  # 1 day trajectory

predictions = predict_eclipses(positions, times, check_bodies=['earth', 'moon'])

for pred in predictions:
    print(f"Eclipse: {pred.eclipse_type.name}")
    print(f"  Body: {pred.eclipsing_body}")
    print(f"  Duration: {pred.duration:.0f} seconds")
    print(f"  Max shadow depth: {pred.max_shadow_depth:.0f} km")
```

---

## Thermal Analysis

Spacecraft thermal environment calculations.

### Solar Flux

```python
from base import solar_flux_at_distance, SOLAR_CONSTANT_1AU

# Solar flux at various distances
flux_earth = solar_flux_at_distance(1.0)    # 1 AU
flux_mars = solar_flux_at_distance(1.524)   # Mars
flux_jupiter = solar_flux_at_distance(5.2)  # Jupiter

print(f"Solar flux at Earth: {flux_earth:.0f} W/m²")
print(f"Solar flux at Mars: {flux_mars:.0f} W/m²")
print(f"Solar flux at Jupiter: {flux_jupiter:.0f} W/m²")
```

### Thermal Environment

```python
from base import compute_thermal_environment, ThermalEnvironment

# LEO thermal environment
env = compute_thermal_environment(
    distance_from_sun_au=1.0,
    altitude_km=400,
    planet='earth',
    absorptivity=0.3,
    emissivity=0.9
)

print(f"Direct solar: {env.solar_flux:.0f} W/m²")
print(f"Earth albedo: {env.albedo_flux:.0f} W/m²")
print(f"Earth IR: {env.planetary_ir:.0f} W/m²")
print(f"Equilibrium temperature: {env.equilibrium_temp:.0f} K ({env.equilibrium_temp - 273:.0f}°C)")

# Deep space (no planet)
env_deep = compute_thermal_environment(
    distance_from_sun_au=5.2,  # Jupiter distance
    altitude_km=None,
    planet=None
)
print(f"Jupiter orbit equilibrium: {env_deep.equilibrium_temp:.0f} K")
```

### Hot/Cold Case Analysis

```python
from base import hot_cold_case_temperatures

# Worst-case thermal analysis for LEO
T_hot, T_cold = hot_cold_case_temperatures(
    distance_from_sun_au=1.0,
    altitude_km=400,
    planet='earth',
    alpha_hot=0.9, epsilon_hot=0.3,   # Hot case: high absorb, low emit
    alpha_cold=0.1, epsilon_cold=0.9  # Cold case: low absorb, high emit
)

print(f"Hot case: {T_hot:.0f} K ({T_hot - 273:.0f}°C)")
print(f"Cold case: {T_cold:.0f} K ({T_cold - 273:.0f}°C)")
```

### Eclipse Thermal Transients

```python
from base import (
    SpacecraftThermal, estimate_heater_power,
    eclipse_cooldown_time, eclipse_warmup_time,
    compute_thermal_environment
)

# Define spacecraft thermal properties
spacecraft = SpacecraftThermal(
    solar_absorptivity=0.3,
    ir_emissivity=0.9,
    area_sun_facing=5.0,      # m²
    area_radiator=10.0,       # m²
    internal_power=500.0,     # W (internal dissipation)
    mass=1000.0,              # kg
    specific_heat=900.0       # J/(kg·K)
)

env = compute_thermal_environment(1.0, 400, 'earth')

# Eclipse cooldown estimate
t_cool = eclipse_cooldown_time(spacecraft, 300.0, 280.0, env)
print(f"Time to cool 20K: {t_cool/60:.0f} minutes")

# Heater power to maintain minimum temperature during eclipse
min_temp = 273  # 0°C
heater_w = estimate_heater_power(spacecraft, min_temp, env)
print(f"Heater power required: {heater_w:.0f} W")
```

### Power System Sizing

```python
from base import battery_sizing_for_eclipse, solar_panel_sizing

# Battery sizing for eclipse survival
spacecraft = SpacecraftThermal(
    solar_absorptivity=0.3, ir_emissivity=0.9,
    area_sun_facing=5.0, area_radiator=10.0,
    internal_power=500.0, mass=1000.0, specific_heat=900.0
)

battery = battery_sizing_for_eclipse(
    spacecraft,
    eclipse_duration_min=35.0,    # LEO eclipse
    heater_power=50.0,            # W
    other_loads_w=200.0,          # W
    depth_of_discharge=0.30
)

print(f"Eclipse duration: {battery['eclipse_duration_min']:.0f} min")
print(f"Total power needed: {battery['total_power_w']:.0f} W")
print(f"Energy required: {battery['energy_required_wh']:.0f} Wh")
print(f"Battery capacity needed: {battery['battery_capacity_wh']:.0f} Wh")

# Solar panel sizing
panels = solar_panel_sizing(
    power_required_w=1000.0,
    distance_au=1.0,
    panel_efficiency=0.30,
    degradation_factor=0.90
)

print(f"Solar flux: {panels['solar_flux_w_m2']:.0f} W/m²")
print(f"Panel area required: {panels['panel_area_m2']:.2f} m²")

# Mars mission panels
panels_mars = solar_panel_sizing(1000.0, 1.524)
print(f"Mars panel area: {panels_mars['panel_area_m2']:.2f} m²")
```

---

## Entry, Descent, and Landing

The Entry, Descent, and Landing (EDL) module provides comprehensive tools for analyzing atmospheric entry trajectories, thermal protection requirements, and EDL event sequencing.

### Atmospheric Models

The module includes exponential atmosphere models for multiple bodies:

```python
from guidance import (
    ATMOSPHERES, atmospheric_density,
    EntryConditions, simulate_entry
)

# Available atmospheres: Earth, Mars, Venus, Titan
earth = ATMOSPHERES['earth']
print(f"Earth surface density: {earth.surface_density:.3f} kg/m³")
print(f"Scale height: {earth.scale_height:.1f} km")
print(f"Entry altitude: {earth.entry_altitude:.0f} km")

mars = ATMOSPHERES['mars']
print(f"Mars surface density: {mars.surface_density:.4f} kg/m³")  # Very thin

# Density at altitude
rho_50km = atmospheric_density(50.0, earth)  # At 50 km
print(f"Density at 50 km: {rho_50km:.2e} kg/m³")
```

### Entry Corridor Analysis

```python
from guidance import (
    entry_corridor_bounds, peak_deceleration_estimate,
    earth_reentry_corridor, ballistic_coefficient
)

# Calculate entry corridor for LEO reentry
steep_limit, shallow_limit = entry_corridor_bounds(
    velocity_km_s=7.8,
    body='earth',
    max_decel_g=10.0
)
print(f"Entry corridor: {steep_limit:.1f}° to {shallow_limit:.1f}°")

# Vehicle type-specific corridors
capsule = earth_reentry_corridor(11.0, 'capsule')  # Lunar return
print(f"Capsule L/D: {capsule['lift_to_drag']:.2f}")
print(f"Corridor width: {capsule['corridor_width_deg']:.1f}°")

shuttle = earth_reentry_corridor(7.8, 'shuttle')
print(f"Shuttle L/D: {shuttle['lift_to_drag']:.2f}")

# Ballistic coefficient
beta = ballistic_coefficient(mass=1000, drag_coef=1.5, area=10)
print(f"Ballistic coefficient: {beta:.1f} kg/m²")

# Peak deceleration estimate
g_peak = peak_deceleration_estimate(
    velocity_km_s=7.8,
    flight_path_angle_deg=-5.0,
    ballistic_coef=100,
    body='earth'
)
print(f"Peak deceleration: {g_peak:.1f} g")
```

### Thermal Protection Analysis

```python
from guidance import peak_heating_rate, total_heat_load

# Peak heating rate (Sutton-Graves correlation)
q_dot = peak_heating_rate(
    velocity_km_s=7.8,
    nose_radius_m=1.0,
    altitude_km=60.0,
    body='earth'
)
print(f"Peak heat rate: {q_dot:.0f} W/cm²")

# Total heat load
Q_total = total_heat_load(
    entry_velocity_km_s=7.8,
    ballistic_coef=100,
    nose_radius_m=1.0,
    body='earth'
)
print(f"Total heat load: {Q_total:.0f} J/cm²")
```

### Entry Simulation

```python
from guidance import simulate_entry, EntryConditions

# Define entry conditions
entry = EntryConditions(
    velocity=7.8,           # km/s
    flight_path_angle=-5.0, # degrees
    altitude=120.0,         # km
    latitude=0.0,           # degrees
    longitude=0.0,          # degrees
    azimuth=90.0            # degrees (eastward)
)

# Simulate entry trajectory
result = simulate_entry(
    entry=entry,
    ballistic_coef=100,     # kg/m²
    nose_radius_m=1.0,
    body='earth',
    dt=1.0                  # 1 second timestep
)

print(f"Peak deceleration: {result.peak_deceleration_g:.1f} g")
print(f"Peak heat rate: {result.peak_heat_rate_w_cm2:.0f} W/cm²")
print(f"Total heat load: {result.total_heat_load_j_cm2:.0f} J/cm²")
print(f"Trajectory points: {len(result.trajectory)}")
```

### Mars EDL Profile

```python
from guidance import mars_edl_profile, parachute_deployment_conditions

# Generate complete Mars EDL profile
profile = mars_edl_profile(
    entry_velocity_km_s=5.5,
    entry_fpa_deg=-12.0,
    vehicle_mass_kg=900.0,
    heatshield_diameter_m=4.5
)

print("Mars EDL Timeline:")
for event, time in profile['timeline'].items():
    print(f"  {event}: T+{time:.0f}s")

print(f"\nPeak deceleration: {profile['entry_result'].peak_deceleration_g:.1f} g")

# Parachute deployment conditions
chute = parachute_deployment_conditions(
    mach_target=2.0,
    altitude_km=10.0,
    body='mars'
)
print(f"Chute deployment velocity: {chute['velocity_m_s']:.0f} m/s")
print(f"Dynamic pressure: {chute['dynamic_pressure_pa']:.0f} Pa")
```

---

## Ground Track and Coverage

The Ground Track module provides tools for satellite ground track computation, ground station visibility analysis, and constellation coverage assessment.

### Ground Stations

```python
from guidance import GROUND_STATIONS, GroundStation

# DSN stations are predefined
goldstone = GROUND_STATIONS['goldstone']
print(f"Goldstone: {goldstone.latitude:.2f}°N, {goldstone.longitude:.2f}°E")
print(f"Minimum elevation: {goldstone.min_elevation}°")

# Get ECEF position
pos_ecef = goldstone.ecef_position()
print(f"ECEF position: {pos_ecef} km")

# Create custom ground station
my_station = GroundStation(
    name="MyStation",
    latitude=40.0,
    longitude=-75.0,
    altitude_m=100.0,
    min_elevation=10.0
)
```

### Ground Track Calculation

```python
import numpy as np
from guidance import (
    ground_track, eci_to_ecef, ecef_to_geodetic,
    R_EARTH, OMEGA_EARTH
)

# Simulate a simple circular orbit
N = 100
T = 90 * 60  # 90-minute orbit period
times = np.linspace(0, T, N)
r = R_EARTH + 400  # 400 km altitude

positions = np.zeros((3, N))
for i, t in enumerate(times):
    omega = 2 * np.pi / T
    positions[0, i] = r * np.cos(omega * t)
    positions[1, i] = r * np.sin(omega * t)

# Compute ground track
lats, lons, alts = ground_track(positions, times)

print(f"Latitude range: {min(lats):.1f}° to {max(lats):.1f}°")
print(f"Altitude range: {min(alts):.1f} to {max(alts):.1f} km")
```

### Visibility Windows

```python
from guidance import (
    find_visibility_windows, compute_access_statistics,
    elevation_angle, azimuth_angle
)

# Find visibility windows for a pass
windows = find_visibility_windows(positions.T, times, goldstone)

for w in windows:
    print(f"Station: {w.station_name}")
    print(f"  Start: {w.start_time:.0f}s, End: {w.end_time:.0f}s")
    print(f"  Duration: {w.duration:.0f}s")
    print(f"  Max elevation: {w.max_elevation:.1f}°")
    print(f"  AOS azimuth: {w.aos_azimuth:.1f}°, LOS azimuth: {w.los_azimuth:.1f}°")

# Compute access statistics
stats = compute_access_statistics(windows, total_time=86400)
print(f"Coverage: {stats['coverage_percent']:.2f}%")
print(f"Number of passes: {stats['num_passes']}")
print(f"Total access time: {stats['total_access_time_s']:.0f}s")
```

### Coverage Analysis

```python
from guidance import (
    swath_width, revisit_time_estimate, horizon_distance,
    line_of_sight_range, communication_window_geometry
)

# Swath width for imaging satellite
sw = swath_width(altitude_km=500, half_cone_angle_deg=30)
print(f"Swath width: {sw:.0f} km")

# Revisit time estimate
rt = revisit_time_estimate(
    altitude_km=500,
    inclination_deg=98,
    swath_width_km=sw
)
print(f"Estimated revisit time: {rt:.1f} hours")

# Horizon distance from orbit
d_horizon = horizon_distance(400)  # 400 km altitude
print(f"Horizon distance: {d_horizon:.0f} km")

# Slant range to ground
rng = line_of_sight_range(400, elevation_deg=10)
print(f"Range at 10° elevation: {rng:.0f} km")

# Communication geometry
geom = communication_window_geometry(400, goldstone)
print(f"Min range: {geom['min_range_km']:.0f} km")
print(f"Max range: {geom['max_range_km']:.0f} km")
```

### Sun-Synchronous Orbits

```python
from guidance import sun_synchronous_inclination

# Calculate sun-sync inclination
inc_600 = sun_synchronous_inclination(600)  # 600 km
inc_800 = sun_synchronous_inclination(800)  # 800 km

print(f"Sun-sync at 600 km: {inc_600:.2f}°")
print(f"Sun-sync at 800 km: {inc_800:.2f}°")
```

### Walker Constellation Coverage

```python
from guidance import walker_constellation_coverage

# Starlink-like constellation
result = walker_constellation_coverage(
    altitude_km=550,
    inclination_deg=53,
    num_planes=6,
    sats_per_plane=22,
    swath_width_km=1000
)

print(f"Total coverage: {result.total_coverage_percent:.1f}%")
print(f"Equatorial coverage: {result.equatorial_coverage_percent:.1f}%")
print(f"Polar coverage: {result.polar_coverage_percent:.1f}%")
print(f"Contacts per day: {result.contacts_per_day:.0f}")
print(f"Mean revisit time: {result.mean_revisit_time_hours:.1f} hours")
```

---

## Launch Windows

The Launch Windows module provides tools for calculating launch opportunities, transfer geometry, and interplanetary mission planning.

### Launch Sites

```python
from guidance import LAUNCH_SITES

# Available launch sites
for name, site in LAUNCH_SITES.items():
    print(f"{name}: {site.latitude:.2f}°N, {site.longitude:.2f}°E")
    print(f"  Min/Max inclination: {site.min_inclination:.1f}° - {site.max_inclination:.1f}°")

# Kennedy Space Center
ksc = LAUNCH_SITES['kennedy']
print(f"\nKSC can launch to inclinations {ksc.min_inclination}° to {ksc.max_inclination}°")
```

### Launch Azimuth

```python
from guidance import launch_azimuth, LAUNCH_SITES

# Calculate launch azimuth for ISS orbit (51.6°)
ksc = LAUNCH_SITES['kennedy']
az_north, az_south = launch_azimuth(ksc.latitude, 51.6)
print(f"Launch azimuth (northerly): {az_north:.1f}°")
print(f"Launch azimuth (southerly): {az_south:.1f}°")

# Launch to equatorial orbit
az_eq_n, az_eq_s = launch_azimuth(ksc.latitude, 28.5)  # Site latitude = min inclination
print(f"Due east launch: {az_eq_n:.1f}°")
```

### RAAN and Launch Time

```python
from guidance import launch_window_raan, raan_to_launch_time
from datetime import datetime

# Calculate required RAAN for target
target_raan = 45.0  # degrees
lst_required = launch_window_raan(target_raan, ksc.longitude)
print(f"Required LST for RAAN {target_raan}°: {lst_required:.2f} hours")

# Convert to UTC launch time
date = datetime(2025, 6, 15)
launch_utc = raan_to_launch_time(target_raan, ksc.longitude, date)
print(f"Launch time (UTC): {launch_utc}")
```

### Synodic Periods and Phase Angles

```python
from guidance import synodic_period, hohmann_phase_angle

# Earth-Mars synodic period
T_mars = synodic_period(1.0, 1.524)  # AU
print(f"Earth-Mars synodic period: {T_mars:.0f} days ({T_mars/365:.2f} years)")

# Earth-Venus synodic period
T_venus = synodic_period(1.0, 0.723)
print(f"Earth-Venus synodic period: {T_venus:.0f} days")

# Hohmann transfer phase angle
phi_mars = hohmann_phase_angle(1.0, 1.524)
print(f"Mars launch phase angle: {phi_mars:.1f}°")

phi_venus = hohmann_phase_angle(1.0, 0.723)
print(f"Venus launch phase angle: {phi_venus:.1f}°")
```

### Interplanetary Launch Windows

```python
from guidance import interplanetary_launch_opportunities

# Find Mars launch windows
mars_windows = interplanetary_launch_opportunities(
    target='mars',
    start_year=2025,
    num_opportunities=3
)

print("Mars Launch Opportunities:")
for w in mars_windows:
    print(f"\n{w['departure_date'].strftime('%Y-%m-%d')}:")
    print(f"  C3: {w['c3_km2_s2']:.1f} km²/s²")
    print(f"  Transfer time: {w['transfer_days']:.0f} days")
    print(f"  Arrival V∞: {w['arrival_v_inf_km_s']:.2f} km/s")

# Venus opportunities
venus_windows = interplanetary_launch_opportunities('venus', 2025, 3)
print(f"\nNext Venus window: {venus_windows[0]['departure_date'].strftime('%Y-%m-%d')}")
```

### GTO Launch Parameters

```python
from guidance import gto_launch_parameters

# Standard GTO from Kennedy
gto = gto_launch_parameters(
    target_longitude=-75.0,  # degrees West
    launch_site='kennedy'
)

print("GTO Launch Parameters:")
print(f"  Parking orbit altitude: {gto['parking_altitude_km']:.0f} km")
print(f"  GTO perigee: {gto['gto_perigee_km']:.0f} km")
print(f"  GTO apogee: {gto['gto_apogee_km']:.0f} km")
print(f"  Transfer time to GEO: {gto['transfer_time_hours']:.1f} hours")
print(f"  Required delta-V: {gto['delta_v_to_geo_km_s']:.3f} km/s")
```

### ISS Rendezvous Windows

```python
from guidance import iss_rendezvous_window
from datetime import datetime

# Find ISS rendezvous opportunities
windows = iss_rendezvous_window(
    launch_site='kennedy',
    start_date=datetime(2025, 7, 1),
    num_days=7
)

print("ISS Rendezvous Windows (next 7 days):")
for w in windows:
    print(f"  {w['launch_time'].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"    Phasing orbits: {w['phasing_orbits']}")
    print(f"    Total time to rendezvous: {w['rendezvous_time_hours']:.1f} hours")
```

### Lunar Mission Windows

```python
from guidance import lunar_launch_window
from datetime import datetime

# Find lunar transfer windows
lunar = lunar_launch_window(
    target_latitude=0.0,      # Equatorial landing
    start_date=datetime(2025, 8, 1),
    mission_type='direct'     # Direct transfer (vs free-return)
)

print("Lunar Launch Window:")
print(f"  Launch date: {lunar['launch_date'].strftime('%Y-%m-%d')}")
print(f"  TLI delta-V: {lunar['tli_delta_v_km_s']:.3f} km/s")
print(f"  Transfer time: {lunar['transfer_time_days']:.1f} days")
print(f"  Moon phase: {lunar['moon_phase']}")
```

---

## Example: Jupiter Gravity Assist Mission

Complete example designing an Earth-Jupiter-Saturn mission trajectory.

```python
import numpy as np
from guidance import (
    compute_flyby, estimate_transfer_delta_v,
    departure_v_infinity, arrival_conditions,
    gravity_assist_sequence_delta_v
)

# Mission parameters
MU_JUPITER = 126686534.0  # km³/s²
MU_SATURN = 37931206.0    # km³/s²
R_JUPITER = 71492.0       # km
R_SATURN = 58232.0        # km

print("=" * 60)
print("EARTH-JUPITER-SATURN GRAVITY ASSIST MISSION")
print("=" * 60)

# Phase 1: Earth to Jupiter
print("\n--- Phase 1: Earth to Jupiter ---")
transfer_ej = estimate_transfer_delta_v('earth', 'jupiter')
print(f"Transfer time: {transfer_ej['transfer_time_days']:.0f} days ({transfer_ej['transfer_time_days']/365:.1f} years)")
print(f"Departure C3: {transfer_ej['c3_departure']:.1f} km²/s²")
print(f"Jupiter arrival V∞: {transfer_ej['v_inf_arrival_km_s']:.2f} km/s")

# Phase 2: Jupiter Gravity Assist
print("\n--- Phase 2: Jupiter Gravity Assist ---")
v_inf_jupiter = transfer_ej['v_inf_arrival_km_s']
rp_jupiter = R_JUPITER + 500000  # 500,000 km above cloud tops

# Create V-infinity vector (simplified)
v_inf_in = np.array([v_inf_jupiter, 0.0, 0.0])

flyby = compute_flyby(v_inf_in, rp_jupiter, MU_JUPITER)
print(f"Periapsis altitude: {(flyby.periapsis - R_JUPITER)/1000:.0f} thousand km")
print(f"Turn angle: {np.degrees(flyby.turn_angle):.1f}°")
print(f"Delta-V gain from flyby: {flyby.delta_v_gain:.2f} km/s")

# Phase 3: Jupiter to Saturn
print("\n--- Phase 3: To Saturn ---")
transfer_js = estimate_transfer_delta_v('jupiter', 'saturn')
print(f"Transfer time: {transfer_js['transfer_time_days']:.0f} days ({transfer_js['transfer_time_days']/365:.1f} years)")
print(f"Saturn arrival V∞: {transfer_js['v_inf_arrival_km_s']:.2f} km/s")

# Summary
print("\n--- MISSION SUMMARY ---")
print(f"Total flight time: ~{(transfer_ej['transfer_time_days'] + transfer_js['transfer_time_days'])/365:.1f} years")
print(f"Jupiter flyby provides: {flyby.delta_v_gain:.2f} km/s free delta-V")

# Multi-flyby sequence analysis
print("\n--- Gravity Assist Sequence Analysis ---")
sequence = gravity_assist_sequence_delta_v(
    body_sequence=['earth', 'jupiter'],
    v_inf_sequence=[3.0, v_inf_jupiter],
    rp_sequence=[6678, rp_jupiter],
    mu_sequence=[398600, MU_JUPITER]
)
print(f"Total delta-V gain from flybys: {sequence['total_delta_v_gain']:.2f} km/s")
```

### Output

```
============================================================
EARTH-JUPITER-SATURN GRAVITY ASSIST MISSION
============================================================

--- Phase 1: Earth to Jupiter ---
Transfer time: 997 days (2.7 years)
Departure C3: 77.3 km²/s²
Jupiter arrival V∞: 5.64 km/s

--- Phase 2: Jupiter Gravity Assist ---
Periapsis altitude: 500 thousand km
Turn angle: 25.3°
Delta-V gain from flyby: 2.47 km/s

--- Phase 3: To Saturn ---
Transfer time: 1188 days (3.3 years)
Saturn arrival V∞: 5.44 km/s

--- MISSION SUMMARY ---
Total flight time: ~6.0 years
Jupiter flyby provides: 2.47 km/s free delta-V

--- Gravity Assist Sequence Analysis ---
Total delta-V gain from flybys: 8.63 km/s
```

---

## API Reference

### base.star_elements

| Function | Description |
|----------|-------------|
| `cartesian_to_keplerian(r, v, mu)` | Convert Cartesian to Keplerian elements |
| `keplerian_to_cartesian(elements, mu)` | Convert Keplerian to Cartesian |
| `propagate_kepler(elements, dt, mu)` | Propagate using Kepler's equation |
| `true_to_mean_anomaly(nu, e)` | Convert true to mean anomaly |
| `mean_to_true_anomaly(M, e)` | Convert mean to true anomaly |
| `vis_viva(r, a, mu)` | Orbital velocity from vis-viva |
| `orbital_period(a, mu)` | Orbital period |
| `escape_velocity(r, mu)` | Escape velocity |
| `circular_velocity(r, mu)` | Circular orbit velocity |

### guidance.flyby

| Function | Description |
|----------|-------------|
| `compute_flyby(v_inf_in, rp, mu)` | Complete flyby calculation |
| `flyby_turn_angle(v_inf, rp, mu)` | Turn angle from periapsis |
| `flyby_periapsis_from_turn_angle(v_inf, delta, mu)` | Periapsis from turn angle |
| `b_plane_from_periapsis(rp, v_inf, mu)` | B-plane parameter |
| `powered_flyby_delta_v(v_inf_in, v_inf_out, rp, mu)` | Powered flyby delta-V |
| `design_flyby_for_target(v_inf_in, target_dir, mu, ...)` | Design flyby for direction |

### guidance.patched_conics

| Function | Description |
|----------|-------------|
| `heliocentric_to_planetocentric(r, v, r_p, v_p)` | Frame transform |
| `planetocentric_to_heliocentric(r, v, r_p, v_p)` | Frame transform |
| `check_soi_entry(r_sc, r_planet, body)` | Check if inside SOI |
| `compute_c3(v_inf)` | Characteristic energy |
| `departure_v_infinity(v_sc, v_planet)` | Departure V-infinity |
| `estimate_transfer_delta_v(body1, body2)` | Transfer estimates |

### guidance.station_keeping

| Function | Description |
|----------|-------------|
| `estimate_geo_station_keeping_budget(...)` | Annual GEO delta-V budget |
| `geo_north_south_delta_v(delta_i)` | N-S inclination correction |
| `geo_inclination_drift(inc, years)` | Inclination drift prediction |
| `sun_earth_lagrange_points()` | Sun-Earth L1-L5 positions |
| `earth_moon_lagrange_points()` | Earth-Moon L1-L5 positions |
| `lagrange_station_keeping_budget(point, system)` | Annual station keeping |
| `low_thrust_delta_v(thrust, mass, time)` | Low-thrust delta-V |
| `estimate_low_thrust_budget(...)` | Low-thrust propellant budget |

### base.star_eclipse

| Function | Description |
|----------|-------------|
| `cylindrical_shadow_check(r_sc, r_body, r_sun, r)` | Fast cylindrical shadow |
| `conical_shadow_check(r_sc, r_body, r_sun, r, r_sun)` | Accurate conical shadow |
| `shadow_cone_geometry(r_sun, r_body, d)` | Umbra/penumbra geometry |
| `compute_eclipse_state(r_sc, time, bodies)` | Full eclipse state |
| `estimate_eclipse_duration_leo(alt, inc, beta)` | LEO eclipse duration |
| `estimate_eclipses_per_orbit_geo()` | GEO eclipse statistics |
| `predict_eclipses(positions, times, bodies)` | Eclipse prediction |

### base.star_thermal

| Function | Description |
|----------|-------------|
| `solar_flux_at_distance(distance_au)` | Solar flux (W/m²) |
| `planetary_view_factor(alt, r_planet)` | View factor to planet |
| `albedo_flux(solar, albedo, vf)` | Reflected sunlight |
| `planetary_ir_flux(temp, vf, eps)` | Planetary IR emission |
| `equilibrium_temperature(flux, eps)` | Radiative equilibrium |
| `compute_thermal_environment(...)` | Complete thermal environment |
| `hot_cold_case_temperatures(...)` | Worst-case analysis |
| `eclipse_cooldown_time(sc, T0, T1, env)` | Eclipse cooling time |
| `estimate_heater_power(sc, T_min, env)` | Heater power needed |
| `battery_sizing_for_eclipse(...)` | Battery capacity sizing |
| `solar_panel_sizing(power, dist, eff, ...)` | Panel area sizing |
