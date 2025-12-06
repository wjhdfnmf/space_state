# Tips for Claude Code: space_state Astrodynamics Framework

A comprehensive guide for Claude Code sessions working on the space_state spacecraft simulation codebase.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Directory Structure](#directory-structure)
4. [Core Concepts](#core-concepts)
5. [Units and Coordinate Systems](#units-and-coordinate-systems)
6. [State Vector Structure](#state-vector-structure)
7. [Key Modules Reference](#key-modules-reference)
8. [Common Patterns](#common-patterns)
9. [Testing Guidelines](#testing-guidelines)
10. [Debugging Tips](#debugging-tips)
11. [Physics References](#physics-references)
12. [Extending the Codebase](#extending-the-codebase)

---

## Project Overview

**space_state** is a high-fidelity orbital mechanics simulation framework featuring:

- **14-element state vector**: Position, velocity, mass, attitude (quaternion), angular velocity
- **N-body propagation**: Using JPL DE440 ephemerides via Skyfield in SSB frame
- **Attitude control**: Reaction wheels + RCS thrusters with realistic physics
- **Finite burn guidance**: ZEM/ZEV algorithm, Hohmann, Lambert solver
- **Mission logging**: Flight Data Recorder (FDR), real-time console display
- **Environmental effects**: Drag, SRP, gravity gradient torques, thermal, eclipse

**Total codebase**: ~98 Python files, ~15,000+ lines of physics code

---

## Environment Setup

### Virtual Environment (Required)

```bash
# ALWAYS activate before running any code
source /home/boss/data/space_state/rocket_sim/bin/activate

# Set PYTHONPATH for imports
export PYTHONPATH=/home/boss/data/space_state/space_state
```

### Running Python Scripts

```bash
# Pattern for running scripts with proper environment
source /home/boss/data/space_state/rocket_sim/bin/activate && \
PYTHONPATH=/home/boss/data/space_state/space_state \
MPLBACKEND=Agg \
LOG_LEVEL=WARNING \
timeout 120 python -c "your_code_here"

# Or run a file
python /home/boss/data/space_state/space_state/examples/moon2u.py
```

### Key Environment Variables

| Variable | Purpose | Values |
|----------|---------|--------|
| `PYTHONPATH` | Module import path | `/home/boss/data/space_state/space_state` |
| `MPLBACKEND` | Matplotlib backend (headless) | `Agg` |
| `LOG_LEVEL` | Logging verbosity | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Directory Structure

```
/home/boss/data/space_state/
├── space_state/                 # Main Python package
│   ├── base/                    # Core simulation modules (25+ files)
│   │   ├── star_propagator.py   # Integration engine (solve_ivp)
│   │   ├── star_ship.py         # Spacecraft models (Starship, Saturn V)
│   │   ├── star_attitude.py     # Quaternion math, PD control, RCS/RW
│   │   ├── star_constants.py    # State indexing, BODIES dict, ephemerides
│   │   ├── star_states.py       # Orbital state generation
│   │   ├── star_dynamics.py     # N-body acceleration, thrust
│   │   ├── star_environment.py  # SRP, atmospheric drag
│   │   ├── star_disturbance.py  # Disturbance torques (GG, aero, SRP)
│   │   ├── star_fdr.py          # Flight Data Recorder
│   │   ├── star_console.py      # Mission console display
│   │   ├── star_soi.py          # Sphere of influence
│   │   ├── star_elements.py     # Orbital elements conversion
│   │   ├── star_eclipse.py      # Eclipse detection
│   │   ├── star_thermal.py      # Thermal environment
│   │   ├── star_power.py        # Solar panels, batteries
│   │   ├── star_comm.py         # Radio communications, DSN
│   │   └── __init__.py          # ~130 public exports
│   ├── guidance/                # Guidance algorithms
│   │   ├── finite_burn_zemzev2.py  # ZEM/ZEV finite burn
│   │   ├── lambert.py           # Lambert problem solver
│   │   ├── proximity_ops.py     # Rendezvous, LVLH frame
│   │   ├── flyby.py             # Gravity assist design
│   │   ├── entry_descent.py     # EDL guidance
│   │   └── station_keeping.py   # GEO, Lagrange point keeping
│   ├── examples/                # Complete mission examples
│   │   ├── moon2u.py            # LEO→Lunar orbit (reference example)
│   │   ├── mars_transfer.py     # Earth→Mars with porkchop
│   │   ├── moon_flyby_reentry.py # Circumlunar free-return
│   │   └── mission_logging_demo.py # FDR + console demo
│   ├── tests/                   # pytest test suite (30+ files)
│   ├── visual/                  # Plotting and visualization
│   ├── mission/                 # Mission planning framework
│   ├── optimization/            # Trajectory optimization
│   └── analysis/                # Monte Carlo analysis
├── ephemeris/                   # JPL ephemeris files
│   ├── de440.bsp                # DE440 planetary positions
│   ├── gm_de440.tpc             # Gravitational parameters
│   └── pck00011.tpc             # Body radii
├── rocket_sim/                  # Python virtual environment
├── README.md                    # Project overview
├── API.md                       # API reference
└── ROADMAP.md                   # Development roadmap
```

---

## Core Concepts

### 1. Propagator

The `Propagator` class is the central integration engine:

```python
from base import Propagator, starship_raptor_vac, BODIES

ship = starship_raptor_vac()
prop = Propagator(
    ship=ship,
    guidance=guidance_controller,
    bodies=BODIES,
    disturbance_model=dist_model,  # Optional disturbance torques
)
prop.result.initialize(y0, t0)
prop.propagate(t_end)
```

### 2. Guidance Controllers

Guidance controllers implement the `.update(t, y)` method:

```python
class MyGuidance:
    def update(self, t: float, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Args:
            t: Time (days from epoch)
            y: State vector (14 elements)

        Returns:
            target_direction: Unit vector (thrust direction, inertial frame)
            desired_attitude: Unit vector (body Z-axis target)
            throttle: 0.0 to 1.0
        """
        return direction, attitude, throttle
```

### 3. Attitude Control Flow

```
Guidance.update() → desired_attitude
                  ↓
attitude_control() → computes quaternion error
                   ↓
PD Controller → control torque τ_control
              ↓
Wheel/RCS Allocation → actuator commands
                     ↓
euler_rotation_equation() → ω_dot
                          ↓
Integration (solve_ivp) → new state
```

### 4. Flight Data Recorder (FDR)

```python
from base import FlightDataRecorder, create_fdr

fdr = create_fdr("mission_name", interval=0.001)
prop.attach_fdr(fdr)

# After simulation
fdr.finalize()
fdr.save_all("/path/to/output")
fdr.generate_plots()
```

---

## Units and Coordinate Systems

### Primary Units (CRITICAL!)

| Quantity | Unit | Notes |
|----------|------|-------|
| **Position** | AU (Astronomical Units) | 1 AU = 1.496e8 km |
| **Velocity** | AU/day | ~1731.46 km/s per AU/day |
| **Time** | Days from epoch | Epoch in CONFIG.start_time |
| **Mass** | kg | Total spacecraft mass |
| **Angle** | radians | For angular velocity |
| **Angular velocity** | rad/day | NOT rad/s! |
| **Torque** | N·m | Standard SI |
| **Inertia** | kg·m² | Standard SI |

### Unit Conversions (star_utils.py)

```python
from base.star_utils import (
    km_to_au, au_to_km,           # Distance
    sec_to_days, day_to_sec,      # Time
    min_to_days,                  # Time
    mps2_apd2,                    # m/s² → AU/day²
    kgps_kgpd,                    # kg/s → kg/day
)

from base.star_constants import (
    AU_TO_KM,      # 1.495978707e8
    DAY_TO_SEC,    # 86400.0
    KM_TO_AU,      # 1.0 / AU_TO_KM
)
```

### Coordinate Frames

1. **SSB (Solar System Barycentric)**: Primary inertial frame
   - Origin: Solar system barycenter
   - Orientation: J2000 ecliptic
   - All positions/velocities in state vector use this frame

2. **Body-Centered**: For local operations
   ```python
   from base.star_utils import ssb_to_body_centered, body_centered_to_ssb
   r_moon, v_moon = ssb_to_body_centered(r_ssb, v_ssb, 'moon', t)
   ```

3. **Body Frame (Spacecraft)**: For attitude
   - +Z axis: Thrust direction (aft)
   - Origin: Center of gravity
   - Quaternion transforms body → inertial

### Quaternion Convention

- **Scalar-first**: `[q0, q1, q2, q3]` where q0 is scalar
- **Identity**: `[1, 0, 0, 0]` (no rotation)
- Transforms body frame to inertial frame

---

## State Vector Structure

```python
# State vector: y[0:14]
# [r_x, r_y, r_z, v_x, v_y, v_z, m, q0, q1, q2, q3, ωx, ωy, ωz]
#  0    1    2    3    4    5   6   7   8   9   10   11  12  13

from base.star_constants import IDX_R, IDX_V, IDX_M, IDX_Q, IDX_OMEGA

# Accessing components
r = y[IDX_R]           # Position [0:3] (AU)
v = y[IDX_V]           # Velocity [3:6] (AU/day)
m = y[IDX_M]           # Mass [6] (kg)
q = y[IDX_Q]           # Quaternion [7:11] (scalar-first)
omega = y[IDX_OMEGA]   # Angular velocity [11:14] (rad/day)

# Creating initial state
y0 = np.concatenate([r0, v0, [mass], q0, omega0])
```

---

## Key Modules Reference

### base/star_states.py - Orbital States

```python
from base import get_circular_state, get_circular_state_at_apoapsis

# Get circular orbit state
r, v, q, omega = get_circular_state(BODIES['earth'], 'leo_iss')  # 400km
r, v, q, omega = get_circular_state(BODIES['moon'], 'lunar_50km')

# Available orbits (star_orbits.py):
# Earth: 'leo_iss', 'leo_400', 'leo_500', 'meo', 'geo', 'gto'
# Moon: 'lunar_50km', 'lunar_100km', 'lunar_200km', 'nrho'
```

### base/star_ship.py - Spacecraft Models

```python
from base import starship_raptor_vac, saturn_v, starship_full_stack

ship = starship_raptor_vac()

# Key properties
ship.mass              # Total mass (kg)
ship.dry_mass          # Dry mass (kg)
ship.propellant_mass   # Main propellant (kg)
ship.rcs_propellant_mass  # RCS propellant (kg)
ship.thrust            # Current thrust (N)
ship.isp               # Specific impulse (s)
ship._inertia          # Inertia tensor (3×3 kg·m²)
ship._wheel_momentum   # Reaction wheel momentum (4,) N·m·s
ship.throttle = 1.0    # Set throttle 0-1
```

### base/star_attitude.py - Attitude Control

```python
from base.star_attitude import (
    quaternion_multiply, quaternion_conjugate, quaternion_error,
    quaternion_to_dcm, quaternion_derivative,
    euler_rotation_equation, attitude_control,
)

# Quaternion error (for control)
q_err = quaternion_error(q_desired, q_current)

# Control torque computation (called automatically by propagator)
tau_control, dm_rcs = attitude_control(t, y, ship, guidance)

# Euler's equation: I·ω̇ + ω×(I·ω) = τ
omega_dot = euler_rotation_equation(omega, I, tau)
```

### base/star_disturbance.py - Disturbance Torques

```python
from base import (
    DisturbanceTorqueModel,
    create_default_disturbance_model,
    create_deep_space_disturbance_model,
)

# Create and attach to propagator
dist_model = create_default_disturbance_model()
prop = Propagator(ship=ship, disturbance_model=dist_model)
prop.set_soi_body('earth')  # Set current SOI for torque calculations

# Disturbance types:
# - Gravity gradient: τ_gg = (3μ/r³) r̂ × (I·r̂)
# - Aerodynamic: τ_aero = r_cp × F_drag
# - Solar radiation pressure: τ_srp = r_srp × F_srp
```

### base/star_constants.py - Bodies and Constants

```python
from base import BODIES

# Access body data
earth = BODIES['earth']
mu = earth['mu']           # AU³/day² (for integration)
mu_km = earth['mu_km3_s2'] # km³/s² (for analysis)
radius = earth['radius']   # AU

# Available bodies:
# 'sun', 'mercury', 'venus', 'earth', 'moon', 'mars', 'jupiter', etc.
# Barycenters: 'mars barycenter', 'jupiter barycenter', etc.

# Hardware constants
from base.star_constants import (
    RW_MAX_MOMENTUM,  # 5000 N·m·s per wheel
    RW_MAX_TORQUE,    # 500 N·m per wheel
    RCS_THRUST,       # 1200 N per thruster
    RCS_ISP,          # 290 s
)
```

---

## Common Patterns

### Pattern 1: Basic Propagation

```python
import numpy as np
from base import BODIES, Propagator, starship_raptor_vac, get_circular_state

# Setup
ship = starship_raptor_vac()
body = BODIES['moon']
r0, v0, q0, omega0 = get_circular_state(body, 'lunar_100km')
y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

# Simple coast guidance (no thrust)
class CoastGuidance:
    def update(self, t, y):
        return np.array([0,0,0]), np.array([0,0,1]), 0.0

# Propagate
prop = Propagator(ship=ship, guidance=CoastGuidance(), bodies=BODIES)
prop.result.initialize(y0, 0.0)
prop.propagate(1.0)  # 1 day

# Results
print(f"Final position: {prop.r} AU")
print(f"Final velocity: {prop.v} AU/day")
```

### Pattern 2: Mission with FDR Logging

```python
from base import (
    Propagator, FlightDataRecorder, create_fdr,
    starship_raptor_vac, BODIES, get_circular_state,
)

ship = starship_raptor_vac()
fdr = create_fdr("my_mission", interval=0.001)

prop = Propagator(ship=ship, guidance=guidance, bodies=BODIES)
prop.attach_fdr(fdr)
prop.result.initialize(y0, 0.0)

# Mission phases...
prop.propagate(t1)
fdr.log_event("TLI_START", "Trans-lunar injection begun")
prop.propagate(t2)
fdr.log_event("TLI_END", "TLI complete")

# Finalize
fdr.finalize()
fdr.save_all("./logs/my_mission")
```

### Pattern 3: With Disturbance Torques

```python
from base import (
    Propagator, create_default_disturbance_model,
    starship_raptor_vac, BODIES,
)

ship = starship_raptor_vac()
dist_model = create_default_disturbance_model()

prop = Propagator(
    ship=ship,
    guidance=guidance,
    bodies=BODIES,
    disturbance_model=dist_model,
)
prop.set_soi_body('earth')  # Important for correct torque calculation

# Propagate - disturbance torques are now active
prop.propagate(t_end)

# Check telemetry
telem = dist_model.get_telemetry()
print(f"GG torque: {telem['tau_gg_mag']:.3f} N·m")
print(f"Aero torque: {telem['tau_aero_mag']:.3f} N·m")
```

---

## Testing Guidelines

### Running Tests

```bash
cd /home/boss/data/space_state
source rocket_sim/bin/activate

# All tests
pytest space_state/tests/ -v

# Specific module
pytest space_state/tests/test_attitude.py -v

# Single test
pytest space_state/tests/test_attitude.py::test_quaternion_multiply -v

# With coverage
pytest space_state/tests/ --cov=space_state/base
```

### Writing Tests

```python
# space_state/tests/test_my_feature.py
import numpy as np
import pytest
from base import my_function, MyClass

def test_basic_functionality():
    result = my_function(input)
    assert np.isclose(result, expected, rtol=1e-6)

def test_edge_case():
    with pytest.raises(ValueError):
        my_function(invalid_input)

@pytest.mark.parametrize("input,expected", [
    (1.0, 2.0),
    (2.0, 4.0),
])
def test_parametrized(input, expected):
    assert my_function(input) == expected
```

---

## Debugging Tips

### 1. Log Level Control

```bash
# Verbose debugging
LOG_LEVEL=DEBUG python script.py

# Quiet (errors only)
LOG_LEVEL=ERROR python script.py
```

### 2. State Vector Inspection

```python
from base.star_constants import IDX_R, IDX_V, IDX_M, IDX_Q, IDX_OMEGA

def inspect_state(y, label=""):
    print(f"=== State {label} ===")
    print(f"  r (AU): {y[IDX_R]}")
    print(f"  v (AU/day): {y[IDX_V]}")
    print(f"  m (kg): {y[IDX_M]}")
    print(f"  q: {y[IDX_Q]} (norm={np.linalg.norm(y[IDX_Q]):.6f})")
    print(f"  ω (rad/day): {y[IDX_OMEGA]}")
    print(f"  ω (deg/s): {y[IDX_OMEGA] * 180 / np.pi / 86400}")
```

### 3. Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: skyfield` | venv not activated | `source rocket_sim/bin/activate` |
| Quaternion drift (norm ≠ 1) | Integration error | Baumgarte stabilization is automatic |
| Zero attitude control | Coast mode or pre-aligned | Check guidance throttle |
| Huge torques | Wrong units | Check rad/day vs rad/s |
| NaN in propagation | Division by zero | Check for zero vectors |

### 4. Checking Attitude Control Activity

```python
# After propagation
print(f"Angular velocity: {prop.omega * 180/np.pi/86400} deg/s")
print(f"Wheel momentum: {ship._wheel_momentum} N·m·s")
print(f"RCS remaining: {ship.rcs_propellant_mass} kg")
```

---

## Physics References

### Orbital Mechanics

- **Vis-viva equation**: v² = μ(2/r - 1/a)
- **Hohmann transfer**: ΔV₁ = √(μ/r₁)(√(2r₂/(r₁+r₂)) - 1)
- **Orbital period**: T = 2π√(a³/μ)

### Attitude Dynamics

- **Euler's equation**: I·ω̇ + ω×(I·ω) = τ
- **Quaternion derivative**: q̇ = ½ q ⊗ [0, ω]
- **Gravity gradient**: τ_gg = (3μ/r³) r̂ × (I·r̂)

### Control

- **PD torque**: τ = Kp·q_err_vec + Kd·ω
- **Reaction wheel**: h_dot = -τ_wheel (momentum exchange)
- **RCS**: τ = Σ(r_i × F_i) for thruster forces

### Key References

1. Wertz, J.R. - "Spacecraft Attitude Determination and Control"
2. Vallado, D.A. - "Fundamentals of Astrodynamics and Applications"
3. Battin, R.H. - "An Introduction to the Mathematics and Methods of Astrodynamics"
4. JPL DE440 Ephemeris Documentation

---

## Extending the Codebase

### Adding a New Module

1. Create file in appropriate directory:
   ```
   space_state/base/star_newfeature.py
   ```

2. Follow module pattern:
   ```python
   # base/star_newfeature.py
   """
   New Feature Module

   Implements XYZ functionality.
   """

   import numpy as np
   from .star_logger import logger
   from .star_constants import BODIES

   def new_function(param: float) -> np.ndarray:
       """Docstring with Args/Returns."""
       pass

   class NewClass:
       """Class docstring."""
       pass
   ```

3. Export in `__init__.py`:
   ```python
   from .star_newfeature import new_function, NewClass

   __all__ = [
       # ... existing exports ...
       "new_function", "NewClass",
   ]
   ```

4. Add tests:
   ```python
   # tests/test_newfeature.py
   ```

### Adding a New Spacecraft

```python
# In star_ship.py

def my_spacecraft() -> Spacecraft:
    """Custom spacecraft configuration."""
    engine = Thruster(thrust_max=1e6, isp=350.0)
    ship = Spacecraft(
        name="MySpacecraft",
        dry_mass=10_000,
        propellant_mass=50_000,
        thrusters=[engine]
    )

    # Attitude control
    ship._inertia = np.diag([10_000, 10_000, 5_000])
    ship._rcs_propellant_mass = 500

    # Disturbance geometry
    ship._cp_offset = np.array([0, 0, -2.0])
    ship._srp_offset = np.array([1, 0, 0])
    ship._srp_area = 50.0
    ship._reflectivity = 1.3
    ship._drag_area = 20.0
    ship._drag_coefficient = 2.2

    return ship
```

### Adding a New Guidance Controller

```python
# In guidance/my_guidance.py

import numpy as np
from typing import Tuple

class MyGuidance:
    """Custom guidance law."""

    def __init__(self, target_state: np.ndarray):
        self.target = target_state

    def update(self, t: float, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute guidance command.

        Args:
            t: Time (days)
            y: State vector (14,)

        Returns:
            direction: Thrust direction (unit vector, inertial)
            attitude: Desired body Z-axis direction
            throttle: 0.0-1.0
        """
        # Your guidance logic here
        direction = self._compute_direction(y)
        throttle = self._compute_throttle(y)

        return direction, direction, throttle
```

---

## Quick Reference Card

```
UNITS:
  Position: AU          (1 AU = 1.496e8 km)
  Velocity: AU/day      (1 AU/day = 1731.46 km/s)
  Time: days            (from CONFIG epoch)
  Mass: kg
  Torque: N·m
  Angular vel: rad/day  (NOT rad/s!)

STATE VECTOR [14]:
  y[0:3]  = r (AU)
  y[3:6]  = v (AU/day)
  y[6]    = m (kg)
  y[7:11] = q [q0,q1,q2,q3] (scalar-first)
  y[11:14]= ω (rad/day)

IMPORTS:
  from base import Propagator, BODIES, starship_raptor_vac
  from base import get_circular_state, FlightDataRecorder
  from base import DisturbanceTorqueModel
  from guidance import BurnController, lambert_solve

FRAME: SSB (Solar System Barycentric, J2000 ecliptic)

QUATERNION: scalar-first [q0, q1, q2, q3], body→inertial

LOGGING:
  LOG_LEVEL=DEBUG|INFO|WARNING|ERROR

ACTIVATE VENV:
  source /home/boss/data/space_state/rocket_sim/bin/activate
```

---

*Last updated: December 2025*
*For space_state astrodynamics simulation framework*
