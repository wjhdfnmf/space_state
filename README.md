# Space State - Orbital Mechanics Simulation

High-fidelity spacecraft dynamics and guidance simulation with attitude control.

## Overview

This project provides a Python-based orbital mechanics simulation framework featuring:

- **14-element state vector**: Position, velocity, mass, attitude (quaternion), and angular velocity
- **High-fidelity attitude control**: Reaction wheels and RCS thrusters with realistic physics
- **N-body propagation**: Solar System Barycentric (SSB) frame using Skyfield ephemerides
- **Finite burn guidance**: Hohmann transfers, inclination changes, lunar orbit insertion
- **Visualization**: 2D/3D orbital plots and attitude metrics

## State Vector Structure

```
[r_x, r_y, r_z, v_x, v_y, v_z, m, q_0, q_1, q_2, q_3, ω_x, ω_y, ω_z]
 0    1    2    3    4    5   6   7    8    9    10    11   12   13
```

- **[0:3]**: Position (AU) in SSB frame
- **[3:6]**: Velocity (AU/day) in SSB frame
- **[6]**: Total mass (kg)
- **[7:11]**: Quaternion (scalar-first: q₀, q₁, q₂, q₃)
- **[11:14]**: Angular velocity (rad/day)

## Project Structure

```
space_state/
├── base/                    # Core simulation modules
│   ├── star_attitude.py     # Attitude control (quaternions, PD control, RCS/RW allocation)
│   ├── star_constants.py    # Physical constants, state indexing
│   ├── star_dynamics.py     # N-body dynamics, thrust acceleration
│   ├── star_propagator.py   # Integration engine (scipy.integrate.solve_ivp)
│   ├── star_ship.py         # Spacecraft model (thrusters, inertia, propellant)
│   ├── star_states.py       # Orbital state generation (circular, Hohmann, LOI)
│   ├── star_structures.py   # Type aliases and PropagationResult
│   └── star_logger.py       # Logging configuration (console + file)
├── guidance/                # Guidance controllers
│   ├── __init__.py          # BurnController export
│   ├── finite_burn_zemzev2.py  # Finite burn guidance law
│   └── star_tools.py        # Orbital mechanics utilities
├── visual/                  # Visualization
│   ├── star_vis2D.py        # 2D orbital metrics plots
│   ├── star_vis3D.py        # 3D orbit visualization
│   └── star_review.py       # Attitude metrics, mission review
├── tests/                   # Test files
│   ├── orbit_moon.py        # Lunar orbit test (50km → 100km Hohmann)
│   ├── guidance_controller.py  # LEO to Moon transfer test
│   ├── test_attitude.py     # Attitude control unit tests (42 tests)
│   └── test_states.py       # State generation tests (18 tests)
└── logs/                    # Runtime log files (auto-generated)
```

## Installation

```bash
python -m venv rocket_sim
source rocket_sim/bin/activate

pip install numpy scipy matplotlib astropy skyfield pytest tqdm
pip install h5py ruff mypy black  # Development tools
```

## Running Tests

```bash
source rocket_sim/bin/activate

# Run all tests
pytest space_state/tests/ -v

# Run specific test files
pytest space_state/tests/test_attitude.py -v    # 42 attitude control tests
pytest space_state/tests/test_states.py -v      # 18 state generation tests

# Run integration tests
pytest space_state/tests/orbit_moon.py -v       # Lunar orbit maneuver
```

## Example Usage

### Basic Lunar Orbit Simulation

```python
from base.star_propagator import Propagator
from base.star_ship import starship_raptor_vac
from base.star_states import get_circular_state, get_hohmann_state_at_time
from base.star_constants import BODIES
from guidance import BurnController
import numpy as np

# Setup
ship = starship_raptor_vac()
body = BODIES['moon']

# Initial state: 50km circular lunar orbit
r0, v0, q0, omega0 = get_circular_state(body, 'lunar_50km')
y0 = np.concatenate([r0, v0, [ship.mass], q0, omega0])

# Initialize propagator with guidance
guid = BurnController(ship, body)
prop = Propagator(ship, guid)
prop.result.initialize(y0, 0.0)

# Get Hohmann transfer target (50km → 100km)
target = get_hohmann_state_at_time(prop, body, 0.0, 'lunar_100km')
guid.set_target(target)

# Propagate through burn
prop.propagate(guid.t_start)
prop.propagate(guid.t_end)
```

## Attitude Control System

### Actuators

1. **Reaction Wheels** (4-wheel tetrahedral pyramid)
   - Max momentum: 50 N·m·s per wheel
   - Max torque: 150 N·m per wheel
   - Used for normal attitude hold and small maneuvers

2. **RCS Thrusters** (16-thruster quad configuration)
   - Thrust: 1,200 N per thruster
   - Isp: 290 s (hypergolic)
   - Propellant budget: 2,000 kg (tracked separately)
   - Bang-bang control with 0.5 N·m deadband

### Control Law

- PD controller with gains tuned for thrust vs coast modes
- Quaternion error: q_err = q_desired ⊗ q_current⁻¹
- Control allocation: Prefers reaction wheels, switches to RCS for large slews or saturation

## Spacecraft Model (Starship)

```python
# Mass Properties
DRY_MASS = 120,000 kg
MAIN_PROPELLANT = 1,100,000 kg
RCS_PROPELLANT = 2,000 kg

# Main Engines (3× Raptor Vacuum)
THRUST = 7,800,000 N total
ISP = 380 s

# Inertia (9m × 50m cylinder)
I = diag([500,000, 500,000, 100,000]) kg·m²
```

## Logs

Runtime logs are saved to `space_state/logs/` with detailed timestamps:
- Console: Brief format (`HH:MM:SS | LEVEL | message`)
- File: Full format with function name and line number

---

## Teensy 4.1 Guidance Computer (PlatformIO)

Embedded guidance computer using Teensy 4.1 with FreeRTOS.

### Building

```bash
# Build guidance firmware
pio run -e guidance

# Upload to Teensy
pio run -e guidance --target upload
```

### Project Structure

```
src/
├── flight/           # Real guidance code (main.cpp)
└── test/
    ├── imu_test.cpp
    ├── quaternion_test.cpp
    └── bench.cpp
```

## Dependencies

- numpy, scipy, matplotlib (core computation)
- astropy, skyfield (ephemeris data)
- pytest (testing)
- tqdm (progress bars)
