# Space State Simulation Roadmap

## Current Architecture (v1.0)
```
┌─────────────────────────────────────────────────────────────┐
│                      Propagator                              │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐        │
│  │Spacecraft│  │ Guidance │  │ PropagationResult   │        │
│  │  (1)     │  │   (1)    │  │  - t[] times        │        │
│  │          │  │          │  │  - y[14,N] states   │        │
│  └──────────┘  └──────────┘  │  - guidance cycles  │        │
│                              └─────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

**State Vector (14 elements):**
- Position: r[3] (AU, SSB frame)
- Velocity: v[3] (AU/day, SSB frame)
- Mass: m[1] (kg)
- Attitude: q[4] (quaternion, scalar-first)
- Angular velocity: ω[3] (rad/day)

---

## Phase 1: Multi-Spacecraft Support

### Design: Simulation Manager Pattern
```
┌────────────────────────────────────────────────────────────────┐
│                        Simulation                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │SpacecraftEntity│ │SpacecraftEntity│ │SpacecraftEntity│      │
│  │  id: "SC-1"    │ │  id: "SC-2"    │ │  id: "SC-3"    │      │
│  │  spacecraft    │ │  spacecraft    │ │  spacecraft    │      │
│  │  propagator    │ │  propagator    │ │  propagator    │      │
│  │  guidance      │ │  guidance      │ │  guidance      │      │
│  │  result        │ │  result        │ │  result        │      │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                 │
│  Methods:                                                       │
│  - add_spacecraft(id, spacecraft, initial_state, guidance)     │
│  - remove_spacecraft(id)                                        │
│  - propagate_all(t_end)                                         │
│  - dock(id1, id2) → new combined entity                        │
│  - undock(id, config) → split into two entities                │
│  - get_relative_state(id1, id2) → position, velocity, range    │
│  - get_all_states(t) → dict of states                          │
└────────────────────────────────────────────────────────────────┘
```

### Implementation Files:
1. `base/star_simulation.py` - New Simulation class
2. `base/star_entity.py` - SpacecraftEntity class
3. Updates to `star_propagator.py` for synchronized propagation

### Key Features:
- Independent spacecraft with own guidance
- Synchronized time advancement
- Relative state queries (range, closing velocity)
- Docking: merge two entities into one
- Separation: split entity with delta-V

---

## Phase 2: Docking & Fine Maneuvering

### Docking Modes:
1. **Approach Phase**: Long-range guidance to target spacecraft
2. **Proximity Ops**: Hold station at standoff distance
3. **Final Approach**: Low-speed approach to docking port
4. **Capture**: Merge spacecraft entities

### Required Components:
- Relative navigation (Hill frame / LVLH)
- Proximity guidance controller
- Docking port geometry
- Capture cone / alignment tolerance

### Implementation:
- `guidance/proximity_ops.py` - Station-keeping & approach
- `guidance/docking_controller.py` - Final approach guidance
- Add `docking_ports` to Spacecraft class

---

## Phase 3: Porkchop Plots

### Purpose:
Visualize delta-V requirements vs departure/arrival dates for interplanetary transfers.

### Implementation:
- Grid search over departure dates × arrival dates
- Compute Lambert solution for each pair
- Generate contour plot of total delta-V
- Mark optimal launch windows

### Files:
- `guidance/lambert.py` - Lambert solver
- `guidance/porkchop.py` - Porkchop plot generator
- `visual/star_porkchop.py` - Visualization

---

## Phase 4: Environmental Effects

### Solar Radiation Pressure (SRP):
- F_srp = (P_sun / c) * A * Cr * (AU/r)²
- Where: P_sun = 1361 W/m², Cr = reflectivity coefficient

### Solar Wind:
- Drag-like force from charged particle flux
- Affects attitude (torque from asymmetric surfaces)

### Atmospheric Drag (LEO):
- Exponential atmosphere model
- F_drag = 0.5 * ρ * v² * Cd * A

### Implementation:
- `base/star_environment.py` - Environmental forces
- Add to `star_dynamics.py` acceleration computation

---

## Phase 5: Power System

### Solar Panel Model:
- Power = η * A * S * cos(θ) * (AU/r)²
- Where: η = efficiency, A = area, S = solar constant, θ = sun angle

### Battery Model:
- State of charge (SOC) tracking
- Charge/discharge rates
- Eclipse detection (Earth/Moon shadow)

### Power Budget:
- Subsystem power draws
- Load shedding logic
- Low-power mode triggers

### Implementation:
- `base/star_power.py` - Power system model
- Add power state to spacecraft
- Eclipse computation in environment

---

## Phase 6: Radio Communications

### Link Budget Model:
- Received power: Pr = Pt * Gt * Gr * (λ/4πR)²
- Data rate based on link margin
- Antenna pointing requirements

### Ground Stations:
- DSN (Goldstone, Madrid, Canberra)
- Earth rotation for visibility
- Handoff between stations

### Relay Satellites:
- TDRS for LEO comms
- Lunar relay concepts

### Implementation:
- `base/star_comm.py` - Communication model
- Ground station database
- Link availability windows

---

## Phase 7: Mission Save/Load

### Save Format: JSON or HDF5
```json
{
  "simulation": {
    "t_current": 5.234,
    "epoch_jd": 2460000.5
  },
  "spacecraft": [
    {
      "id": "SC-1",
      "name": "Starship",
      "state": [r, v, m, q, ω],
      "spacecraft_config": {...},
      "guidance_config": {...}
    }
  ],
  "history": {
    "SC-1": {"t": [...], "y": [...]}
  }
}
```

### Implementation:
- `base/star_io.py` - Save/load functions
- Serialize all simulation state
- Restore spacecraft, guidance, propagators

---

## Phase 8: Mission Examples

### Example Missions:
1. **Apollo 11 Recreation** - TLI, LOI, lunar landing, return
2. **Mars Transfer** - Hohmann to Mars with porkchop optimization
3. **ISS Rendezvous** - LEO docking simulation
4. **Artemis** - Gateway orbit, lunar landing
5. **Starship Lunar** - Full stack launch, landing, return

### Format:
- Jupyter notebooks with explanations
- Python scripts for CI testing
- Visualization outputs

---

## Phase 9: Teensy C++ Port (Hardware-in-Loop)

### Architecture:
```
┌─────────────┐     UART/SPI      ┌─────────────────┐
│  Teensy 4.1 │◄──────────────────►│  Host Computer  │
│  FreeRTOS   │                    │  Python GUI     │
│             │                    │                 │
│ - Dynamics  │  State updates     │ - Visualization │
│ - Guidance  │  ◄───────────────► │ - Commands      │
│ - Attitude  │                    │ - Telemetry     │
└─────────────┘                    └─────────────────┘
```

### Teensy Components:
- Real-time dynamics integration (fixed timestep)
- Guidance algorithms
- Attitude control
- Telemetry downlink

### FreeRTOS Tasks:
- Navigation (10 Hz)
- Guidance (1 Hz)
- Telemetry (10 Hz)
- Command processing

### Implementation:
- `/teensy/` directory with C++ code
- CMake build system
- Serial protocol definition
- Python host interface

---

## Priority Order

| Phase | Feature | Complexity | Dependencies |
|-------|---------|------------|--------------|
| 1 | Multi-Spacecraft | Medium | None |
| 2 | Docking | Medium | Phase 1 |
| 3 | Porkchop | Low | None |
| 4 | Environment | Low | None |
| 5 | Power | Medium | Phase 4 |
| 6 | Communications | Medium | Phase 1 |
| 7 | Save/Load | Low | Phase 1 |
| 8 | Examples | Low | All above |
| 9 | Teensy Port | High | Core stable |

---

## Implementation Status

### Completed:
- [x] Flight plan optimization (combined burns)
- [x] Rocket staging
- [x] Multi-spacecraft support (star_simulation.py)
- [x] Docking/separation operations
- [x] Porkchop plots (lambert.py, porkchop.py)
- [x] Save/load (star_io.py)
- [x] Environmental effects (SRP, multi-body atmosphere)
- [x] Power system (solar panels, batteries, eclipse)
- [x] SOI detection and integration events
- [x] Attitude control (quaternions, RCS, reaction wheels)
- [x] Physics validation test suite (27 tests)
- [x] Mission planning framework (plug-and-play)
- [x] Example notebooks (4 missions)
- [x] Proximity guidance controller (Hill/CW, V-bar/R-bar, station-keeping)

### Remaining:
- [ ] Radio communications (link budget, ground stations)
- [ ] Navigation & estimation (EKF, orbit determination)
- [ ] Teensy C++ port (Hardware-in-Loop)
