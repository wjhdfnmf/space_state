# Disturbance Torque Implementation Plan

## Overview
Implement realistic disturbance torques for the space_state simulation to create non-zero attitude errors during coast phases and require active attitude control engagement.

## Disturbance Torques to Implement

### 1. Gravity Gradient Torque
**Physics**: Differential gravitational pull on different parts of spacecraft creates torque.
```
τ_gg = (3μ/r³) × r̂_body × (I · r̂_body)
```
Where:
- μ = gravitational parameter of central body
- r = distance from body center
- r̂_body = unit vector from body to spacecraft in body frame
- I = spacecraft inertia tensor

**When significant**: Low orbit (LEO ~400km), elongated spacecraft

### 2. Aerodynamic Torque
**Physics**: Atmospheric drag acts at center of pressure (CP), offset from center of gravity (CG).
```
τ_aero = r_cp × F_drag
F_drag = -0.5 × ρ × v² × Cd × A × v̂
```
Where:
- r_cp = CP offset from CG in body frame
- ρ = atmospheric density
- v = velocity relative to atmosphere
- Cd, A = drag coefficient and cross-sectional area

**When significant**: Below ~500km altitude at Earth

### 3. Solar Radiation Pressure Torque
**Physics**: Solar photon pressure acts at center of pressure, offset from CG.
```
τ_srp = r_srp × F_srp
F_srp = P_sun × A × (1 + ρ_reflect) × n̂ × cos(θ)
```
Where:
- P_sun = solar pressure (~4.56e-6 N/m² at 1 AU)
- r_srp = solar pressure CP offset from CG
- A = illuminated area
- ρ_reflect = reflectivity coefficient

**When significant**: Always present in sunlight, more noticeable at higher altitudes

### 4. Residual Magnetic Torque (Optional/Future)
**Physics**: Spacecraft dipole interacts with planetary magnetic field.
```
τ_mag = m × B
```
Where m = spacecraft magnetic moment, B = local magnetic field

**When significant**: Earth orbit with uncompensated magnetic moment

## Implementation Architecture

### New File: `star_disturbance.py`

```python
class DisturbanceTorqueModel:
    """Computes environmental disturbance torques on spacecraft."""

    def __init__(self, config: dict = None):
        self.enable_gravity_gradient = True
        self.enable_aerodynamic = True
        self.enable_srp = True
        # Configuration from dict...

    def compute_gravity_gradient(self, r_body_inertial, q, I, mu, r_mag) -> np.ndarray:
        """Gravity gradient torque in body frame."""

    def compute_aerodynamic_torque(self, v_rel, rho, q, cp_offset, Cd, A) -> np.ndarray:
        """Aerodynamic torque from drag at CP offset."""

    def compute_srp_torque(self, sun_dir, q, srp_offset, A_srp, reflectivity) -> np.ndarray:
        """Solar radiation pressure torque."""

    def total_disturbance(self, t, state, ship, soi_body, sun_pos) -> np.ndarray:
        """Sum of all enabled disturbance torques."""
```

### Modifications to `star_ship.py`

Add spacecraft geometry properties:
```python
class Spacecraft:
    # Existing properties...

    # New geometry properties
    @property
    def length(self) -> float:
        """Spacecraft length (m) for gravity gradient."""

    @property
    def cp_offset(self) -> np.ndarray:
        """Center of pressure offset from CG in body frame (m)."""

    @property
    def srp_offset(self) -> np.ndarray:
        """SRP center of pressure offset from CG (m)."""

    @property
    def srp_area(self) -> float:
        """Effective SRP area (m²)."""

    @property
    def reflectivity(self) -> float:
        """Surface reflectivity coefficient (0-1)."""
```

Starship values:
- Length: ~50m (Starship spacecraft section)
- CP offset: [0, 0, -5] m (nose-forward, CP ahead of CG)
- SRP area: ~250 m² (side area)
- Reflectivity: 0.3 (stainless steel)

### Modifications to `star_propagator.py`

In `dynamics()` method:
```python
def dynamics(self, t, y):
    # ... existing code to extract state ...

    # Compute control torque
    tau_control, dm_rcs = attitude_control(...)

    # NEW: Compute disturbance torques
    if self.disturbance_model is not None:
        tau_disturbance = self.disturbance_model.total_disturbance(
            t, y, self.ship, self.current_soi_body, sun_position
        )
    else:
        tau_disturbance = np.zeros(3)

    # Total torque
    tau_total = tau_control + tau_disturbance

    # Attitude dynamics with total torque
    omega_dot = euler_rotation_equation(omega, I, tau_total)
```

## Implementation Steps

### Step 1: Create star_disturbance.py
- [ ] Implement `DisturbanceTorqueModel` class
- [ ] Implement `compute_gravity_gradient()` with frame transformation
- [ ] Implement `compute_aerodynamic_torque()` using existing atmosphere model
- [ ] Implement `compute_srp_torque()` using existing SRP force pattern
- [ ] Add unit tests for each torque calculation

### Step 2: Extend star_ship.py
- [ ] Add `_length`, `_cp_offset`, `_srp_offset`, `_srp_area`, `_reflectivity` attributes
- [ ] Add properties with getters
- [ ] Add constructor parameters with defaults
- [ ] Update Starship preset with realistic values

### Step 3: Integrate into star_propagator.py
- [ ] Add `disturbance_model` parameter to `Propagator.__init__()`
- [ ] Modify `dynamics()` to compute and add disturbance torques
- [ ] Add disturbance torque to FDR telemetry logging
- [ ] Add configuration option to enable/disable

### Step 4: Add Telemetry
- [ ] Log individual disturbance torque components
- [ ] Log total disturbance torque magnitude
- [ ] Add to FDR column set

### Step 5: Test Standalone
- [ ] Create test script demonstrating each torque type
- [ ] Verify torque magnitudes are physically reasonable
- [ ] Test with different altitudes and orientations

### Step 6: Integrate into Example Missions
- [ ] Update moon_flyby_reentry.py with disturbance model
- [ ] Update moon2u.py with disturbance model
- [ ] Verify missions still complete successfully
- [ ] Verify attitude control now shows engagement

### Step 7: Document
- [ ] Add docstrings to all new functions
- [ ] Add physics references in comments
- [ ] Update any relevant documentation

## Expected Torque Magnitudes (Starship)

| Torque Type | LEO (400km) | Lunar Transfer | Lunar Orbit |
|-------------|-------------|----------------|-------------|
| Gravity Gradient | ~0.1-1 N·m | ~1e-6 N·m | ~0.001 N·m |
| Aerodynamic | ~0.01-0.1 N·m | ~0 | ~0 |
| SRP | ~0.001 N·m | ~0.001 N·m | ~0.001 N·m |

These are significant enough to require attitude control but small enough not to overwhelm the reaction wheels.

## Validation Criteria

1. **Physical Validity**: Torque magnitudes match analytical predictions
2. **Frame Correctness**: Torques computed in correct frames and transformed properly
3. **Attitude Response**: Attitude error grows during coast without control
4. **Control Engagement**: Reaction wheels and RCS show activity countering disturbances
5. **Mission Success**: Example missions still complete within delta-V budget
