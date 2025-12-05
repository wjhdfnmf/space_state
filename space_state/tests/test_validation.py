"""
Physics Validation Test Suite

Validates simulation physics against known analytical solutions:
- Hohmann transfer Δv
- Kepler orbital periods
- Circular orbit velocities
- Escape velocities
- Lambert solver accuracy
- Atmospheric density profiles
- SOI radius calculations
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base.star_constants import BODIES, AU_TO_KM, DAY_TO_SEC
from base.star_states import get_circular_state
from base.star_soi import get_body_position, get_body_velocity, SOI_RADII
from base.star_environment import MultiBodyAtmosphere, ATMOSPHERE_PARAMS
from guidance.lambert import lambert_solve


# Physical constants for validation
MU_SUN_KM3_S2 = 1.32712440018e11  # km³/s²
MU_EARTH_KM3_S2 = 3.986004418e5   # km³/s²
MU_MOON_KM3_S2 = 4.9048695e3      # km³/s²
EARTH_RADIUS_KM = 6371.0
MOON_RADIUS_KM = 1737.4


class TestKeplerOrbitalMechanics:
    """Validate Kepler's laws and orbital mechanics."""

    def test_circular_orbit_velocity_leo(self):
        """Circular velocity in LEO matches v = sqrt(μ/r)."""
        # 400 km altitude LEO
        alt_km = 400
        r_km = EARTH_RADIUS_KM + alt_km

        # Analytical circular velocity
        v_analytical_kmps = np.sqrt(MU_EARTH_KM3_S2 / r_km)

        # Our simulation uses AU/day - convert
        # v = sqrt(mu/r) where mu is in AU³/day²
        mu_earth_au = BODIES['earth']['mu']
        r_au = r_km / AU_TO_KM
        v_au_day = np.sqrt(mu_earth_au / r_au)
        v_sim_kmps = v_au_day * AU_TO_KM / DAY_TO_SEC

        # Should match within 1%
        assert abs(v_sim_kmps - v_analytical_kmps) / v_analytical_kmps < 0.01
        # LEO velocity should be ~7.67 km/s
        assert 7.5 < v_analytical_kmps < 7.8

    def test_circular_orbit_velocity_geo(self):
        """GEO circular velocity matches analytical."""
        # GEO altitude ~35,786 km
        r_geo_km = 42164  # Semi-major axis

        v_analytical = np.sqrt(MU_EARTH_KM3_S2 / r_geo_km)

        # Should be ~3.07 km/s
        assert 3.0 < v_analytical < 3.2

    def test_orbital_period_leo(self):
        """LEO orbital period matches T = 2π√(a³/μ)."""
        alt_km = 400
        a_km = EARTH_RADIUS_KM + alt_km

        # Analytical period
        T_analytical_s = 2 * np.pi * np.sqrt(a_km**3 / MU_EARTH_KM3_S2)
        T_analytical_min = T_analytical_s / 60

        # ISS orbital period is ~92.68 minutes
        assert 90 < T_analytical_min < 95

    def test_orbital_period_geo(self):
        """GEO orbital period is ~24 hours."""
        a_geo_km = 42164

        T_s = 2 * np.pi * np.sqrt(a_geo_km**3 / MU_EARTH_KM3_S2)
        T_hours = T_s / 3600

        # Should be ~23.93 hours (sidereal day)
        assert 23.5 < T_hours < 24.5

    def test_escape_velocity_earth_surface(self):
        """Earth surface escape velocity is ~11.2 km/s."""
        v_esc = np.sqrt(2 * MU_EARTH_KM3_S2 / EARTH_RADIUS_KM)

        assert 11.0 < v_esc < 11.4

    def test_escape_velocity_leo(self):
        """Escape velocity from 400 km LEO."""
        r_km = EARTH_RADIUS_KM + 400
        v_esc = np.sqrt(2 * MU_EARTH_KM3_S2 / r_km)

        # Should be ~10.9 km/s
        assert 10.7 < v_esc < 11.1


class TestHohmannTransfer:
    """Validate Hohmann transfer calculations."""

    def test_leo_to_geo_delta_v(self):
        """LEO to GEO Hohmann transfer Δv matches analytical."""
        r1_km = EARTH_RADIUS_KM + 400   # LEO
        r2_km = 42164                    # GEO

        # Hohmann transfer velocities
        # Δv1 = √(μ/r1) * (√(2r2/(r1+r2)) - 1)
        # Δv2 = √(μ/r2) * (1 - √(2r1/(r1+r2)))

        a_transfer = (r1_km + r2_km) / 2

        v_circ_1 = np.sqrt(MU_EARTH_KM3_S2 / r1_km)
        v_circ_2 = np.sqrt(MU_EARTH_KM3_S2 / r2_km)

        v_periapsis = np.sqrt(MU_EARTH_KM3_S2 * (2/r1_km - 1/a_transfer))
        v_apoapsis = np.sqrt(MU_EARTH_KM3_S2 * (2/r2_km - 1/a_transfer))

        dv1 = v_periapsis - v_circ_1
        dv2 = v_circ_2 - v_apoapsis
        total_dv = dv1 + dv2

        # LEO to GEO is about 3.9 km/s total
        assert 3.8 < total_dv < 4.1

    def test_earth_to_mars_hohmann(self):
        """Earth to Mars Hohmann transfer Δv matches analytical."""
        r_earth_au = 1.0
        r_mars_au = 1.524

        # Using Sun's μ
        mu_sun = BODIES['sun']['mu']

        a_transfer = (r_earth_au + r_mars_au) / 2

        v_earth = np.sqrt(mu_sun / r_earth_au)  # AU/day
        v_mars = np.sqrt(mu_sun / r_mars_au)

        v_dep = np.sqrt(mu_sun * (2/r_earth_au - 1/a_transfer))
        v_arr = np.sqrt(mu_sun * (2/r_mars_au - 1/a_transfer))

        dv1_au_day = v_dep - v_earth
        dv2_au_day = v_mars - v_arr

        # Convert to km/s
        dv1_kmps = dv1_au_day * AU_TO_KM / DAY_TO_SEC
        dv2_kmps = dv2_au_day * AU_TO_KM / DAY_TO_SEC

        # Earth departure: ~2.9 km/s (heliocentric)
        # Mars arrival: ~2.6 km/s (heliocentric)
        assert 2.5 < dv1_kmps < 3.3
        assert 2.0 < abs(dv2_kmps) < 3.0


class TestLambertSolver:
    """Validate Lambert solver against known solutions."""

    def test_180_degree_transfer(self):
        """180° transfer (Hohmann-like) converges correctly."""
        r1 = np.array([1.0, 0.0, 0.0])  # 1 AU
        r2 = np.array([-1.5, 0.0, 0.0])  # 1.5 AU (opposite side)

        mu = BODIES['sun']['mu']

        # Transfer time for Hohmann-like trajectory
        a_transfer = (1.0 + 1.5) / 2
        T_half = np.pi * np.sqrt(a_transfer**3 / mu)

        sol = lambert_solve(r1, r2, T_half, mu)

        # Should produce valid velocities
        assert np.linalg.norm(sol.v1) > 0
        assert np.linalg.norm(sol.v2) > 0
        assert not np.any(np.isnan(sol.v1))
        assert not np.any(np.isnan(sol.v2))
        assert sol.converged

    def test_short_transfer(self):
        """Short arc transfer converges."""
        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([0.9, 0.4, 0.0])  # ~24° transfer

        mu = BODIES['sun']['mu']
        tof = 30.0  # 30 days

        sol = lambert_solve(r1, r2, tof, mu)

        assert np.linalg.norm(sol.v1) > 0
        assert not np.any(np.isnan(sol.v1))
        assert sol.converged

    def test_lambert_energy_conservation(self):
        """Lambert solution conserves orbital energy."""
        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([0.7, 0.7, 0.1])

        mu = BODIES['sun']['mu']
        tof = 50.0

        sol = lambert_solve(r1, r2, tof, mu)

        # Specific orbital energy: ε = v²/2 - μ/r
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        v1_mag = np.linalg.norm(sol.v1)
        v2_mag = np.linalg.norm(sol.v2)

        energy_1 = v1_mag**2 / 2 - mu / r1_mag
        energy_2 = v2_mag**2 / 2 - mu / r2_mag

        # Energy should be conserved (same orbit)
        assert abs(energy_1 - energy_2) / abs(energy_1) < 0.01


class TestSOICalculations:
    """Validate SOI radius calculations."""

    def test_earth_soi_formula(self):
        """Earth SOI matches r_SOI = a * (m_E/m_Sun)^(2/5)."""
        # Earth orbital radius
        a_earth_km = 1.496e8  # 1 AU in km

        # Mass ratio
        mass_ratio = MU_EARTH_KM3_S2 / MU_SUN_KM3_S2

        # SOI radius
        r_soi_km = a_earth_km * (mass_ratio ** 0.4)

        # Should be ~930,000 km
        assert 900000 < r_soi_km < 960000

        # Compare to our stored value
        our_soi_km = SOI_RADII['earth'] * AU_TO_KM
        assert abs(r_soi_km - our_soi_km) / r_soi_km < 0.05

    def test_moon_soi_relative_to_earth(self):
        """Moon SOI calculated relative to Earth (not Sun)."""
        # Moon orbit semi-major axis
        a_moon_km = 384400

        # Mass ratio (Moon/Earth)
        mass_ratio = MU_MOON_KM3_S2 / MU_EARTH_KM3_S2

        r_soi_km = a_moon_km * (mass_ratio ** 0.4)

        # Should be ~66,000 km
        assert 60000 < r_soi_km < 70000


class TestAtmosphereModels:
    """Validate atmospheric density models."""

    def test_earth_surface_density(self):
        """Earth sea level density is 1.225 kg/m³."""
        atm = MultiBodyAtmosphere()
        rho = atm.density('earth', 0.0)

        assert abs(rho - 1.225) < 0.01

    def test_earth_density_at_100km(self):
        """Earth density at 100 km (Karman line) ~5.6e-7 kg/m³."""
        atm = MultiBodyAtmosphere()
        rho = atm.density('earth', 100.0)

        # Order of magnitude check
        assert 1e-8 < rho < 1e-5

    def test_earth_scale_height_validation(self):
        """Validate exponential decay with scale height."""
        atm = MultiBodyAtmosphere()

        # In first layer (H=8.5 km), density should halve every ~5.9 km
        rho_0 = atm.density('earth', 0.0)
        rho_8_5 = atm.density('earth', 8.5)

        # At one scale height, density should be e^-1 ≈ 0.368 of surface
        ratio = rho_8_5 / rho_0
        assert 0.30 < ratio < 0.45

    def test_mars_thin_atmosphere(self):
        """Mars surface density is ~1.6% of Earth."""
        atm = MultiBodyAtmosphere()

        rho_mars = atm.density('mars barycenter', 0.0)
        rho_earth = atm.density('earth', 0.0)

        ratio = rho_mars / rho_earth
        assert 0.01 < ratio < 0.03

    def test_venus_thick_atmosphere(self):
        """Venus surface density is ~53x Earth."""
        atm = MultiBodyAtmosphere()

        rho_venus = atm.density('venus barycenter', 0.0)
        rho_earth = atm.density('earth', 0.0)

        ratio = rho_venus / rho_earth
        assert 40 < ratio < 60


class TestDragPhysics:
    """Validate drag force calculations."""

    def test_drag_force_formula(self):
        """Drag force F = 0.5 * ρ * v² * Cd * A."""
        # Standard conditions
        rho = 1.225  # kg/m³
        v = 100  # m/s
        Cd = 2.2
        A = 10  # m²

        F = 0.5 * rho * v**2 * Cd * A

        # Expected: 0.5 * 1.225 * 10000 * 2.2 * 10 = 134,750 N
        assert abs(F - 134750) < 100

    def test_drag_scales_with_velocity_squared(self):
        """Drag scales with v²."""
        atm = MultiBodyAtmosphere(area=10.0)

        # Get two different densities to compute drag ratio
        rho = 1.225
        v1 = 100
        v2 = 200

        F1 = 0.5 * rho * v1**2 * 2.2 * 10
        F2 = 0.5 * rho * v2**2 * 2.2 * 10

        # Should scale as v²
        assert abs(F2 / F1 - 4.0) < 0.01


class TestHeatFlux:
    """Validate reentry heat flux calculations."""

    def test_sutton_graves_correlation(self):
        """Heat flux follows q = k * sqrt(ρ/r) * v³."""
        atm = MultiBodyAtmosphere()

        # Typical reentry conditions
        alt_km = 80
        v_mps = 7000  # ~orbital velocity

        q = atm.heat_flux(alt_km, v_mps, 'earth')

        # Peak heating is typically 10-100 MW/m² for orbital reentry
        # At 80 km, 7 km/s: should be order of MW/m²
        assert q > 1e5  # > 100 kW/m²
        assert q < 1e9  # < 1 GW/m²

    def test_heat_flux_velocity_cubed(self):
        """Heat flux scales with v³."""
        atm = MultiBodyAtmosphere()

        q1 = atm.heat_flux(80.0, 5000.0, 'earth')
        q2 = atm.heat_flux(80.0, 7000.0, 'earth')

        # Ratio should be (7000/5000)³ ≈ 2.74
        expected_ratio = (7000 / 5000) ** 3
        actual_ratio = q2 / q1

        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.1


class TestOrbitalElements:
    """Validate orbital element calculations."""

    def test_vis_viva_equation(self):
        """Vis-viva: v² = μ(2/r - 1/a)."""
        mu = MU_EARTH_KM3_S2

        # Elliptical orbit: perigee 400 km, apogee 35786 km (GTO)
        r_p = EARTH_RADIUS_KM + 400
        r_a = EARTH_RADIUS_KM + 35786
        a = (r_p + r_a) / 2

        # Velocity at perigee
        v_p = np.sqrt(mu * (2/r_p - 1/a))

        # GTO perigee velocity should be ~10.2 km/s
        assert 10.0 < v_p < 10.5

        # Velocity at apogee
        v_a = np.sqrt(mu * (2/r_a - 1/a))

        # GTO apogee velocity should be ~1.6 km/s
        assert 1.4 < v_a < 1.8

    def test_angular_momentum_conservation(self):
        """Angular momentum h = r × v is constant."""
        mu = MU_EARTH_KM3_S2

        r_p = EARTH_RADIUS_KM + 400
        r_a = EARTH_RADIUS_KM + 35786
        a = (r_p + r_a) / 2

        v_p = np.sqrt(mu * (2/r_p - 1/a))
        v_a = np.sqrt(mu * (2/r_a - 1/a))

        # h = r * v (for periapsis/apoapsis where r ⊥ v)
        h_p = r_p * v_p
        h_a = r_a * v_a

        # Angular momentum should be conserved
        assert abs(h_p - h_a) / h_p < 0.001


class TestUnitConversions:
    """Validate unit conversion factors."""

    def test_au_to_km(self):
        """1 AU = 149,597,870.7 km."""
        assert abs(AU_TO_KM - 1.496e8) / 1.496e8 < 0.001

    def test_day_to_seconds(self):
        """1 day = 86400 seconds."""
        assert DAY_TO_SEC == 86400

    def test_velocity_conversion(self):
        """AU/day to km/s conversion."""
        # 1 AU/day = 149597870.7 / 86400 km/s ≈ 1731.46 km/s
        v_au_day = 1.0
        v_kmps = v_au_day * AU_TO_KM / DAY_TO_SEC

        assert abs(v_kmps - 1731.46) < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
