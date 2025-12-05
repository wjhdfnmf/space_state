# base/star_environment.py
"""
Environmental Forces Module

Models perturbation forces from:
- Solar Radiation Pressure (SRP)
- Atmospheric drag (for LEO)
- Solar wind (optional)
"""

import numpy as np
from typing import Optional, Tuple

from .star_constants import BODIES
from .star_logger import logger


# Physical constants
SPEED_OF_LIGHT = 299792.458  # km/s
SOLAR_FLUX = 1361.0  # W/m² at 1 AU (solar constant)
SOLAR_PRESSURE = SOLAR_FLUX / (SPEED_OF_LIGHT * 1000)  # N/m² at 1 AU (~4.56e-6)

# Earth atmosphere model (exponential)
EARTH_SURFACE_DENSITY = 1.225  # kg/m³ at sea level
ATMOSPHERE_SCALE_HEIGHT = 8.5  # km
EARTH_RADIUS_KM = 6371.0  # km


class SolarRadiationPressure:
    """
    Solar Radiation Pressure model.

    SRP force per unit area at distance r from Sun:
        P = P_1AU * (1 AU / r)²

    Force on spacecraft:
        F_srp = -P * A * Cr * r̂_sun

    Where:
        A = cross-sectional area (m²)
        Cr = reflectivity coefficient (1.0 = absorbing, 2.0 = specular reflection)
        r̂_sun = unit vector from spacecraft toward Sun
    """

    def __init__(
        self,
        area: float = 100.0,
        reflectivity: float = 1.5,
        enabled: bool = True,
    ):
        """
        Initialize SRP model.

        Args:
            area: Cross-sectional area facing Sun (m²)
            reflectivity: Cr coefficient (1.0-2.0, typical 1.2-1.5)
            enabled: Whether SRP is active
        """
        self.area = area
        self.reflectivity = reflectivity
        self.enabled = enabled

    def acceleration(
        self,
        r_sc: np.ndarray,
        mass: float,
        time: float = 0.0,
    ) -> np.ndarray:
        """
        Compute SRP acceleration on spacecraft.

        Args:
            r_sc: Spacecraft position in SSB frame (AU)
            mass: Spacecraft mass (kg)
            time: Simulation time (days from epoch) - for Sun position

        Returns:
            Acceleration vector (AU/day²)
        """
        if not self.enabled or mass <= 0:
            return np.zeros(3)

        # Sun position (at SSB origin for simplicity)
        r_sun = np.zeros(3)

        # Vector from Sun to spacecraft
        r_sun_to_sc = r_sc - r_sun
        dist_au = np.linalg.norm(r_sun_to_sc)

        if dist_au < 0.01:  # Too close to Sun
            return np.zeros(3)

        r_hat = r_sun_to_sc / dist_au

        # Solar pressure at this distance (N/m²)
        P = SOLAR_PRESSURE * (1.0 / dist_au) ** 2

        # Force magnitude (N)
        F = P * self.area * self.reflectivity

        # Acceleration (m/s²)
        a_mps2 = F / mass

        # Convert to AU/day²
        # 1 AU = 1.496e11 m, 1 day = 86400 s
        # a [AU/day²] = a [m/s²] * (86400² / 1.496e11)
        conversion = 86400**2 / 1.496e11
        a_au_day2 = a_mps2 * conversion

        # SRP pushes away from Sun
        return a_au_day2 * r_hat


class AtmosphericDrag:
    """
    Atmospheric drag model for Earth LEO.

    Uses exponential atmosphere model:
        ρ(h) = ρ_0 * exp(-h / H)

    Drag force:
        F_drag = -0.5 * ρ * v² * Cd * A * v̂

    Note: This is Earth-centric and assumes spacecraft is
    orbiting Earth (uses velocity relative to Earth's rotation).
    """

    def __init__(
        self,
        area: float = 100.0,
        drag_coefficient: float = 2.2,
        enabled: bool = True,
    ):
        """
        Initialize drag model.

        Args:
            area: Cross-sectional area (m²)
            drag_coefficient: Cd (typical 2.0-2.5 for spacecraft)
            enabled: Whether drag is active
        """
        self.area = area
        self.drag_coefficient = drag_coefficient
        self.enabled = enabled

    def density(self, altitude_km: float) -> float:
        """
        Get atmospheric density at altitude.

        Args:
            altitude_km: Altitude above Earth surface (km)

        Returns:
            Density (kg/m³)
        """
        if altitude_km < 0:
            return EARTH_SURFACE_DENSITY
        if altitude_km > 1000:  # Above ~1000 km, negligible
            return 0.0

        return EARTH_SURFACE_DENSITY * np.exp(-altitude_km / ATMOSPHERE_SCALE_HEIGHT)

    def acceleration(
        self,
        r_sc: np.ndarray,
        v_sc: np.ndarray,
        mass: float,
        r_earth: np.ndarray,
        v_earth: np.ndarray,
    ) -> np.ndarray:
        """
        Compute atmospheric drag acceleration.

        Args:
            r_sc: Spacecraft position in SSB frame (AU)
            v_sc: Spacecraft velocity in SSB frame (AU/day)
            mass: Spacecraft mass (kg)
            r_earth: Earth position in SSB frame (AU)
            v_earth: Earth velocity in SSB frame (AU/day)

        Returns:
            Acceleration vector (AU/day²)
        """
        if not self.enabled or mass <= 0:
            return np.zeros(3)

        # Position relative to Earth
        r_rel = r_sc - r_earth
        dist_au = np.linalg.norm(r_rel)

        # Convert to km
        dist_km = dist_au * 1.496e8
        altitude_km = dist_km - EARTH_RADIUS_KM

        # Only apply drag in LEO
        if altitude_km < 0 or altitude_km > 1000:
            return np.zeros(3)

        # Velocity relative to Earth (ignoring Earth rotation for now)
        v_rel = v_sc - v_earth
        v_rel_kps = v_rel * 1731.46  # AU/day to km/s
        v_mag = np.linalg.norm(v_rel_kps) * 1000  # m/s

        if v_mag < 1.0:  # Near stationary
            return np.zeros(3)

        v_hat = v_rel / np.linalg.norm(v_rel)

        # Atmospheric density
        rho = self.density(altitude_km)

        if rho < 1e-20:  # Negligible
            return np.zeros(3)

        # Drag force magnitude (N)
        F = 0.5 * rho * v_mag**2 * self.drag_coefficient * self.area

        # Acceleration (m/s²)
        a_mps2 = F / mass

        # Convert to AU/day²
        conversion = 86400**2 / 1.496e11
        a_au_day2 = a_mps2 * conversion

        # Drag opposes velocity
        return -a_au_day2 * v_hat


class EnvironmentModel:
    """
    Combined environmental perturbation model.

    Aggregates multiple perturbation sources:
    - Solar Radiation Pressure
    - Atmospheric Drag
    - (Future: solar wind, magnetic field effects)
    """

    def __init__(
        self,
        srp: Optional[SolarRadiationPressure] = None,
        drag: Optional[AtmosphericDrag] = None,
    ):
        """
        Initialize environment model.

        Args:
            srp: SRP model (or None to disable)
            drag: Drag model (or None to disable)
        """
        self.srp = srp
        self.drag = drag

    @classmethod
    def default(cls, spacecraft_area: float = 100.0) -> "EnvironmentModel":
        """
        Create default environment with typical parameters.

        Args:
            spacecraft_area: Cross-sectional area (m²)

        Returns:
            EnvironmentModel with SRP and drag enabled
        """
        return cls(
            srp=SolarRadiationPressure(area=spacecraft_area),
            drag=AtmosphericDrag(area=spacecraft_area),
        )

    @classmethod
    def deep_space(cls, spacecraft_area: float = 100.0) -> "EnvironmentModel":
        """
        Create environment for deep space (SRP only, no drag).
        """
        return cls(
            srp=SolarRadiationPressure(area=spacecraft_area),
            drag=None,
        )

    @classmethod
    def leo(cls, spacecraft_area: float = 100.0) -> "EnvironmentModel":
        """
        Create environment for LEO (both SRP and drag).
        """
        return cls(
            srp=SolarRadiationPressure(area=spacecraft_area),
            drag=AtmosphericDrag(area=spacecraft_area),
        )

    def total_acceleration(
        self,
        r_sc: np.ndarray,
        v_sc: np.ndarray,
        mass: float,
        time: float = 0.0,
        r_earth: Optional[np.ndarray] = None,
        v_earth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute total environmental perturbation acceleration.

        Args:
            r_sc: Spacecraft position (AU)
            v_sc: Spacecraft velocity (AU/day)
            mass: Spacecraft mass (kg)
            time: Simulation time (days)
            r_earth: Earth position (AU) - needed for drag
            v_earth: Earth velocity (AU/day) - needed for drag

        Returns:
            Total perturbation acceleration (AU/day²)
        """
        a_total = np.zeros(3)

        if self.srp is not None:
            a_total += self.srp.acceleration(r_sc, mass, time)

        if self.drag is not None and r_earth is not None and v_earth is not None:
            a_total += self.drag.acceleration(r_sc, v_sc, mass, r_earth, v_earth)

        return a_total


def compute_srp_acceleration(
    r_sc: np.ndarray,
    mass: float,
    area: float = 100.0,
    reflectivity: float = 1.5,
) -> np.ndarray:
    """
    Convenience function for one-off SRP calculation.

    Args:
        r_sc: Spacecraft position (AU)
        mass: Mass (kg)
        area: Cross-sectional area (m²)
        reflectivity: Cr coefficient

    Returns:
        SRP acceleration (AU/day²)
    """
    srp = SolarRadiationPressure(area=area, reflectivity=reflectivity)
    return srp.acceleration(r_sc, mass)


def estimate_srp_delta_v(
    duration_days: float,
    distance_au: float = 1.0,
    mass: float = 50000.0,
    area: float = 100.0,
    reflectivity: float = 1.5,
) -> float:
    """
    Estimate cumulative delta-V from SRP over a time period.

    Useful for mission planning to understand SRP perturbation magnitude.

    Args:
        duration_days: Time period
        distance_au: Average distance from Sun
        mass: Spacecraft mass (kg)
        area: Cross-sectional area (m²)
        reflectivity: Cr coefficient

    Returns:
        Approximate delta-V (m/s)
    """
    # Pressure at distance
    P = SOLAR_PRESSURE * (1.0 / distance_au) ** 2

    # Force
    F = P * area * reflectivity

    # Acceleration (m/s²)
    a = F / mass

    # Delta-V over time
    duration_s = duration_days * 86400
    dv = a * duration_s

    return dv


# ------------------- MULTI-BODY ATMOSPHERE MODELS -------------------

# Atmosphere parameters for each body
# Format: (surface_density_kg_m3, scale_heights_km, density_floor_km)
# Scale heights is list of (altitude_km, H_km) tuples for multi-layer model
ATMOSPHERE_PARAMS = {
    'earth': {
        'surface_density': 1.225,  # kg/m³
        'radius_km': 6371.0,
        'scale_heights': [  # Multi-layer model
            (0, 8.5),      # 0-25 km: H = 8.5 km
            (25, 7.5),     # 25-50 km: H = 7.5 km
            (50, 6.5),     # 50-80 km: H = 6.5 km
            (80, 6.0),     # 80-120 km: H = 6.0 km
            (120, 30.0),   # 120-300 km: H = 30 km (thermosphere)
            (300, 60.0),   # 300-500 km: H = 60 km
            (500, 100.0),  # 500+ km: H = 100 km
        ],
        'max_altitude': 1000,  # km - negligible above this
    },
    'mars barycenter': {
        'surface_density': 0.020,  # kg/m³ (~1.6% of Earth)
        'radius_km': 3389.5,
        'scale_heights': [
            (0, 11.1),     # 0-50 km
            (50, 10.0),    # 50-100 km
            (100, 15.0),   # 100+ km
        ],
        'max_altitude': 250,
    },
    'venus barycenter': {
        'surface_density': 65.0,   # kg/m³ (~53x Earth!)
        'radius_km': 6051.8,
        'scale_heights': [
            (0, 15.9),     # Dense lower atmosphere
            (50, 5.0),     # Middle atmosphere
            (70, 4.0),     # Upper atmosphere
        ],
        'max_altitude': 250,
    },
}


class MultiBodyAtmosphere:
    """
    Multi-body atmospheric drag model.

    Supports Earth, Mars, Venus atmospheres with multi-layer
    scale height models for accurate reentry simulation.
    """

    def __init__(
        self,
        area: float = 100.0,
        drag_coefficient: float = 2.2,
        enabled: bool = True,
    ):
        self.area = area
        self.drag_coefficient = drag_coefficient
        self.enabled = enabled

    def density(self, body_name: str, altitude_km: float) -> float:
        """
        Get atmospheric density at altitude for specified body.

        Uses multi-layer exponential model for each body.

        Args:
            body_name: 'earth', 'mars barycenter', 'venus barycenter'
            altitude_km: Altitude above surface (km)

        Returns:
            Density (kg/m³)
        """
        params = ATMOSPHERE_PARAMS.get(body_name)
        if params is None:
            return 0.0

        if altitude_km < 0:
            return params['surface_density']
        if altitude_km > params['max_altitude']:
            return 0.0

        # Multi-layer exponential model
        scale_heights = params['scale_heights']
        rho = params['surface_density']

        prev_alt = 0.0
        for i, (layer_alt, H) in enumerate(scale_heights):
            if altitude_km <= layer_alt or i == len(scale_heights) - 1:
                # In this layer
                dh = altitude_km - prev_alt
                return rho * np.exp(-dh / H)
            else:
                # Pass through this layer
                dh = layer_alt - prev_alt
                rho = rho * np.exp(-dh / H)
                prev_alt = layer_alt

        return rho

    def acceleration(
        self,
        r_sc: np.ndarray,
        v_sc: np.ndarray,
        mass: float,
        body_name: str,
        r_body: np.ndarray,
        v_body: np.ndarray,
    ) -> np.ndarray:
        """
        Compute atmospheric drag acceleration for any body.

        Args:
            r_sc: Spacecraft position (AU, SSB)
            v_sc: Spacecraft velocity (AU/day, SSB)
            mass: Spacecraft mass (kg)
            body_name: Central body name
            r_body: Body position (AU, SSB)
            v_body: Body velocity (AU/day, SSB)

        Returns:
            Drag acceleration (AU/day²)
        """
        if not self.enabled or mass <= 0:
            return np.zeros(3)

        params = ATMOSPHERE_PARAMS.get(body_name)
        if params is None:
            return np.zeros(3)

        # Position relative to body
        r_rel = r_sc - r_body
        dist_km = np.linalg.norm(r_rel) * 1.496e8  # AU to km
        altitude_km = dist_km - params['radius_km']

        # Check if in atmosphere
        if altitude_km < 0 or altitude_km > params['max_altitude']:
            return np.zeros(3)

        # Velocity relative to body (includes body rotation for accurate drag)
        v_rel = v_sc - v_body
        v_rel_mps = np.linalg.norm(v_rel) * 1731.46 * 1000  # AU/day to m/s

        if v_rel_mps < 1.0:
            return np.zeros(3)

        v_hat = v_rel / np.linalg.norm(v_rel)

        # Atmospheric density
        rho = self.density(body_name, altitude_km)

        if rho < 1e-20:
            return np.zeros(3)

        # Drag force: F = 0.5 * ρ * v² * Cd * A
        F = 0.5 * rho * v_rel_mps**2 * self.drag_coefficient * self.area

        # Acceleration (m/s²)
        a_mps2 = F / mass

        # Convert to AU/day²
        conversion = 86400**2 / 1.496e11
        a_au_day2 = a_mps2 * conversion

        # Drag opposes velocity
        return -a_au_day2 * v_hat

    def heat_flux(
        self,
        altitude_km: float,
        velocity_mps: float,
        body_name: str = 'earth',
    ) -> float:
        """
        Estimate convective heat flux for reentry planning.

        Uses Sutton-Graves correlation:
            q = k * sqrt(ρ/r_nose) * v³

        Args:
            altitude_km: Altitude (km)
            velocity_mps: Velocity relative to atmosphere (m/s)
            body_name: Central body

        Returns:
            Heat flux (W/m²)
        """
        rho = self.density(body_name, altitude_km)
        if rho < 1e-20:
            return 0.0

        # Sutton-Graves constant (depends on gas composition)
        # k ≈ 1.83e-4 for Earth air (varies with composition)
        k = 1.83e-4 if body_name == 'earth' else 1.9e-4

        # Assume nose radius of 1 m for estimation
        r_nose = 1.0

        q = k * np.sqrt(rho / r_nose) * velocity_mps**3

        return q


def compute_drag_acceleration(
    r_sc: np.ndarray,
    v_sc: np.ndarray,
    mass: float,
    time: float,
    area: float = 100.0,
    drag_coefficient: float = 2.2,
) -> np.ndarray:
    """
    Compute atmospheric drag with automatic body detection.

    Uses SOI to determine which body's atmosphere to use.

    Args:
        r_sc: Spacecraft position (AU, SSB)
        v_sc: Spacecraft velocity (AU/day, SSB)
        mass: Mass (kg)
        time: Simulation time (days from epoch)
        area: Cross-sectional area (m²)
        drag_coefficient: Cd

    Returns:
        Drag acceleration (AU/day²)
    """
    from .star_soi import determine_soi, get_body_position, get_body_velocity

    # Determine which body we're near
    body_name, regime, distance = determine_soi(r_sc, time)

    # Only apply drag for bodies with atmospheres
    if body_name not in ATMOSPHERE_PARAMS:
        return np.zeros(3)

    # Get body state
    r_body = get_body_position(body_name, time)
    v_body = get_body_velocity(body_name, time)

    # Compute drag
    drag = MultiBodyAtmosphere(area=area, drag_coefficient=drag_coefficient)
    return drag.acceleration(r_sc, v_sc, mass, body_name, r_body, v_body)
