# base/star_disturbance.py
"""
Disturbance Torque Module

Models environmental disturbance torques acting on spacecraft:
- Gravity Gradient Torque
- Aerodynamic Torque
- Solar Radiation Pressure Torque

All torques are computed in the body frame for direct use in
Euler's rotation equations.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

from .star_logger import logger


# Physical constants
SOLAR_PRESSURE = 4.56e-6  # N/m² at 1 AU
AU_TO_M = 1.496e11  # meters per AU
DAY_TO_SEC = 86400.0


class DisturbanceTorqueModel:
    """
    Computes environmental disturbance torques on spacecraft.

    All torques are returned in the body frame (N·m).
    """

    def __init__(
        self,
        enable_gravity_gradient: bool = True,
        enable_aerodynamic: bool = True,
        enable_srp: bool = True,
        scale_factor: float = 1.0,
    ):
        """
        Initialize disturbance torque model.

        Args:
            enable_gravity_gradient: Enable gravity gradient torque
            enable_aerodynamic: Enable aerodynamic torque
            enable_srp: Enable solar radiation pressure torque
            scale_factor: Global scaling factor for all torques (for testing)
        """
        self.enable_gravity_gradient = enable_gravity_gradient
        self.enable_aerodynamic = enable_aerodynamic
        self.enable_srp = enable_srp
        self.scale_factor = scale_factor

        # Telemetry tracking
        self._last_tau_gg = np.zeros(3)
        self._last_tau_aero = np.zeros(3)
        self._last_tau_srp = np.zeros(3)
        self._last_tau_total = np.zeros(3)

    @staticmethod
    def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to direction cosine matrix.

        Args:
            q: [q0, q1, q2, q3] scalar-first quaternion

        Returns:
            R: 3×3 rotation matrix (body → inertial)
        """
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return np.eye(3)
        q = q / q_norm
        q0, q1, q2, q3 = q

        return np.array([
            [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
            [    2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
            [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])

    def compute_gravity_gradient(
        self,
        r_body_to_sc_inertial: np.ndarray,
        q: np.ndarray,
        I: np.ndarray,
        mu: float,
    ) -> np.ndarray:
        """
        Compute gravity gradient torque.

        τ_gg = (3μ/r³) × r̂_body × (I · r̂_body)

        This torque arises from the differential gravitational pull on
        different parts of the spacecraft due to the 1/r² gravity field.
        It tends to align the spacecraft's minimum inertia axis with
        the local vertical.

        Args:
            r_body_to_sc_inertial: Vector from central body to spacecraft (AU, inertial)
            q: Quaternion [q0, q1, q2, q3] scalar-first (body → inertial)
            I: Inertia tensor (3×3, kg·m²)
            mu: Gravitational parameter (AU³/day²)

        Returns:
            Gravity gradient torque in body frame (N·m)
        """
        r_mag_au = np.linalg.norm(r_body_to_sc_inertial)
        if r_mag_au < 1e-10:
            return np.zeros(3)

        # Unit vector from body to spacecraft in inertial frame
        r_hat_inertial = r_body_to_sc_inertial / r_mag_au

        # Transform to body frame: R^T * r_inertial = r_body
        # R is body→inertial, so R^T is inertial→body
        R = self.quaternion_to_dcm(q)
        r_hat_body = R.T @ r_hat_inertial

        # Convert distance to meters
        r_mag_m = r_mag_au * AU_TO_M

        # Convert mu from AU³/day² to m³/s²
        # 1 AU³ = (1.496e11)³ m³, 1 day² = 86400² s²
        mu_si = mu * (AU_TO_M ** 3) / (DAY_TO_SEC ** 2)

        # Gravity gradient torque: τ = (3μ/r³) * r̂ × (I · r̂)
        # This is the standard formulation from spacecraft dynamics texts
        coeff = 3.0 * mu_si / (r_mag_m ** 3)

        I_r = I @ r_hat_body  # Inertia tensor operating on radial direction
        tau_gg = coeff * np.cross(r_hat_body, I_r)

        return tau_gg

    def compute_aerodynamic_torque(
        self,
        v_rel_inertial: np.ndarray,
        rho: float,
        q: np.ndarray,
        cp_offset: np.ndarray,
        drag_coefficient: float,
        area: float,
    ) -> np.ndarray:
        """
        Compute aerodynamic torque from drag.

        τ_aero = r_cp × F_drag

        Drag force acts at the center of pressure (CP), which is offset
        from the center of gravity (CG). This offset creates a torque.

        Args:
            v_rel_inertial: Velocity relative to atmosphere (AU/day, inertial)
            rho: Atmospheric density (kg/m³)
            q: Quaternion [q0, q1, q2, q3] scalar-first
            cp_offset: CP offset from CG in body frame (m)
            drag_coefficient: Cd coefficient
            area: Cross-sectional area (m²)

        Returns:
            Aerodynamic torque in body frame (N·m)
        """
        if rho < 1e-20:
            return np.zeros(3)

        # Convert velocity to m/s
        v_rel_mps = v_rel_inertial * AU_TO_M / DAY_TO_SEC
        v_mag = np.linalg.norm(v_rel_mps)

        if v_mag < 1.0:  # Near stationary
            return np.zeros(3)

        v_hat_inertial = v_rel_mps / v_mag

        # Drag force magnitude (N)
        F_drag_mag = 0.5 * rho * v_mag**2 * drag_coefficient * area

        # Drag force in inertial frame (opposes velocity)
        F_drag_inertial = -F_drag_mag * v_hat_inertial

        # Transform force to body frame
        R = self.quaternion_to_dcm(q)
        F_drag_body = R.T @ F_drag_inertial

        # Torque = r × F (both in body frame)
        tau_aero = np.cross(cp_offset, F_drag_body)

        return tau_aero

    def compute_srp_torque(
        self,
        r_sun_to_sc_inertial: np.ndarray,
        q: np.ndarray,
        srp_offset: np.ndarray,
        srp_area: float,
        reflectivity: float,
    ) -> np.ndarray:
        """
        Compute solar radiation pressure torque.

        τ_srp = r_srp × F_srp

        SRP force acts at the center of solar pressure, which may be
        offset from CG due to non-uniform spacecraft geometry.

        Args:
            r_sun_to_sc_inertial: Vector from Sun to spacecraft (AU, inertial)
            q: Quaternion [q0, q1, q2, q3] scalar-first
            srp_offset: SRP center of pressure offset from CG (m, body frame)
            srp_area: Effective SRP area (m²)
            reflectivity: Reflectivity coefficient (1.0=absorbing, 2.0=specular)

        Returns:
            SRP torque in body frame (N·m)
        """
        dist_au = np.linalg.norm(r_sun_to_sc_inertial)
        if dist_au < 0.01:  # Too close to Sun
            return np.zeros(3)

        # Sun direction (from spacecraft toward Sun, for force direction)
        sun_dir_inertial = -r_sun_to_sc_inertial / dist_au

        # Solar pressure at this distance (N/m²)
        P = SOLAR_PRESSURE * (1.0 / dist_au) ** 2

        # SRP force magnitude (N)
        # Force pushes away from Sun
        F_srp_mag = P * srp_area * reflectivity

        # SRP force in inertial frame (pushes away from Sun)
        F_srp_inertial = -F_srp_mag * sun_dir_inertial

        # Transform force to body frame
        R = self.quaternion_to_dcm(q)
        F_srp_body = R.T @ F_srp_inertial

        # Torque = r × F
        tau_srp = np.cross(srp_offset, F_srp_body)

        return tau_srp

    def total_disturbance(
        self,
        t: float,
        r_sc: np.ndarray,
        v_sc: np.ndarray,
        q: np.ndarray,
        ship,
        soi_body_name: str,
        bodies: dict,
    ) -> np.ndarray:
        """
        Compute total disturbance torque from all sources.

        Args:
            t: Time (days from epoch)
            r_sc: Spacecraft position (AU, SSB inertial)
            v_sc: Spacecraft velocity (AU/day, SSB inertial)
            q: Quaternion [q0, q1, q2, q3] scalar-first
            ship: Spacecraft object with geometry properties
            soi_body_name: Name of body whose SOI spacecraft is in
            bodies: Dictionary of celestial bodies

        Returns:
            Total disturbance torque in body frame (N·m)
        """
        tau_total = np.zeros(3)
        self._last_tau_gg = np.zeros(3)
        self._last_tau_aero = np.zeros(3)
        self._last_tau_srp = np.zeros(3)

        # Check if ship has required properties
        if ship is None:
            self._last_tau_total = tau_total
            return tau_total

        I = getattr(ship, '_inertia', np.eye(3))

        # Get spacecraft geometry properties (with defaults)
        cp_offset = getattr(ship, '_cp_offset', np.zeros(3))
        srp_offset = getattr(ship, '_srp_offset', np.zeros(3))
        srp_area = getattr(ship, '_srp_area', 100.0)
        reflectivity = getattr(ship, '_reflectivity', 1.5)
        drag_area = getattr(ship, '_drag_area', 100.0)
        drag_coefficient = getattr(ship, '_drag_coefficient', 2.2)

        # --- Gravity Gradient Torque ---
        if self.enable_gravity_gradient and soi_body_name:
            body = bodies.get(soi_body_name)
            if body is not None:
                from .star_soi import get_body_position
                r_body = get_body_position(soi_body_name, t)
                r_body_to_sc = r_sc - r_body
                mu = body.get('mu', 0.0)

                if mu > 0:
                    tau_gg = self.compute_gravity_gradient(r_body_to_sc, q, I, mu)
                    self._last_tau_gg = tau_gg
                    tau_total += tau_gg

        # --- Aerodynamic Torque ---
        if self.enable_aerodynamic and soi_body_name:
            from .star_environment import ATMOSPHERE_PARAMS

            if soi_body_name in ATMOSPHERE_PARAMS:
                from .star_soi import get_body_position, get_body_velocity
                from .star_environment import MultiBodyAtmosphere

                params = ATMOSPHERE_PARAMS[soi_body_name]
                r_body = get_body_position(soi_body_name, t)
                v_body = get_body_velocity(soi_body_name, t)

                # Position relative to body
                r_rel = r_sc - r_body
                dist_km = np.linalg.norm(r_rel) * 1.496e8  # AU to km
                altitude_km = dist_km - params['radius_km']

                if 0 <= altitude_km <= params['max_altitude']:
                    # Get atmospheric density
                    atmo = MultiBodyAtmosphere()
                    rho = atmo.density(soi_body_name, altitude_km)

                    # Velocity relative to atmosphere
                    v_rel = v_sc - v_body

                    tau_aero = self.compute_aerodynamic_torque(
                        v_rel, rho, q, cp_offset, drag_coefficient, drag_area
                    )
                    self._last_tau_aero = tau_aero
                    tau_total += tau_aero

        # --- Solar Radiation Pressure Torque ---
        if self.enable_srp:
            # Sun is at SSB origin (approximately)
            r_sun_to_sc = r_sc  # Vector from Sun to spacecraft

            tau_srp = self.compute_srp_torque(
                r_sun_to_sc, q, srp_offset, srp_area, reflectivity
            )
            self._last_tau_srp = tau_srp
            tau_total += tau_srp

        # Apply global scale factor
        tau_total *= self.scale_factor
        self._last_tau_total = tau_total

        return tau_total

    def get_telemetry(self) -> Dict[str, Any]:
        """Get last computed torque values for telemetry."""
        return {
            'tau_gg': self._last_tau_gg.copy(),
            'tau_aero': self._last_tau_aero.copy(),
            'tau_srp': self._last_tau_srp.copy(),
            'tau_total': self._last_tau_total.copy(),
            'tau_gg_mag': np.linalg.norm(self._last_tau_gg),
            'tau_aero_mag': np.linalg.norm(self._last_tau_aero),
            'tau_srp_mag': np.linalg.norm(self._last_tau_srp),
            'tau_total_mag': np.linalg.norm(self._last_tau_total),
        }


# Convenience functions

def create_default_disturbance_model() -> DisturbanceTorqueModel:
    """Create disturbance model with default settings (all enabled)."""
    return DisturbanceTorqueModel(
        enable_gravity_gradient=True,
        enable_aerodynamic=True,
        enable_srp=True,
    )


def create_deep_space_disturbance_model() -> DisturbanceTorqueModel:
    """Create disturbance model for deep space (GG + SRP only)."""
    return DisturbanceTorqueModel(
        enable_gravity_gradient=True,
        enable_aerodynamic=False,
        enable_srp=True,
    )


def create_leo_disturbance_model() -> DisturbanceTorqueModel:
    """Create disturbance model for LEO (all enabled, aero dominant)."""
    return DisturbanceTorqueModel(
        enable_gravity_gradient=True,
        enable_aerodynamic=True,
        enable_srp=True,
    )
