# base/star_utils.py
from astropy import units as u
from astropy.time import Time
import numpy as np

from .star_structures import ControllerCycle

## ---- Unit conversion functions ---- ##
# Time conversions
def sec_to_days(seconds: float) -> float:
    # Convert seconds to days
    return (seconds * u.second).to(u.day).value

def day_to_sec(days: float) -> float:
    # Convert days to seconds
    return (days * u.day).to(u.second).value

def min_to_days(minutes: float) -> float:
    # Convert minutes to days
    return (minutes * u.minute).to(u.day).value

# Body conversions
def mean_radius_km(radii: np.ndarray) -> float:
    # Calculate the mean radius from the three principal radii
    return (radii[0] * radii[1] * radii[2]) ** (1/3)

def convert_gm(gm_km3_s2: float) -> float:
    # Convert gravitational parameter from km^3/s^2 to AU^3/day^2
    return (gm_km3_s2 * u.km**3 / u.s**2).to(u.au**3 / u.day**2).value

def au3d2_km3s2(au3_day2: float) -> float:
    # Convert from AU³/day² to km³/s²
    return (au3_day2 * u.au**3 / u.day**2).to(u.km**3 / u.s**2).value

# Mass conversions
def kgps_kgpd(kg_per_s: float) -> float:
    # Convert mass flow rate from kg/s to kg/day
    return (kg_per_s * u.kg / u.s).to(u.kg / u.day).value

# Length conversions
def km_to_au(km: float) -> float:
    # Convert kilometers to astronomical units
    return (km * u.km).to(u.au).value

def au_to_km(au: float) -> float:
    # Convert astronomical units to kilometers
    return (au * u.au).to(u.km).value

# Velocity conversions
def apd_kps(au_per_day: float) -> float:
    # Convert velocity from AU/day to km/s
    return (au_per_day * u.au / u.day).to(u.km / u.s).value

def apd_to_mps(au_per_day: float) -> float:
    # Convert velocity from AU/day to m/s
    return (au_per_day * u.au / u.day).to(u.m / u.s).value

def mps_to_apd(m_per_s: float) -> float:
    # Convert velocity from m/s to AU/day
    return (m_per_s * u.m / u.s).to(u.au / u.day).value

# Acceleration conversions
def apd2_mps2(au_per_day2: float) -> float:
    # Convert acceleration from AU/day² to m/s²
    return (au_per_day2 * u.au / u.day**2).to(u.m / u.s**2).value

def mps2_apd2(m_per_s2: float) -> float:
    # Convert acceleration from m/s² to AU/day²
    return (m_per_s2 * u.m / u.s**2).to(u.au / u.day**2).value

def mps2_to_kps2(m_per_s2: float) -> float:
    # Convert acceleration from m/s² to km/s²
    return (m_per_s2 * u.m / u.s**2).to(u.km / u.s**2).value

# Energy conversions
def au2day2_mjkg(au2_day: float) -> float:
    # Convert from AU²/day² to MJ/kg
    return (au2_day * u.au**2 / u.day**2).to(u.MJ / u.kg).value


## ---- Starfield helper functions ---- ##
def body_position(BODY: dict, time: Time) -> np.ndarray:
    # Return the position of a celestial body at a given time
    return BODY['skyfield'].at(time).position.au

def body_velocity(BODY: dict, time: Time) -> np.ndarray:
    # Return the velocity of a celestial body at a given time
    return BODY['skyfield'].at(time).velocity.au_per_d

# ----------------------------------------------
# Convert SSB state to body-centered state
def ssb_to_body_centered(time: Time, y_ssb: np.ndarray, BODY: dict) -> dict:
    r_body_au = body_position(BODY, time)
    v_body_au_d = body_velocity(BODY, time)
    
    r_rel_au = y_ssb[:3] - r_body_au
    v_rel_aud = y_ssb[3:6] - v_body_au_d
    
    # Normalize
    r_mag = np.linalg.norm(r_rel_au)
    r_hat = r_rel_au / r_mag  # radial unit vector (body-centered X)
    
    # Angular momentum vector
    h_vec = np.cross(r_rel_au, v_rel_aud)
    h_mag = np.linalg.norm(h_vec)
    if h_mag == 0:
        h_hat = np.array([0.0, 0.0, 1.0])  # fallback
    else:
        h_hat = h_vec / h_mag  # orbit normal (Z)
    
    # Tangential unit vector (Y)
    t_hat = np.cross(h_hat, r_hat)
    t_hat /= np.linalg.norm(t_hat)
    
    # Body-centered basis vectors in SSB frame
    R_bc_to_ssb = np.column_stack([r_hat, t_hat, h_hat])  # 3×3 rotation matrix
    
    # Inverse: SSB → body-centered
    R_ssb_to_bc = R_bc_to_ssb.T
    
    return {
        'r_au': r_rel_au,
        'v_aud': v_rel_aud,
        'r_km': au_to_km(r_rel_au),
        'v_kms': apd_kps(v_rel_aud),
        'y_km_kms': np.concatenate([au_to_km(r_rel_au), apd_kps(v_rel_aud), y_ssb[6:]]),
        'R_bc_to_ssb': R_bc_to_ssb,   # ← THIS IS THE KEY
        'R_ssb_to_bc': R_ssb_to_bc,
    }


## ---- Orbital mechanics helper functions ---- ##
def calculate_eccentricity(r_vec: np.ndarray, v_vec: np.ndarray, mu: float) -> float:
    # Calculate the eccentricity of an orbit given position and velocity vectors relative to the central body
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    e_vec = (np.cross(v_vec, h_vec) / mu) - (r_vec / r)
    return np.linalg.norm(e_vec)


#
# LOG GUIDANCE CYCLES
#
def log_guidance(sol: object, guidance, result):
    # Log guidance cycles
    if guidance is not None:
        for i in range(len(sol.t)):
            t_i = sol.t[i]
            y_i = sol.y[:, i]
            vgo, thrust_dir, throttle = guidance.update(t_i, y_i)

            result.log_cycle(t_i, ControllerCycle(
                vgo=vgo,
                thrust_dir=thrust_dir,
                throttle=throttle,
            ))
    # No guidance - default to zero thrust
    else:
        for i in range(len(sol.t)):
            t_i = sol.t[i]
            result.log_cycle(t_i, ControllerCycle(
                vgo=0.0,
                thrust_dir=np.array([0.0, 0.0, 0.0]),
                throttle=0.0,
            ))
    
    