# star_validate.py
import numpy as np

from star_utils import au_to_km, apd_kps
from star_utils import body_position, body_velocity
from star_constants import BODIES, ts, epoch_jd


def validate_orbit(result, tolerance_km=1.0, tolerance_kms=0.01):
    print("=== ORBIT VALIDATION ===")

    t_days = result.t
    r_au = result.y[:3, :].T
    v_au_day = result.y[3:6, :].T

    # --- Earth (EMB) state over time ---
    earth_r_au = np.array([body_position(BODIES['earth'], ts.tt_jd(epoch_jd + t)) for t in t_days])
    earth_v_au_day = np.array([body_velocity(BODIES['earth'], ts.tt_jd(epoch_jd + t)) for t in t_days])

    # --- Relative state ---
    r_rel_au = r_au - earth_r_au
    v_rel_au_day = v_au_day - earth_v_au_day

    r_rel_km = au_to_km(r_rel_au)
    v_rel_kms = apd_kps(v_rel_au_day)

    R_EARTH_KM = au_to_km(BODIES['earth']['radius'])
    MU_EARTH = BODIES['earth']['mu']

    # --- 1. Altitude ---
    alt0 = np.linalg.norm(r_rel_km[0]) - R_EARTH_KM
    altf = np.linalg.norm(r_rel_km[-1]) - R_EARTH_KM
    print(f"Initial altitude: {alt0:.1f} km")
    print(f"Final altitude:   {altf:.1f} km")
    print(f"ΔAltitude:        {abs(altf - alt0):.3f} km", end="")
    print(" [PASS]" if abs(altf - alt0) < tolerance_km else " [FAIL]")

    # --- 2. Energy (Earth-relative) ---
    r_norm_rel = np.linalg.norm(r_rel_au, axis=1)
    v2_rel = np.sum(v_rel_au_day**2, axis=1)
    energy = 0.5 * v2_rel - MU_EARTH / r_norm_rel

    energy0, energyf = energy[0], energy[-1]
    print(f"Initial energy: {energy0:.3e} AU²/day²")
    print(f"Final energy:   {energyf:.3e} AU²/day²")
    dE_pct = abs(energyf - energy0) / abs(energy0)
    print(f"ΔEnergy:        {dE_pct:.3e}%", end="")
    print(" [PASS]" if dE_pct < 1e-6 else " [FAIL]")

    # --- 3. Angular Momentum ---
    # h_vec = np.cross(r_au, v_au_day)
    # h = np.linalg.norm(h_vec, axis=1)
    # h0, hf = h[0], h[-1]
    # print(f"Initial h: {h0:.3e} AU²/day")
    # print(f"Final h:   {hf:.3e} AU²/day")
    # print(f"Δh:        {abs(hf - h0)/h0*100:.3e}%", end="")
    # if abs(hf - h0)/h0 < 1e-8:
    #     print(" [PASS]")
    # else:
    #     print(" [FAIL]")

    # # --- 4. Period (if >1 orbit) ---
    # if len(t_days) > 10:
    #     peaks = np.where(np.diff(np.sign(np.diff(alt_km))) > 0)[0]
    #     if len(peaks) >= 2:
    #         period_sec = (t_days[peaks[1]] - t_days[peaks[0]]) * 86400
    #         expected = 2 * np.pi * np.sqrt((R_EARTH + 400e3)**3 / 3.986e14)
    #         print(f"Measured period: {period_sec:.1f} s")
    #         print(f"Expected (400km): {expected:.1f} s")
    #         print(f"Error: {(period_sec - expected)/expected*100:.2f}%", end="")
    #         if abs(period_sec - expected) < 10:
    #             print(" [PASS]")
    #         else:
    #             print(" [FAIL]")

    print("=== END VALIDATION ===\n")