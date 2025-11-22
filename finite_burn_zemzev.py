# finite_burn_zemzev.py
import numpy as np
from scipy.integrate import solve_ivp
from star_utils import days_to_seconds, sec_to_days, mps_to_apd, mps2_apd2
from star_config import g0, SECONDS_PER_DAY
from star_dynamics import n_body_acceleration

AU = 149597870.7e3                          # m per AU
SEC_PER_DAY = SECONDS_PER_DAY

class FiniteBurnZEMZEV:
    def __init__(self, target_r, target_v, spacecraft, bodies,
                 vgo_threshold=2.0, cycle_sec=2.0):
        self.r_t = np.asarray(target_r, dtype=float)
        self.v_t = np.asarray(target_v, dtype=float)
        self.sc = spacecraft
        self.bodies = bodies
        self.vgo_thr = vgo_threshold
        self.cycle = sec_to_days(cycle_sec)
        self.last_t = -99999.0
        self.dir = np.array([0., 0., 1.])

    def _primer_magnitude(self, Tgo_sec: float, m0: float) -> tuple[float, float]:
        """Battin/D’Souza exact variable-mass primer (seconds·m/s) and remaining Δv (m/s)"""
        ve = self.sc.total_isp * g0
        a0 = self.sc.total_thrust / m0
        if a0 <= 0 or Tgo_sec <= 0:
            return 0.0, 0.0
        τ = ve / a0
        α = np.exp(Tgo_sec / τ)
        if α > 1e20:                              # burn essentially finished
            return Tgo_sec / 2.0, 0.0
        lam_sec = τ * (α - 1.0 - np.log(α))        # exact primer magnitude
        vgo = ve * np.log(α)
        return lam_sec, vgo

    def _predict_powered(self, t0, r0, v0, m0, direction_unit, Tgo_days):
        if Tgo_days <= 0:
            return r0.copy(), v0.copy(), m0

        ve = self.sc.total_isp * g0
        F = self.sc.total_thrust
        mdot = F / ve                                 # positive kg/s

        def eom(t, y):
            r, v, m = y[:3], y[3:6], y[6]
            if m <= self.sc.dry_mass:
                return np.concatenate([v, n_body_acceleration(t, y, self.bodies), [0.0]])

            a_grav = n_body_acceleration(t, y, self.bodies)
            a_thrust_si = (F / m) * direction_unit                    # m/s²
            a_thrust = mps2_apd2(a_thrust_si)                       # AU/day²
            return np.concatenate([v, a_grav + a_thrust, [-mdot]])

        sol = solve_ivp(eom, (t0, t0 + Tgo_days),
                        np.hstack([r0, v0, m0]),
                        method='DOP853', rtol=1e-12, atol=1e-12,
                        max_step=30*sec_to_days(1))
        yf = sol.y[:, -1]
        return yf[:3], yf[3:6], yf[6]

    def update(self, t, y):
        if t - self.last_t < self.cycle:
            return self.vgo_thr + 10.0, self.dir.copy()

        r, v, m = y[:3], y[3:6], y[6]
        self.last_t = t

        # Initial guess – 3-minute horizon
        Tgo_guess = sec_to_days(180.0)
        dir_guess = np.array([0., 0., 1.])

        for _ in range(4):                                 # 3–4 iterations converge < 0.1 m/s
            rf, vf, mf = self._predict_powered(t, r, v, m, dir_guess, Tgo_guess)

            zem = rf - self.r_t
            zev = vf - self.v_t

            lam_sec, vgo = self._primer_magnitude(days_to_seconds(Tgo_guess), m)
            lam = lam_sec * SEC_PER_DAY                    # → AU·day

            primer = zem + lam * zev
            dir_guess = -primer / (np.linalg.norm(primer) + 1e-20)

            # Update horizon with exact remaining Δv
            if vgo > 0.1:
                ve = self.sc.total_isp * g0
                mf_target = max(m * np.exp(-vgo / ve), self.sc.dry_mass)
                Tgo_guess = sec_to_days((m - mf_target) / (self.sc.total_thrust / ve))
            else:
                Tgo_guess = sec_to_days(1.0)

        # Final prediction & output
        rf, vf, mf = self._predict_powered(t, r, v, m, dir_guess, Tgo_guess)
        lam_sec, vgo = self._primer_magnitude(days_to_seconds(Tgo_guess), m)
        lam = lam_sec * SEC_PER_DAY
        primer = (rf - self.r_t) + lam * (vf - self.v_t)
        self.dir = -primer / (np.linalg.norm(primer) + 1e-20)

        if vgo < self.vgo_thr or days_to_seconds(Tgo_guess) < 8.0:
            self.sc.set_throttle(0.0)
            vgo = 0.0

        return vgo, self.dir.copy()