"""
Lorenz96 混沌控制环境与统一底层计算模块
v22: init_scale 默认 3.0（论文标准强扰动场景）
"""

import numpy as np

class Lorenz96Env:
    def __init__(self, N=40, F=8.0, dt=0.01, max_steps=800):
        self.N = N
        self.F = F
        self.dt = dt
        self.max_steps = max_steps
        self.fixed_point = np.ones(N) * F
        self.state = None
        self.step_count = 0

    def reset(self, init=None, warmup=10, init_scale=3.0):
        """v22: 默认 init_scale 提升至 3.0，触发明显混沌"""
        if init is not None:
            self.state = np.array(init, dtype=np.float64)
        else:
            self.state = self.fixed_point + np.random.uniform(-init_scale, init_scale, self.N)
            for _ in range(warmup):
                self.state += self.dt * self._rhs(self.state, np.zeros(self.N))
        self.step_count = 0
        return self.state.copy()

    def _rhs(self, s, u):
        return (np.roll(s, -1) - np.roll(s, 2)) * np.roll(s, 1) - s + self.F + u

    def _integrate(self, u):
        u = np.clip(u, -50.0, 50.0)
        k1 = self._rhs(self.state, u)
        k2 = self._rhs(self.state + .5*self.dt*k1, u)
        k3 = self._rhs(self.state + .5*self.dt*k2, u)
        k4 = self._rhs(self.state + self.dt*k3, u)
        self.state += (self.dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        if np.any(np.abs(self.state) > 200.0):
            self.state = np.clip(self.state, -200.0, 200.0)
            self.state = self.state * 0.99 + self.fixed_point * 0.01

    def step(self, u, n_substeps=1):
        for _ in range(n_substeps):
            self._integrate(u)
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        en = np.linalg.norm(self.state - self.fixed_point) / np.sqrt(self.N)
        return self.state.copy(), en, done

    def get_error(self):
        return self.state - self.fixed_point


class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=0.01, lim=50.0, N=40):
        self.N = N
        self.base_Kp = np.array(Kp, dtype=np.float64)
        self.base_Ki = np.array(Ki, dtype=np.float64)
        self.base_Kd = np.array(Kd, dtype=np.float64)
        self.dt = dt
        self.lim = lim
        self.reset()

    def reset(self):
        self.integ  = np.zeros(self.N)
        self.prev_e = np.zeros(self.N)
        self.inited = False

    def compute(self, e, Kp=None, Ki=None, Kd=None):
        Kp_use = np.broadcast_to(Kp if Kp is not None else self.base_Kp, self.N)
        Ki_use = np.broadcast_to(Ki if Ki is not None else self.base_Ki, self.N)
        Kd_use = np.broadcast_to(Kd if Kd is not None else self.base_Kd, self.N)
        self.integ = np.clip(self.integ + e * self.dt, -50.0, 50.0)
        d = np.zeros(self.N) if not self.inited else (e - self.prev_e) / self.dt
        self.inited = True
        self.prev_e = e.copy()
        u = -(Kp_use * e + Ki_use * self.integ + Kd_use * d)
        return np.clip(u, -self.lim, self.lim)


def compute_base_reward(e_next, u, N=40, energy_coef=0.05):
    """统一基础奖励，所有项 ≤ 0，最优 → 0"""
    en = np.linalg.norm(e_next) / np.sqrt(N)
    un = np.linalg.norm(u) / np.sqrt(N)
    err_r = -0.1 * (en ** 2)
    energy_r = -energy_coef * (un ** 2) / 20.0
    explode_r = -5.0 if np.any(np.abs(e_next) > 100.0) else 0.0
    r = err_r + energy_r + explode_r
    return float(np.clip(r, -10.0, 0.0))