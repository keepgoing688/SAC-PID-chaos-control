"""
独立的 Lorenz96 时空图像绘制脚本
仅生成时空图，无其他内容
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # 无 GUI 后端，适合服务器或脚本运行
import matplotlib.pyplot as plt

# ---------- 文字设定（五号字体为主）----------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10.5           # 五号 ≈ 10.5 pt
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 9       # 刻度稍小，常为小五号
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10.5

# ---------- Lorenz96 环境定义（自包含）----------
class Lorenz96Env:
    def __init__(self, N=40, F=8.0, dt=0.01, max_steps=10000):
        self.N = N
        self.F = F
        self.dt = dt
        self.max_steps = max_steps
        self.fixed_point = np.full(N, F, dtype=np.float64)
        self.state = None
        self.step_count = 0

    def reset(self, init=None):
        if init is not None:
            self.state = np.array(init, dtype=np.float64)
        else:
            self.state = self.fixed_point + np.random.uniform(-1.0, 1.0, self.N)
        self.step_count = 0
        return self.state.copy()

    def _rhs(self, s, u):
        N = self.N
        F = self.F
        ds = np.zeros(N)
        for i in range(N):
            ds[i] = (s[(i+1)%N] - s[(i-2)%N]) * s[(i-1)%N] - s[i] + F
        return ds + u

    def step(self, u):
        u = np.clip(u, -50, 50)
        k1 = self._rhs(self.state, u)
        k2 = self._rhs(self.state + 0.5*self.dt*k1, u)
        k3 = self._rhs(self.state + 0.5*self.dt*k2, u)
        k4 = self._rhs(self.state + self.dt*k3, u)
        self.state += (self.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        self.step_count += 1
        en = np.linalg.norm(self.state - self.fixed_point)
        done = (self.step_count >= self.max_steps or
                np.any(np.abs(self.state) > 200))
        return self.state.copy(), en, done

# ---------- 绘图函数 ----------
def plot_spacetime(N=40, F=8.0, dt=0.01,
                   n_transient=1000, n_steps=5000,
                   save_path='figure5.png'):
    """
    绘制 Lorenz96 时空图像（仅时空图，无其他子图或文字说明）
    N : 格点数
    F : 强迫参数
    dt : 时间步长
    n_transient : 暂态步数（不记录）
    n_steps : 记录步数
    save_path : 图片保存路径
    """
    # 创建环境并跑暂态
    env = Lorenz96Env(N=N, F=F, dt=dt, max_steps=n_transient + n_steps + 100)
    env.reset()
    for _ in range(n_transient):
        env.step(np.zeros(N))

    # 记录轨迹
    traj = []
    for _ in range(n_steps):
        env.step(np.zeros(N))
        traj.append(env.state.copy())
    traj = np.array(traj)          # shape: (n_steps, N)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(traj.T,          # 转置使 y 轴为空间格点
                   aspect='auto',
                   cmap='RdBu_r',
                   origin='lower',
                   extent=[0, n_steps * dt, 0, N - 1])
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Grid index i', fontsize=12)
    ax.set_title(f'Lorenz96 (N={N}, F={F}) Space-Time Plot', fontsize=12)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 时空图像已保存至 {save_path}")

# ---------- 主程序入口 ----------
if __name__ == "__main__":
    # 可在此修改参数
    plot_spacetime(N=40, F=8.0, dt=0.01,
                   n_transient=1000, n_steps=5000,
                   save_path='figure6.png')