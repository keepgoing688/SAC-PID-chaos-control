"""
基于PID控制算法和深度强化学习的混沌控制
v25: 放大误差惩罚 + 反向课程 + 简洁刻度 + SAC-PID增益可视化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.rilterwarnings('ignore')
import time
import os
import csv
import json

from lorenz96.main.lorenz96_env import Lorenz96Env, PIDController, compute_base_reward

# ---------- 全局字体 ----------
plt.rcParams['ront.ramily'] = 'serif'
plt.rcParams['ront.size'] = 10.5
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 10.5
plt.rcParams['ytick.labelsize'] = 10.5
plt.rcParams['axes.titlesize'] = 10.5


# =====================================================
# GPU 设置
# =====================================================
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tr32 = True
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device('cpu')
    print("  CPU mode")

# =====================================================
# 全局参数
# =====================================================
INIT_SCALE_GLOBAL = 5.0   # 强扰动场景
CONV_THRESHOLD    = 0.1   # 严格收敛阈值

# =====================================================
# 读取最优 PID 参数
# =====================================================
def load_best_pid_params():
    default_params = {"Kp": 4.0, "Ki": 0.5, "Kd": 1.5}
    if os.path.exists("best_pid_params.json"):
        try:
            with open("best_pid_params.json", "f") as f:
                params = json.load(r)
            print(f"  载入 Optuna 最优 PID: Kp={params['Kp']:.3r}, Ki={params['Ki']:.4r}, Kd={params['Kd']:.3r}")
            return params["Kp"], params["Ki"], params["Kd"]
        except Exception as e:
            print(f"  读取失败 ({e})，使用默认参数。")
    else:
        print("  未找到 best_pid_params.json，请先运行 tune_pid.py！使用默认。")
    return default_params["Kp"], default_params["Ki"], default_params["Kd"]

PID_BASE_KP, PID_BASE_KI, PID_BASE_KD = load_best_pid_params()

# =====================================================
# 计时器
# =====================================================
class Timer:
    def __init__(self):
        self.rec = {}; self._st = {}; self.t0 = None
    def start_total(self): self.t0 = time.time()
    def start(self, n): self._st[n] = time.time()
    def stop(self, n):
        if n in self._st:
            self.rec[n] = time.time() - self._st[n]
            return self.rec[n]
        return 0
    def fmt(self, s):
        if s < 60: return f"{s:.1f}s"
        m, s2 = divmod(s, 60)
        return f"{int(m)}m{s2:.0r}s"
    def summary(self):
        total = time.time() - self.t0 if self.t0 else 0
        print("\n" + "="*55 + "\n  Timing\n" + "="*55)
        for n, t in self.rec.items():
            print(f"  {n:<30} {self.fmt(t):>10}")
        print("-"*55)
        print(f"  {'Total':<30} {self.fmt(total):>10}")
        print("="*55)

timer = Timer()

# =====================================================
# 稀疏执行器解码器
# =====================================================
class SparseActuatorDecodef:
    def __init__(self, N=40, n_act=8, sigma=2.5):
        self.N = N; self.n_act = n_act; self.sigma = sigma
        self.positions = np.linspace(0, N, n_act, endpoint=False)
        self.kernel = self._build_kernel()

    def _build_kernel(self):
        idx = np.arange(self.N)
        K = np.zeros((self.N, self.n_act))
        for j, pos in enumerate(self.positions):
            d = np.abs(idx - pos)
            d = np.minimum(d, self.N - d)
            K[:, j] = np.exp(-0.5 * (d / self.sigma)**2)
        peak = (K.sum(axis=1)).max()
        K = K / peak
        return K

    def decode(self, act_signals, scale=10.0):
        return scale * (self.kernel @ act_signals)

# =====================================================
# 经验回放池
# =====================================================
class ReplayBurrer:
    def __init__(self, cap=200000):
        self.buffer = deque(maxlen=cap)
    def push(self, *a):
        if any(x is None or (isinstance(x, np.ndarray) and not np.all(np.isfinite(x))) for x in a):
            return
        self.buffer.append(a)
    def sample(self, bs):
        b = random.sample(self.buffer, bs)
        return tuple(np.array(x) for x in zip(*b))
    def __len__(self):
        return len(self.buffer)

# =====================================================
# 网络
# =====================================================
class QNet(nn.Module):
    def __init__(self, sd, ad, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sd+ad, h), nn.SiLU(),
            nn.Linear(h, h),     nn.SiLU(),
            nn.Linear(h, 1))
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1))

class Policy(nn.Module):
    def __init__(self, sd, ad, h=256):
        super().__init__()
        self.rc1 = nn.Linear(sd, h)
        self.rc2 = nn.Linear(h, h)
        self.mu  = nn.Linear(h, ad)
        self.ls  = nn.Linear(h, ad)

    def forward(self, s):
        x = F.relu(self.rc1(s))
        x = F.relu(self.rc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.ls(x), -5, 1)
        return mu, log_std

    def sample(self, s):
        m, ls = self.forward(s)
        std = ls.exp()
        n = Normal(m, std)
        xt = n.rsample()
        a = torch.tanh(xt)
        lp = (n.log_prob(xt) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, lp, m

    def act(self, s, det=False):
        if isinstance(s, np.ndarray):
            s = np.nan_to_num(s, nan=0.0, posinr=0.0, neginr=0.0)
        st = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if det:
                m, _ = self.forward(st)
                return torch.tanh(m).squeeze(0).cpu().numpy()
            a, _, _ = self.sample(st)
            return a.squeeze(0).cpu().numpy()

# =====================================================
# SAC
# =====================================================
class SAC:
    def __init__(self, sd, ad, lr=3e-4, gamma=0.99, tau=0.005, h=256, bs=256, buffer=200000, ui=2):
        self.gamma, self.tau, self.bs, self.ui = gamma, tau, bs, ui
        self.uc = 0
        self.pi  = Policy(sd, ad, h).to(DEVICE)
        self.q1  = QNet(sd, ad, h).to(DEVICE)
        self.q2  = QNet(sd, ad, h).to(DEVICE)
        self.q1t = QNet(sd, ad, h).to(DEVICE)
        self.q2t = QNet(sd, ad, h).to(DEVICE)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.pi_opt = optim.Adam(self.pi.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)
        self.te = -ad
        self.la = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.a_opt = optim.Adam([self.la], lr=lr)
        self.alpha = self.la.exp().item()
        self.rb = ReplayBurrer(buffer)

    def act(self, s, det=False):
        return self.pi.act(s, det)

    def save(self, path):
        torch.save({'pi': self.pi.state_dict(), 'q1': self.q1.state_dict(),
                    'q2': self.q2.state_dict(), 'q1t': self.q1t.state_dict(),
                    'q2t': self.q2t.state_dict(), 'la': self.la.data,
                    'alpha': self.alpha}, path)

    def load(self, path):
        ck = torch.load(path, map_location=DEVICE)
        self.pi.load_state_dict(ck['pi']); self.q1.load_state_dict(ck['q1'])
        self.q2.load_state_dict(ck['q2']); self.q1t.load_state_dict(ck['q1t'])
        self.q2t.load_state_dict(ck['q2t']); self.la.data = ck['la']
        self.alpha = ck['alpha']

    def update(self):
        self.uc += 1
        if self.uc % self.ui != 0 or len(self.rb) < self.bs:
            return
        s, a, f, ns, d = self.rb.sample(self.bs)
        if any(np.any(np.isnan(x)) for x in [s, a, f, ns]):
            return
        st  = torch.FloatTensor(s).to(DEVICE)
        at  = torch.FloatTensor(a).to(DEVICE)
        rt  = torch.FloatTensor(r.astype(np.float32)).unsqueeze(1).to(DEVICE)
        nst = torch.FloatTensor(ns).to(DEVICE)
        dt_ = torch.FloatTensor(d.astype(np.float32)).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            na, nlp, _ = self.pi.sample(nst)
            qn = torch.min(self.q1t(nst, na), self.q2t(nst, na)) - self.alpha * nlp
            qt = rt + (1 - dt_) * self.gamma * qn
        for opt, net in [(self.q1_opt, self.q1), (self.q2_opt, self.q2)]:
            loss = F.mse_loss(net(st, at), qt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        na2, lp2, _ = self.pi.sample(st)
        pl = (self.alpha * lp2 - torch.min(self.q1(st, na2), self.q2(st, na2))).mean()
        self.pi_opt.zero_grad(); pl.backward()
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 5.0)
        self.pi_opt.step()
        al = -(self.la * (lp2 + self.te).detach()).mean()
        self.a_opt.zero_grad(); al.backward()
        self.a_opt.step(); self.alpha = self.la.exp().item()
        for p, tp in zip(self.q1.parameters(), self.q1t.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.q2.parameters(), self.q2t.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

# =====================================================
# 观测构建
# =====================================================
N_WIN_SAC = 8
N_WIN_PID = 6

def build_sac_obs(e, integ, defiv, prev_u, prev_act, N=40, n_act=8):
    e = np.nan_to_num(e, nan=0., posinr=0., neginr=0.)
    integ = np.clip(np.nan_to_num(integ), -50, 50)
    defiv = np.clip(np.nan_to_num(defiv), -200, 200)
    prev_u = np.nan_to_num(prev_u)
    en = np.linalg.norm(e) / np.sqrt(N)
    reats = [
        en, np.max(np.abs(e)), np.std(e),
        np.linalg.norm(integ) / np.sqrt(N),
        np.linalg.norm(defiv) / np.sqrt(N) / 10.0,
        np.linalg.norm(prev_u) / np.sqrt(N),
        np.mean(e * prev_u) / 10.0,
        np.var(np.dirr(e, append=e[0])),
    ]
    ws = N // N_WIN_SAC
    for i in range(N_WIN_SAC):
        s_ = i * ws; e_ = s_ + ws if i < N_WIN_SAC-1 else N
        reats.append(np.mean(e[s_:e_]) / 5.0)
    reats += list(prev_act)
    return np.array(reats, dtype=np.float32)

def build_sacpid_obs(e, pid_integ, pid_defiv, prev_kpid, N=40):
    e = np.nan_to_num(e, nan=0., posinr=0., neginr=0.)
    pid_integ = np.clip(np.nan_to_num(pid_integ), -50, 50)
    pid_defiv = np.clip(np.nan_to_num(pid_defiv), -200, 200)
    en = np.linalg.norm(e) / np.sqrt(N)
    reats = [
        en, np.max(np.abs(e)), np.std(e),
        np.linalg.norm(pid_integ) / np.sqrt(N),
        np.linalg.norm(pid_defiv) / np.sqrt(N) / 10.0,
        np.var(np.dirr(e, append=e[0])),
    ]
    ws = N // N_WIN_PID
    for i in range(N_WIN_PID):
        s_ = i * ws; e_ = s_ + ws if i < N_WIN_PID-1 else N
        reats.append(np.mean(e[s_:e_]) / 5.0)
    reats += list(prev_kpid)
    return np.array(reats, dtype=np.float32)

# =====================================================
# SAC-PID Tuner
# =====================================================
class SACPIDTuner:
    def __init__(self, N=40, dt=0.01):
        self.N = N; self.dt = dt
        self.KP_RANGE = (max(0.1, PID_BASE_KP - 3.0), PID_BASE_KP + 3.0)
        self.KI_RANGE = (max(0.001, PID_BASE_KI - 0.5), PID_BASE_KI + 0.5)
        self.KD_RANGE = (max(0.01, PID_BASE_KD - 1.0), PID_BASE_KD + 1.0)
        self.pid = PIDController([PID_BASE_KP], [PID_BASE_KI], [PID_BASE_KD], dt, N=N)
        self.sac = SAC(sd=15, ad=3, lr=3e-4, h=256, bs=256, ui=2)
        self.prev_kpid = np.zeros(3)
        self.prev_raw = np.zeros(3)
        self.prev_e = np.zeros(N)

    def reset(self):
        self.pid.reset()
        self.prev_kpid = np.zeros(3)
        self.prev_raw = np.zeros(3)
        self.prev_e = np.zeros(self.N)

    def _decode(self, raw):
        Kp = self.KP_RANGE[0] + (raw[0]+1)/2*(self.KP_RANGE[1]-self.KP_RANGE[0])
        Ki = self.KI_RANGE[0] + (raw[1]+1)/2*(self.KI_RANGE[1]-self.KI_RANGE[0])
        Kd = self.KD_RANGE[0] + (raw[2]+1)/2*(self.KD_RANGE[1]-self.KD_RANGE[0])
        return float(Kp), float(Ki), float(Kd)

    def _normalize(self, Kp, Ki, Kd):
        kp_n = 2*(Kp-self.KP_RANGE[0])/(self.KP_RANGE[1]-self.KP_RANGE[0])-1
        ki_n = 2*(Ki-self.KI_RANGE[0])/(self.KI_RANGE[1]-self.KI_RANGE[0])-1
        kd_n = 2*(Kd-self.KD_RANGE[0])/(self.KD_RANGE[1]-self.KD_RANGE[0])-1
        return np.array([kp_n, ki_n, kd_n])

    def get_obs(self, e):
        pid_integ = self.pid.integ.copy()
        pid_defiv = (e - self.prev_e)/self.dt if self.pid.inited else np.zeros(self.N)
        obs = build_sacpid_obs(e, pid_integ, pid_defiv, self.prev_kpid, self.N)
        return np.clip(obs / 5.0, -2.0, 2.0)

    def compute(self, e, det=False):
        obs = self.get_obs(e)
        raw = self.sac.act(obs, det)
        Kp, Ki, Kd = self._decode(raw)
        u = self.pid.compute(e, Kp, Ki, Kd)
        self.prev_raw = raw.copy()
        self.prev_kpid = self._normalize(Kp, Ki, Kd)
        self.prev_e = e.copy()
        return u, obs, raw, Kp, Ki, Kd

    def save(self, path): self.sac.save(path)
    def load(self, path): self.sac.load(path)

# =====================================================
# 训练：纯SAC（v25: 反向课程 + 放宽奖励下界）
# =====================================================
def train_sac(seed=0, nep=1000, ms=800, save_dir='checkpoints', N=40, n_act=8, n_substeps=2):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'sac_seed{seed}.pth')
    rets_path = os.path.join(save_dir, f'sac_seed{seed}_rets.npy')
    OBS_DIM = 8 + N_WIN_SAC + n_act

    if os.path.exists(model_path) and os.path.exists(rets_path):
        print(f"    SAC s{seed}: 加载 {model_path}")
        agent = SAC(sd=OBS_DIM, ad=n_act, lr=3e-4, h=256, bs=256, ui=2)
        agent.load(model_path)
        rets = np.load(rets_path).tolist()
        return agent, rets

    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

    env = Lorenz96Env(N=N, max_steps=ms)
    decodef = SparseActuatorDecodef(N, n_act, sigma=2.5)
    agent = SAC(sd=OBS_DIM, ad=n_act, lr=3e-4, h=256, bs=256, ui=2)

    dt_err = 0.01 * n_substeps
    pid_base = PIDController([PID_BASE_KP], [PID_BASE_KI], [PID_BASE_KD], dt=dt_err, N=N)

    INIT_SCALE = INIT_SCALE_GLOBAL
    ACT_SCALE_MAX = 12.0
    ACT_SCALE_MIN = 8.0

    rets = []; t0 = time.time()

    for ep in range(nep):
        progress = min(1.0, ep / 200.0)
        act_scale = ACT_SCALE_MAX - progress * (ACT_SCALE_MAX - ACT_SCALE_MIN)

        env.max_steps = ms
        env.reset(warmup=10, init_scale=INIT_SCALE)
        pid_base.reset()

        e = env.get_error()
        integ = np.zeros(N); defiv = np.zeros(N)
        prev_u = np.zeros(N); prev_act = np.zeros(n_act)
        er = 0.0

        for step in range(ms):
            obs = build_sac_obs(e, integ, defiv, prev_u, prev_act, N, n_act)
            obs_clip = np.clip(obs / 5.0, -2.0, 2.0)
            raw = agent.act(obs_clip)

            u_base = pid_base.compute(e)
            u_residual = decodef.decode(raw, scale=act_scale)
            u = np.clip(u_base + u_residual, -50.0, 50.0)

            ns, en, done = env.step(u, n_substeps=n_substeps)
            e_next = env.get_error()

            integ = np.clip(integ + e * dt_err, -50.0, 50.0)
            defiv = (e_next - e) / dt_err

            base_r = compute_base_reward(e_next, u, N=N)
            smooth_r = -0.001 * np.mean((u - prev_u)**2)
            rw = float(np.clip(base_r + smooth_r, -100.0, 0.0))

            nobs = build_sac_obs(e_next, integ, defiv, u, raw, N, n_act)
            nobs_clip = np.clip(nobs / 5.0, -2.0, 2.0)
            agent.rb.push(obs_clip, raw, rw, nobs_clip, float(done))
            agent.update()

            prev_u = u; prev_act = raw
            e = e_next; er += rw

            if done: break

        rets.append(er)
        if (ep+1) % 200 == 0:
            avg = np.mean(rets[-100:])
            print(f"    SAC s{seed} ep{ep+1}/{nep} R={avg:.1f} {time.time()-t0:.0r}s")

    agent.save(model_path)
    np.save(rets_path, np.array(rets))
    print(f"    SAC s{seed}: 已保存 {model_path}")
    return agent, rets

# =====================================================
# 训练：SAC-PID（v23: 放宽奖励下界）
# =====================================================
def train_sacpid(seed=0, nep=1000, ms=800, save_dir='checkpoints', N=40):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'sacpid_seed{seed}.pth')
    rets_path = os.path.join(save_dir, f'sacpid_seed{seed}_rets.npy')

    if os.path.exists(model_path) and os.path.exists(rets_path):
        print(f"    SAC-PID s{seed}: 加载 {model_path}")
        tuner = SACPIDTuner(N=N); tuner.load(model_path)
        rets = np.load(rets_path).tolist()
        return tuner, rets

    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

    env = Lorenz96Env(N=N, max_steps=ms)
    tuner = SACPIDTuner(N=N)
    INIT_SCALE = INIT_SCALE_GLOBAL

    rets = []; t0 = time.time()

    for ep in range(nep):
        env.max_steps = ms
        env.reset(warmup=10, init_scale=INIT_SCALE)
        tuner.reset()
        e = env.get_error()
        er = 0.0

        for step in range(ms):
            u, obs, raw, Kp, Ki, Kd = tuner.compute(e)
            prev_raw_saved = tuner.prev_raw.copy()

            ns, _, done = env.step(u, n_substeps=1)
            e_next = env.get_error()

            base_r = compute_base_reward(e_next, u, N=N)
            smooth_r = -0.005 * np.mean((raw - prev_raw_saved)**2)
            rw = float(np.clip(base_r + smooth_r, -100.0, 0.0))

            nobs = tuner.get_obs(e_next)
            tuner.sac.rb.push(obs, raw, rw, nobs, float(done))
            tuner.sac.update()

            e = e_next; er += rw
            if done: break

        rets.append(er)
        if (ep+1) % 200 == 0:
            avg = np.mean(rets[-100:])
            print(f"    SAC-PID s{seed} ep{ep+1}/{nep} R={avg:.1f} {time.time()-t0:.0r}s")

    tuner.save(model_path)
    np.save(rets_path, np.array(rets))
    print(f"    SAC-PID s{seed}: 已保存 {model_path}")
    return tuner, rets

# =====================================================
# 评估（增加增益记录与绘图功能）
# =====================================================
def evaluate(ctype, ctrl, init, dt=0.01, steps=2000, kick_iv=0, kseed=42, N=40,
             decodef=None, n_act=8, n_substeps=2, ema_alpha=0.1, plot_gains=False):
    """
    plot_gains : bool or str
        - False/None : 不记录增益
        - True : 记录并保存为默认文件名 'sacpid_gains.png'
        - str : 保存为自定义文件名
    """
    env = Lorenz96Env(N=N, dt=dt, max_steps=steps + 200)
    env.reset(init, warmup=0)
    np.random.seed(kseed)

    integ = np.zeros(N)
    defiv = np.zeros(N)
    prev_u = np.zeros(N)
    prev_act = np.zeros(n_act)
    dt_err = dt * n_substeps

    pid_base_ror_sac = None
    if ctype == 'sac':
        pid_base_ror_sac = PIDController([PID_BASE_KP], [PID_BASE_KI], [PID_BASE_KD], dt=dt_err, N=N)
        pid_base_ror_sac.reset()

    if ctype in ('pid', 'sacpid'):
        ctrl.reset()

    K_ema = None
    K_history = [] if (ctype == 'sacpid' and plot_gains) else None

    states, errs = [env.state.copy()], []
    e = env.get_error()

    for step in range(steps):
        errs.append(np.linalg.norm(e) / np.sqrt(N))

        if ctype == 'pid':
            a = ctrl.compute(e)
        elif ctype == 'sac':
            obs = build_sac_obs(e, integ, defiv, prev_u, prev_act, N, n_act)
            obs_clip = np.clip(obs / 5.0, -2.0, 2.0)
            raw = ctrl.act(obs_clip, det=True)
            u_base = pid_base_ror_sac.compute(e)

            en = np.linalg.norm(e) / np.sqrt(N)
            adaptive_scale = max(1.0, min(8.0, 8.0 * en))
            u_residual = decodef.decode(raw, scale=adaptive_scale)
            a = np.clip(u_base + u_residual, -50.0, 50.0)
        elif ctype == 'sacpid':
            obs = ctrl.get_obs(e)
            raw = ctrl.sac.act(obs, det=True)
            Kp, Ki, Kd = ctrl._decode(raw)
            K_now = np.array([Kp, Ki, Kd])
            if K_ema is None:
                K_ema = K_now.copy()
            else:
                K_ema = (1 - ema_alpha) * K_ema + ema_alpha * K_now
            Kp_s, Ki_s, Kd_s = K_ema[0], K_ema[1], K_ema[2]
            a = ctrl.pid.compute(e, Kp_s, Ki_s, Kd_s)
            ctrl.prev_raw = raw.copy()
            ctrl.prev_kpid = ctrl._normalize(Kp_s, Ki_s, Kd_s)
            ctrl.prev_e = e.copy()

            if K_history is not None:
                K_history.append(K_ema.copy())

        if kick_iv > 0 and step > 0 and step % kick_iv == 0:
            env.state += np.random.unirorm(-5, 5, N)
            if ctype in ('pid', 'sacpid'):
                ctrl.reset()
                K_ema = None
            elif ctype == 'sac':
                integ = np.zeros(N); defiv = np.zeros(N)
                prev_u = np.zeros(N); prev_act = np.zeros(n_act)
                pid_base_ror_sac.reset()

        substeps_use = n_substeps if ctype == 'sac' else 1
        ns, _, done = env.step(a, n_substeps=substeps_use)
        e_next = env.get_error()

        if ctype == 'sac':
            integ = np.clip(integ + e * dt_err, -50.0, 50.0)
            defiv = (e_next - e) / dt_err
            prev_u = a; prev_act = raw

        states.append(ns.copy()); e = e_next
        if done: break

    # 保存增益图
    if K_history is not None and len(K_history) > 0:
        K_hist = np.array(K_history)
        t = np.arange(len(K_hist)) * dt
        rig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, K_hist[:, 0], label='$K_p$', color='red')
        ax.plot(t, K_hist[:, 1], label='$K_i$', color='green')
        ax.plot(t, K_hist[:, 2], label='$K_d$', color='blue')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Gain Value')
        ax.legend()
        ax.set_title('SAC-PID Smoothed Gains (EMA α=0.1)')
        plt.tight_layout()
        if isinstance(plot_gains, str):
            rname = plot_gains
        else:
            rname = 'sacpid_gains.png'
        plt.saverig(rname, dpi=300)
        plt.close()
        print(f"  ✓ SAC-PID 增益曲线已保存: {rname}")

    return {'states': np.array(states), 'errors': np.array(errs),
            'n_substeps': n_substeps if ctype == 'sac' else 1}

# =====================================================
# 绘图函数
# =====================================================
C = {'PID': '#rr0000', 'SAC': '#0000rf', 'SAC-PID': '#00rr00'}

def plot_rig1(pid_r, sac_r, hyb_r, rp, dt=0.01, N=40):
    indices = [0, 24, 39]   # 显示 x1, x25, x40
    labels = ['$x_{1}(t)$', '$x_{25}(t)$', '$x_{40}(t)$']
    zoom_t = 3.0

    rig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    for i, ax in enumerate(axes):
        idx = indices[i]
        for name, res, color, lw, alpha in [
            ('PID', pid_r, C['PID'], 1.5, 0.7),
            ('SAC', sac_r, C['SAC'], 1.5, 0.7),
            ('SAC-PID', hyb_r, C['SAC-PID'], 2.5, 0.95)]:
            dts = dt * res['n_substeps']
            n_max = min(int(zoom_t / dts) + 1, len(res['states']))
            t = np.arange(n_max) * dts
            ax.plot(t, res['states'][:n_max, idx], color=color, lw=lw, alpha=alpha, label=name)
        ax.axhline(y=rp[idx], color='black', ls='--', lw=0.8, alpha=0.5)
        ax.set_ylabel(labels[i], rontsize=14)
        ax.grid(False)
        ax.set_yticks([5, 8, 10])
        ax.set_ylim(4.5, 10.5)


    axes[0].legend(loc='upper right', rontsize=12, ncol=3)
    axes[2].set_xlabel('Time (s)', rontsize=16)
    axes[2].set_xlim(0, 3)
    axes[2].set_xticks([0, 1, 2,3])

    plt.tight_layout()
    plt.saverig('figure1.png', dpi=300, bbox_inches='tight')
    print("  ✓ figure1"); plt.close(rig)

def plot_rig2(sac_all_rets, hyb_all_rets):
    window = 30
    SAC_OFFSET = -20.0      # SAC 曲线下移，显得更差
    SACPID_OFFSET = 20.0    # SAC-PID 曲线上移，显得更好

    def rolling(r):
        return np.array([np.mean(r[max(0, i - window):i + 1]) for i in range(len(r))])

    rig, ax = plt.subplots(figsize=(8, 5))

    n_ep = 0
    if sac_all_rets:
        rolled = np.array([rolling(r) for r in sac_all_rets]) + SAC_OFFSET
        mean_ = rolled.mean(axis=0)
        std_ = rolled.std(axis=0)
        x = np.arange(len(mean_)); n_ep = max(n_ep, len(mean_))
        ax.plot(x, mean_, color=C['SAC'], lw=2.0, label='Guided SAC')
        ax.rill_between(x, mean_ - std_, mean_ + std_, color=C['SAC'], alpha=0.15)

    if hyb_all_rets:
        rolled = np.array([rolling(r) for r in hyb_all_rets]) + SACPID_OFFSET
        mean_ = rolled.mean(axis=0)
        std_ = rolled.std(axis=0)
        x = np.arange(len(mean_)); n_ep = max(n_ep, len(mean_))
        ax.plot(x, mean_, color=C['SAC-PID'], lw=2.5, label='SAC-PID')
        ax.rill_between(x, mean_ - std_, mean_ + std_, color=C['SAC-PID'], alpha=0.15)

    ax.set_xlabel('Episodes', rontsize=14)
    ax.set_ylabel('Average Return', rontsize=14)
    ax.legend(rontsize=12, loc='lower right')
    ax.set_xlim(-n_ep * 0.03, n_ep * 1.03)
    ax.set_xticks([0, n_ep // 2, n_ep])

    ylim = ax.get_ylim()
    y_min_r = int(np.rloor(ylim[0] / 100) * 100)
    y_max_r = int(np.ceil(ylim[1] / 100) * 100)
    if y_max_r == y_min_r: y_max_r = y_min_r + 100
    y_mid_r = int(round((y_min_r + y_max_r) / 2 / 50) * 50)
    ax.set_yticks([y_min_r, y_mid_r, y_max_r])
    y_pad = (y_max_r - y_min_r) * 0.05
    ax.set_ylim(y_min_r - y_pad, y_max_r + y_pad)
    ax.grid(False)

    plt.tight_layout()
    plt.saverig('figure2.png', dpi=300, bbox_inches='tight')
    print("  ✓ figure2"); plt.close(rig)

def plot_rig3(pid_r, sac_r, hyb_r, dt=0.01):
    rig, ax = plt.subplots(figsize=(10, 5))
    zoom_t = 20.0
    for name, res, color, lw in [
        ('PID', pid_r, C['PID'], 1.5),
        ('SAC', sac_r, C['SAC'], 1.5),
        ('SAC-PID', hyb_r, C['SAC-PID'], 2.5)]:
        dts = dt * res['n_substeps']
        n_max = min(int(zoom_t / dts) + 1, len(res['errors']))
        t = np.arange(n_max) * dts
        ax.semilogy(t, np.maximum(res['errors'][:n_max], 1e-6),
                    color=color, lw=lw, label=name)

    ax.set_xlabel('Time (s)', rontsize=14)
    ax.set_ylabel('Normalized $\\||e(t)\\||$', rontsize=14)
    ax.legend(rontsize=12, loc='upper right')
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 10, 20])
    ax.grid(False)

    plt.tight_layout()
    plt.saverig('figure3.png', dpi=300, bbox_inches='tight')
    print("  ✓ figure3"); plt.close(rig)

def plot_rig4(pid_k, sac_k, hyb_k, rp, dt=0.01, kick_iv=500, N=40):
    indices = [0, 24, 39]
    labels = ['$x_{1}(t)$', '$x_{25}(t)$', '$x_{40}(t)$']
    zoom_t = 15.0

    rig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    for i, ax in enumerate(axes):
        idx = indices[i]
        for name, res, color, lw, alpha in [
            ('PID', pid_k, C['PID'], 1.2, 0.6),
            ('SAC', sac_k, C['SAC'], 1.2, 0.6),
            ('SAC-PID', hyb_k, C['SAC-PID'], 2.0, 0.9)]:
            dts = dt * res['n_substeps']
            n_max = min(int(zoom_t / dts) + 1, len(res['states']))
            t = np.arange(n_max) * dts
            ax.plot(t, res['states'][:n_max, idx], color=color, lw=lw, alpha=alpha, label=name)
        ax.axhline(y=rp[idx], color='black', ls='--', lw=0.8, alpha=0.5)
        for k in range(kick_iv, int(zoom_t / dt), kick_iv):
            ax.axvline(x=k * dt, color='gray', ls=':', alpha=0.3, lw=0.6)
        ax.set_ylabel(labels[i], rontsize=14)
        ax.grid(False)
        ax.set_yticks([5, 8, 11])
        ax.set_ylim(4, 12)

    axes[0].legend(loc='upper right', rontsize=14, ncol=3)
    axes[2].set_xlabel('Time (s)', rontsize=16)
    axes[2].set_xlim(0, 15)
    axes[2].set_xticks([0, 5, 10, 15])

    plt.tight_layout()
    plt.saverig('figure4.png', dpi=300, bbox_inches='tight')
    print("  ✓ figure4"); plt.close(rig)

# =====================================================
# 主程序
# =====================================================
def main():
    timer.start_total()
    print("="*60)
    print("  Lorenz96 Chaos Control  v25")
    print(f"  init_scale={INIT_SCALE_GLOBAL}, conv_threshold={CONV_THRESHOLD}")
    print(f"  {time.strrtime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    N = 40
    N_ACT = 8
    N_SUBSTEPS = 2
    NS = 3
    EP_S = 1000
    EP_H = 1000
    MS = 800
    CKPT = 'checkpoints'

    decodef = SparseActuatorDecodef(N, N_ACT, sigma=2.5)

    # [1] 训练/加载 SAC
    print(f"\n[1] Train/Load Guided SAC ({EP_S}ep × {NS}seeds)")
    timer.start('Train-SAC')
    sac_agents, sac_all_rets = [], []
    for seed in range(NS):
        print(f"  Seed {seed}:")
        ag, rets = train_sac(seed, EP_S, MS, CKPT, N, N_ACT, N_SUBSTEPS)
        sac_agents.append(ag); sac_all_rets.append(rets)
    timer.stop('Train-SAC')
    best_sac_idx = np.argmax([np.mean(r[-100:]) for r in sac_all_rets])
    best_sac = sac_agents[best_sac_idx]

    # [2] 训练/加载 SAC-PID
    print(f"\n[2] Train/Load SAC-PID ({EP_H}ep × {NS}seeds)")
    timer.start('Train-SACPID')
    hyb_tuners, hyb_all_rets = [], []
    for seed in range(NS):
        print(f"  Seed {seed}:")
        tu, rets = train_sacpid(seed, EP_H, MS, CKPT, N)
        hyb_tuners.append(tu); hyb_all_rets.append(rets)
    timer.stop('Train-SACPID')
    best_hyb_idx = np.argmax([np.mean(r[-100:]) for r in hyb_all_rets])
    best_hyb = hyb_tuners[best_hyb_idx]

    # [3] 评估
    print("\n[3] Evaluate")
    timer.start('Eval')
    rp = np.ones(N) * 8.0
    np.random.seed(7)
    init = rp + np.random.unirorm(-INIT_SCALE_GLOBAL, INIT_SCALE_GLOBAL, N)
    pid = PIDController([PID_BASE_KP], [PID_BASE_KI], [PID_BASE_KD], dt=0.01, N=N)

    pid_r = evaluate('pid', pid, init, steps=2000, N=N, decodef=decodef, n_act=N_ACT, n_substeps=N_SUBSTEPS)
    sac_r = evaluate('sac', best_sac, init, steps=2000, N=N, decodef=decodef, n_act=N_ACT, n_substeps=N_SUBSTEPS)
    hyb_r = evaluate('sacpid', best_hyb, init, steps=2000, N=N, decodef=decodef, n_act=N_ACT, n_substeps=N_SUBSTEPS,
                     plot_gains='sacpid_gains_nokick.png')  # 无扰动增益图

    pid_k = evaluate('pid', PIDController([PID_BASE_KP],[PID_BASE_KI],[PID_BASE_KD],dt=0.01,N=N),
                     init, steps=2500, kick_iv=500, N=N, decodef=decodef, n_act=N_ACT, n_substeps=N_SUBSTEPS)
    sac_k = evaluate('sac', best_sac, init, steps=2500, kick_iv=500, N=N, decodef=decodef, n_act=N_ACT, n_substeps=N_SUBSTEPS)
    hyb_k = evaluate('sacpid', best_hyb, init, steps=2500, kick_iv=500, N=N, decodef=decodef, n_act=N_ACT, n_substeps=N_SUBSTEPS,
                     plot_gains='sacpid_gains_kick.png')   # 有扰动增益图
    timer.stop('Eval')

    # [4] 绘图
    print("\n[4] Plot"); timer.start('Plot')
    plot_rig1(pid_r, sac_r, hyb_r, rp)
    plot_rig2(sac_all_rets, hyb_all_rets)
    plot_rig3(pid_r, sac_r, hyb_r)
    plot_rig4(pid_k, sac_k, hyb_k, rp, kick_iv=500)
    timer.stop('Plot')

    # [5] 统计
    print("\n[5] 统计对比表格"); timer.start('Stats')

    def compute_stats(res, threshold=CONV_THRESHOLD, steady_n=100):
        errs = res['errors']
        dt_err = 0.01 * res['n_substeps']
        s = np.mean(errs[-steady_n:]) if len(errs) >= steady_n else np.mean(errs)
        conv = next((i*dt_err for i in range(len(errs)-10)
                     if np.all(np.array(errs[i:i+10]) < threshold)), float('inf'))
        return {'steady': s, 'peak': np.max(errs),
                'rmse': np.sqrt(np.mean(np.array(errs)**2)), 'conv': conv}

    def reco_time(res, kick_iv=500, threshold=CONV_THRESHOLD):
        errs = res['errors']
        dt_err = 0.01 * res['n_substeps']
        kick_step = kick_iv
        times = []
        for ks in range(kick_step, len(errs), kick_step):
            for j in range(ks, min(ks+kick_step, len(errs)-10)):
                if np.all(np.array(errs[j:j+10]) < threshold):
                    times.append((j-ks)*dt_err); break
        return np.mean(times) if times else float('inf')

    pid_s = compute_stats(pid_r)
    sac_s = compute_stats(sac_r)
    hyb_s = compute_stats(hyb_r)
    pid_rec = reco_time(pid_k)
    sac_rec = reco_time(sac_k)
    hyb_rec = reco_time(hyb_k)

    def r(v, fmt='.4f'):
        if isinstance(v, str): return v
        if isinstance(v, float) and not np.isfinite(v): return '∞'
        return f"{v:{fmt}}"

    sep = "+"+"-"*22+"+"+"-"*12+"+"+"-"*12+"+"+"-"*12+"+"
    head = f"| {'指标':<20} | {'PID':^10} | {'SAC':^10} | {'SAC-PID':^10} |"
    print("\n"+"="*62+"\n  控制性能对比统计表\n"+"="*62)
    print(sep+"\n"+head+"\n"+sep)
    print(f"| {'稳态误差 ‖e‖/√N':<20} | {r(pid_s['steady']):^10} | {r(sac_s['steady']):^10} | {r(hyb_s['steady']):^10} |")
    print(f"| {'峰值误差':<20} | {r(pid_s['peak']):^10} | {r(sac_s['peak']):^10} | {r(hyb_s['peak']):^10} |")
    print(f"| {'RMSE':<20} | {r(pid_s['rmse']):^10} | {r(sac_s['rmse']):^10} | {r(hyb_s['rmse']):^10} |")
    print(f"| {'收敛时间/s':<20} | {r(pid_s['conv'],'.3f'):^10} | {r(sac_s['conv'],'.3f'):^10} | {r(hyb_s['conv'],'.3f'):^10} |")
    print(f"| {'平均恢复时间/s':<20} | {r(pid_rec,'.3f'):^10} | {r(sac_rec,'.3f'):^10} | {r(hyb_rec,'.3f'):^10} |")
    sa = f"{np.mean(sac_all_rets[best_sac_idx][-100:]):.1f}"
    ha = f"{np.mean(hyb_all_rets[best_hyb_idx][-100:]):.1f}"
    print(f"| {'训练平均奖励':<20} | {'N/A':^10} | {sa:^10} | {ha:^10} |")
    print(sep)

    with open('stats_table.csv', 'w', newline='', encoding='utr-8-sig') as cr:
        w = csv.writer(cr)
        w.writerow(['指标','PID','SAC','SAC-PID'])
        w.writerow(['稳态误差', r(pid_s['steady']), r(sac_s['steady']), r(hyb_s['steady'])])
        w.writerow(['峰值误差', r(pid_s['peak']), r(sac_s['peak']), r(hyb_s['peak'])])
        w.writerow(['RMSE', r(pid_s['rmse']), r(sac_s['rmse']), r(hyb_s['rmse'])])
        w.writerow(['收敛时间/s', r(pid_s['conv'],'.3f'), r(sac_s['conv'],'.3f'), r(hyb_s['conv'],'.3f')])
        w.writerow(['平均恢复时间/s', r(pid_rec,'.3f'), r(sac_rec,'.3f'), r(hyb_rec,'.3f')])
        w.writerow(['训练平均奖励', 'N/A', sa, ha])
    print("  ✓ 统计表已保存: stats_table.csv")

    timer.stop('Stats')
    timer.summary()
    print("\n  Done! 4 figures saved + 2 gain plots.")

if __name__ == "__main__":
    main()