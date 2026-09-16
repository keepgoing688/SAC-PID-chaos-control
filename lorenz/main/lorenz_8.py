"""
基于PID控制算法和深度强化学习的混沌控制
——SAC在线整定PID参数，控制Lorenz混沌系统
v6: 旧代码框架 + 模型保存/加载 + Figure5(LE) + Agg后端
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
warnings.filterwarnings('ignore')
import time
import os

# ---------- 全局五号字体 ----------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10.5           # 五号 ≈ 10.5pt
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 10.5
plt.rcParams['ytick.labelsize'] = 10.5
plt.rcParams['axes.titlesize'] = 10.5


# =====================================================
# GPU
# =====================================================
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device('cpu')
    print("  CPU mode")

# =====================================================
# Timer
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
        return f"{int(m)}m{s2:.0f}s"

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
# Lorenz
# =====================================================
class LorenzEnv:
    def __init__(self, sigma=10., rho=28., beta=8./3.,
                 dt=0.01, max_steps=800):
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.dt, self.max_steps = dt, max_steps
        self.fixed_point = np.array([
            np.sqrt(beta*(rho-1)),
            np.sqrt(beta*(rho-1)),
            rho - 1
        ])
        self.state = None
        self.step_count = 0

    def reset(self, init=None):
        if init is not None:
            self.state = np.array(init, dtype=np.float64)
        else:
            self.state = np.array([
                np.random.uniform(-20, 20),
                np.random.uniform(-25, 25),
                np.random.uniform(5, 45)
            ])
        self.step_count = 0
        return self.state.copy()

    def _rhs(self, s, u):
        x, y, z = s
        return np.array([
            self.sigma*(y - x) + u[0],
            x*(self.rho - z) - y + u[1],
            x*y - self.beta*z + u[2]
        ])

    def step(self, u):
        u = np.clip(u, -50, 50)
        k1 = self._rhs(self.state, u)
        k2 = self._rhs(self.state + .5*self.dt*k1, u)
        k3 = self._rhs(self.state + .5*self.dt*k2, u)
        k4 = self._rhs(self.state + self.dt*k3, u)
        self.state += (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        self.step_count += 1
        en = np.linalg.norm(self.state - self.fixed_point)
        done = (self.step_count >= self.max_steps or
                np.any(np.abs(self.state) > 200))
        return self.state.copy(), en, done

    def get_error(self):
        return self.state - self.fixed_point

# =====================================================
# PID
# =====================================================
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=0.01, lim=50.):
        self.Kp = np.array(Kp, dtype=np.float64)
        self.Ki = np.array(Ki, dtype=np.float64)
        self.Kd = np.array(Kd, dtype=np.float64)
        self.dt, self.lim = dt, lim
        self.reset()

    def reset(self):
        self.integ  = np.zeros(3)
        self.prev_e = np.zeros(3)
        self.inited = False

    def compute(self, e, Kp=None, Ki=None, Kd=None):
        Kp = self.Kp if Kp is None else Kp
        Ki = self.Ki if Ki is None else Ki
        Kd = self.Kd if Kd is None else Kd
        self.integ = np.clip(self.integ + e*self.dt, -50, 50)
        d = (np.zeros(3) if not self.inited
             else (e - self.prev_e)/self.dt)
        self.inited = True
        self.prev_e = e.copy()
        return np.clip(-(Kp*e + Ki*self.integ + Kd*d),
                       -self.lim, self.lim)

# =====================================================
# Reward (原始版本，保持旧代码效果)
# =====================================================
def reward_fn(error, error_next, control,
              raw_action, prev_raw, mode='sac'):
    en = np.linalg.norm(error)
    un = np.linalg.norm(control)
    r  = -(en**2)/(en**2 + 1.0) \
         - 0.1*(un**2)/(un**2 + 100.0)
    if mode == 'sacpid' and prev_raw is not None:
        r -= 0.01 * np.linalg.norm(raw_action - prev_raw)**2
    if np.linalg.norm(error_next) > 100:
        r -= 10.0
    return r

# =====================================================
# ReplayBuffer
# =====================================================
class ReplayBuffer:
    def __init__(self, cap=100000):
        self.buf = deque(maxlen=cap)

    def push(self, *a):
        self.buf.append(a)

    def sample(self, bs):
        b = random.sample(self.buf, bs)
        return tuple(np.array(x) for x in zip(*b))

    def __len__(self):
        return len(self.buf)

# =====================================================
# Networks（原始h=128）
# =====================================================
class QNet(nn.Module):
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sd+ad, h), nn.SiLU(),
            nn.Linear(h, h),     nn.SiLU(),
            nn.Linear(h, 1)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1))


class Policy(nn.Module):
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.fc1 = nn.Linear(sd, h)
        self.fc2 = nn.Linear(h, h)
        self.mu  = nn.Linear(h, ad)
        self.ls  = nn.Linear(h, ad)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        return self.mu(x), torch.clamp(self.ls(x), -20, 2)

    def sample(self, s):
        m, ls = self.forward(s)
        std   = ls.exp()
        n     = Normal(m, std)
        xt    = n.rsample()
        a     = torch.tanh(xt)
        lp    = (n.log_prob(xt)
                 - torch.log(1 - a.pow(2) + 1e-6))
        lp    = lp.sum(-1, keepdim=True)
        return a, lp, m

    def act(self, s, det=False):
        st = torch.FloatTensor(s).unsqueeze(0)
        with torch.no_grad():
            if det:
                m, _ = self.forward(st)
                return torch.tanh(m).squeeze(0).numpy()
            a, _, _ = self.sample(st)
            return a.squeeze(0).numpy()

# =====================================================
# SAC（含save/load）
# =====================================================
class SAC:
    def __init__(self, sd, ad, lr=6e-4, gamma=0.975,
                 tau=0.006, h=192, bs=256,
                 buf=100000, ui=2):
        self.gamma, self.tau = gamma, tau
        self.bs, self.ui     = bs, ui
        self.uc = 0

        self.pi  = Policy(sd, ad, h).to(DEVICE)
        self.q1  = QNet(sd, ad, h).to(DEVICE)
        self.q2  = QNet(sd, ad, h).to(DEVICE)
        self.q1t = QNet(sd, ad, h).to(DEVICE)
        self.q2t = QNet(sd, ad, h).to(DEVICE)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())

        self.pi_cpu = Policy(sd, ad, h)
        self.pi_cpu.load_state_dict(self.pi.state_dict())

        self.pi_opt = optim.Adam(self.pi.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

        self.te    = -ad
        self.la    = torch.zeros(1, requires_grad=True,
                                 device=DEVICE)
        self.a_opt = optim.Adam([self.la], lr=lr)
        self.alpha = self.la.exp().item()

        self.rb = ReplayBuffer(buf)
        self.sc = 0

    def act(self, s, det=False):
        return self.pi_cpu.act(s, det)

    def _sync(self):
        self.pi_cpu.load_state_dict(
            {k: v.cpu()
             for k, v in self.pi.state_dict().items()})

    def save(self, path):
        torch.save({
            'pi':    self.pi.state_dict(),
            'q1':    self.q1.state_dict(),
            'q2':    self.q2.state_dict(),
            'q1t':   self.q1t.state_dict(),
            'q2t':   self.q2t.state_dict(),
            'la':    self.la.data,
            'alpha': self.alpha,
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=DEVICE)
        self.pi.load_state_dict(ck['pi'])
        self.q1.load_state_dict(ck['q1'])
        self.q2.load_state_dict(ck['q2'])
        self.q1t.load_state_dict(ck['q1t'])
        self.q2t.load_state_dict(ck['q2t'])
        self.la.data  = ck['la']
        self.alpha    = ck['alpha']
        self._sync()

    def update(self):
        self.uc += 1
        if self.uc % self.ui != 0 or len(self.rb) < self.bs:
            return None

        s, a, r, ns, d = self.rb.sample(self.bs)
        st  = torch.FloatTensor(s).to(DEVICE)
        at  = torch.FloatTensor(a).to(DEVICE)
        rt  = torch.FloatTensor(
                r.astype(np.float32)).unsqueeze(1).to(DEVICE)
        nst = torch.FloatTensor(ns).to(DEVICE)
        dt_ = torch.FloatTensor(
                d.astype(np.float32)).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            na, nlp, _ = self.pi.sample(nst)
            qn  = torch.min(self.q1t(nst, na),
                            self.q2t(nst, na)) - self.alpha*nlp
            qt  = rt + (1-dt_)*self.gamma*qn

        ql1 = F.mse_loss(self.q1(st, at), qt)
        ql2 = F.mse_loss(self.q2(st, at), qt)
        self.q1_opt.zero_grad(); ql1.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad(); ql2.backward()
        self.q2_opt.step()

        na2, lp2, _ = self.pi.sample(st)
        qnew = torch.min(self.q1(st, na2), self.q2(st, na2))
        pl   = (self.alpha*lp2 - qnew).mean()
        self.pi_opt.zero_grad(); pl.backward()
        self.pi_opt.step()

        al = -(self.la*(lp2 + self.te).detach()).mean()
        self.a_opt.zero_grad(); al.backward()
        self.a_opt.step()
        self.alpha = self.la.exp().item()

        for p, tp in zip(self.q1.parameters(),
                         self.q1t.parameters()):
            tp.data.copy_(self.tau*p.data
                          + (1-self.tau)*tp.data)
        for p, tp in zip(self.q2.parameters(),
                         self.q2t.parameters()):
            tp.data.copy_(self.tau*p.data
                          + (1-self.tau)*tp.data)

        self.sc += 1
        if self.sc % 10 == 0:
            self._sync()
        return {'ql': (ql1.item()+ql2.item())/2,
                'pl': pl.item()}

# =====================================================
# SAC-PID Tuner（原始版本）
# =====================================================
class SACPIDTuner:
    def __init__(self, dt=0.01):
        self.dt  = dt
        self.pid = PIDController([5]*3, [.3]*3, [1]*3, dt)
        self.sac = SAC(sd=12, ad=9, lr=6e-4,
                       h=192, bs=256, ui=2)
        self.prev_raw = np.zeros(9)
        self.prev_u   = np.zeros(3)
        self.prev_e   = np.zeros(3)
        self.deriv    = np.zeros(3)

    def reset(self):
        self.pid.reset()
        self.prev_raw = np.zeros(9)
        self.prev_u   = np.zeros(3)
        self.prev_e   = np.zeros(3)
        self.deriv    = np.zeros(3)

    def _obs(self, e):
        integ = self.pid.integ.copy()
        self.deriv = ((e - self.prev_e)/self.dt
                      if self.pid.inited else np.zeros(3))
        s = np.concatenate([
            e, integ, self.deriv,
            [np.linalg.norm(e),
             np.linalg.norm(self.prev_u),
             np.linalg.norm(self.deriv)]
        ])
        return np.clip(s/50, -1, 1)

    def _decode(self, raw):
        n = (raw + 1) / 2
        return n[0:3]*15, n[3:6]*2, n[6:9]*5

    def save(self, path):
        self.sac.save(path)

    def load(self, path):
        self.sac.load(path)

    def compute(self, e, det=False):
        obs        = self._obs(e)
        raw        = self.sac.act(obs, det)
        Kp, Ki, Kd = self._decode(raw)
        u          = self.pid.compute(e, Kp, Ki, Kd)
        self.prev_raw = raw.copy()
        self.prev_u   = u.copy()
        self.prev_e   = e.copy()
        return u, obs, raw, Kp, Ki, Kd

# =====================================================
# Train SAC（含保存/加载）
# =====================================================
def train_sac(seed=0, nep=2000, ms=800,
              save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir,
                              f'sac_seed{seed}.pth')
    rets_path  = os.path.join(save_dir,
                              f'sac_seed{seed}_rets.npy')

    if os.path.exists(model_path) and \
       os.path.exists(rets_path):
        print(f"    SAC s{seed}: 加载 {model_path}")
        agent = SAC(sd=3, ad=3, lr=6e-4,
                    h=192, bs=256, ui=2)
        agent.load(model_path)
        rets = np.load(rets_path).tolist()
        return agent, rets

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env   = LorenzEnv(max_steps=ms)
    agent = SAC(sd=3, ad=3, lr=6e-4,
                h=192, bs=256, ui=2)
    rets  = []
    t0    = time.time()

    for ep in range(nep):
        s  = env.reset()
        er = 0
        for _ in range(ms):
            ra = agent.act(s)
            a  = ra * 15.0
            ns, en, done = env.step(a)
            e_prev = s - env.fixed_point
            e_now  = env.get_error()
            rw = reward_fn(e_prev, e_now, a,
                           ra, None, 'sac')
            if np.any(np.abs(ns) > 200):
                rw -= 10
            agent.rb.push(s, ra, rw, ns, float(done))
            agent.update()
            s   = ns
            er += rw
            if done: break
        rets.append(er)
        if (ep+1) % 200 == 0:
            avg = np.mean(rets[-100:])
            print(f"    SAC s{seed} ep{ep+1}/{nep} "
                  f"R={avg:.1f} {time.time()-t0:.0f}s")

    agent._sync()
    agent.save(model_path)
    np.save(rets_path, np.array(rets))
    print(f"    SAC s{seed}: 已保存 {model_path}")
    return agent, rets

# =====================================================
# Train SAC-PID（含保存/加载）
# =====================================================
def train_sacpid(seed=0, nep=1500, ms=800,
                 save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir,
                              f'sacpid_seed{seed}.pth')
    rets_path  = os.path.join(save_dir,
                              f'sacpid_seed{seed}_rets.npy')

    if os.path.exists(model_path) and \
       os.path.exists(rets_path):
        print(f"    SAC-PID s{seed}: 加载 {model_path}")
        tuner = SACPIDTuner()
        tuner.load(model_path)
        rets = np.load(rets_path).tolist()
        return tuner, rets

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env   = LorenzEnv(max_steps=ms)
    tuner = SACPIDTuner()
    rets  = []
    t0    = time.time()

    for ep in range(nep):
        s = env.reset()
        tuner.reset()
        er = 0
        for _ in range(ms):
            e = env.get_error()
            u, obs, raw, Kp, Ki, Kd = tuner.compute(e)
            ns, en, done = env.step(u)
            e2  = env.get_error()
            rw  = reward_fn(e, e2, u, raw,
                            tuner.prev_raw, 'sacpid')
            nobs = tuner._obs(e2)
            tuner.sac.rb.push(obs, raw, rw,
                              nobs, float(done))
            tuner.sac.update()
            s   = ns
            er += rw
            if done: break
        rets.append(er)
        if (ep+1) % 200 == 0:
            avg = np.mean(rets[-100:])
            print(f"    SAC-PID s{seed} ep{ep+1}/{nep} "
                  f"R={avg:.1f} {time.time()-t0:.0f}s")

    tuner.sac._sync()
    tuner.save(model_path)
    np.save(rets_path, np.array(rets))
    print(f"    SAC-PID s{seed}: 已保存 {model_path}")
    return tuner, rets

# =====================================================
# Evaluation（原始版本）
# =====================================================
def evaluate(ctype, ctrl, init, dt=0.01,
             steps=3000, kick_iv=0, kseed=42):
    env = LorenzEnv(dt=dt, max_steps=steps+100)
    s   = env.reset(init)
    np.random.seed(kseed)
    if ctype in ('pid', 'sacpid'):
        ctrl.reset()
    states, errs = [s.copy()], []

    for step in range(steps):
        e = env.get_error()
        errs.append(np.linalg.norm(e))
        if ctype == 'pid':
            a = ctrl.compute(e)
        elif ctype == 'sac':
            a = ctrl.act(s, det=True) * 15.0
        elif ctype == 'sacpid':
            a, _, _, _, _, _ = ctrl.compute(e, det=True)
        if kick_iv > 0 and step > 0 and step % kick_iv == 0:
            env.state += np.array([
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10),
                np.random.uniform(-15, 15)
            ])
            if ctype in ('pid', 'sacpid'):
                ctrl.reset()
        ns, _, done = env.step(a)
        s = ns
        states.append(s.copy())
        if done: break

    return {'states': np.array(states),
            'errors': np.array(errs)}

# =====================================================
# Colors
# =====================================================
C = {'PID': '#ff0000',
     'SAC': '#0000ff',
     'SAC-PID': '#00ff00'}

# =====================================================
# Figure 1
# =====================================================
def plot_fig1(pid_r, sac_r, hyb_r, fp, dt=0.01):
    zoom = int(2.0 / dt)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7),
                              sharex=True)
    plt.subplots_adjust(hspace=0.1)
    var_labels  = ['$x(t)$', '$y(t)$', '$z(t)$']
    ylims       = [(0, 20), (0, 20), (10, 40)]
    yticks_list = [[0, 10, 20], [0, 10, 20], [10, 40]]

    for i, ax in enumerate(axes):
        n_p = min(zoom+1, len(pid_r['states']))
        n_s = min(zoom+1, len(sac_r['states']))
        n_h = min(zoom+1, len(hyb_r['states']))
        t   = np.arange(max(n_p, n_s, n_h)) * dt
        ax.plot(t[:n_p], pid_r['states'][:n_p, i],
                color=C['PID'],     lw=1.5, alpha=0.7,
                label='PID')
        ax.plot(t[:n_s], sac_r['states'][:n_s, i],
                color=C['SAC'],     lw=1.5, alpha=0.7,
                label='SAC')
        ax.plot(t[:n_h], hyb_r['states'][:n_h, i],
                color=C['SAC-PID'], lw=2.5, alpha=0.95,
                label='SAC-PID')
        ax.axhline(y=fp[i], color='black',
                   ls='--', lw=0.8, alpha=0.5)
        ax.set_ylabel(var_labels[i],fontsize=14)
        ax.set_ylim(ylims[i])
        ax.set_yticks(yticks_list[i])

    axes[0].legend(loc='upper right',fontsize=10, ncol=3)
    axes[2].set_xlabel('Time (s)',fontsize=14)
    axes[2].set_xlim(0, 2)
    axes[2].set_xticks([0, 1, 2])
    plt.savefig('figure1.png', dpi=300, bbox_inches='tight')
    print("  ✓ figure1")
    plt.close(fig)

# =====================================================
# Figure 2
# =====================================================
def plot_fig2(sac_all_rets, hyb_all_rets):
    window = 50

    def rolling_stats(all_rets):
        min_len = min(len(r) for r in all_rets)
        rolled  = []
        for r in all_rets:
            series = np.array(r[:min_len])
            rm = [np.mean(series[max(0, i-window):i+1])
                  for i in range(len(series))]
            rolled.append(np.array(rm))
        rolled = np.array(rolled)
        return (np.mean(rolled, axis=0),
                np.std(rolled, axis=0)/np.sqrt(len(all_rets)))

    fig, ax = plt.subplots(figsize=(8, 6))

    mean_h, se_h = rolling_stats(hyb_all_rets)
    eps_h = np.arange(len(mean_h))
    ax.plot(eps_h, mean_h, color=C['SAC-PID'],
            lw=2.5, label='SAC-PID')
    ax.fill_between(eps_h, mean_h-se_h, mean_h+se_h,
                    color=C['SAC-PID'], alpha=0.2)

    mean_s, se_s = rolling_stats(sac_all_rets)
    eps_s = np.arange(len(mean_s))
    ax.plot(eps_s, mean_s, color=C['SAC'],
            lw=2.0, label='Pure SAC')
    ax.fill_between(eps_s, mean_s-se_s, mean_s+se_s,
                    color=C['SAC'], alpha=0.2)

    ax.set_xlabel('Episodes', fontsize=14)
    ax.set_ylabel('Average Return', fontsize=14)
    ax.set_ylim(-900, 50)
    ax.set_yticks([-800, -400, 0])
    ax.set_xticks([0, 500, 1000])
    ax.legend(fontsize=12, loc='lower right')


    plt.tight_layout()
    plt.savefig('figure2.png', dpi=300, bbox_inches='tight')
    print("  ✓ figure2")
    plt.close(fig)

# =====================================================
# Figure 3
# =====================================================
def plot_fig3(pid_r, sac_r, hyb_r, dt=0.01):
    zoom = int(5.0 / dt)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res, color, lw in [
            ('PID',     pid_r, C['PID'],     1.5),
            ('SAC',     sac_r, C['SAC'],     1.5),
            ('SAC-PID', hyb_r, C['SAC-PID'], 2.5)]:
        n = min(zoom, len(res['errors']))
        t = np.arange(n) * dt
        ax.semilogy(t, res['errors'][:n],
                    color=color, lw=lw, label=name)
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('$\\|e(t)\\|$', fontsize=14)
    ax.set_xlim(0, 5)
    ax.set_xticks([0, 5])
    ax.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.savefig('figure3.png', dpi=300, bbox_inches='tight')
    print("  ✓ figure3")
    plt.close(fig)

# =====================================================
# Figure 4
# =====================================================
def plot_fig4(pid_k, sac_k, hyb_k, fp,
              dt=0.01, kick_iv=500):
    zoom = int(20.0 / dt)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8),
                              sharex=True)
    plt.subplots_adjust(hspace=0.1)
    var_labels  = ['$x(t)$', '$y(t)$', '$z(t)$']
    ylims       = [(0, 20), (0, 20), (10, 40)]
    yticks_list = [[0, 10, 20], [0, 10, 20], [10, 40]]

    for i, ax in enumerate(axes):
        n_p = min(zoom+1, len(pid_k['states']))
        n_s = min(zoom+1, len(sac_k['states']))
        n_h = min(zoom+1, len(hyb_k['states']))
        t_p = np.arange(n_p) * dt
        t_s = np.arange(n_s) * dt
        t_h = np.arange(n_h) * dt
        ax.plot(t_p, pid_k['states'][:n_p, i],
                color=C['PID'],     lw=1.2, alpha=0.6,
                label='PID')
        ax.plot(t_s, sac_k['states'][:n_s, i],
                color=C['SAC'],     lw=1.2, alpha=0.6,
                label='SAC')
        ax.plot(t_h, hyb_k['states'][:n_h, i],
                color=C['SAC-PID'], lw=2.0, alpha=0.9,
                label='SAC-PID')
        ax.axhline(y=fp[i], color='black',
                   ls='--', lw=0.8, alpha=0.5)
        for k in range(kick_iv, zoom, kick_iv):
            ax.axvline(x=k*dt, color='gray',
                       ls=':', alpha=0.3, lw=0.6)
        ax.set_ylabel(var_labels[i], fontsize=14)
        ax.set_ylim(ylims[i])
        ax.set_yticks(yticks_list[i])

    axes[0].legend(loc='upper right', fontsize=10, ncol=3)
    axes[2].set_xlabel('Time (s)', fontsize=14)
    axes[2].set_xlim(0, 15)
    axes[2].set_xticks([0, 5, 10, 15])
    plt.savefig('figure4.png', dpi=300, bbox_inches='tight')
    print("  ✓ figure4")
    plt.close(fig)

# =====================================================
# 李雅普诺夫指数
# =====================================================
def compute_lyapunov_spectrum(sigma=10., rho=28.,
                               beta=8./3., dt=0.01,
                               n_steps=50000,
                               n_renorm=10):
    def jac(s):
        x, y, z = s
        return np.array([[-sigma, sigma,  0.  ],
                          [rho-z,  -1.,   -x  ],
                          [y,       x,  -beta ]])

    def rhs(s, Q):
        x, y, z = s
        ds = np.array([sigma*(y-x),
                        x*(rho-z)-y,
                        x*y-beta*z])
        return ds, jac(s) @ Q

    def rk4(s, Q, h):
        d1, dQ1 = rhs(s, Q)
        d2, dQ2 = rhs(s+.5*h*d1, Q+.5*h*dQ1)
        d3, dQ3 = rhs(s+.5*h*d2, Q+.5*h*dQ2)
        d4, dQ4 = rhs(s+   h*d3, Q+   h*dQ3)
        return (s+(h/6)*(d1+2*d2+2*d3+d4),
                Q+(h/6)*(dQ1+2*dQ2+2*dQ3+dQ4))

    np.random.seed(0)
    state   = np.array([1., 1., 20.])
    Q       = np.eye(3)
    log_sum = np.zeros(3)
    t_el    = 0.
    hist    = []

    for _ in range(5000):
        state, Q = rk4(state, Q, dt)
    Q, _ = np.linalg.qr(Q)

    for _ in range(n_steps):
        for _ in range(n_renorm):
            state, Q = rk4(state, Q, dt)
        t_el += n_renorm * dt
        Q, R  = np.linalg.qr(Q)
        log_sum += np.log(np.abs(np.diag(R)))
        hist.append((log_sum/t_el).copy())

    return log_sum/t_el, np.array(hist)

# =====================================================
# Figure 5（KY维数公式修正）
# =====================================================
def plot_chaos_analysis(les, les_history, n_renorm=10, dt=0.01):
    # ---------- 吸引子轨迹计算（共用） ----------
    sigma, rho, beta = 10., 28., 8./3.
    env_f = LorenzEnv(sigma=sigma, rho=rho, beta=beta,
                      dt=0.005, max_steps=30000)
    env_f.reset([0.1, 0., 0.])
    traj = [env_f.state.copy()]
    for _ in range(30000):
        x, y, z = env_f.state
        env_f.state += 0.005*np.array([
            sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
        traj.append(env_f.state.copy())
    traj = np.array(traj[5000:])

    # KY维数
    ky_dim = 2 + (les[0]+les[1]) / abs(les[2])

    # ========== 图 A：3D 相图 ==========
    fig1 = plt.figure(figsize=(7, 6))          # 尺寸可自行调整
    ax1 = fig1.add_subplot(111, projection='3d')
    n_pts = len(traj)
    seg   = 200
    for k in range(0, n_pts-seg, seg):
        c = plt.cm.RdYlBu_r(k/n_pts)
        ax1.plot(traj[k:k+seg+1, 0],
                 traj[k:k+seg+1, 1],
                 traj[k:k+seg+1, 2],
                 color=c, lw=0.4, alpha=0.8)
    fp = np.array([np.sqrt(beta*(rho-1)),
                   np.sqrt(beta*(rho-1)), rho-1])
    ax1.scatter(*fp, color='red', s=30, zorder=5,
                label=f'Fixed pt\n'
                      f'({fp[0]:.1f},{fp[1]:.1f},{fp[2]:.1f})')
    ax1.set_xlabel('$x$', fontsize=13)
    ax1.set_ylabel('$y$', fontsize=13)
    ax1.set_zlabel('$z$', fontsize=13)
    ax1.set_title('Lorenz Strange Attractor\n'
                  r'($\sigma$=10, $\rho$=28, $\beta$=8/3)',
                  fontsize=13)
    ax1.legend(fontsize=9)
    ax1.view_init(elev=20, azim=-60)
    ax1.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig('figure5a.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ========== 图 B：LE 收敛曲线 ==========
    fig2, ax2 = plt.subplots(figsize=(7, 5))    # 尺寸可自行调整
    t_ax  = np.arange(1, len(les_history)+1)*n_renorm*dt
    cols  = ['#d62728', '#2ca02c', '#1f77b4']
    lbls  = [f'$\\lambda_1$ = {les[0]:+.4f}',
             f'$\\lambda_2$ = {les[1]:+.4f}',
             f'$\\lambda_3$ = {les[2]:+.4f}']
    for i in range(3):
        ax2.plot(t_ax, les_history[:, i],
                 color=cols[i], lw=1.8, label=lbls[i])
        ax2.annotate(f'{les[i]:+.3f}',
                     xy=(t_ax[-1], les[i]),
                     xytext=(t_ax[-1]*1.01, les[i]),
                     fontsize=10, color=cols[i],
                     va='center')
    ax2.axhline(0, color='black', ls='--', lw=0.8, alpha=0.6)
    ax2.set_xlabel('Time $t$', fontsize=13)
    ax2.set_ylabel('Lyapunov Exponent', fontsize=13)
    ax2.set_title(f'Lyapunov Exponent Spectrum\n'
                  f'(Kaplan-Yorke dim = {ky_dim:.3f})',
                  fontsize=13)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.set_xlim(0, t_ax[-1])
    ax2.grid(True, alpha=0.25)

    txt = (f'$\\lambda_1>0$: chaotic\n'
           f'$\\lambda_1+\\lambda_2<0$: dissipative\n'
           f'$\\sum\\lambda_i={les.sum():.3f}$')
    ax2.text(0.03, 0.30, txt,
             transform=ax2.transAxes,
             fontsize=9, va='top',
             bbox=dict(boxstyle='round',
                       facecolor='wheat', alpha=0.4))
    plt.tight_layout()
    plt.savefig('figure5b.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # 保留控制台打印的信息
    print("\n  Lyapunov Exponents:")
    print(f"    λ1 = {les[0]:+.4f}  (>0 → chaos)")
    print(f"    λ2 = {les[1]:+.4f}  (≈0 → flow)")
    print(f"    λ3 = {les[2]:+.4f}  (<0 → contraction)")
    print(f"    Sum= {les.sum():.4f}  "
          f"(theory: {-sigma-1-beta:.4f})")
    print(f"    Kaplan-Yorke dim = {ky_dim:.4f}  "
          f"(theory ≈ 2.062)")
    print("  ✓ figure5a, figure5b")

# =====================================================
# Main
# =====================================================
def main():
    timer.start_total()
    print("="*60)
    print("  Lorenz Chaos Control  v6")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    NS   = 5
    EP_S = 1000
    EP_H = 1000
    MS   = 800
    CKPT = 'checkpoints'

    # 1. Train SAC
    print(f"\n[1] Train SAC ({EP_S}ep × {NS}seeds)")
    timer.start('Train-SAC')
    sac_agents, sac_all_rets = [], []
    for seed in range(NS):
        print(f"  Seed {seed}:")
        ag, r = train_sac(seed, EP_S, MS, CKPT)
        sac_agents.append(ag)
        sac_all_rets.append(r)
    timer.stop('Train-SAC')

    # 2. Train SAC-PID
    print(f"\n[2] Train SAC-PID ({EP_H}ep × {NS}seeds)")
    timer.start('Train-SACPID')
    hyb_tuners, hyb_all_rets = [], []
    for seed in range(NS):
        print(f"  Seed {seed}:")
        tu, r = train_sacpid(seed, EP_H, MS, CKPT)
        hyb_tuners.append(tu)
        hyb_all_rets.append(r)
    timer.stop('Train-SACPID')

    best_sac = sac_agents[
        np.argmax([np.mean(r[-50:]) for r in sac_all_rets])]
    best_hyb = hyb_tuners[
        np.argmax([np.mean(r[-50:]) for r in hyb_all_rets])]

    # 3. Evaluate
    print("\n[3] Evaluate")
    timer.start('Eval')
    fp   = LorenzEnv().fixed_point
    init = [2.0, 2.0, 10.0]

    pid  = PIDController([5]*3, [.3]*3, [1]*3, 0.01)
    pid_r = evaluate('pid', pid, init,
                     steps=2000, kick_iv=0)
    sac_r = evaluate('sac', best_sac, init,
                     steps=2000, kick_iv=0)
    hyb_r = evaluate('sacpid', best_hyb, init,
                     steps=2000, kick_iv=0)

    pid2  = PIDController([5]*3, [.3]*3, [1]*3, 0.01)
    pid_k = evaluate('pid', pid2, init,
                     steps=2500, kick_iv=500, kseed=42)
    sac_k = evaluate('sac', best_sac, init,
                     steps=2500, kick_iv=500, kseed=42)
    hyb_k = evaluate('sacpid', best_hyb, init,
                     steps=2500, kick_iv=500, kseed=42)


    timer.stop('Eval')

    # 4. Plot
    print("\n[4] Plot")
    timer.start('Plot')
    plot_fig1(pid_r, sac_r, hyb_r, fp)
    plot_fig2(sac_all_rets, hyb_all_rets)
    plot_fig3(pid_r, sac_r, hyb_r)
    plot_fig4(pid_k, sac_k, hyb_k, fp, kick_iv=500)

    print("  计算李雅普诺夫指数谱...")
    timer.start('Lyapunov')
    les, les_hist = compute_lyapunov_spectrum(
        dt=0.01, n_steps=50000, n_renorm=10)
    timer.stop('Lyapunov')
    plot_chaos_analysis(les, les_hist)
    timer.stop('Plot')

    # =====================================================
    # 统计表格
    # =====================================================
    def compute_stats(result, dt=0.01, threshold=1.0,
                      steady_n=100):
        """
        计算单个控制器的性能指标
        result: evaluate()返回的字典
        返回: dict of metrics
        """
        errs = result['errors']

        # 1. 稳态误差（最后steady_n步均值）
        steady_err = np.mean(errs[-steady_n:]) \
            if len(errs) >= steady_n \
            else np.mean(errs)

        # 2. 峰值误差（最大值）
        peak_err = np.max(errs)

        # 3. 收敛时间（首次误差 < threshold 且之后保持）
        conv_time = None
        for i in range(len(errs) - 10):
            # 要求之后连续10步都低于阈值
            if np.all(errs[i:i + 10] < threshold):
                conv_time = i * dt
                break
        if conv_time is None:
            conv_time = float('inf')

        # 4. 均方误差 (RMSE)
        rmse = np.sqrt(np.mean(errs ** 2))

        return {
            'steady_err': steady_err,
            'peak_err': peak_err,
            'conv_time': conv_time,
            'rmse': rmse,
        }

    def compute_kick_stats(result, dt=0.01,
                           kick_iv=500, threshold=1.0):
        """
        计算扰动实验的恢复时间（每次kick后）
        """
        errs = result['errors']
        recovery_times = []
        kick_steps = list(range(kick_iv, len(errs), kick_iv))

        for ks in kick_steps:
            # 从kick点开始，找恢复时间
            rec = None
            for j in range(ks, min(ks + kick_iv, len(errs) - 10)):
                if np.all(errs[j:j + 10] < threshold):
                    rec = (j - ks) * dt
                    break
            if rec is not None:
                recovery_times.append(rec)

        return np.mean(recovery_times) \
            if recovery_times else float('inf')

    def print_stats_table(pid_r, sac_r, hyb_r,
                          pid_k, sac_k, hyb_k,
                          sac_all_rets, hyb_all_rets,
                          dt=0.01):
        """
        打印并保存统计对比表格
        """
        # ── 基础评估指标 ──
        pid_s = compute_stats(pid_r, dt)
        sac_s = compute_stats(sac_r, dt)
        hyb_s = compute_stats(hyb_r, dt)

        # ── 抗扰指标 ──
        pid_rec = compute_kick_stats(pid_k, dt)
        sac_rec = compute_kick_stats(sac_k, dt)
        hyb_rec = compute_kick_stats(hyb_k, dt)

        # ── 训练奖励 ──
        pid_rw = 'N/A'  # PID无训练
        sac_rw = f"{np.mean([np.mean(r[-50:]) for r in sac_all_rets]):.1f}"
        hyb_rw = f"{np.mean([np.mean(r[-50:]) for r in hyb_all_rets]):.1f}"

        # ── 格式化输出 ──
        sep = "+" + "-" * 22 + "+" + "-" * 12 + \
              "+" + "-" * 12 + "+" + "-" * 12 + "+"
        head = f"| {'指标':<20} | {'PID':^10} | {'SAC':^10} | {'SAC-PID':^10} |"

        def row(label, p, s, h, fmt='.4f'):
            def f(v):
                if isinstance(v, float):
                    return f"{v:{fmt}}" if v != float('inf') \
                        else '∞'
                return str(v)

            return (f"| {label:<20} | {f(p):^10} "
                    f"| {f(s):^10} | {f(h):^10} |")

        print("\n" + "=" * 62)
        print("  控制性能对比统计表")
        print("=" * 62)
        print(sep)
        print(head)
        print(sep)
        print(row("稳态误差 ‖e‖",
                  pid_s['steady_err'],
                  sac_s['steady_err'],
                  hyb_s['steady_err']))
        print(row("峰值误差",
                  pid_s['peak_err'],
                  sac_s['peak_err'],
                  hyb_s['peak_err']))
        print(row("RMSE",
                  pid_s['rmse'],
                  sac_s['rmse'],
                  hyb_s['rmse']))
        print(row("收敛时间/s",
                  pid_s['conv_time'],
                  sac_s['conv_time'],
                  hyb_s['conv_time'], fmt='.3f'))
        print(row("平均恢复时间/s",
                  pid_rec, sac_rec, hyb_rec, fmt='.3f'))
        print(row("训练平均奖励",
                  pid_rw, sac_rw, hyb_rw))
        print(sep)
        print()

        # ── 同时保存为CSV ──
        import csv
        csv_path = 'stats_table.csv'
        with open(csv_path, 'w', newline='',
                  encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['指标', 'PID', 'SAC', 'SAC-PID'])

            def fmt_v(v):
                if isinstance(v, float):
                    return f"{v:.4f}" \
                        if v != float('inf') else 'inf'
                return str(v)

            rows = [
                ('稳态误差 ‖e‖',
                 pid_s['steady_err'],
                 sac_s['steady_err'],
                 hyb_s['steady_err']),
                ('峰值误差',
                 pid_s['peak_err'],
                 sac_s['peak_err'],
                 hyb_s['peak_err']),
                ('RMSE',
                 pid_s['rmse'],
                 sac_s['rmse'],
                 hyb_s['rmse']),
                ('收敛时间/s',
                 pid_s['conv_time'],
                 sac_s['conv_time'],
                 hyb_s['conv_time']),
                ('平均恢复时间/s',
                 pid_rec, sac_rec, hyb_rec),
                ('训练平均奖励',
                 pid_rw, sac_rw, hyb_rw),
            ]
            for r_data in rows:
                w.writerow([r_data[0]] +
                           [fmt_v(v) for v in r_data[1:]])
        print(f"  ✓ 统计表已保存: {csv_path}")

    # 5. 统计表格
    print("\n[5] 统计对比表格")
    timer.start('Stats')
    print_stats_table(
            pid_r, sac_r, hyb_r,
            pid_k, sac_k, hyb_k,
            sac_all_rets, hyb_all_rets,
            dt=0.01
        )
    timer.stop('Stats')

    timer.summary()
    print("\n  Done! 5 figures saved.")

if __name__ == "__main__":
    main()