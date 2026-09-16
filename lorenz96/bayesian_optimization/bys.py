"""
Optuna优化SAC和SAC-PID共享超参数 (Lorenz96版本)
目标函数 = 0.5*R_SAC + 0.5*R_SACPID
带剪枝功能
"""

import json
import time
import gc
import numpy as np
import torch
import random
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lorenz96.main.lorenz96_env import Lorenz96Env, PIDController, compute_base_reward
from lorenz96.main.lorenz96_main import (
    SAC, SACPIDTuner, SparseActuatorDecodef,
    build_sac_obs, build_sacpid_obs,
    N_WIN_SAC, N_WIN_PID,
    INIT_SCALE_GLOBAL, CONV_THRESHOLD,
    PID_BASE_KP, PID_BASE_KI, PID_BASE_KD,
)

# =====================================================
# 缩减规模
# =====================================================
NEP_SEARCH = 500
MS_SEARCH = 400
REPORT_INTERVAL = 100
EVAL_STEPS = 500
N_DIM = 40
N_ACT = 8
N_SUBSTEPS = 2
OBS_DIM_SAC = 8 + N_WIN_SAC + N_ACT
OBS_DIM_SACPID = 15

np.random.seed(42)
EVAL_INITS = []
for _ in range(3):
    EVAL_INITS.append(
        np.ones(N_DIM) * 8.0
        + np.random.uniform(-INIT_SCALE_GLOBAL, INIT_SCALE_GLOBAL, N_DIM))
np.random.seed(None)

SEED = 0
_N_TRIALS = 50

# =====================================================
# ETA回调
# =====================================================
_study_start = None


def _eta_callback(study, trial):
    elapsed = time.time() - _study_start
    done = trial.number + 1
    remaining = _N_TRIALS - done
    eta_sec = (elapsed / done) * remaining if remaining > 0 else 0
    pruned_trials = study.get_trials(
        deepcopy=False,
        states=[optuna.trial.TrialState.PRUNED])
    print(f"    ── 进度 {done}/{_N_TRIALS} "
          f"(Pruned: {len(pruned_trials)})  "
          f"已用 {elapsed/60:.1f}min  "
          f"预计剩余 {eta_sec/60:.1f}min  "
          f"当前最优 J={study.best_value:.2f}")


# =====================================================
# 固定随机种子
# =====================================================
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


# =====================================================
# 训练并评估 SAC (Lorenz96)
# =====================================================
def run_sac(trial, lr, gamma, tau, h, bs):
    t0 = time.time()
    set_seed(SEED)

    env = Lorenz96Env(N=N_DIM, max_steps=MS_SEARCH)
    decoder = SparseActuatorDecodef(N_DIM, N_ACT, sigma=2.5)
    agent = SAC(sd=OBS_DIM_SAC, ad=N_ACT,
                lr=lr, gamma=gamma, tau=tau,
                h=h, bs=bs, buf=50000, ui=2)

    dt_eff = 0.01 * N_SUBSTEPS
    pid_base = PIDController(
        [PID_BASE_KP], [PID_BASE_KI], [PID_BASE_KD],
        dt=dt_eff, N=N_DIM)

    ACT_SCALE_MAX = 12.0
    ACT_SCALE_MIN = 8.0

    ep_rewards = []
    for ep in range(NEP_SEARCH):
        progress = min(1.0, ep / 200.0)
        act_scale = ACT_SCALE_MAX - progress * (ACT_SCALE_MAX - ACT_SCALE_MIN)

        env.max_steps = MS_SEARCH
        env.reset(warmup=10, init_scale=INIT_SCALE_GLOBAL)
        pid_base.reset()

        e = env.get_error()
        integ = np.zeros(N_DIM)
        deriv = np.zeros(N_DIM)
        prev_u = np.zeros(N_DIM)
        prev_act = np.zeros(N_ACT)
        ep_rw = 0.0

        for _ in range(MS_SEARCH):
            obs = build_sac_obs(e, integ, deriv, prev_u, prev_act,
                                N_DIM, N_ACT)
            obs_clip = np.clip(obs / 5.0, -2.0, 2.0)
            raw = agent.act(obs_clip)

            u_base = pid_base.compute(e)
            u_residual = decoder.decode(raw, scale=act_scale)
            u = np.clip(u_base + u_residual, -50.0, 50.0)

            ns, en, done = env.step(u, n_substeps=N_SUBSTEPS)
            e_next = env.get_error()

            integ = np.clip(integ + e * dt_eff, -50.0, 50.0)
            deriv = (e_next - e) / dt_eff

            base_r = compute_base_reward(e_next, u, N=N_DIM)
            smooth_r = -0.001 * np.mean((u - prev_u)**2)
            rw = float(np.clip(base_r + smooth_r, -100.0, 0.0))

            nobs = build_sac_obs(e_next, integ, deriv, u, raw,
                                 N_DIM, N_ACT)
            nobs_clip = np.clip(nobs / 5.0, -2.0, 2.0)
            agent.rb.push(obs_clip, raw, rw, nobs_clip, float(done))
            agent.update()

            prev_u = u
            prev_act = raw
            e = e_next
            ep_rw += rw
            if done:
                break

        ep_rewards.append(ep_rw)

        if (ep + 1) % REPORT_INTERVAL == 0:
            intermediate_value = np.mean(
                ep_rewards[-REPORT_INTERVAL:])
            trial.report(intermediate_value, ep)
            if trial.should_prune():
                del agent, env, decoder, pid_base
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()

    # ── 评估（多初始条件） ──
    total = 0.0
    for init in EVAL_INITS:
        env_eval = Lorenz96Env(N=N_DIM,
                               max_steps=EVAL_STEPS + 100)
        env_eval.reset(init, warmup=0)
        pid_eval = PIDController(
            [PID_BASE_KP], [PID_BASE_KI], [PID_BASE_KD],
            dt=dt_eff, N=N_DIM)
        pid_eval.reset()

        e_e = env_eval.get_error()
        integ_e = np.zeros(N_DIM)
        deriv_e = np.zeros(N_DIM)
        prev_u_e = np.zeros(N_DIM)
        prev_act_e = np.zeros(N_ACT)

        for _ in range(EVAL_STEPS):
            obs_e = build_sac_obs(e_e, integ_e, deriv_e,
                                  prev_u_e, prev_act_e,
                                  N_DIM, N_ACT)
            obs_clip_e = np.clip(obs_e / 5.0, -2.0, 2.0)
            raw_e = agent.act(obs_clip_e, det=True)

            u_base_e = pid_eval.compute(e_e)
            en_e = np.linalg.norm(e_e) / np.sqrt(N_DIM)
            adaptive_scale = max(1.0, min(8.0, 8.0 * en_e))
            u_residual_e = decoder.decode(raw_e,
                                          scale=adaptive_scale)
            a_e = np.clip(u_base_e + u_residual_e,
                          -50.0, 50.0)

            ns_e, _, done_e = env_eval.step(
                a_e, n_substeps=N_SUBSTEPS)
            e_next_e = env_eval.get_error()

            total += compute_base_reward(e_next_e, a_e, N=N_DIM)

            integ_e = np.clip(
                integ_e + e_e * dt_eff, -50.0, 50.0)
            deriv_e = (e_next_e - e_e) / dt_eff
            prev_u_e = a_e
            prev_act_e = raw_e
            e_e = e_next_e
            if done_e:
                break

        del env_eval, pid_eval

    result = total / len(EVAL_INITS)

    del agent, env, decoder, pid_base
    torch.cuda.empty_cache()
    gc.collect()

    print(f"    [SAC]    耗时 {time.time()-t0:.1f}s  "
          f"R={result:.2f}")
    return result


# =====================================================
# 训练并评估 SAC-PID (Lorenz96)
# =====================================================
def run_sacpid(trial, lr, gamma, tau, h, bs):
    t0 = time.time()
    set_seed(SEED)

    env = Lorenz96Env(N=N_DIM, max_steps=MS_SEARCH)
    tuner = SACPIDTuner(N=N_DIM)
    tuner.sac = SAC(sd=OBS_DIM_SACPID, ad=3,
                    lr=lr, gamma=gamma, tau=tau,
                    h=h, bs=bs, buf=50000, ui=2)

    ep_rewards = []
    for ep in range(NEP_SEARCH):
        env.max_steps = MS_SEARCH
        env.reset(warmup=10, init_scale=INIT_SCALE_GLOBAL)
        tuner.reset()
        e = env.get_error()
        ep_rw = 0.0

        for _ in range(MS_SEARCH):
            u, obs, raw, Kp, Ki, Kd = tuner.compute(e)
            prev_raw_saved = tuner.prev_raw.copy()

            ns, _, done = env.step(u, n_substeps=1)
            e_next = env.get_error()

            base_r = compute_base_reward(e_next, u, N=N_DIM)
            smooth_r = -0.005 * np.mean(
                (raw - prev_raw_saved)**2)
            rw = float(np.clip(base_r + smooth_r,
                               -100.0, 0.0))

            nobs = tuner.get_obs(e_next)
            tuner.sac.rb.push(obs, raw, rw, nobs, float(done))
            tuner.sac.update()

            e = e_next
            ep_rw += rw
            if done:
                break

        ep_rewards.append(ep_rw)

        if (ep + 1) % REPORT_INTERVAL == 0:
            intermediate_value = np.mean(
                ep_rewards[-REPORT_INTERVAL:])
            trial.report(intermediate_value, ep)
            if trial.should_prune():
                del tuner, env
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()

    # ── 评估（多初始条件） ──
    total = 0.0
    ema_alpha = 0.1
    for init in EVAL_INITS:
        env_eval = Lorenz96Env(N=N_DIM,
                               max_steps=EVAL_STEPS + 100)
        env_eval.reset(init, warmup=0)
        tuner.reset()
        e_e = env_eval.get_error()
        K_ema = None

        for _ in range(EVAL_STEPS):
            obs_e = tuner.get_obs(e_e)
            raw_e = tuner.sac.act(obs_e, det=True)
            Kp_e, Ki_e, Kd_e = tuner._decode(raw_e)
            K_now = np.array([Kp_e, Ki_e, Kd_e])
            if K_ema is None:
                K_ema = K_now.copy()
            else:
                K_ema = ((1 - ema_alpha) * K_ema
                         + ema_alpha * K_now)
            a_e = tuner.pid.compute(
                e_e, K_ema[0], K_ema[1], K_ema[2])
            tuner.prev_raw = raw_e.copy()
            tuner.prev_kpid = tuner._normalize(
                K_ema[0], K_ema[1], K_ema[2])
            tuner.prev_e = e_e.copy()

            ns_e, _, done_e = env_eval.step(a_e, n_substeps=1)
            e_next_e = env_eval.get_error()

            total += compute_base_reward(
                e_next_e, a_e, N=N_DIM)
            e_e = e_next_e
            if done_e:
                break

        del env_eval

    result = total / len(EVAL_INITS)

    del tuner, env
    torch.cuda.empty_cache()
    gc.collect()

    print(f"    [SACPID] 耗时 {time.time()-t0:.1f}s  "
          f"R={result:.2f}")
    return result


# =====================================================
# Optuna目标函数
# J = 0.5*R_SAC + 0.5*R_SACPID
# =====================================================
def objective(trial):
    lr = trial.suggest_float('lf', 1e-4, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999,
                                log=True)
    tau = trial.suggest_float('tau', 0.001, 0.02, log=True)
    h = trial.suggest_categorical('h', [128, 192, 256])
    bs = trial.suggest_categorical('bs', [128, 256, 512])

    print(f"\n  Trial {trial.number:>2d}: "
          f"lr={lr:.2e}  gamma={gamma:.4f}  "
          f"tau={tau:.4f}  h={h}  bs={bs}")

    t0 = time.time()

    try:
        r_sac = run_sac(trial, lr, gamma, tau, h, bs)

        torch.cuda.empty_cache()
        gc.collect()

        r_sacpid = run_sacpid(trial, lr, gamma, tau, h, bs)

        J = 0.5 * r_sac + 0.5 * r_sacpid

        torch.cuda.empty_cache()
        gc.collect()

        print(f"    R_SAC={r_sac:.2f}  "
              f"R_SACPID={r_sacpid:.2f}  "
              f"J={J:.2f}  "
              f"耗时={time.time()-t0:.1f}s")
        return J

    except optuna.TrialPruned:
        print(f"    Trial {trial.number} PRUNED.")
        raise


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()

    print("=" * 55)
    print("  Optuna 优化 SAC/SAC-PID 共享超参数 "
          "(Lorenz96, 带剪枝)")
    print(f"  搜索维度: 5  总Trial数: {_N_TRIALS}")
    print(f"  训练规模: {NEP_SEARCH}ep × {MS_SEARCH}steps")
    print(f"  评估条件: {len(EVAL_INITS)}个初始点")
    print(f"  系统维度: N={N_DIM}, n_act={N_ACT}")
    print("=" * 55)

    sampler = TPESampler(seed=42, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=20,
                          n_warmup_steps=100)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='sac_sacpid_hyperparams_lorenz96'
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    _study_start = time.time()

    study.optimize(objective,
                   n_trials=_N_TRIALS,
                   callbacks=[_eta_callback])

    total_elapsed = time.time() - _study_start

    best = study.best_trial
    print(f"\n{'=' * 55}")
    if best is not None:
        print(f"  最优超参数 (Trial {best.number}):")
        for k, v in best.params.items():
            print(f"    {k:6s} = {v}")
        print(f"  目标函数 J = {best.value:.4f}")
    else:
        print("  没有找到完成的Trial，无法确定最优参数。")

    print(f"  总耗时:      {total_elapsed/60:.1f} min")
    print(f"  平均每Trial: {total_elapsed/_N_TRIALS:.1f}s")

    if best is not None:
        best_params = {
            'lf': float(best.params['lf']),
            'gamma': float(best.params['gamma']),
            'tau': float(best.params['tau']),
            'h': int(best.params['h']),
            'bs': int(best.params['bs']),
            'best_J': float(best.value),
            'total_time_sec': round(total_elapsed, 1),
            'avg_trial_sec': round(
                total_elapsed / _N_TRIALS, 1),
            'note': ('Optuna TPE (with Pruning) Lorenz96, '
                     'J=0.5*R_SAC+0.5*R_SACPID')
        }
        with open('best_hyperparams.json', 'w',
                  encoding='utf-8') as f:
            json.dump(best_params, f, indent=2,
                      ensure_ascii=False)
        print("\n  已保存 best_hyperparams.json")

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("\n  没有完成的Trial，跳过可视化。")
    else:
        # ── 可视化1：收敛曲线 ──
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(
            [t.number for t in completed_trials],
            [t.value for t in completed_trials],
            color='steelblue', alpha=0.6, s=30,
            label='Completed Trial value')

        best_so_far = []
        cur_best = -np.inf
        for t in study.trials:
            if (t.state == optuna.trial.TrialState.COMPLETE
                    and t.value > cur_best):
                cur_best = t.value
            best_so_far.append(cur_best)
        ax1.plot(range(len(study.trials)), best_so_far,
                 color='red', lw=2, label='Best so faf')

        ax1.set_xlabel('Trial', fontsize=13)
        ax1.set_ylabel(
            'J = 0.5·R_SAC + 0.5·R_SACPID', fontsize=13)
        ax1.set_title(
            'Hyperparameter Optimization Convergence '
            '(Lorenz96)', fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        fig1.savefig('optuna_conv_hp.png', dpi=300,
                     bbox_inches='tight')
        plt.close(fig1)
        print("  optuna_conv_hp.png 已保存")

        # ── 可视化2：参数重要性 ──
        try:
            importances = optuna.importance.get_param_importances(
                study)
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            bars = ax2.barh(list(importances.keys()),
                            list(importances.values()),
                            color='steelblue', alpha=0.8)
            ax2.bar_label(bars, fmt='%.3f', fontsize=10)
            ax2.set_xlabel('Importance', fontsize=13)
            ax2.set_title(
                'Hyperparameter Importance (Lorenz96)',
                fontsize=13)
            ax2.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            fig2.savefig('optuna_importance_hp.png', dpi=300,
                         bbox_inches='tight')
            plt.close(fig2)
            print("  optuna_importance_hp.png 已保存")
        except Exception as e:
            print(f"  参数重要性图跳过: {e}")

        # ── 可视化3：参数散点图 ──
        fig3, axes3 = plt.subplots(1, 5, figsize=(18, 4))
        for ax, pn in zip(
                axes3, ['lf', 'gamma', 'tau', 'h', 'bs']):
            xs = [t.params[pn] for t in completed_trials]
            ys = [t.value for t in completed_trials]
            ax.scatter(xs, ys, color='steelblue',
                       alpha=0.6, s=30)
            ax.axvline(x=best.params[pn],
                       color='red', ls='--', lw=1.5,
                       label='best')
            ax.set_xlabel(pn, fontsize=11)
            ax.set_ylabel('J', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        plt.suptitle(
            'Parameter vs Objective (Lorenz96)',
            fontsize=13)
        plt.tight_layout()
        fig3.savefig('optuna_scatter_hp.png', dpi=300,
                     bbox_inches='tight')
        plt.close(fig3)
        print("  optuna_scatter_hp.png 已保存")

    print("\n  完成！使用流程：")
    print("  1. 查看 best_hyperparams.json")
    print("  2. 将参数填入主文件 lorenz96_main.py")
