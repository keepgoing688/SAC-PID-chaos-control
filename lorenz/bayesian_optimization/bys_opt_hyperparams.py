# bys_opt_hyperparams.py
"""
Optuna优化SAC和SAC-PID共享超参数
目标函数 = 0.5*R_SAC + 0.5*R_SACPID
修复版：bs范围/资源释放/逻辑清晰化/ETA修复
+ 增加了Pruning剪枝功能
"""

import json
import time
import gc
import numpy as np
import torch
import random
import optuna
from optuna.pruners import MedianPruner  # <--- MODIFIED: 导入Pruner
from optuna.samplers import TPESampler
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lorenz.main.lorenz_8 import (LorenzEnv, SACPIDTuner, SAC, reward_fn)

# =====================================================
# 缩减规模
# =====================================================
NEP_SEARCH = 500
MS_SEARCH = 400
REPORT_INTERVAL = 100  # <--- MODIFIED: 每100个episodes报告一次
EVAL_STEPS = 500
EVAL_INITS = [
    [2.0, 2.0, 10.0],
    [10.0, -5.0, 30.0],
    [-8.0, 12.0, 20.0],
]
SEED = 0
_N_TRIALS = 50

# =====================================================
# ETA回调（修复版：在optimize前计时）
# =====================================================
_study_start = None


def _eta_callback(study, trial):
    elapsed = time.time() - _study_start
    done = trial.number + 1
    remaining = _N_TRIALS - done
    eta_sec = (elapsed / done) * remaining if remaining > 0 else 0

    # <--- MODIFIED: 统计并显示剪枝数量 --->
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])

    print(f"    ── 进度 {done}/{_N_TRIALS} (Pruned: {len(pruned_trials)})  "
          f"已用 {elapsed / 60:.1f}min  "
          f"预计剩余 {eta_sec / 60:.1f}min  "
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
# 训练并评估 SAC
# =====================================================
def run_sac(trial, lr, gamma, tau, h, bs):  # <--- MODIFIED: 增加 trial 参数
    t0 = time.time()
    set_seed(SEED)

    env = LorenzEnv(max_steps=MS_SEARCH)
    agent = SAC(sd=3, ad=3,
                lr=lr, gamma=gamma, tau=tau,
                h=h, bs=bs, buf=30000, ui=2)

    # ── 训练 ──
    ep_rewards = []
    for ep in range(NEP_SEARCH):
        s = env.reset()
        prev_e = env.get_error()
        ep_rw = 0.0

        for _ in range(MS_SEARCH):
            ra = agent.act(s)
            a = ra * 15.0
            ns, _, done = env.step(a)
            e_now = env.get_error()

            rw = reward_fn(prev_e, e_now, a,
                           ra, None, 'sac')
            if np.any(np.abs(ns) > 200):
                rw -= 10

            agent.rb.push(s, ra, rw, ns, float(done))
            agent.update()

            s = ns
            prev_e = e_now.copy()
            ep_rw += rw
            if done:
                break

        ep_rewards.append(ep_rw)

        # <--- MODIFIED: 剪枝逻辑 --->
        if (ep + 1) % REPORT_INTERVAL == 0:
            intermediate_value = np.mean(ep_rewards[-REPORT_INTERVAL:])
            trial.report(intermediate_value, ep)
            if trial.should_prune():
                del agent, env
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()

    agent._sync()

    # ── 评估（多初始条件） ──
    total = 0.0
    for init in EVAL_INITS:
        env_eval = LorenzEnv(max_steps=EVAL_STEPS + 100)
        s_e = env_eval.reset(init)
        prev_e_e = env_eval.get_error()

        for _ in range(EVAL_STEPS):
            a_e = agent.act(s_e, det=True) * 15.0
            ns_e, _, done_e = env_eval.step(a_e)
            e_e = env_eval.get_error()

            total += reward_fn(prev_e_e, e_e, a_e,
                               None, None, 'sac')
            prev_e_e = e_e.copy()
            s_e = ns_e
            if done_e:
                break

        del env_eval

    result = total / len(EVAL_INITS)

    del agent, env
    torch.cuda.empty_cache()
    gc.collect()

    print(f"    [SAC]    耗时 {time.time() - t0:.1f}s  "
          f"R={result:.2f}")
    return result


# =====================================================
# 训练并评估 SAC-PID
# =====================================================
def run_sacpid(trial, lr, gamma, tau, h, bs):  # <--- MODIFIED: 增加 trial 参数
    t0 = time.time()
    set_seed(SEED)

    env = LorenzEnv(max_steps=MS_SEARCH)
    tuner = SACPIDTuner()
    tuner.sac = SAC(sd=12, ad=9,
                    lr=lr, gamma=gamma, tau=tau,
                    h=h, bs=bs, buf=30000, ui=2)

    # ── 训练 ──
    ep_rewards = []
    for ep in range(NEP_SEARCH):
        env.reset()
        tuner.reset()
        ep_rw = 0.0

        for _ in range(MS_SEARCH):
            e = env.get_error()
            u, obs, raw, Kp, Ki, Kd = tuner.compute(e)
            ns, _, done = env.step(u)
            e2 = env.get_error()

            rw = reward_fn(e, e2, u, raw,
                           tuner.prev_raw, 'sacpid')
            nobs = tuner._obs(e2)

            tuner.sac.rb.push(obs, raw, rw,
                              nobs, float(done))
            tuner.sac.update()
            ep_rw += rw
            if done:
                break

        ep_rewards.append(ep_rw)

        # <--- MODIFIED: 剪枝逻辑 --->
        if (ep + 1) % REPORT_INTERVAL == 0:
            intermediate_value = np.mean(ep_rewards[-REPORT_INTERVAL:])
            trial.report(intermediate_value, ep)
            if trial.should_prune():
                del tuner, env
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()

    tuner.sac._sync()

    # ── 评估（多初始条件） ──
    total = 0.0
    for init in EVAL_INITS:
        env_eval = LorenzEnv(max_steps=EVAL_STEPS + 100)
        env_eval.reset(init)
        tuner.reset()
        prev_e_e = env_eval.get_error()

        for _ in range(EVAL_STEPS):
            e_e = env_eval.get_error()
            u_e, _, _, _, _, _ = tuner.compute(e_e, det=True)
            _, _, done_e = env_eval.step(u_e)
            e_next = env_eval.get_error()

            total += reward_fn(prev_e_e, e_next, u_e,
                               None, None, 'sacpid')
            prev_e_e = e_next.copy()
            if done_e:
                break

        del env_eval

    result = total / len(EVAL_INITS)

    del tuner, env
    torch.cuda.empty_cache()
    gc.collect()

    print(f"    [SACPID] 耗时 {time.time() - t0:.1f}s  "
          f"R={result:.2f}")
    return result


# =====================================================
# Optuna目标函数
# J = 0.5*R_SAC + 0.5*R_SACPID
# =====================================================
def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999, log=True)
    tau = trial.suggest_float('tau', 0.001, 0.02, log=True)
    h = trial.suggest_categorical('h', [64, 128, 192])
    bs = trial.suggest_categorical('bs', [128, 256])

    print(f"\n  Trial {trial.number:>2d}: "
          f"lr={lr:.2e}  gamma={gamma:.4f}  "
          f"tau={tau:.4f}  h={h}  bs={bs}")

    t0 = time.time()

    # <--- MODIFIED: 使用try-except捕获剪枝异常 --->
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
              f"耗时={time.time() - t0:.1f}s")
        return J

    except optuna.TrialPruned:
        # 如果被剪枝，捕获异常并重新抛出，Optuna会处理它
        print(f"    Trial {trial.number} PRUNED.")
        raise


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    # 启动时强制清理
    torch.cuda.empty_cache()
    gc.collect()

    print("=" * 55)
    print("  Optuna 优化 SAC/SAC-PID 共享超参数 (带剪枝功能)")
    print(f"  搜索维度: 5  总Trial数: {_N_TRIALS}")
    print(f"  训练规模: {NEP_SEARCH}ep × {MS_SEARCH}steps")
    print(f"  评估条件: {len(EVAL_INITS)}个初始点")
    print("=" * 55)

    sampler = TPESampler(seed=42, n_startup_trials=10)

    # <--- MODIFIED: 配置并应用Pruner --->
    # 在前20次实验完成后，才激活中位数修剪器。
    # n_warmup_steps=100 表示每个Trial的前100个episode(即第一个报告点)不剪枝，给模型热身。
    pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=100)

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,  # 应用剪枝器
        study_name='sac_sacpid_hyperparams'
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── 修复：在optimize前开始计时 ──
    _study_start = time.time()

    study.optimize(objective,
                   n_trials=_N_TRIALS,
                   callbacks=[_eta_callback])

    total_elapsed = time.time() - _study_start

    # ── 最优结果 ──
    best = study.best_trial
    print(f"\n{'=' * 55}")
    if best is not None:
        print(f"  最优超参数 (Trial {best.number}):")
        for k, v in best.params.items():
            print(f"    {k:6s} = {v}")
        print(f"  目标函数 J = {best.value:.4f}")
    else:
        print("  没有找到完成的Trial，无法确定最优参数。")

    print(f"  总耗时:      {total_elapsed / 60:.1f} min")
    print(f"  平均每Trial: {total_elapsed / _N_TRIALS:.1f}s")

    # ── 保存JSON ──
    if best is not None:
        best_params = {
            'lr': float(best.params['lr']),
            'gamma': float(best.params['gamma']),
            'tau': float(best.params['tau']),
            'h': int(best.params['h']),
            'bs': int(best.params['bs']),
            'best_J': float(best.value),
            'total_time_sec': round(total_elapsed, 1),
            'avg_trial_sec': round(total_elapsed / _N_TRIALS, 1),
            'note': 'Optuna TPE (with Pruning), J=0.5*R_SAC+0.5*R_SACPID'
        }
        with open('best_hyperparams.json', 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        print("\n  已保存 best_hyperparams.json")

    # <--- MODIFIED: 调整可视化逻辑以适应剪枝 --->
    # 筛选出已完成（未被剪枝）的Trial用于绘图
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("\n  没有完成的Trial，跳过可视化。")
    else:
        # ── 可视化1：收敛曲线 ──
        fig1, ax1 = plt.subplots(figsize=(8, 5))

        # 绘制所有完成的Trial的最终结果
        ax1.scatter([t.number for t in completed_trials],
                    [t.value for t in completed_trials],
                    color='steelblue', alpha=0.6, s=30,
                    label='Completed Trial value')

        # 绘制历史最优值的演变曲线（包含所有Trial，即使是剪枝的）
        best_so_far = []
        cur_best = -np.inf
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE and t.value > cur_best:
                cur_best = t.value
            best_so_far.append(cur_best)
        ax1.plot(range(len(study.trials)), best_so_far,
                 color='red', lw=2, label='Best so far')

        ax1.set_xlabel('Trial', fontsize=13)
        ax1.set_ylabel('J = 0.5·R_SAC + 0.5·R_SACPID', fontsize=13)
        ax1.set_title('Hyperparameter Optimization Convergence', fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        fig1.savefig('optuna_conv_hp.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("  optuna_conv_hp.png 已保存")

        # ── 可视化2：参数重要性 ──
        try:
            importances = optuna.importance.get_param_importances(study)
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            bars = ax2.barh(list(importances.keys()),
                            list(importances.values()),
                            color='steelblue', alpha=0.8)
            ax2.bar_label(bars, fmt='%.3f', fontsize=10)
            ax2.set_xlabel('Importance', fontsize=13)
            ax2.set_title('Hyperparameter Importance', fontsize=13)
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
        for ax, pn in zip(axes3, ['lr', 'gamma', 'tau', 'h', 'bs']):
            xs = [t.params[pn] for t in completed_trials]
            ys = [t.value for t in completed_trials]
            ax.scatter(xs, ys, color='steelblue', alpha=0.6, s=30)
            ax.axvline(x=best.params[pn],
                       color='red', ls='--', lw=1.5, label='best')
            ax.set_xlabel(pn, fontsize=11)
            ax.set_ylabel('J', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        plt.suptitle('Parameter vs Objective', fontsize=13)
        plt.tight_layout()
        fig3.savefig('optuna_scatter_hp.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print("  optuna_scatter_hp.png 已保存")

    print("\n  完成！使用流程：")
    print("  1. 查看 best_hyperparams.json")
    print("  2. 将参数填入主文件 lorenz_8.py")
