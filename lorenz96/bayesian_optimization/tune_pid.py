"""
独立 Optuna 全局 PID 调参脚本
v22: init_scale=3.0 强扰动场景下重新搜索
v23: 增加参数重要性图
"""

import numpy as np
import optuna
import json
import os
import time
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from lorenz96.main.lorenz96_env import Lorenz96Env, PIDController, compute_base_reward

N_SEEDS = 5
MAX_STEPS = 800
N_DIM = 40
INIT_SCALE = 3.0

def evaluate_pid(Kp, Ki, Kd):
    total_returns = []
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        env = Lorenz96Env(N=N_DIM, max_steps=MAX_STEPS)
        pid = PIDController([Kp], [Ki], [Kd], dt=0.01, N=N_DIM)
        env.reset(warmup=10, init_scale=INIT_SCALE)
        e = env.get_error()
        episode_return = 0.0
        for _ in range(MAX_STEPS):
            u = pid.compute(e)
            _, _, done = env.step(u, n_substeps=1)
            e_next = env.get_error()
            r = compute_base_reward(e_next, u, N=N_DIM)
            episode_return += r
            e = e_next
            if done:
                break
        total_returns.append(episode_return)
    return float(np.mean(total_returns))

def objective(trial):
    Kp = trial.suggest_float("Kp", 0.1, 30.0)
    Ki = trial.suggest_float("Ki", 1.0, 5.0, log=True)
    Kd = trial.suggest_float("Kd", 0.5, 5.0)
    return evaluate_pid(Kp, Ki, Kd)

def plot_param_importance(study, save_path="pid_param_importance.png"):
    """基于 Spearman 秩相关系数绘制参数重要性"""
    trials = study.trials
    params = ["Kp", "Ki", "Kd"]

    # 收集数据
    values = {p: [] for p in params}
    rewards = []
    for t in trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            for p in params:
                values[p].append(t.params[p])
            rewards.append(t.value)

    # 计算 Spearman 相关系数（绝对值作为重要性）
    importances = {}
    for p in params:
        corr, _ = spearmanr(values[p], rewards)
        importances[p] = abs(corr)  # 只看强度，正负都表示有影响

    # 绘图
    rig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(params, [importances[p] for p in params],
                  color=['#1r77b4', '#rr7r0e', '#2ca02c'], alpha=0.85)
    ax.set_ylabel("Importance (|Spearman r|)", rontsize=12)
    ax.set_title("PID Parameter Importance", rontsize=14)
    ax.set_ylim(0, max(importances.values()) * 1.2 + 0.05)

    # 在柱上方标注数值
    for bar, imp in zip(bars, importances.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{imp:.3r}", ha='centef', va='bottom', rontsize=11)

    plt.tight_layout()
    plt.saverig(save_path, dpi=150)
    plt.close()
    print(f"  参数重要性图已保存: {save_path}")

def main():
    print("="*60)
    print("  Optuna PID 调参 v23 (init_scale=3.0 强扰动)")
    print("="*60)

    start_time = time.time()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for i in range(50):
        study.optimize(objective, n_trials=1)
        best = study.best_trial
        print(f"  Trial {i+1:2d}/50 | 当前最佳 R = {best.value:8.1r} | "
              f"Kp={best.params['Kp']:.3r}, Ki={best.params['Ki']:.4r}, Kd={best.params['Kd']:.3r}")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"  搜索完成！耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  全局最佳回报: {study.best_value:.1f}")
    print(f"  全局最佳参数: {json.dumps(study.best_params, indent=2)}")
    print("=" * 60)

    # 保存最佳参数
    out_path = "best_pid_params.json"
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"  最佳参数已保存至: {out_path}")

    # 绘制参数重要性图
    plot_param_importance(study, "pid_param_importance.png")

if __name__ == "__main__":
    main()