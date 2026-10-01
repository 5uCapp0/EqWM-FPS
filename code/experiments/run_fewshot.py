"""
experiments/run_fewshot.py
==========================
实验二：少样本想象训练样本效率。

对比：
  - Eq-Imagination: 等变世界模型 + 想象 PPO
  - Std-Imagination: 普通世界模型 + 想象 PPO
  - Real-PPO: 纯真实 PPO（无世界模型）

输出样本效率曲线：平均回报 vs 环境真实步数。

输出：
  results/fewshot.json
  results/fewshot.png
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.imagination_ppo import run_imagination
from training.train_baselines import run_real_ppo


def run(env="health_gathering", seeds=(0, 1), init_steps=10000,
        outer_iters=6, real_ppo_budgets=(10000, 50000, 100000)):
    curves = {"eq_imagine": [], "std_imagine": [], "real_ppo": []}
    for seed in seeds:
        print(f"\n===== fewshot seed {seed} =====")
        t0 = time.time()
        eq_curve = run_imagination(env_name=env, init_steps=init_steps,
                                   outer_iters=outer_iters, seed=seed, equivariant=True)
        curves["eq_imagine"].append(eq_curve)
        print(f"  eq_imagine done {time.time()-t0:.0f}s")

        t0 = time.time()
        std_curve = run_imagination(env_name=env, init_steps=init_steps,
                                    outer_iters=outer_iters, seed=seed, equivariant=False)
        curves["std_imagine"].append(std_curve)
        print(f"  std_imagine done {time.time()-t0:.0f}s")

        t0 = time.time()
        # 纯真实 PPO 跑到 100k 步，中间记录曲线
        real_curve = run_real_ppo(env_name=env, total_steps=max(real_ppo_budgets),
                                  seed=seed, equivariant=False, eval_every=20000)
        curves["real_ppo"].append(real_curve)
        print(f"  real_ppo done {time.time()-t0:.0f}s")

    os.makedirs("results", exist_ok=True)
    with open("results/fewshot.json", "w") as f:
        json.dump(curves, f, indent=2)
    print("Saved results/fewshot.json")

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, style, color in [("eq_imagine", "-o", "#2a9d8f"),
                                  ("std_imagination" if False else "std_imagine", "-s", "#e9c46a"),
                                  ("real_ppo", "--^", "#e76f51")]:
            arr = curves[key]
            # 对齐 x 轴（取每条曲线的 x）
            for c in arr:
                xs = [p[0] for p in c]
                ys = [p[1] for p in c]
                ax.plot(xs, ys, style, color=color, alpha=0.5)
            # 均值
            min_len = min(len(c) for c in arr)
            xs = [arr[0][i][0] for i in range(min_len)]
            ys = np.mean([[c[i][1] for i in range(min_len)] for c in arr], axis=0)
            ax.plot(xs, ys, style + "-", color=color, lw=2,
                    label={"eq_imagine": "Eq-Imagination",
                           "std_imagine": "Std-Imagination",
                           "real_ppo": "Real-PPO"}[key])
        ax.set_xlabel("Environment steps (real)")
        ax.set_ylabel("Episode return")
        ax.set_title(f"Sample Efficiency ({env})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("results/fewshot.png", dpi=120)
        print("Saved results/fewshot.png")
    except Exception as e:
        print("plot failed:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="health_gathering")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--init_steps", type=int, default=10000)
    ap.add_argument("--outer_iters", type=int, default=6)
    args = ap.parse_args()
    run(env=args.env, seeds=tuple(args.seeds), init_steps=args.init_steps,
        outer_iters=args.outer_iters)
