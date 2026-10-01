"""
experiments/run_ablation.py
===========================
实验三：等变部件消融。

四个配置：
  [eq_wm, eq_policy]   全部等变
  [eq_wm, std_policy]  仅世界模型等变
  [std_wm, eq_policy]  仅策略等变
  [std_wm, std_policy] 都不等变（基线）

输出最终（训练结束时）评估回报对比。
  results/ablation.json
  results/ablation.png
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.imagination_ppo import run_imagination


CONFIGS = [
    ("both_eq", True, True),
    ("wm_eq_only", True, False),
    ("policy_eq_only", False, True),
    ("none_eq", False, False),
]


def run(env="health_gathering", seeds=(0, 1), init_steps=10000, outer_iters=6):
    results = {name: [] for name, _, _ in CONFIGS}
    for seed in seeds:
        for name, wm_eq, pol_eq in CONFIGS:
            print(f"\n===== ablation seed={seed} cfg={name} =====")
            t0 = time.time()
            curve = run_imagination(env_name=env, init_steps=init_steps,
                                    outer_iters=outer_iters, seed=seed,
                                    wm_equivariant=wm_eq, policy_equivariant=pol_eq)
            final_ret = curve[-1][1]
            results[name].append(final_ret)
            print(f"  {name}: final_return={final_ret:.2f}  ({time.time()-t0:.0f}s)")

    summary = {name: {"mean": float(np.mean(v)), "std": float(np.std(v))}
               for name, v in results.items()}
    os.makedirs("results", exist_ok=True)
    with open("results/ablation.json", "w") as f:
        json.dump({"per_seed": results, "summary": summary}, f, indent=2)
    print("Saved results/ablation.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(summary.keys())
        means = [summary[n]["mean"] for n in names]
        stds = [summary[n]["std"] for n in names]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(names)), means, yerr=stds, capsize=4,
               color=["#2a9d8f", "#8ab17d", "#e9c46a", "#e76f51"])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylabel("Final episode return")
        ax.set_title(f"Ablation: which parts should be equivariant? ({env})")
        fig.tight_layout()
        fig.savefig("results/ablation.png", dpi=120)
        print("Saved results/ablation.png")
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
