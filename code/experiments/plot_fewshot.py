"""从 results/fewshot.json 重画 fewshot.png（修复 linestyle bug）。"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/fewshot.json"))
fig, ax = plt.subplots(figsize=(8, 5))
styles = [("eq_imagine", "o", "#2a9d8f", "Eq-Imagination"),
          ("std_imagine", "s", "#e9c46a", "Std-Imagination"),
          ("real_ppo", "^", "#e76f51", "Real-PPO")]
for key, marker, color, label in styles:
    arr = d[key]
    for c in arr:
        xs = [p[0] for p in c]; ys = [p[1] for p in c]
        ax.plot(xs, ys, marker=marker, color=color, alpha=0.35, lw=1)
    min_len = min(len(c) for c in arr)
    xs = [arr[0][i][0] for i in range(min_len)]
    ys = np.mean([[c[i][1] for i in range(min_len)] for c in arr], axis=0)
    ax.plot(xs, ys, marker=marker, color=color, lw=2, label=label)
ax.set_xlabel("Environment steps (real)")
ax.set_ylabel("Episode return")
ax.set_title("Sample Efficiency (health_gathering, 2 seeds)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/fewshot.png", dpi=120)
print("Saved results/fewshot.png")
