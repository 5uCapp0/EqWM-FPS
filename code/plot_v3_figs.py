# -*- coding: utf-8 -*-
"""
plot_v3_figs.py
===============
v3 论文图（matplotlib 精确绘制，直接读 data/ablation_pooled_v3.json 等数据，
无任何 AI 生图成分）：
  1) paper/figs/fig_ablation.pdf  —— 消融 4 配置：本机 5 种子均值±95%CI +
     逐种子散点 + 云端种子对（菱形）标注；配对 p 值注记。
  2) paper/figs/fig_wm_reward.pdf —— 逐种子 reward MAE（Eq vs Std，3 seeds）
     + 常数基线两条水平线（恒报均值 0.167 / 恒报众数 0.084，标准化空间）。
输出矢量 PDF（也存 PNG 供快速预览）。
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

with open(os.path.join(DATA, "ablation_pooled_v3.json"), "r", encoding="utf-8") as f:
    S = json.load(f)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans",
})

# ---------------------------------------------------------------------------
# 图 1：消融（聚合 5 本机种子 + 云端种子对）
# ---------------------------------------------------------------------------
CONFIGS = ["both_eq", "wm_eq_only", "policy_eq_only", "none_eq"]
CLABELS = ["both-eq\n(WM+policy)", "wm-eq-only", "policy-eq-only", "none-eq"]
COLORS = ["#2a9d8f", "#457b9d", "#e9a23b", "#e76f51"]

loc = S["ablation"]["primary_local5"]
pool = S["ablation"]["pooled_6units"]
paired = S["ablation"]["paired_local5"]
cloud = {"both_eq": 439.333, "wm_eq_only": 378.667, "policy_eq_only": 440.667, "none_eq": 403.333}

fig, ax = plt.subplots(figsize=(5.4, 3.6))
x = np.arange(len(CONFIGS))
for i, cfg in enumerate(CONFIGS):
    s = loc[cfg]
    ax.bar(x[i], s["mean"], width=0.55, yerr=[[s["mean"] - s["ci95_lo"]], [s["ci95_hi"] - s["mean"]]],
           color=COLORS[i], alpha=0.82, capsize=3.5, error_kw=dict(lw=1.0),
           label=CLABELS[i])
    # 本机逐种子散点
    ax.scatter(np.full(5, x[i]) + np.linspace(-0.13, 0.13, 5),
               s["per_unit"], s=14, color="black", zorder=5, alpha=0.85)
    # 云端种子对（单个观测）
    ax.scatter([x[i] + 0.22], [cloud[cfg]], marker="D", s=26,
               facecolor="none", edgecolor="#333333", zorder=5, linewidths=1.1)
    # 均值标注
    ax.text(x[i], s["ci95_hi"] + 18, f"{s['mean']:.0f}", ha="center", fontsize=8.5,
            fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(CLABELS)
ax.set_ylabel("Robust episode return")
ax.set_ylim(300, 760)
ax.set_title("Ablation of equivariant components (pooled over 5 local train seeds; "
             "open diamond = 2 cloud seeds, pooled)", fontsize=9)

# 配对 p 值注记（both vs 其余）
p_notes = {
    "both_eq": ("vs none p=0.25", (0, 0.62)),
}
for cfg, (txt, (xx, yy)) in p_notes.items():
    ax.annotate(txt, xy=(x[0], 620), xytext=(-0.55, 0.75), xycoords="data",
                textcoords="axes fraction", fontsize=8, color="#333333",
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#999999"))

ax.legend(loc="lower left", frameon=True, fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_ablation.pdf"))
fig.savefig(os.path.join(FIGS, "fig_ablation.png"), dpi=150)
plt.close(fig)
print("Saved fig_ablation.pdf / .png")

# ---------------------------------------------------------------------------
# 图 2：逐种子 reward MAE + 常数基线
# ---------------------------------------------------------------------------
R = S["reward_head"]
eq_mae = [0.0723, 0.1069, 0.1324]
std_mae = [0.0888, 0.2557, 0.3238]
xr = np.array([0, 1, 2])

fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.plot(xr, eq_mae, "-o", color="#2a9d8f", lw=1.6, ms=5, label="Equivariant reward head (per-seed MAE)")
ax.plot(xr, std_mae, "-s", color="#e76f51", lw=1.6, ms=5, label="Standard reward head (per-seed MAE)")

cb = R["constant_baselines"]
ax.axhline(cb["predict_mean_mae"], color="#457b9d", ls="--", lw=1.2,
           label=f"Constant baseline (predict mean) MAE={cb['predict_mean_mae']:.3f}")
ax.axhline(cb["predict_mode_mae"], color="#6a4c93", ls=":", lw=1.2,
           label=f"Constant baseline (predict mode) MAE={cb['predict_mode_mae']:.3f}")

for xi, v in zip(xr, eq_mae):
    ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(4, 6), fontsize=7.5)
for xi, v in zip(xr, std_mae):
    ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(4, -11), fontsize=7.5)

ax.set_xticks(xr)
ax.set_xticklabels(["seed 0", "seed 1", "seed 2"])
ax.set_xlabel("Training seed")
ax.set_ylabel("Reward-prediction MAE (standardized reward space)")
ax.set_ylim(0.0, 0.36)
ax.set_title("Reward-head MAE vs constant predictors (health_gathering, 3 seeds)")
ax.legend(loc="upper left", fontsize=7.5, frameon=True)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_wm_reward.pdf"))
fig.savefig(os.path.join(FIGS, "fig_wm_reward.png"), dpi=150)
plt.close(fig)
print("Saved fig_wm_reward.pdf / .png")
