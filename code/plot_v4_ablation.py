# -*- coding: utf-8 -*-
"""plot_v4_ablation.py — 仅重画 fig_ablation.pdf（n=7 本机种子，matplotlib）。"""
import json, os
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

plt.rcParams.update({"font.size":9,"axes.titlesize":10,"axes.labelsize":9,
    "xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8,
    "axes.grid":True,"grid.alpha":0.35,"grid.linestyle":"--","font.family":"DejaVu Sans"})

CONFIGS=["both_eq","wm_eq_only","policy_eq_only","none_eq"]
CLABELS=["both-eq\n(WM+policy)","wm-eq-only","policy-eq-only","none-eq"]
COLORS=["#2a9d8f","#457b9d","#e9a23b","#e76f51"]
loc=S["ablation"]["primary_local9"]
N=len(loc["both_eq"]["per_unit"])
# 云盘 cross-check 中心（A10 合并对，作为参考菱形）
cloud={"both_eq":439.333,"wm_eq_only":378.667,"policy_eq_only":440.667,"none_eq":403.333}

fig,ax=plt.subplots(figsize=(5.4,3.6))
x=np.arange(len(CONFIGS))
for i,cfg in enumerate(CONFIGS):
    s=loc[cfg]
    ax.bar(x[i],s["mean"],width=0.55,yerr=[[s["mean"]-s["ci95_lo"]],[s["ci95_hi"]-s["mean"]]],
           color=COLORS[i],alpha=0.82,capsize=3.5,error_kw=dict(lw=1.0),label=CLABELS[i])
    ax.scatter(np.full(N,x[i])+np.linspace(-0.15,0.15,N),s["per_unit"],s=13,color="black",zorder=5,alpha=0.85)
    ax.scatter([x[i]+0.24],[cloud[cfg]],marker="D",s=24,facecolor="none",edgecolor="#333333",zorder=5,linewidths=1.1)
    ax.text(x[i],s["ci95_hi"]+18,f"{s['mean']:.0f}",ha="center",fontsize=8.5,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(CLABELS)
ax.set_ylabel("Robust episode return"); ax.set_ylim(300,760)
ax.set_title(f"Ablation of equivariant components (pooled over {N} local train seeds; "
             "open diamond = A10 cloud pair, pooled)",fontsize=9)
ax.annotate("vs none p=0.41",xy=(x[0],560),xytext=(-0.55,0.78),xycoords="data",
            textcoords="axes fraction",fontsize=8,color="#333333",
            arrowprops=dict(arrowstyle="-",lw=0.8,color="#999999"))
ax.legend(loc="lower left",frameon=True,fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGS,"fig_ablation.pdf"))
fig.savefig(os.path.join(FIGS,"fig_ablation.png"),dpi=150)
plt.close(fig)
print(f"Saved fig_ablation.pdf (n={N})")
