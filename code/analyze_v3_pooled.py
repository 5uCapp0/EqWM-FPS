# -*- coding: utf-8 -*-
"""
analyze_v3_pooled.py
====================
v3 论文数据补强分析（2026-10-02）：
  1) 消融实验聚合：5 个本机种子 (0,1,2,13,14) + 云端种子对 (3,4)，
     逐配置 mean/std/95%CI，配对差值（both vs none / wm vs none / pol vs none），
     Cohen's d、配对 t 检验、Wilcoxon。
  2) 奖励头常数基线：从经验奖励分布 (99.3% +4 / 0.7% -96) 计算
     标准化空间下的常数预测器 MAE/MSE，与等变/普通模型对比。
  3) Exp2 样本效率区间：想象@15k vs PPO@20k/40k 的 95% CI 与步数比。

所有原始值均注明来源（日志/JSON 文件），输出 data/ablation_pooled_v3.json
与 data/reward_head_analysis_v3.json。只读分析，不修改任何源数据。
"""

import json
import math
import os

from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


# ----------------------------------------------------------------------------
# 1. 消融：逐种子 robust 均值（来源见注释）
# ----------------------------------------------------------------------------
# 本机 seeds 0,1,2：来自 code/logs/ablation_run.log 的逐种子 "robust=..." 行
local_seeds_012 = {
    0: {"both_eq": 490.7, "wm_eq_only": 401.3, "policy_eq_only": 454.7, "none_eq": 402.7},
    1: {"both_eq": 398.7, "wm_eq_only": 409.3, "policy_eq_only": 390.7, "none_eq": 513.3},
    2: {"both_eq": 452.0, "wm_eq_only": 465.3, "policy_eq_only": 434.7, "none_eq": 433.3},
}
# 本机 seeds 13,14：来自当前 data/ablation.json 的 robust_mean 数组
local_seeds_1314 = {
    13: {"both_eq": 577.333, "wm_eq_only": 461.333, "policy_eq_only": 402.667, "none_eq": 392.0},
    14: {"both_eq": 682.667, "wm_eq_only": 414.667, "policy_eq_only": 424.0, "none_eq": 404.0},
}
# 云端 seeds 3+4：来自 data/ablation_cloud_s3s4.json 的 summary（两种子合并均值，
# 逐种子明细未保留 → 论文中作为 1 个单位观测，跨机器异质性披露）
cloud_pair_34 = {"both_eq": 439.333, "wm_eq_only": 378.667, "policy_eq_only": 440.667, "none_eq": 403.333}

CONFIGS = ["both_eq", "wm_eq_only", "policy_eq_only", "none_eq"]


def t_ci95(xs):
    """95% t-置信区间 (mean, lo, hi)。"""
    n = len(xs)
    m = float(sum(xs)) / n
    if n < 2:
        return m, m, m
    s = math.sqrt(sum((v - m) ** 2 for v in xs) / (n - 1))
    t = stats.t.ppf(0.975, n - 1)
    se = s / math.sqrt(n)
    return m, m - t * se, m + t * se


def summarize_config(values, label):
    n = len(values)
    m, lo, hi = t_ci95(values)
    sd = (math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0)
    return {
        "label": label, "n": n, "mean": round(m, 1), "std": round(sd, 1),
        "ci95_lo": round(lo, 1), "ci95_hi": round(hi, 1),
        "min": round(min(values), 1), "max": round(max(values), 1),
        "per_unit": [round(v, 1) for v in values],
    }


def paired_stats(a, b, label):
    """配对差值 a-b：mean diff、sd、Cohen's d、配对 t 检验、Wilcoxon。"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    md = sum(d) / n
    sd = math.sqrt(sum((v - md) ** 2 for v in d) / (n - 1)) if n > 1 else 0.0
    d_cohen = md / sd if sd > 0 else 0.0
    t_stat, p_t = stats.ttest_rel(a, b)
    try:
        w_stat, p_w = stats.wilcoxon(a, b)
    except ValueError:  # 全零差值
        w_stat, p_w = 0.0, 1.0
    return {
        "label": label, "n": n, "mean_diff": round(md, 1), "sd_diff": round(sd, 1),
        "cohens_d": round(d_cohen, 2), "t": round(t_stat, 3), "p_twosided_t": round(p_t, 4),
        "wilcoxon_p": round(p_w, 4),
        "per_unit_diff": [round(v, 1) for v in d],
    }


# --- 主分析：5 个本机种子 ---
local_all = {}
local_all.update(local_seeds_012)
local_all.update(local_seeds_1314)
local_seeds = sorted(local_all.keys())

ablation = {"primary_local5": {}, "pooled_6units": {}, "paired_local5": {}, "paired_pooled6": {}}
for cfg in CONFIGS:
    vals_local = [local_all[s][cfg] for s in local_seeds]
    ablation["primary_local5"][cfg] = summarize_config(vals_local, cfg)
    vals_pooled = vals_local + [cloud_pair_34[cfg]]
    ablation["pooled_6units"][cfg] = summarize_config(vals_pooled, cfg)

# 配对检验（local 5）
for a_cfg, b_cfg, lab in [("both_eq", "none_eq", "both_eq - none_eq"),
                          ("wm_eq_only", "none_eq", "wm_eq_only - none_eq"),
                          ("policy_eq_only", "none_eq", "policy_eq_only - none_eq"),
                          ("both_eq", "wm_eq_only", "both_eq - wm_eq_only"),
                          ("both_eq", "policy_eq_only", "both_eq - policy_eq_only")]:
    a = [local_all[s][a_cfg] for s in local_seeds]
    b = [local_all[s][b_cfg] for s in local_seeds]
    ablation["paired_local5"][lab] = paired_stats(a, b, lab)
    a6 = a + [cloud_pair_34[a_cfg]]
    b6 = b + [cloud_pair_34[b_cfg]]
    ablation.setdefault("paired_pooled6", {})[lab] = paired_stats(a6, b6, lab)

ablation["_provenance"] = {
    "local_seeds_012": "code/logs/ablation_run.log 逐种子 robust 行",
    "local_seeds_1314": "data/ablation.json robust_mean（seeds 13,14，文件 21:14 覆盖后版本）",
    "cloud_pair_34": "data/ablation_cloud_s3s4.json summary（两种子合并，无逐种子明细）",
    "note": "primary_local5 = 同机（本机 RTX3060）5 种子；pooled_6units 加入云端种子对作为次级披露（跨机器异质性）。"
}

# ----------------------------------------------------------------------------
# 2. 奖励头常数基线（标准化空间）
# ----------------------------------------------------------------------------
p4, r4 = 0.993, 4.0      # 99.3% 步 +4（经验分布，来源：项目进度 17:0x 记录）
pd, rd = 0.007, -96.0    # 0.7% 步 -96（死亡）
mu = p4 * r4 + pd * rd
sigma = math.sqrt(p4 * (r4 - mu) ** 2 + pd * (rd - mu) ** 2)
z4 = (r4 - mu) / sigma
zd = (rd - mu) / sigma

const_mean_mae = p4 * abs(z4) + pd * abs(zd)          # 恒报均值(0)
const_mean_mse = p4 * z4 ** 2 + pd * zd ** 2          # 应≈1.0（与日志 reward loss 对照）
const_mode_mae = pd * abs(zd - z4)                    # 恒报众数(z4)

reward = {
    "dist": {"p_plus4": p4, "r_plus4": r4, "p_death": pd, "r_death": rd},
    "zspace": {"mu": round(mu, 4), "sigma": round(sigma, 4),
               "z_plus4": round(z4, 4), "z_death": round(zd, 4)},
    "constant_baselines": {
        "predict_mean_mae": round(const_mean_mae, 4),
        "predict_mean_mse": round(const_mean_mse, 4),
        "predict_mode_mae": round(const_mode_mae, 4),
    },
    "model_mae": {
        "equivariant": {"per_seed": [0.0723, 0.1069, 0.1324], "mean": 0.1039, "std": 0.0246},
        "standard": {"per_seed": [0.0888, 0.2557, 0.3238], "mean": 0.2228, "std": 0.0987},
    },
    "interpretation": (
        "两模型 reward head 的训练 loss≈0.97-1.0，等于恒报标准化均值的 MSE≈1.0："
        "奖励头基本退化为预测均值，未学会死亡惩罚。等变头 MAE 0.104 优于恒报均值基线 0.167 "
        "但劣于恒报众数基线 0.084；普通头均值 0.223 劣于恒报均值基线且两种子漂移至 0.26-0.32。"
        "结论：0.104 vs 0.223 是'稳定近众数预测器 vs 漂移预测器'的稳定性对比，"
        "不是任务相关奖励理解的证据。"
    ),
    "_source": "code/results/wm_comparison.json + code/logs/ablation_run.log reward loss 行",
}

# ----------------------------------------------------------------------------
# 3. Exp2 样本效率（fewshot.json）
# ----------------------------------------------------------------------------
imag15k = [316.0, 364.0, 508.0, 428.0]        # eq 2 seeds + std 2 seeds @15k
ppo20k = [476.0, 364.0]
ppo40k = [572.0, 284.0]
m_i, lo_i, hi_i = t_ci95(imag15k)
m_20, _, _ = t_ci95(ppo20k)
m_40, _, _ = t_ci95(ppo40k)

exp2 = {
    "imag_15k": {"values": imag15k, "mean": round(m_i, 1), "ci95": [round(lo_i, 1), round(hi_i, 1)]},
    "ppo_20k": {"values": ppo20k, "mean": round(m_20, 1)},
    "ppo_40k": {"values": ppo40k, "mean": round(m_40, 1)},
    "step_ratio": {"lo": round(20.0 / 15.0, 2), "hi": round(40.0 / 15.0, 2)},
    "variant_at_15k": {"eq_imagine": [316.0, 364.0], "std_imagine": [508.0, 428.0]},
    "_source": "code/results/fewshot.json",
}

# ----------------------------------------------------------------------------
out = {
    "generated": "2026-10-02 (analyze_v3_pooled.py)",
    "ablation": ablation,
    "reward_head": reward,
    "exp2": exp2,
}
os.makedirs(DATA, exist_ok=True)
with open(os.path.join(DATA, "ablation_pooled_v3.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# 控制台摘要
print("==== 消融聚合（本机 5 种子）====")
for cfg in CONFIGS:
    s = ablation["primary_local5"][cfg]
    print(f"{cfg:14s} mean={s['mean']:7.1f} ± {s['std']:6.1f}  CI95=[{s['ci95_lo']:7.1f},{s['ci95_hi']:7.1f}]  per={s['per_unit']}")
print("\n==== 配对检验（本机 5 种子）====")
for lab, s in ablation["paired_local5"].items():
    print(f"{lab:28s} d={s['mean_diff']:7.1f}  cohen_d={s['cohens_d']:5.2f}  t={s['t']:6.3f}  p={s['p_twosided_t']:.4f}  wilcoxon_p={s['wilcoxon_p']:.4f}")
print("\n==== 云端种子对（次级披露）====")
print(ablation["pooled_6units"]["both_eq"]["per_unit"][-1], cloud_pair_34)

print("\n==== 奖励头常数基线（标准化空间）====")
print(f"mu={mu:.3f} sigma={sigma:.3f}  z(+4)={z4:.4f}  z(-96)={zd:.3f}")
print(f"恒报均值: MAE={const_mean_mae:.4f}  MSE={const_mean_mse:.4f}")
print(f"恒报众数: MAE={const_mode_mae:.4f}")
print(f"模型: EQ={reward['model_mae']['equivariant']['mean']:.4f}  STD={reward['model_mae']['standard']['mean']:.4f}")

print("\n==== Exp2 ====")
print(f"想象@15k: mean={m_i:.1f}  CI95=[{lo_i:.1f},{hi_i:.1f}]   PPO@20k={m_20:.1f} @40k={m_40:.1f}  ratio=[{20/15:.2f},{40/15:.2f}]")
print(f"\nSaved: {os.path.join(DATA, 'ablation_pooled_v3.json')}")
