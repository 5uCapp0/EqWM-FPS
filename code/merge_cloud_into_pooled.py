# -*- coding: utf-8 -*-
"""
merge_cloud_into_pooled.py  (夜间值守，只增不改主表)
加载 data/ablation_pooled_v3.json，追加云盘 cross-check 单位（s9/s10 优云逐种子）
与 provenance。primary_local5 / paired_local5 / pooled_6units 保持逐字节不变。
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
POOLED = os.path.join(DATA, "ablation_pooled_v3.json")

with open(POOLED, "r", encoding="utf-8") as f:
    S = json.load(f)

# 云盘逐种子（优云 RTX4090，来自 data/cloud_pulls/ablation_s9s10_youyun_cloud.json）
cloud_seeds_910 = {
    9:  {"both_eq": 385.333, "wm_eq_only": 416.0,   "policy_eq_only": 422.667, "none_eq": 424.0},
    10: {"both_eq": 457.333, "wm_eq_only": 460.0,   "policy_eq_only": 490.333, "none_eq": 501.333},
}

units = {}
both_none_diffs = []
for s in sorted(cloud_seeds_910):
    v = cloud_seeds_910[s]
    diff = round(v["both_eq"] - v["none_eq"], 1)
    both_none_diffs.append(diff)
    units[f"seed{s}_youyun_RTX4090"] = {
        "platform": "优云智算 RTX4090 24G",
        "source_file": "data/cloud_pulls/ablation_s9s10_youyun_cloud.json",
        "both_eq": round(v["both_eq"], 1),
        "wm_eq_only": round(v["wm_eq_only"], 1),
        "policy_eq_only": round(v["policy_eq_only"], 1),
        "none_eq": round(v["none_eq"], 1),
        "both_minus_none": diff,
    }

cloud_crosscheck = {
    "units": units,
    "pair_34_A10_pooled": S["ablation"]["_provenance"].get("cloud_pair_34", ""),
    "note": ("云盘单位作为跨机器异质性披露，不并入本机主表 primary_local5。"
             "s9/s10 有逐种子明细；3/4 为 A10 合并单位。云盘 both-none 均值="
             f"{round(sum(both_none_diffs)/len(both_none_diffs),1)}（{both_none_diffs}），"
             "与本机 both-none=+91.2 方向相反 → 跨机器不稳定，结论仍为无显著返回增益。"),
}

S["ablation"]["cloud_crosscheck"] = cloud_crosscheck

# provenance 追加（不删旧键）
S["ablation"]["_provenance"]["cloud_seeds_910"] = (
    "data/cloud_pulls/ablation_s9s10_youyun_cloud.json robust_mean 逐种子（优云 RTX4090）"
)
S["ablation"]["_provenance"]["merge_log"] = (
    "夜间值守合并：追加云盘 s9/s10 cross-check；本机主表 primary_local5 未变（待本机 s15/s16）。"
)

with open(POOLED, "w", encoding="utf-8") as f:
    json.dump(S, f, ensure_ascii=False, indent=2)

# 校验主表未变
print("primary_local5 both_eq mean =", S["ablation"]["primary_local5"]["both_eq"]["mean"])
print("paired_local5 both-none mean_diff =", S["ablation"]["paired_local5"]["both_eq - none_eq"]["mean_diff"])
print("cloud_crosscheck units:", list(units.keys()))
print("both-none diffs (cloud):", both_none_diffs)
print("OK merged ->", POOLED)
