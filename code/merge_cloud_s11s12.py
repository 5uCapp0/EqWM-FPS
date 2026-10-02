# -*- coding: utf-8 -*-
"""追加 s11/s12 真实云盘数据到 ablation_pooled_v3.json 的 cloud_crosscheck（只增不改主表）。"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOLED = os.path.join(ROOT, "data", "ablation_pooled_v3.json")
with open(POOLED, "r", encoding="utf-8") as f:
    S = json.load(f)

cloud_seeds_1112 = {
    11: {"both_eq": 482.667, "wm_eq_only": 430.667, "policy_eq_only": 472.0, "none_eq": 456.0},
    12: {"both_eq": 401.333, "wm_eq_only": 418.667, "policy_eq_only": 428.0, "none_eq": 385.333},
}
cc = S["ablation"]["cloud_crosscheck"]
diffs = []
for s in sorted(cloud_seeds_1112):
    v = cloud_seeds_1112[s]
    diff = round(v["both_eq"] - v["none_eq"], 1)
    diffs.append(diff)
    cc["units"][f"seed{s}_youyun_RTX4090"] = {
        "platform": "优云智算 RTX4090 24G",
        "source_file": "data/cloud_pulls/ablation_s11s12_youyun_cloud.json",
        "both_eq": round(v["both_eq"], 1),
        "wm_eq_only": round(v["wm_eq_only"], 1),
        "policy_eq_only": round(v["policy_eq_only"], 1),
        "none_eq": round(v["none_eq"], 1),
        "both_minus_none": diff,
    }
cc["note"] = (
    "云盘单位作为跨机器异质性披露，不并入本机主表 primary_local5。"
    "s9/s10、s11/s12 有逐种子明细；3/4 为 A10 合并单位。"
    "s9/s10 both-none=[-38.7,-44.0]（both<none）；s11/s12 both-none="
    f"{diffs}（both 略高）。云盘方向不一致、幅度小，与本机 +91.2 共同支持"
    "'跨机器不稳定、无稳健返回增益'。"
)
S["ablation"]["_provenance"]["cloud_seeds_1112"] = (
    "data/cloud_pulls/ablation_s11s12_youyun_cloud.json robust_mean 逐种子（优云 RTX4090；"
    "曾因浏览器缓存误得 s9/s10 副本，03:39 已重拉为真实数据，SHA256 已与 s9s10 不同）"
)
with open(POOLED, "w", encoding="utf-8") as f:
    json.dump(S, f, ensure_ascii=False, indent=2)

print("primary both_eq mean =", S["ablation"]["primary_local5"]["both_eq"]["mean"])
print("primary both-none diff =", S["ablation"]["paired_local5"]["both_eq - none_eq"]["mean_diff"])
print("cloud units:", list(cc["units"].keys()))
print("s11/s12 both-none:", diffs)
print("OK")
