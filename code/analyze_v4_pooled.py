# -*- coding: utf-8 -*-
"""
analyze_v4_pooled.py  (夜间值守 · 本机 7 种子主表重算)
在 v3 方法上扩展：本机同机种子 0,1,2,13,14,15,16 (n=7) 作为主表 primary_local7，
重算逐配置 mean/std/95%CI 与配对统计；云盘单位(s9-s12 + A10对3/4 + 优云15/16)
仅作 cloud_crosscheck，不进主表。reward_head / exp2 块沿用现有 json（不变）。
"""
import json, math, os
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
POOLED = os.path.join(DATA, "ablation_pooled_v3.json")

# ---- 本机同机 (RTX3060) 逐种子 robust mean ----
local_seeds = {
    0:  {"both_eq": 490.7,   "wm_eq_only": 401.3, "policy_eq_only": 454.7, "none_eq": 402.7},
    1:  {"both_eq": 398.7,   "wm_eq_only": 409.3, "policy_eq_only": 390.7, "none_eq": 513.3},
    2:  {"both_eq": 452.0,   "wm_eq_only": 465.3, "policy_eq_only": 434.7, "none_eq": 433.3},
    13: {"both_eq": 577.333, "wm_eq_only": 461.333, "policy_eq_only": 402.667, "none_eq": 392.0},
    14: {"both_eq": 682.667, "wm_eq_only": 414.667, "policy_eq_only": 424.0,  "none_eq": 404.0},
    15: {"both_eq": 377.333, "wm_eq_only": 473.333, "policy_eq_only": 478.333, "none_eq": 465.333},
    16: {"both_eq": 408.0,   "wm_eq_only": 578.667, "policy_eq_only": 446.667, "none_eq": 445.333},
}
# ---- 云盘 cross-check 单位（不进主表）----
cloud_units = {
    "pair_34_A10":       {"both_eq": 439.333, "wm_eq_only": 378.667, "policy_eq_only": 440.667, "none_eq": 403.333, "note": "A10 两合并并为1单位(无逐种子)"},
    "seed9_youyun":      {"both_eq": 385.333, "wm_eq_only": 416.0,   "policy_eq_only": 422.667, "none_eq": 424.0},
    "seed10_youyun":     {"both_eq": 457.333, "wm_eq_only": 460.0,   "policy_eq_only": 490.333, "none_eq": 501.333},
    "seed11_youyun":     {"both_eq": 482.667, "wm_eq_only": 430.667, "policy_eq_only": 472.0,   "none_eq": 456.0},
    "seed12_youyun":     {"both_eq": 401.333, "wm_eq_only": 418.667, "policy_eq_only": 428.0,   "none_eq": 385.333},
    "seed15_youyun":     {"both_eq": 428.0,   "wm_eq_only": 418.667, "policy_eq_only": 436.0,   "none_eq": 510.667, "note": "优云并行 s15/16，与本机 s15/16 独立样本"},
    "seed16_youyun":     {"both_eq": 429.333, "wm_eq_only": 350.667, "policy_eq_only": 394.667, "none_eq": 420.0,   "note": "优云并行 s15/16，与本机 s15/16 独立样本"},
}
CONFIGS = ["both_eq", "wm_eq_only", "policy_eq_only", "none_eq"]
PAIRS = [("both_eq","none_eq","both_eq - none_eq"),
         ("wm_eq_only","none_eq","wm_eq_only - none_eq"),
         ("policy_eq_only","none_eq","policy_eq_only - none_eq"),
         ("both_eq","wm_eq_only","both_eq - wm_eq_only"),
         ("both_eq","policy_eq_only","both_eq - policy_eq_only")]

def t_ci95(xs):
    n=len(xs); m=sum(xs)/n
    if n<2: return m,m,m
    s=math.sqrt(sum((v-m)**2 for v in xs)/(n-1)); tcrit=stats.t.ppf(0.975,n-1); se=s/math.sqrt(n)
    return m, m-tcrit*se, m+tcrit*se

def summarize(vals,label):
    n=len(vals); m,lo,hi=t_ci95(vals)
    sd=math.sqrt(sum((v-m)**2 for v in vals)/(n-1)) if n>1 else 0.0
    return {"label":label,"n":n,"mean":round(m,1),"std":round(sd,1),
            "ci95_lo":round(lo,1),"ci95_hi":round(hi,1),
            "min":round(min(vals),1),"max":round(max(vals),1),"per_unit":[round(v,1) for v in vals]}

def paired(a,b,label):
    d=[x-y for x,y in zip(a,b)]; n=len(d); md=sum(d)/n
    sd=math.sqrt(sum((v-md)**2 for v in d)/(n-1)) if n>1 else 0.0
    dc=md/sd if sd>0 else 0.0
    tt,pt=stats.ttest_rel(a,b)
    try: _,pw=stats.wilcoxon(a,b)
    except ValueError: pw=1.0
    return {"label":label,"n":n,"mean_diff":round(md,1),"sd_diff":round(sd,1),
            "cohens_d":round(dc,2),"t":round(tt,3),"p_twosided_t":round(pt,4),
            "wilcoxon_p":round(pw,4),"per_unit_diff":[round(v,1) for v in d]}

seeds=sorted(local_seeds)
primary={}
for cfg in CONFIGS:
    primary[cfg]=summarize([local_seeds[s][cfg] for s in seeds],cfg)
paired_stats={}
for a,b,lab in PAIRS:
    paired_stats[lab]=paired([local_seeds[s][a] for s in seeds],
                       [local_seeds[s][b] for s in seeds],lab)

# cloud cross-check both-none per unit
cloud_cc={"units":{},"both_minus_none":{}}
for name,v in cloud_units.items():
    cloud_cc["units"][name]={k:round(v[k],1) for k in CONFIGS}
    if "note" in v: cloud_cc["units"][name]["note"]=v["note"]
    cloud_cc["both_minus_none"][name]=round(v["both_eq"]-v["none_eq"],1)

# load existing to preserve reward_head/exp2
with open(POOLED,"r",encoding="utf-8") as f: S=json.load(f)
old5=S["ablation"].get("primary_local5"); oldp5=S["ablation"].get("paired_local5")

S["ablation"]={
    "primary_local7": primary,
    "paired_local7": paired_stats,
    "_previous_local5_archive": {"primary_local5": old5, "paired_local5": oldp5},
    "cloud_crosscheck": cloud_cc,
    "_provenance": {
        "primary_local7": "本机 RTX3060 7 种子 0,1,2,13,14,15,16；15/16 来自 data/ablation_s15.json / ablation_s16.json 及 code/logs robust 行",
        "cloud_crosscheck": "云盘异质性披露，不进主表；s9-s12 来自 data/cloud_pulls/ablation_s{9,10,11,12}_youyun_cloud.json；3/4=A10合并对；优云 s15/s16=ablation_s15s16_youyun_cloud.json（与本机15/16并行独立样本）",
        "note": "v4 扩展：主表由 5→7 本机种子；云盘仅 cross-check。",
    },
}
with open(POOLED,"w",encoding="utf-8") as f:
    json.dump(S,f,ensure_ascii=False,indent=2)

print("==== primary_local7 (n=%d) ===="%len(seeds))
for cfg in CONFIGS:
    s=primary[cfg]; print(f"{cfg:14s} {s['mean']:7.1f} +/- {s['std']:6.1f}  CI[{s['ci95_lo']},{s['ci95_hi']}]  per={s['per_unit']}")
print("\n==== paired_local7 ====")
for lab,s in paired_stats.items():
    print(f"{lab:26s} d={s['mean_diff']:7.1f} cohen={s['cohens_d']:5.2f} t={s['t']:6.3f} p={s['p_twosided_t']:.4f} wilcox={s['wilcoxon_p']:.4f}")
print("\ncloud both-none:", cloud_cc["both_minus_none"])
print("OK ->", POOLED)
