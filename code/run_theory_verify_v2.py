"""
code/run_theory_verify_v2.py
============================
实验四：M_eff bootstrap 缩放实验（充分收敛后重测）。

背景（eqwm_fps/results/theory_verify.json）：
  B=15, epochs=5 时 Var(std)/Var(eq)=0.39（clip 1.0），即等变模型误差方差反而更大，
  M_eff>1 未验证出。教材 ch14 的 M_eff≈15.7 与实测矛盾——本脚本不再预设结论，
  而是沿训练量 epochs 做缩放，观察 Var(std)/Var(eq) 是否随收敛翻转为 >1。

方法：
  1. 固定收集一份数据 D（seed=0, n_steps=8000），切出固定测试集 D_test（n_test=1000）。
  2. 对每个 epochs 水平 e ∈ {5,20,40}：
       - 对训练集做 B 次 bootstrap 有放回重采样；
       - 每次分别训练 等变 / 普通 WM（epochs=e）；
       - 在 D_test 上测综合误差 e = MSE(latent_dyn) + MSE(recon)；
       - 得 eq_errs[e] (B个), std_errs[e] (B个)。
  3. ratio(e) = Var(std_errs[e]) / Var(eq_errs[e])，clip 到 [1, |G|=2]。
  4. 画 ratio-vs-epochs 曲线，判断是否随训练量进入 >1。

输出：
  data/theory_verify_v2.json
  figs/theory_verify_v2.png
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# Repo-relative paths: works on any machine after `git clone`.
HERE = os.path.dirname(os.path.abspath(__file__))
EQWM = HERE          # code/ mirrors the eqwm_fps package (training/, utils/, models/, env/)
PAPER = os.path.dirname(HERE)
sys.path.insert(0, EQWM)
os.chdir(EQWM)

from training.train_world_model import collect_data, train_world_model
from utils.meff_estimation import estimate_meff_ratio


@torch.no_grad()
def test_error(model, obs, act, nobs, device, batch=256):
    """与 run_theory_verify.test_error 相同：latent 动力学 + 重建综合 MSE。"""
    model.eval()
    errs = []
    n = obs.shape[0]
    for i in range(0, n, batch):
        o = torch.from_numpy(obs[i:i+batch]).float().to(device)
        a = torch.from_numpy(act[i:i+batch]).float().to(device)
        no = torch.from_numpy(nobs[i:i+batch]).float().to(device)
        h = model.encode(o)
        hn = model.encode(no)
        hn_pred = model.step(h, a)
        recon = model.decode(h)
        e = float(((hn_pred - hn) ** 2).mean() + ((recon - o) ** 2).mean())
        errs.append(e)
    return float(np.mean(errs))


def run(n_steps=8000, n_test=1000, B=15, epoch_grid=(5, 20, 40), seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    print("[theory_v2] collecting fixed data ...", flush=True)
    data = collect_data("health_gathering", n_steps=n_steps, seed=seed)
    N = data["obs"].shape[0]
    perm = np.random.RandomState(seed).permutation(N)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    test = {k: v[test_idx] for k, v in data.items()}

    out = {"n_steps": n_steps, "n_test": n_test, "B": B,
           "epoch_grid": list(epoch_grid), "levels": []}
    t_start = time.time()
    for e in epoch_grid:
        rng = np.random.default_rng(seed + e)
        eq_errs, std_errs = [], []
        t0 = time.time()
        for b in range(B):
            idx = rng.choice(train_idx, size=len(train_idx), replace=True)
            boot = {k: v[idx] for k, v in data.items()}
            for eq in (True, False):
                model = train_world_model(boot, equivariant=eq, epochs=e,
                                          device=device, log_every=100000)
                er = test_error(model, test["obs"], test["act"],
                                test["next_obs"], device)
                (eq_errs if eq else std_errs).append(er)
        var_eq = float(np.var(eq_errs))
        var_std = float(np.var(std_errs))
        meff = estimate_meff_ratio(var_std, var_eq, group_size=2)
        rec = {
            "epochs": e,
            "eq_err_mean": float(np.mean(eq_errs)),
            "std_err_mean": float(np.mean(std_errs)),
            "eq_err_var": var_eq,
            "std_err_var": var_std,
            "ratio": meff["ratio"],
            "clipped_ratio": meff["clipped_ratio"],
            "eq_errs": eq_errs,
            "std_errs": std_errs,
            "secs": round(time.time() - t0, 1),
        }
        out["levels"].append(rec)
        print(f"[theory_v2] epochs={e:3d}  "
              f"eq_var={var_eq:.3e} std_var={var_std:.3e}  "
              f"ratio={meff['ratio']:.3f} (clip {meff['clipped_ratio']:.2f})  "
              f"({rec['secs']:.0f}s)", flush=True)
        _dump(out)
    out["total_secs"] = round(time.time() - t_start, 1)
    _dump(out)
    _plot(out)
    print("[theory_v2] DONE", flush=True)


def _dump(out):
    os.makedirs(os.path.join(PAPER, "data"), exist_ok=True)
    with open(os.path.join(PAPER, "data", "theory_verify_v2.json"), "w") as f:
        json.dump(out, f, indent=2)


def _plot(out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        es = [l["epochs"] for l in out["levels"]]
        ratios = [l["ratio"] for l in out["levels"]]
        clips = [l["clipped_ratio"] for l in out["levels"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(es, ratios, "-o", color="#2a9d8f", label="raw ratio = Var(std)/Var(eq)")
        ax.plot(es, clips, "--s", color="#e76f51", label="clipped to [1, |G|=2]")
        ax.axhline(1.0, color="gray", ls=":", label="M_eff=1 (no gain)")
        ax.axhline(2.0, color="#e9c46a", ls=":", label="|G|=2 upper bound")
        ax.set_xlabel("WM training epochs (per bootstrap)")
        ax.set_ylabel("Var(std error) / Var(eq error)")
        ax.set_title("Exp4: M_eff bootstrap ratio vs training amount (B=%d)" % out["B"])
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.join(PAPER, "figs"), exist_ok=True)
        fig.savefig(os.path.join(PAPER, "figs", "theory_verify_v2.png"), dpi=120)
        print("Saved figs/theory_verify_v2.png", flush=True)
    except Exception as e:
        print("plot failed:", e, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=15)
    ap.add_argument("--epoch_grid", type=int, nargs="+", default=[5, 20, 40])
    args = ap.parse_args()
    run(B=args.B, epoch_grid=tuple(args.epoch_grid))
