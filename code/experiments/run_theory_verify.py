"""
experiments/run_theory_verify.py
================================
实验四：理论验证 —— 等变世界模型的有效样本量 M_eff。

方法（bootstrap）：
  1. 收集一份固定数据集 D，分出固定测试集 D_test。
  2. 对 D 做 B 次 bootstrap 重采样（有放回）。
  3. 每次重采样分别训练小模型（等变 / 普通），在 D_test 上测预测 MSE。
  4. 比较两种模型测试误差的方差：
        M_eff / M ≈ Var(普通误差) / Var(等变误差)
     群大小 |G|=2（Z2），理论上限为 2。

输出：
  results/theory_verify.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_world_model import collect_data, train_world_model
from utils.meff_estimation import estimate_meff_ratio


@torch.no_grad()
def test_error(model, obs, act, nobs, device, batch=256):
    """在测试集上算单步 latent + 重建综合 MSE。"""
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


def run(env="health_gathering", n_steps=8000, n_test=1000, B=30,
        epochs=10, seed=0, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    print("[theory] collecting data ...")
    data = collect_data(env, n_steps=n_steps, seed=seed)
    # 划分训练 / 测试
    N = data["obs"].shape[0]
    perm = np.random.permutation(N)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    test = {k: v[test_idx] for k, v in data.items()}

    rng = np.random.default_rng(seed)
    eq_errs, std_errs = [], []
    t0 = time.time()
    for b in range(B):
        idx = rng.choice(train_idx, size=len(train_idx), replace=True)
        boot = {k: v[idx] for k, v in data.items()}
        for eq in (True, False):
            model = train_world_model(boot, equivariant=eq, epochs=epochs,
                                      device=device, log_every=1000)
            e = test_error(model, test["obs"], test["act"], test["next_obs"], device)
            (eq_errs if eq else std_errs).append(e)
        if (b + 1) % 5 == 0:
            print(f"  bootstrap {b+1}/{B}  ({time.time()-t0:.0f}s)")

    var_eq = float(np.var(eq_errs))
    var_std = float(np.var(std_errs))
    meff = estimate_meff_ratio(var_std, var_eq, group_size=2)
    out = {
        "B": B, "n_steps": n_steps, "epochs": epochs,
        "eq_errors_mean": float(np.mean(eq_errs)),
        "std_errors_mean": float(np.mean(std_errs)),
        "eq_error_var": var_eq,
        "std_error_var": var_std,
        "eq_errors": eq_errs,
        "std_errors": std_errs,
        "meff": meff,
        "note": "M_eff ratio = Var(std)/Var(eq), clipped to [1, |G|=2].",
    }
    os.makedirs("results", exist_ok=True)
    with open("results/theory_verify.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(meff, indent=2))
    print("Saved results/theory_verify.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="health_gathering")
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    run(B=args.B, epochs=args.epochs)
