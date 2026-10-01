"""
experiments/run_wm_comparison.py
================================
实验一：世界模型预测准确性对比（等变 vs 普通）。

流程：
  - 在 defend_the_center / health_gathering 上用随机策略收集 10k 步。
  - 分别训练等变世界模型与普通世界模型（同架构，50 epochs）。
  - 在测试集上评估：
      * 单步重建 PSNR / SSIM
      * 奖励预测 MAE
      * 开环 10/20/50 步预测帧 MSE
  - 多个种子取均值与 std。

输出：
  results/wm_comparison.json
  results/wm_comparison.png
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
from utils.metrics import psnr, ssim
from utils.bootstrap import bootstrap_ci


@torch.no_grad()
def evaluate_wm(model, data, device, horizons=(10, 20, 50), n_eval=200):
    """评估世界模型。

    Returns:
        dict(psnr, ssim, reward_mae, mse_h10, mse_h20, mse_h50)
    """
    model.eval()
    n_actions = data["act"].shape[1]
    idx = np.random.choice(data["obs"].shape[0], size=min(n_eval, data["obs"].shape[0]),
                           replace=False)
    obs = torch.from_numpy(data["obs"][idx]).float().to(device)
    nobs = torch.from_numpy(data["next_obs"][idx]).float().to(device)

    # 单步重建
    recon = model.decode(model.encode(obs)).cpu().numpy()
    obs_np = obs.cpu().numpy()
    psnrs, ssims = [], []
    for i in range(recon.shape[0]):
        psnrs.append(psnr(recon[i], obs_np[i]))
        ssims.append(ssim(recon[i], obs_np[i]))

    # 奖励 MAE（训练时奖励被标准化，这里在标准化空间比较）
    r_mean = float(data["rew"].mean())
    r_std = float(data["rew"].std() + 1e-6)
    r_pred = model.predict_reward(model.encode(obs)).cpu().numpy().reshape(-1)
    r_true = (data["rew"][idx].reshape(-1) - r_mean) / r_std
    r_mae = float(np.mean(np.abs(r_pred - r_true)))

    # 开环多步预测 MSE：从 obs[idx[0]] 出发，沿数据序列逐帧预测并对比真值
    h = model.encode(obs[:1])
    act = torch.from_numpy(data["act"][idx]).float().to(device)
    mse_h = {}
    hh = model.encode(obs[:1])
    start = idx[0]
    for step in range(1, max(horizons) + 1):
        a_oh = act[step - 1:step]
        hh = model.step(hh, a_oh)
        if start + step < data["obs"].shape[0]:
            pred_frame = model.decode(hh).cpu().numpy()[0]
            gt = data["obs"][start + step]
            mse_h[step] = float(np.mean((pred_frame - gt) ** 2))
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "reward_mae": r_mae,
        "mse_h10": mse_h.get(10, float("nan")),
        "mse_h20": mse_h.get(20, float("nan")),
        "mse_h50": mse_h.get(50, float("nan")),
    }


def run(env="health_gathering", steps=10000, epochs=50, seeds=(0, 1, 2),
        device="cuda", out="results/wm_comparison.json"):
    device = device if torch.cuda.is_available() else "cpu"
    results = {"equivariant": [], "standard": []}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        print(f"\n=== seed {seed} ===")
        data = collect_data(env, n_steps=steps, seed=seed)
        for eq in (True, False):
            model = train_world_model(data, equivariant=eq, epochs=epochs,
                                      device=device, log_every=epochs)
            m = evaluate_wm(model, data, device)
            key = "equivariant" if eq else "standard"
            results[key].append(m)
            print(f"  {key:11s}: " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
        print(f"  seed {seed} done in {time.time()-t0:.1f}s")

    # 聚合
    summary = {}
    for key in ("equivariant", "standard"):
        keys = results[key][0].keys()
        summary[key] = {k: {
            "mean": float(np.mean([r[k] for r in results[key]])),
            "std": float(np.std([r[k] for r in results[key]])),
        } for k in keys}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"per_seed": results, "summary": summary}, f, indent=2)
    print("\nSaved", out)

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        metrics = ["psnr", "ssim", "reward_mae", "mse_h10", "mse_h20", "mse_h50"]
        labels = ["PSNR(dB)", "SSIM", "RewardMAE", "MSE@10", "MSE@20", "MSE@50"]
        eq_means = [summary["equivariant"][m]["mean"] for m in metrics]
        std_means = [summary["standard"][m]["mean"] for m in metrics]
        x = np.arange(len(metrics)); w = 0.35
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, met, lab, em, sm in zip(axes.flat, metrics, labels, eq_means, std_means):
            es = summary["equivariant"][met]["std"]
            ss = summary["standard"][met]["std"]
            ax.bar([0, 1], [em, sm], yerr=[es, ss], color=["#2a9d8f", "#e76f51"],
                   capsize=4)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Eq-WM", "Std-WM"])
            ax.set_title(lab)
        fig.suptitle(f"World Model Prediction Accuracy ({env}, {steps} steps, {len(seeds)} seeds)")
        fig.tight_layout()
        png = out.replace(".json", ".png")
        fig.savefig(png, dpi=120)
        print("Saved", png)
    except Exception as e:
        print("plot failed:", e)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="health_gathering")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    run(env=args.env, steps=args.steps, epochs=args.epochs, seeds=tuple(args.seeds))
