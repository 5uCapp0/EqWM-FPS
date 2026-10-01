"""
code/run_ablation_v2.py
=======================
实验三：等变部件消融（多种子 + 稳健评估）。

与 eqwm_fps/experiments/run_ablation.py 同样的四配置：
  both_eq         : WM 等变 + 策略等变
  wm_eq_only      : WM 等变 + 策略普通
  policy_eq_only  : WM 普通 + 策略等变
  none_eq         : 都不等变（基线）

与原版 run_ablation.py 的差异（为降低评估噪声）：
  1. 忠实复制 training/imagination_ppo.py:run_imagination 的训练超参，
     但额外返回最终策略 policy（原版只返回曲线）。
  2. 最终回报用「多 eval-seed × 多 episode」稳健估计（默认 3 seeds × 8 episodes），
     而非原版 evaluate_real 的单 seed × 2 episodes。
  3. 同时记录原版 curve[-1][1]（单 seed 2 episode）作为对照。

输出（绝对路径，写到论文项目 data/）：
  data/ablation.json
  figs/ablation.png
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
os.chdir(EQWM)  # vizdoom 场景用绝对路径；chdir 仅为兼容库内相对输出

from training.train_world_model import collect_data, train_world_model
from models.policy import EquivPolicy, StandardPolicy
from training.imagination_ppo import imagine_rollouts, ppo_update
from env.vizdoom_env import ViZDoomEnv


CONFIGS = [
    ("both_eq", True, True),
    ("wm_eq_only", True, False),
    ("policy_eq_only", False, True),
    ("none_eq", False, False),
]


def run_imagination_ret(env_name="health_gathering", init_steps=10000,
                        outer_iters=6, real_per_iter=1000, wm_epochs_init=30,
                        wm_epochs_finetune=5, imagine_per_iter=200, horizon=20,
                        seed=0, device="cuda", wm_equivariant=True, policy_equivariant=True):
    """与 imagination_ppo.run_imagination 同超参；返回 (curve, policy, wm)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device if torch.cuda.is_available() else "cpu"

    data = collect_data(env_name, init_steps, seed=seed)
    n_actions = data["act"].shape[1]
    wm = train_world_model(data, equivariant=wm_equivariant,
                           epochs=wm_epochs_init, device=device)
    policy = (EquivPolicy(n_actions) if policy_equivariant
              else StandardPolicy(n_actions)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

    curve = []
    r0 = evaluate_real_once(env_name, policy, device, episodes=2, eval_seed=12345)
    curve.append((0, r0))

    replay_obs = data["obs"]
    real_total = init_steps
    for it in range(outer_iters):
        new_data = collect_data(env_name, real_per_iter, seed=seed + it,
                                policy=policy, device=device)
        real_total += real_per_iter
        replay_obs = np.concatenate([replay_obs, new_data["obs"]], axis=0)
        wm = train_world_model(new_data, equivariant=wm_equivariant,
                               epochs=wm_epochs_finetune, device=device, log_every=1000)
        idx = np.random.choice(replay_obs.shape[0],
                               size=min(imagine_per_iter, replay_obs.shape[0]),
                               replace=False)
        init_obs = [replay_obs[i] for i in idx]
        o_l, a_l, lp_l, r_l, ad_l = imagine_rollouts(
            wm, policy, init_obs, horizon, device, n_actions)
        ppo_update(policy, o_l, a_l, lp_l, r_l, ad_l, opt, device=device)
        ev = evaluate_real_once(env_name, policy, device, episodes=2, eval_seed=12345)
        curve.append((real_total, ev))
        print(f"    [ablation] real_steps={real_total} eval(2ep)={ev:.1f}", flush=True)
    return curve, policy, wm


@torch.no_grad()
def evaluate_real_once(env_name, policy, device, episodes=2, eval_seed=12345):
    """与 imagination_ppo.evaluate_real 相同，但可指定 eval_seed / episodes。"""
    env = ViZDoomEnv(env_name, frame_skip=4, seed=eval_seed)
    totals = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        tot = 0.0
        while not done:
            ot = torch.from_numpy(obs).unsqueeze(0).float().to(device)
            logits, _ = policy(ot)
            a = int(torch.distributions.Categorical(logits=logits).sample().item())
            obs, r, done, _ = env.step(a)
            tot += r
        totals.append(tot)
    env.close()
    return float(np.mean(totals))


@torch.no_grad()
def robust_eval(policy, device, eval_seeds=(12345, 222, 333), episodes_per_seed=8):
    """多 eval-seed × 多 episode 稳健估计最终回报。返回 (mean, std, all)。"""
    all_returns = []
    for es in eval_seeds:
        env = ViZDoomEnv("health_gathering", frame_skip=4, seed=es)
        for _ in range(episodes_per_seed):
            obs = env.reset()
            done = False
            tot = 0.0
            while not done:
                ot = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                logits, _ = policy(ot)
                a = int(torch.distributions.Categorical(logits=logits).sample().item())
                obs, r, done, _ = env.step(a)
                tot += r
            all_returns.append(tot)
        env.close()
    all_returns = np.array(all_returns, dtype=np.float64)
    return float(all_returns.mean()), float(all_returns.std()), all_returns.tolist()


def run(seeds=(0, 1, 2), init_steps=10000, outer_iters=6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {name: {"curve2ep": [], "robust_mean": [], "robust_std": [],
                      "robust_all": []} for name, _, _ in CONFIGS}
    for seed in seeds:
        for name, wm_eq, pol_eq in CONFIGS:
            print(f"\n===== ablation seed={seed} cfg={name} "
                  f"(wm_eq={wm_eq}, pol_eq={pol_eq}) =====", flush=True)
            t0 = time.time()
            curve, policy, wm = run_imagination_ret(
                env_name="health_gathering", init_steps=init_steps,
                outer_iters=outer_iters, seed=seed, device=device,
                wm_equivariant=wm_eq, policy_equivariant=pol_eq)
            # 原版口径：curve 最后一点（单 eval seed × 2 episode）
            final_2ep = curve[-1][1]
            # 稳健口径：多 seed × 多 episode
            rm, rs, rall = robust_eval(policy, device)
            results[name]["curve2ep"].append(final_2ep)
            results[name]["robust_mean"].append(rm)
            results[name]["robust_std"].append(rs)
            results[name]["robust_all"].append(rall)
            print(f"  {name}: final(2ep)={final_2ep:.1f}  "
                  f"robust={rm:.1f}±{rs:.1f}  ({time.time()-t0:.0f}s)", flush=True)
            # 及时落盘（防中断丢数据）
            _dump(results)

    summary = {}
    for name in results:
        rm = np.array(results[name]["robust_mean"])
        c2 = np.array(results[name]["curve2ep"])
        summary[name] = {
            "robust_mean_over_seeds": float(rm.mean()),
            "robust_std_over_seeds": float(rm.std()),
            "curve2ep_mean": float(c2.mean()),
            "curve2ep_std": float(c2.std()),
        }
    results["_summary"] = summary
    _dump(results)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    _plot(summary)


def _dump(results):
    os.makedirs(os.path.join(PAPER, "data"), exist_ok=True)
    with open(os.path.join(PAPER, "data", "ablation.json"), "w") as f:
        json.dump(results, f, indent=2)


def _plot(summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [n for n in summary]
        means = [summary[n]["robust_mean_over_seeds"] for n in names]
        stds = [summary[n]["robust_std_over_seeds"] for n in names]
        cmeans = [summary[n]["curve2ep_mean"] for n in names]
        cstds = [summary[n]["curve2ep_std"] for n in names]
        x = np.arange(len(names)); w = 0.35
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(x - w/2, means, w, yerr=stds, capsize=4,
               color="#2a9d8f", label="Robust eval (3 eval-seed x 8 ep)")
        ax.bar(x + w/2, cmeans, w, yerr=cstds, capsize=4,
               color="#e9c46a", label="Original curve[-1] (1 seed x 2 ep)")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=12)
        ax.set_ylabel("Final episode return")
        ax.set_title("Exp3 Ablation: which parts should be equivariant? (health_gathering)")
        ax.legend()
        fig.tight_layout()
        os.makedirs(os.path.join(PAPER, "figs"), exist_ok=True)
        fig.savefig(os.path.join(PAPER, "figs", "ablation.png"), dpi=120)
        print("Saved figs/ablation.png")
    except Exception as e:
        print("plot failed:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--init_steps", type=int, default=10000)
    ap.add_argument("--outer_iters", type=int, default=6)
    args = ap.parse_args()
    run(seeds=tuple(args.seeds), init_steps=args.init_steps,
        outer_iters=args.outer_iters)
