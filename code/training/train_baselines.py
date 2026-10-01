"""
training/train_baselines.py
===========================
纯真实环境 PPO 基线（不使用世界模型/想象）。

直接在 ViZDoom 上 rollout + PPO 更新，记录累计真实步数 vs 评估回报，
作为少样本想象训练的对照。
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.vizdoom_env import ViZDoomEnv
from models.policy import EquivPolicy, StandardPolicy
from training.imagination_ppo import ppo_update, compute_gae, evaluate_real


def run_real_ppo(env_name="health_gathering", total_steps=50000, rollout_steps=1000,
                 n_actions=3, equivariant=False, seed=0, device="cuda",
                 eval_every=10000):
    """纯真实 PPO。

    Returns:
        curve: [(real_steps, eval_return), ...]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device if torch.cuda.is_available() else "cpu"

    env = ViZDoomEnv(env_name, frame_skip=4, seed=seed)
    policy = (EquivPolicy(n_actions) if equivariant else StandardPolicy(n_actions)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

    obs = env.reset()
    curve = []
    obs_b, act_b, logp_b, rew_b, val_b = [], [], [], [], []
    steps = 0
    while steps < total_steps:
        # 收集一段 rollout
        for _ in range(rollout_steps):
            ot = torch.from_numpy(obs).unsqueeze(0).float().to(device)
            logits, value = policy(ot)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = float(dist.log_prob(a).item())
            a_i = int(a.item())
            nobs, r, done, _ = env.step(a_i)
            obs_b.append(obs); act_b.append(a_i); logp_b.append(logp); rew_b.append(r); val_b.append(float(value.item()))
            obs = nobs
            steps += 1
            if done:
                obs = env.reset()
        # PPO 更新
        rets, advs = compute_gae(rew_b, val_b)
        ppo_update(policy, obs_b, act_b, logp_b, rets.tolist(), advs.tolist(), opt, device=device)
        obs_b.clear(); act_b.clear(); logp_b.clear(); rew_b.clear(); val_b.clear()
        if steps % eval_every == 0 or steps >= total_steps:
            ev = evaluate_real(env_name, policy, device)
            curve.append((steps, ev))
            print(f"[real-ppo] real_steps={steps}  eval_return={ev:.2f}")
    env.close()
    return curve


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="health_gathering")
    ap.add_argument("--total_steps", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--equivariant", type=int, default=0)
    args = ap.parse_args()
    curve = run_real_ppo(env_name=args.env, total_steps=args.total_steps, seed=args.seed,
                         equivariant=bool(args.equivariant))
    print("curve:", curve)
