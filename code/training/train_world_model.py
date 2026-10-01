"""
training/train_world_model.py
==============================
世界模型数据收集与训练。

数据收集：用随机策略（或给定策略）在 ViZDoom 上 rollout，
存储 (obs_t, action_t, reward_t, obs_{t+1}) 四元组。

训练：对 WorldModel（等变或普通）做监督回归。

CLI:
    python -m training.train_world_model --env health_gathering \
        --steps 10000 --epochs 50 --seed 0 --out results/wm.pt
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.vizdoom_env import ViZDoomEnv
from models.world_model import WorldModel


def collect_data(env_name: str = "health_gathering", n_steps: int = 10000,
                 seed: int = 0, policy=None, device="cpu") -> dict:
    """用随机策略（或给定策略）收集转移数据。

    Args:
        env_name: 场景名。
        n_steps: 收集的环境步数。
        seed: 随机种子。
        policy: 可选策略，调用 policy(obs)->logits；None 表示均匀随机。
        device: 策略推理设备。

    Returns:
        dict(obs, act, rew, next_obs) 均为 numpy 数组：
            obs: (N,3,64,64) float32
            act: (N, n_actions) float32 one-hot
            rew: (N,1) float32
            next_obs: (N,3,64,64) float32
    """
    env = ViZDoomEnv(env_name, frame_skip=4, seed=seed)
    n_actions = env.n_actions
    obs_l, act_l, rew_l, nobs_l = [], [], [], []
    obs = env.reset()
    step = 0
    while step < n_steps:
        if policy is None:
            a = np.random.randint(n_actions)
        else:
            with torch.no_grad():
                ot = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                logits, _ = policy(ot)
                a = int(torch.distributions.Categorical(logits=logits).sample().item())
        nobs, r, done, _ = env.step(a)
        obs_l.append(obs)
        act_oh = np.zeros(n_actions, dtype=np.float32)
        act_oh[a] = 1.0
        act_l.append(act_oh)
        rew_l.append([r])
        nobs_l.append(nobs)
        obs = nobs
        step += 1
        if done:
            obs = env.reset()
    env.close()
    return {
        "obs": np.stack(obs_l).astype(np.float32),
        "act": np.stack(act_l).astype(np.float32),
        "rew": np.stack(rew_l).astype(np.float32),
        "next_obs": np.stack(nobs_l).astype(np.float32),
    }


def make_loader(data: dict, batch_size: int = 256, shuffle=True):
    """简单 numpy -> torch DataLoader 生成器。"""
    n = data["obs"].shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield {k: torch.from_numpy(v[j]) for k, v in data.items()}


def train_world_model(data: dict, equivariant: bool = True, epochs: int = 50,
                      batch_size: int = 256, lr: float = 1e-3,
                      lambda_cons: float = 1.0, device: str = "cuda",
                      base_ch: int = 16, z_channels: int = 32,
                      log_every: int = 10) -> WorldModel:
    """训练世界模型。

    Returns:
        训练好的 WorldModel。
    """
    n_actions = data["act"].shape[1]
    # 奖励标准化（health_gathering 每步 ~4，死亡惩罚 -100*frame_skip，
    # 不标准化会让 reward MSE 压过 recon）
    r_mean = float(data["rew"].mean())
    r_std = float(data["rew"].std() + 1e-6)
    data = dict(data)
    data["rew_raw"] = data["rew"].copy()
    data["rew"] = (data["rew"] - r_mean) / r_std
    model = WorldModel(n_actions=n_actions, base_ch=base_ch, z_channels=z_channels,
                       equivariant=equivariant).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        tot = {}
        nb = 0
        for batch in make_loader(data, batch_size):
            obs = batch["obs"].to(device)
            act = batch["act"].to(device)
            rew = batch["rew"].to(device)
            nobs = batch["next_obs"].to(device)
            losses = model.compute_loss(obs, act, rew, nobs, lambda_cons=lambda_cons)
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            for k, v in losses.items():
                tot[k] = tot.get(k, 0.0) + float(v)
            nb += 1
        if (ep + 1) % log_every == 0 or ep == 0:
            msg = " ".join(f"{k}={tot[k]/nb:.4f}" for k in tot)
            print(f"  [{'EQ' if equivariant else 'STD'}] epoch {ep+1}/{epochs} {msg}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="health_gathering")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--equivariant", type=int, default=1)
    ap.add_argument("--out", default="results/wm.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"Collecting {args.steps} steps on {args.env} ...")
    data = collect_data(args.env, args.steps, seed=args.seed)
    print(f"Collected. obs {data['obs'].shape}  rew mean {data['rew'].mean():.3f}")
    model = train_world_model(data, equivariant=bool(args.equivariant),
                              epochs=args.epochs, device=device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "n_actions": data["act"].shape[1],
                "equivariant": bool(args.equivariant)}, args.out)
    print(f"Saved {args.out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
