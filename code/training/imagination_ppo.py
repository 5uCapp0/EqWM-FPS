"""
training/imagination_ppo.py
===========================
想象训练 PPO（Dreamer 风格的简化实现）。

流程：
  1. 用随机策略收集初始真实数据，训练世界模型。
  2. 外层循环：
     a. 用当前策略在真实环境收集少量新数据（真实步数计数）。
     b. 在回放池上微调世界模型。
     c. 在世界模型中 "想象" rollout：从真实 obs 编码出发，用策略选动作，
        由 WM 预测下一隐状态与奖励。
     d. 在想象数据上做 PPO 更新。
     e. 周期性在真实环境评估平均回报。

输出样本效率曲线（真实步数 vs 评估回报）。

参考：DreamerV3 (arXiv:2301.04104)；EMERALD (arXiv:2507.04075)。
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.vizdoom_env import ViZDoomEnv
from models.world_model import WorldModel
from models.policy import EquivPolicy, StandardPolicy
from training.train_world_model import collect_data, train_world_model


def select_action(policy, obs, device):
    """根据策略采样动作，返回 (action_idx, log_prob, value)。"""
    ot = torch.from_numpy(obs).unsqueeze(0).float().to(device)
    logits, value = policy(ot)
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return int(a.item()), float(dist.log_prob(a).item()), float(value.item())


def ppo_update(policy, obs_list, act_list, old_logp_list, ret_list, adv_list,
               optimizer, n_epochs=4, clip=0.2, device="cuda"):
    """在一批 (obs, act, logp, return, advantage) 上做 PPO 更新。"""
    obs_t = torch.from_numpy(np.stack(obs_list)).float().to(device)
    act_t = torch.tensor(act_list, dtype=torch.long, device=device)
    old_logp = torch.tensor(old_logp_list, dtype=torch.float32, device=device)
    ret_t = torch.tensor(ret_list, dtype=torch.float32, device=device)
    adv_t = torch.tensor(adv_list, dtype=torch.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    bs = 256
    idx = np.arange(len(obs_list))
    for _ in range(n_epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            logits, value = policy(obs_t[j])
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(act_t[j])
            ratio = torch.exp(new_logp - old_logp[j])
            s1 = ratio * adv_t[j]
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[j]
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = F.mse_loss(value.squeeze(-1), ret_t[j])
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """计算折扣回报与 GAE 优势。"""
    n = len(rewards)
    rets = np.zeros(n, dtype=np.float32)
    advs = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in reversed(range(n)):
        next_val = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        last = delta + gamma * lam * last
        advs[t] = last
        rets[t] = advs[t] + values[t]
    return rets, advs


@torch.no_grad()
def imagine_rollouts(wm, policy, init_obs, horizon, device, n_actions):
    """在世界模型中想象 rollout。

    Args:
        wm: 训练好的世界模型。
        policy: 当前策略。
        init_obs: list of (3,64,64) np arrays（起始真实 obs）。
        horizon: 每条想象轨迹长度。

    Returns:
        obs_list, act_list, logp_list, ret_list, adv_list（展平）。
    """
    all_obs, all_act, all_logp, all_ret, all_adv = [], [], [], [], []
    for obs0 in init_obs:
        ot = torch.from_numpy(obs0).unsqueeze(0).float().to(device)
        h = wm.encode(ot)
        rewards, values, logps, acts, obses = [], [], [], [], []
        for _ in range(horizon):
            # 用解码出的画面作为策略输入（策略吃像素）
            frame = wm.decode(h)
            logits, value = policy(frame)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            a_oh = torch.zeros(1, n_actions, device=device)
            a_oh[0, int(a.item())] = 1.0
            r_pred = wm.predict_reward(h).item()
            h = wm.step(h, a_oh)
            obses.append(frame.squeeze(0).cpu().numpy())
            acts.append(int(a.item()))
            logps.append(float(dist.log_prob(a).item()))
            values.append(float(value.item()))
            rewards.append(r_pred)
        rets, advs = compute_gae(rewards, values)
        all_obs.extend(obses)
        all_act.extend(acts)
        all_logp.extend(logps)
        all_ret.extend(rets.tolist())
        all_adv.extend(advs.tolist())
    return all_obs, all_act, all_logp, all_ret, all_adv


def evaluate_real(env_name, policy, device, episodes=2):
    """在真实环境跑若干 episode，返回平均总回报。"""
    env = ViZDoomEnv(env_name, frame_skip=4, seed=12345)
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


def run_imagination(env_name="health_gathering", init_steps=10000,
                    outer_iters=8, real_per_iter=1000, wm_epochs_init=30,
                    wm_epochs_finetune=5, imagine_per_iter=200,
                    horizon=20, equivariant=True, seed=0, device="cuda",
                    wm_equivariant=None, policy_equivariant=None):
    """运行完整想象训练，返回样本效率曲线 [(real_steps, return), ...]。

    Args:
        equivariant: 同时设置 WM 和策略等变（默认）。
        wm_equivariant / policy_equivariant: 若显式给出，则单独覆盖（消融用）。
    """
    if wm_equivariant is None:
        wm_equivariant = equivariant
    if policy_equivariant is None:
        policy_equivariant = equivariant
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device if torch.cuda.is_available() else "cpu"

    # 1. 初始真实数据 + 世界模型
    print("[imagine] collecting initial random data ...")
    data = collect_data(env_name, init_steps, seed=seed)
    n_actions = data["act"].shape[1]
    wm = train_world_model(data, equivariant=wm_equivariant,
                           epochs=wm_epochs_init, device=device)

    # 2. 策略
    policy = (EquivPolicy(n_actions) if policy_equivariant else StandardPolicy(n_actions)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

    curve = []
    # 初始评估（随机策略水平）
    r0 = evaluate_real(env_name, policy, device)
    curve.append((0, r0))
    print(f"[imagine] real_steps=0  eval_return={r0:.2f}")

    replay_obs = data["obs"]
    real_total = init_steps
    for it in range(outer_iters):
        # a. 用当前策略收集新真实数据
        new_data = collect_data(env_name, real_per_iter, seed=seed + it, policy=policy, device=device)
        real_total += real_per_iter
        replay_obs = np.concatenate([replay_obs, new_data["obs"]], axis=0)
        # 合并到回放（简单起见，这里用新数据微调 WM）
        # b. 微调 WM
        wm = train_world_model(new_data, equivariant=wm_equivariant,
                               epochs=wm_epochs_finetune, device=device, log_every=1000)
        # c. 想象 rollout
        idx = np.random.choice(replay_obs.shape[0], size=min(imagine_per_iter, replay_obs.shape[0]),
                               replace=False)
        init_obs = [replay_obs[i] for i in idx]
        o_l, a_l, lp_l, r_l, ad_l = imagine_rollouts(
            wm, policy, init_obs, horizon, device, n_actions)
        # d. PPO 更新
        ppo_update(policy, o_l, a_l, lp_l, r_l, ad_l, opt, device=device)
        # e. 评估
        ev = evaluate_real(env_name, policy, device)
        curve.append((real_total, ev))
        print(f"[imagine] real_steps={real_total}  eval_return={ev:.2f}")
    return curve


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="health_gathering")
    ap.add_argument("--init_steps", type=int, default=10000)
    ap.add_argument("--outer_iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--equivariant", type=int, default=1)
    args = ap.parse_args()
    curve = run_imagination(env_name=args.env, init_steps=args.init_steps,
                            outer_iters=args.outer_iters, seed=args.seed,
                            equivariant=bool(args.equivariant))
    print("curve:", curve)
