"""
models/policy.py
================
PPO 策略网络：等变版 + 普通版。

等变动作头设计（Z2）：
- 不变量特征 inv = gap(h)：镜像后不变，驱动 "对称动作"（前进/射击）。
- 反对称特征 anti = mean(左半) - mean(右半)：镜像后取负。
- logits = [ inv_mlp(inv) + anti_mlp(anti),
            inv_mlp(inv) - anti_mlp(anti),
            inv_mlp2(inv) ]
  即左/右动作镜像互换，前进/射击不变。
价值头为不变量 MLP。
"""

import torch
import torch.nn as nn

from .equivariant_layers import (
    EquivConv2d, EquivReLU, invariant_gap, antisymmetric_half_diff,
)


class EquivPolicy(nn.Module):
    """等变 PPO 策略（演员-评论家）。"""

    def __init__(self, n_actions: int = 3, base_ch: int = 16, z_channels: int = 32):
        super().__init__()
        self.n_actions = n_actions
        self.enc = nn.Sequential(
            EquivConv2d(3, base_ch, 4, stride=2, padding=1), EquivReLU(),
            EquivConv2d(base_ch, base_ch * 2, 4, stride=2, padding=1), EquivReLU(),
            EquivConv2d(base_ch * 2, z_channels, 4, stride=2, padding=1), EquivReLU(),
        )
        # 不变量分支（驱动对称动作 + 价值）
        self.inv_mlp = nn.Sequential(
            nn.Linear(z_channels, 64), nn.ReLU(), nn.Linear(64, 2))
        # 反对称分支（区分左右）
        self.anti_mlp = nn.Sequential(
            nn.Linear(z_channels, 64), nn.ReLU(), nn.Linear(64, 1))
        # 价值头（不变量）
        self.value_head = nn.Sequential(
            nn.Linear(z_channels, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs):
        """
        Args:
            obs: (B,3,64,64)
        Returns:
            logits: (B, n_actions)
            value: (B,1)
        """
        h = self.enc(obs)
        inv = invariant_gap(h)
        anti = antisymmetric_half_diff(h)
        s = self.inv_mlp(inv)          # (B,2): [sym_left_right, sym_forward]
        a = self.anti_mlp(anti)        # (B,1)
        logit_lr = s[:, 0:1]           # 左右共享对称部分
        logit_fwd = s[:, 1:2]           # 前进/射击对称部分
        logit_left = logit_lr + a
        logit_right = logit_lr - a
        logits = torch.cat([logit_left, logit_right, logit_fwd], dim=-1)
        value = self.value_head(inv)
        return logits, value


class StandardPolicy(nn.Module):
    """普通 PPO 策略基线。"""

    def __init__(self, n_actions: int = 3, base_ch: int = 16, z_channels: int = 32):
        super().__init__()
        self.n_actions = n_actions
        self.enc = nn.Sequential(
            nn.Conv2d(3, base_ch, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(base_ch * 2, z_channels, 4, stride=2, padding=1), nn.ReLU(),
        )
        feat = z_channels * 8 * 8
        self.actor = nn.Sequential(
            nn.Linear(feat, 128), nn.ReLU(), nn.Linear(128, n_actions))
        self.critic = nn.Sequential(
            nn.Linear(feat, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, obs):
        h = self.enc(obs)
        flat = h.flatten(1)
        logits = self.actor(flat)
        value = self.critic(flat)
        return logits, value
