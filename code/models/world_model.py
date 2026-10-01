"""
models/world_model.py
======================
等变世界模型（Equivariant World Model）。

组成：
- Encoder: 等变 CNN，3 层下采样，(B,3,64,64) -> 隐特征图 (B, zch, 8, 8)
- Dynamics: 等变 ConvGRU，输入 (h_t, a_t one-hot broadcast) -> h_{t+1}
- Decoder: 等变转置 CNN，(B,zch,8,8) -> (B,3,64,64)
- RewardHead: 不变量 MLP（全局平均池化后），从 h_t 预测标量奖励

损失：
    L = MSE(recon) + MSE(reward) + MSE(latent_dynamics) + lambda_c * L_consistency

其中 L_consistency 显式惩罚 f(flip x) != flip f(x) 的偏差，
作为硬等变约束的软监督（group averaging 已近似满足，此项进一步收紧）。

参考：DreamerV3 (arXiv:2301.04104) 的 RSSM 结构；EqMBRL (OpenReview 2025)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .equivariant_layers import (
    EquivConv2d, EquivConvTranspose2d, EquivReLU, flip_w, invariant_gap,
)


def _conv(eq, *args, **kwargs):
    return EquivConv2d(*args, **kwargs) if eq else nn.Conv2d(*args, **kwargs)


def _deconv(eq, *args, **kwargs):
    return EquivConvTranspose2d(*args, **kwargs) if eq else nn.ConvTranspose2d(*args, **kwargs)


class ConvGRUCell(nn.Module):
    """等变（或普通）ConvGRU 单元，作用于空间特征图 (B,C,H,W)。"""

    def __init__(self, channels, n_actions, equivariant=True):
        super().__init__()
        self.equivariant = equivariant
        # 输入 = 隐状态 + 动作广播通道
        in_ch = channels + n_actions
        self.conv_zr = _conv(equivariant, in_ch, 2 * channels, 3, padding=1)
        self.conv_n = _conv(equivariant, in_ch + channels, channels, 3, padding=1)

    def forward(self, h, a_bcast):
        """
        Args:
            h: (B, C, H, W) 隐状态
            a_bcast: (B, n_actions, H, W) 动作广播特征
        Returns:
            h_next: (B, C, H, W)
        """
        x = torch.cat([h, a_bcast], dim=1)
        zr = torch.sigmoid(self.conv_zr(x))
        z, r = torch.chunk(zr, 2, dim=1)
        n = torch.tanh(self.conv_n(torch.cat([x, r * h], dim=1)))
        h_next = (1 - z) * n + z * h
        return h_next


class WorldModel(nn.Module):
    def __init__(self, n_actions: int = 3, base_ch: int = 16, z_channels: int = 32,
                 equivariant: bool = True):
        """
        Args:
            n_actions: 离散动作数。
            base_ch: 首层通道基数（小模型用 16）。
            z_channels: 隐特征图通道数（默认 32 -> 8x8x32 latent）。
            equivariant: True=等变模型；False=普通基线。
        """
        super().__init__()
        self.n_actions = n_actions
        self.z_channels = z_channels
        self.equivariant = equivariant
        R = EquivReLU() if equivariant else nn.ReLU()

        # Encoder: 64 -> 32 -> 16 -> 8
        self.enc = nn.Sequential(
            _conv(equivariant, 3, base_ch, 4, stride=2, padding=1), R,
            _conv(equivariant, base_ch, base_ch * 2, 4, stride=2, padding=1), R,
            _conv(equivariant, base_ch * 2, z_channels, 4, stride=2, padding=1), R,
        )

        # 动作 -> 空间嵌入（广播到 8x8）
        self.action_proj = _conv(equivariant, n_actions, z_channels, 1)

        # Dynamics
        self.gru = ConvGRUCell(z_channels, n_actions, equivariant=equivariant)

        # Decoder: 8 -> 16 -> 32 -> 64
        self.dec = nn.Sequential(
            _deconv(equivariant, z_channels, base_ch * 2, 4, stride=2, padding=1), R,
            _deconv(equivariant, base_ch * 2, base_ch, 4, stride=2, padding=1), R,
            _deconv(equivariant, base_ch, 3, 4, stride=2, padding=1),
        )

        # Reward head: 不变量 MLP
        self.reward_head = nn.Sequential(
            nn.Linear(z_channels, 32), nn.ReLU(), nn.Linear(32, 1))

    # ------------------------------------------------------------------
    def encode(self, obs):
        """obs (B,3,64,64) in [0,1] -> h (B,zch,8,8)。"""
        return self.enc(obs)

    def decode(self, h):
        """h -> recon (B,3,64,64)，sigmoid 约束到 [0,1]。"""
        return torch.sigmoid(self.dec(h))

    def predict_reward(self, h):
        """h -> reward (B,1)。"""
        return self.reward_head(invariant_gap(h))

    def step(self, h, a_onehot):
        """单步动力学：h_t + a_t -> h_{t+1}。

        Args:
            h: (B,zch,8,8)
            a_onehot: (B, n_actions)
        Returns:
            h_next: (B,zch,8,8)
        """
        B, _, H, W = h.shape
        a_map = a_onehot.view(B, self.n_actions, 1, 1).expand(-1, -1, H, W)
        return self.gru(h, a_map)

    # ------------------------------------------------------------------
    def compute_loss(self, obs_t, a_t, r_t, obs_next, lambda_cons=1.0):
        """计算训练损失。

        Args:
            obs_t: (B,3,64,64)
            a_t: (B, n_actions) one-hot
            r_t: (B,1)
            obs_next: (B,3,64,64)
            lambda_cons: 等变一致性损失权重。
        Returns:
            dict of losses.
        """
        h_t = self.encode(obs_t)
        recon = self.decode(h_t)
        L_recon = F.mse_loss(recon, obs_t)

        r_pred = self.predict_reward(h_t)
        L_reward = F.mse_loss(r_pred, r_t)

        h_next_pred = self.step(h_t, a_t)
        with torch.no_grad():
            h_next_true = self.encode(obs_next)
        L_dyn = F.mse_loss(h_next_pred, h_next_true)

        out = {"recon": L_recon, "reward": L_reward, "dyn": L_dyn}

        if self.equivariant and lambda_cons > 0:
            # 一致性：encoder 翻转等变
            h_flip_in = self.encode(flip_w(obs_t))
            L_enc = F.mse_loss(h_flip_in, flip_w(h_t))
            # 一致性：dynamics 翻转等变（动作也翻转）
            from utils.transforms import flip_action_onehot
            a_flip = flip_action_onehot(a_t, self.n_actions)
            h_next_pred_flip = self.step(flip_w(h_t), a_flip)
            L_dyn_c = F.mse_loss(h_next_pred_flip, flip_w(h_next_pred))
            L_cons = L_enc + L_dyn_c
            out["cons"] = L_cons
            out["loss"] = L_recon + L_reward + L_dyn + lambda_cons * L_cons
        else:
            out["loss"] = L_recon + L_reward + L_dyn
        return out

    @torch.no_grad()
    def open_loop(self, obs_t, actions, horizon):
        """开环 rollout：从 obs_t 出发，按给定动作序列预测未来帧。

        Args:
            obs_t: (1,3,64,64)
            actions: list[int] 长度 horizon
            horizon: 步数
        Returns:
            list of (3,64,64) np arrays（预测帧，含 t 时刻）。
        """
        h = self.encode(obs_t)
        frames = [self.decode(h).cpu()]
        for a in actions:
            a_oh = torch.zeros(1, self.n_actions, device=obs_t.device)
            a_oh[0, a] = 1.0
            h = self.step(h, a_oh)
            frames.append(self.decode(h).cpu())
        return frames
