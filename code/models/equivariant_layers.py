"""
models/equivariant_layers.py
==============================
Z2 群（恒等 e + 水平翻转 g）等变 CNN 层。

核心思想（group averaging，权重共享）：
    对任意层 f，定义
        Eqf(x) = 0.5 * ( f(x) + flip_W( f(flip_W(x)) ) )
    则 Eqf 满足严格等变性：
        Eqf(flip_W(x)) = flip_W(Eqf(x))
    且只使用一份卷积权重（f 被调用两次但参数共享），参数效率不损失。

参考：Cohen & Welling 2016 (group equivariant CNN)；
      EqMBRL (OpenReview 2025) 在 MBRL 中采用 group averaging。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def flip_w(x: torch.Tensor) -> torch.Tensor:
    """沿宽度维 W 翻转特征图 x: (B,C,H,W) -> (B,C,H,W)。"""
    return torch.flip(x, dims=[-1])


class EquivConv2d(nn.Module):
    """群平均等变 2D 卷积（权重共享）。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = flip_w(self.conv(flip_w(x)))
        return 0.5 * (y1 + y2)


class EquivConvTranspose2d(nn.Module):
    """群平均等变转置卷积（解码器用）。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, bias=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, bias=bias)

    def forward(self, x):
        y1 = self.deconv(x)
        y2 = flip_w(self.deconv(flip_w(x)))
        return 0.5 * (y1 + y2)


class EquivReLU(nn.Module):
    """ReLU 与 W 翻转交换，因此天然等变。"""

    def forward(self, x):
        return F.relu(x)


class EquivAvgPool2d(nn.Module):
    """平均池化：对 W 翻转等变（池化核对称）。"""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)


def invariant_gap(x: torch.Tensor) -> torch.Tensor:
    """全局平均池化 -> 不变量向量。gap(flip x)=gap(x)。

    Args:
        x: (B, C, H, W)
    Returns:
        (B, C) 不变量特征。
    """
    return x.mean(dim=[2, 3])


def antisymmetric_half_diff(x: torch.Tensor) -> torch.Tensor:
    """左右半图差分 -> 反对称量，flip 后取负。

    h = mean(左半) - mean(右半)。flip(x) 左右互换 -> h -> -h。
    用于策略动作头中区分左右动作。

    Args:
        x: (B, C, H, W)
    Returns:
        (B, C) 反对称特征。
    """
    W = x.shape[-1]
    half = W // 2
    left = x[..., :half].mean(dim=[2, 3])
    right = x[..., W - half:].mean(dim=[2, 3])
    return left - right
