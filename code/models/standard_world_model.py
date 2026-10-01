"""
models/standard_world_model.py
==============================
普通（非等变）世界模型基线。与 WorldModel 结构完全一致，
但不施加等变约束（use_equiv=False），用于对照实验。
"""

from .world_model import WorldModel


class StandardWorldModel(WorldModel):
    """普通 CNN 世界模型（对照组）。"""

    def __init__(self, n_actions: int = 3, base_ch: int = 16, z_channels: int = 32):
        super().__init__(n_actions=n_actions, base_ch=base_ch,
                         z_channels=z_channels, equivariant=False)
