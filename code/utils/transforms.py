"""
utils/transforms.py
====================
等变变换工具：Z2 群（恒等 + 水平翻转）。

约定：
- 画面 x 的形状为 (..., C, H, W)，水平翻转沿最后一维 W。
- ViZDoom 离散动作为 one-hot 向量，顺序固定为：
    [TURN_LEFT, TURN_RIGHT, (MOVE_FORWARD | ATTACK), ...]
  水平翻转时左右动作互换，前后/射击等对称动作保持不变。

参考：DreamerV3 (arXiv:2301.04104) 中对数据增广与对称性的讨论；
      EqMBRL (OpenReview 2025) 将动作镜像纳入等变世界模型。
"""

import torch


def flip_obs(x: torch.Tensor) -> torch.Tensor:
    """水平翻转图像/特征图。

    Args:
        x: 张量，最后一维为宽度 W，形状 (..., C, H, W)。

    Returns:
        沿 W 维翻转后的同形状张量。
    """
    return torch.flip(x, dims=[-1])


def make_action_flip_map(n_actions: int) -> list:
    """生成动作索引翻转映射表。

    约定索引 0=TURN_LEFT, 1=TURN_RIGHT，其余索引保持不变。

    Args:
        n_actions: 动作总数（通常为 3）。

    Returns:
        list flipped[i] = 动作 i 在水平翻转后对应的新动作索引。
        例如 n_actions=3 时返回 [1, 0, 2]。
    """
    m = list(range(n_actions))
    if n_actions >= 2:
        m[0], m[1] = 1, 0
    return m


def flip_action_index(a: int, n_actions: int = 3) -> int:
    """翻转单个离散动作索引。"""
    return make_action_flip_map(n_actions)[a]


def flip_action_onehot(a_oh: torch.Tensor, n_actions: int = 3) -> torch.Tensor:
    """翻转 one-hot 动作向量。

    Args:
        a_oh: 形状 (..., n_actions) 的 one-hot 动作。
        n_actions: 动作总数。

    Returns:
        左右动作互换后的 one-hot 张量。
    """
    perm = make_action_flip_map(n_actions)
    # 按 perm 重排最后一维：out[..., j] = a_oh[..., perm[j]]
    return a_oh[..., perm]
