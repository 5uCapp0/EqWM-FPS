"""
utils/meff_estimation.py
=========================
M_eff（有效样本量）估计。

思想：等变模型利用对称性，相当于把训练集 "复制" 到对称群上。
若群大小为 G（此处 Z2 群 |G|=2：恒等 + 水平翻转），理论上样本效率
可提升至多 G 倍。M_eff 估计通过比较 "等变模型" 与 "普通模型" 在
同等训练数据量下的预测误差方差，反推等效样本倍数：

    M_eff / M = Var(普通模型误差) / Var(等变模型误差)

该实现提供一个根据误差方差比计算 M_eff 倍数的工具函数，供实验四调用。
参考：EqMBRL (OpenReview 2025) 中关于 symmetry-induced data reuse 的讨论。
"""

import numpy as np


def estimate_meff_ratio(var_std_model: float, var_eq_model: float,
                        group_size: int = 2) -> dict:
    """由预测误差方差比估计有效样本量倍数。

    Args:
        var_std_model: 普通（非等变）模型在测试集上的预测误差方差。
        var_eq_model: 等变模型在同一测试集上的预测误差方差。
        group_size: 对称群大小（Z2=2）。

    Returns:
        dict(ratio=..., clipped_ratio=..., group_size=...)
        ratio = var_std / var_eq；clipped_ratio 限制在 [1, group_size]。
    """
    ratio = float(var_std_model / max(var_eq_model, 1e-12))
    clipped = float(np.clip(ratio, 1.0, float(group_size)))
    return {
        "ratio": ratio,
        "clipped_ratio": clipped,
        "group_size": group_size,
    }
