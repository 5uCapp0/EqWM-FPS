"""
utils/bootstrap.py
===================
Bootstrap 统计工具：对一组评估指标做有放回重采样，估计均值置信区间。
"""

import numpy as np


def bootstrap_ci(values, n_boot=200, alpha=0.05, seed=0):
    """对一维数组做 bootstrap，返回均值的点估计与置信区间。

    Args:
        values: array-like, 每个元素是一次实验/一个样本的指标。
        n_boot: bootstrap 重采样次数。
        alpha: 显著性水平（默认 0.05 -> 95% CI）。
        seed: 随机种子。

    Returns:
        dict(mean, lo, hi, se)：点估计、CI 下界、上界、标准误。
    """
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean()
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "se": float(means.std(ddof=1)),
    }
