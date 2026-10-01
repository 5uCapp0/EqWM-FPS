"""
utils/metrics.py
================
图像预测质量指标：PSNR / SSIM / MAE，全部用 numpy/torch 自实现，
避免依赖 skimage。
"""

import numpy as np
import torch


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """计算 Peak Signal-to-Noise Ratio (dB)。

    Args:
        pred: 预测图像，形状 (H,W,3) 或 (3,H,W)，取值 [0, data_range]。
        target: 真值图像，同形状。
        data_range: 像素动态范围（归一化图为 1.0）。

    Returns:
        PSNR 分贝值。
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse = np.mean((pred - target) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((data_range ** 2) / mse)


def _ssim_single_channel(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    """单通道 SSIM，高斯窗近似为均匀 7x7 窗。"""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    # 简单 box 滤波（用 numpy 卷积近似）
    k = 7
    if x.shape[0] < k or x.shape[1] < k:
        k = min(x.shape[0], x.shape[1])
        if k < 3:
            return 1.0
    mu_x = _box_filter(x, k)
    mu_y = _box_filter(y, k)
    sx2 = _box_filter(x * x, k) - mu_x ** 2
    sy2 = _box_filter(y * y, k) - mu_y ** 2
    sxy = _box_filter(x * y, k) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sx2 + sy2 + C2))
    return float(ssim_map.mean())


def _box_filter(a: np.ndarray, k: int) -> np.ndarray:
    """均匀盒式滤波（valid 区域）。"""
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(a, (k, k))
    return win.mean(axis=(-1, -2))


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """多通道 SSIM，对通道取平均。

    Args:
        pred/target: 形状 (H,W,3)。
    Returns:
        平均 SSIM。
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim == 3 and pred.shape[0] == 3:  # (3,H,W) -> (H,W,3)
        pred = np.transpose(pred, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))
    vals = []
    for c in range(pred.shape[-1]):
        vals.append(_ssim_single_channel(pred[..., c], target[..., c], data_range))
    return float(np.mean(vals))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """平均绝对误差。"""
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))))


def torch_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """batch 版 PSNR，输入 (B,3,H,W)。"""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return 10.0 * torch.log10((data_range ** 2) / (mse + 1e-12))
