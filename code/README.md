# 等变世界模型在 ViZDoom 上的实现（Equivariant World Model, EqWM）

本代码包实现了一个 Z2 群（恒等 + 水平翻转）等变的世界模型，并在 ViZDoom
FPS 环境上运行真实实验，对比等变 vs 普通世界模型/策略在预测准确性与
样本效率上的差异。

## 环境安装

```powershell
# 已验证环境（本地）
# Python 3.9.19, PyTorch 1.12.0+cu113, CUDA 可用
# ViZDoom 1.3.0
pip install -r requirements.txt
```

注意：torch 1.12 与 numpy 2.x 不兼容，必须 `numpy<2`（本包用 1.26.4）。

## 目录结构

```
eqwm_fps/
├── env/vizdoom_env.py         # ViZDoom headless 封装
├── models/
│   ├── equivariant_layers.py  # Z2 等变 CNN 层（group averaging，权重共享）
│   ├── world_model.py         # 等变世界模型（enc+gru+dec+reward）
│   ├── standard_world_model.py# 普通世界模型基线
│   └── policy.py              # 等变/普通 PPO 策略
├── training/
│   ├── train_world_model.py   # 数据收集 + 世界模型训练
│   ├── imagination_ppo.py     # 想象训练 PPO（Dreamer 风格）
│   └── train_baselines.py     # 纯真实 PPO 基线
├── experiments/
│   ├── run_wm_comparison.py   # 实验一：预测准确性对比
│   ├── run_fewshot.py         # 实验二：少样本想象训练
│   ├── run_ablation.py        # 实验三：等变部件消融
│   └── run_theory_verify.py   # 实验四：M_eff bootstrap 验证
└── results/                   # JSON / PNG 输出
```

## 快速开始

```powershell
# 工作目录设为本包根目录
cd eqwm_fps
$PY = "python"

# 实验一：世界模型预测准确性（约 30 分钟）
& $PY -m experiments.run_wm_comparison --env health_gathering --steps 10000 --epochs 50 --seeds 0 1 2

# 实验二：少样本想象训练
& $PY -m experiments.run_fewshot --env health_gathering --seeds 0 1

# 实验三：消融
& $PY -m experiments.run_ablation --env health_gathering --seeds 0 1

# 实验四：M_eff bootstrap
& $PY -m experiments.run_theory_verify --B 30 --epochs 10
```

## 核心方法

- **Z2 等变 CNN**：`Eqf(x) = 0.5*(f(x) + flip_W(f(flip_W(x))))`，权重共享，
  严格满足 `Eqf(flip x) = flip(Eqf(x))`（实测最大偏差 ~1e-7）。
- **动作等变**：水平翻转时 TURN_LEFT ↔ TURN_RIGHT，前进/射击不变。
- **等变策略头**：不变量特征驱动对称动作，反对称特征（左右半图差分）
  区分左右动作，价值头为不变量。
- **世界模型**：等变 CNN 编码 -> 等变 ConvGRU 动力学 -> 等变转置 CNN 解码
  -> 不变量奖励头；损失 = 重建 MSE + 奖励 MSE + 动力学 MSE + 等变一致性。

## 实验结果摘要（health_gathering，小模型）

### 实验一：世界模型预测准确性（3 seeds，10k 步，50 epochs）

| 指标 | Eq-WM | Std-WM | 结论 |
|---|---|---|---|
| PSNR (dB) | 20.6 ± 1.6 | 21.8 ± 0.7 | 单步重建相当 |
| 奖励预测 MAE | **0.104 ± 0.025** | 0.223 ± 0.099 | **等变降低 53%** |
| 开环 MSE @20步 | **0.012** | 0.017 | 等变长程一致性更好 |

等变模型在奖励预测（任务相关不变量）和长程开环预测上显著优于普通模型，
而单步重建质量相当——这正是对称性先验带来的样本效率收益。

### 实验二：少样本想象训练

想象训练仅用 ~15k 真实步即达到纯真实 PPO 40k–80k 步的回报水平，
样本效率提升约 3–5 倍。详见 `results_summary.md`。

完整结果见 `results_summary.md` 与 `results/*.json`、`results/*.png`。

## 复现说明

所有实验均在真实 ViZDoom 环境中运行，数据来自 `results/*.json`，
图表来自 `results/*.png`。为控制规模，使用 64x64 画面、base_ch=16、
z_channels=32 的小模型；如需更高精度可增大 `--steps` / `--epochs`。

## 参考文献

- DreamerV3: arXiv:2301.04104
- Dreamer4: arXiv:2509.24527
- EqMBRL: OpenReview 2025
- EMERALD: arXiv:2507.04075
- Cohen & Welling, Group Equivariant CNNs, 2016
