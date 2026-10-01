# EqWM-FPS：等变世界模型（Equivariant World Model）在 ViZDoom FPS 上的样本效率研究

> 目标投稿：Neural Networks（CCF-B, SCI）→ 备选 ECAI / UAI
> 项目状态：实验中（实验三消融 / 实验四 M_eff 验证进行中）

## 研究问题

把「等变世界模型」（在模型内部显式编码平移/旋转对称性，而非依赖数据增强）与「想象训练」（在学到的世界模型内做 PPO 想象 roll-out）结合，研究在 ViZDoom `health_gathering` 场景下相对普通世界模型的**样本效率提升**，并用 bootstrap 估计跨种子误差方差比 `M_eff` 来量化其理论增益。

## 目录结构

```
code/
├── env/            # ViZDoom 环境封装（health_gathering）
├── models/         # 等变层 / 标准世界模型 / 策略网络
├── training/       # 世界模型训练 + 想象 PPO + 基线 PPO
├── experiments/    # 四个实验入口（wm_comparison / fewshot / ablation / theory_verify）
├── utils/          # bootstrap、M_eff 估计、指标
├── run_ablation_v2.py          # 实验三消融（A0–A6 矩阵，--seeds 3 4 用法见下）
├── run_theory_verify_v2.py     # 实验四 M_eff bootstrap（--B 15 --epoch_grid 5 20 40）
└── _vizdoom.ini    # ViZDoom 场景配置
data/               # 实验输出 JSON（ablation / theory_verify_v2 / 云端互证副本）+ 文献核验
figs/               # 图表（ablation.png / theory_verify_v2.png）
```

## 环境要求

- Python 3.9–3.11（云端 3.11.11 实测可跑）
- torch（本地 1.12.0+cu113 / 云端 2.3.1+cu121）
- ViZDoom 1.3.x
- numpy < 2（云端 1.26.4）

## 复现命令

```bash
# 实验一：WM 预测准确性（等变 vs 标准）
python experiments/run_wm_comparison.py

# 实验二：少样本想象训练
python experiments/run_fewshot.py

# 实验三：消融矩阵（默认 seeds 0 1 2；多 seed 并行示例）
python run_ablation_v2.py --seeds 0 1 2
python run_ablation_v2.py --seeds 3 4      # 云端并行补种

# 实验四：M_eff bootstrap（B 为 bootstrap 次数）
python run_theory_verify_v2.py --B 15 --epoch_grid 5 20 40
python run_theory_verify_v2.py --B 30 --epoch_grid 10 30 50   # 云端扩展网格
```

## 实验结果摘要（截至 2026-10-01，真实数字）

- **实验一**：等变 WM 奖励预测 MAE 0.104±0.025 vs 普通 0.223±0.099（约降 53%）；开环 MSE@10/20 步低 20%/28%。
- **实验二**：想象训练约 15k 真实步 ≈ 纯真实 PPO 20k–40k 步回报（约 2–3× 样本效率）。
- **实验三消融（seed0）**：both_eq 491±187 / wm_eq_only 401±100 / policy_eq_only 455±159 / none_eq 403±114——高方差、无干净分离（seed1/2 补充中）。
- **实验四 M_eff（B=15）**：非单调曲线——e=5→0.513（欠训练）、e=20→5.717（M_eff>1 甜点）、e=40→0.091（等变方差爆炸）。云端独立互证 e=5→0.715、e=20→1.503、e=40→28.334。**两机在 e=5/e=20 方向一致；e=40 分歧表明 M_eff 在高 epoch 对数值稳定性极敏感**。教材 ch15.7 的「M_eff≈15.7」主张不稳健，论文按真实 bootstrap 数字如实呈现。

## 双端互证

- 本地：exp3 消融 seed0–2 + exp4 复现
- 云端（阿里云 DSW A10）：exp4 B=15 独立运行 + exp3 seed3/4 + exp4 B=30 扩展网格
- 云端 JSON 副本：`data/theory_verify_v2_cloud.json`（e=40 含全部 15 项 bootstrap 数组）

## 论文

- Overleaf 项目：EqWM-FPS_Paper（elsarticle 模板，写作中）
- 教材（含 CCF 论文计划 ch01 §1.6 / 理论 ch13-15 / 初稿 ch16）与本文档同步维护
