# EqWM-FPS：等变世界模型（Equivariant World Model）在 ViZDoom FPS 上的样本效率研究

> 目标投稿：Neural Networks（CCF-B, SCI）→ 备选 ECAI / UAI
> 项目状态：M0–M4 实验与论文 v2 润色完成，稿件可投稿；主版本 = acmart 双栏 eqwm_acmart_v2（含修正版方法总览图 fig:overview）

## 研究问题

把「等变世界模型」（在模型内部显式编码水平翻转对称性 Z_2，而非依赖数据增强）与「想象训练」（在学到的世界模型内做 PPO 想象 roll-out）结合，研究在 ViZDoom `health_gathering` 场景下相对普通世界模型的**样本效率**，并用 bootstrap 估计有效样本乘数 `M_eff` 作为理论增益的经验探针（该估计不稳健，结论见实验四）。

## 目录结构

```
code/
├── env/            # ViZDoom 环境封装（health_gathering）
├── models/        # 等变层 / 标准世界模型 / 策略网络
├── training/       # 世界模型训练 + 想象 PPO + 基线 PPO
├── experiments/    # 四个实验入口（wm_comparison / fewshot / ablation / theory_verify）
├── utils/         # bootstrap、M_eff 估计、指标
├── run_ablation_v2.py        # 实验三消融
├── run_theory_verify_v2.py   # 实验四 M_eff bootstrap
└── _vizdoom.ini    # ViZDoom 场景配置
data/               # 实验输出 JSON（wm_comparison / fewshot / ablation / theory_verify_v2* 等）
paper/              # 论文稿（tex + pdf + references.bib）与矢量图（figs/）
```

## 环境要求

- Python 3.9–3.11
- torch（CUDA 11.x / 12.x 均可）
- ViZDoom 1.3.x
- numpy < 2

## 复现命令

```bash
# 实验一：WM 预测准确性（等变 vs 标准）
python experiments/run_wm_comparison.py

# 实验二：少样本想象训练
python experiments/run_fewshot.py

# 实验三：消融（指定训练种子）
python run_ablation_v2.py --seeds 0 1 2

# 实验四：M_eff bootstrap（B 为 bootstrap 次数）
python run_theory_verify_v2.py --B 15 --epoch_grid 5 20 40
```

## 实验结果摘要（截至 2026-10-01，真实数字）

- **实验一（世界模型预测准确性）**：奖励预测 MAE 0.104±0.025 vs 标准 WM 0.223±0.099——等变奖励头跨种子方差更小（更稳定）；画面重建质量 PSNR 20.62±1.59 vs 21.78±0.66、SSIM 0.272±0.069 vs 0.321±0.026，两者大体相当、标准模型在重建上略占优。论文据此把等变收益定位为**奖励头跨种子稳定性**，而非整体预测精度提升。
- **实验二（少样本想象训练）**：想象训练约 15k 真实步即达回报区间 316–508，对照纯真实 PPO 在 20k–40k 步达回报 284–572；步长比约 1.3–2.7×（量级约 2×）。该结论基于 2 个训练种子，等变与标准变体互有胜负、未达统计可分，仅作 order-of-2× 的初步证据。
- **实验三（等变部件消融）**：以三训练种子稳健评估为准，四配置最终回报区间重叠（425–450），等变部件未带来显著最终回报增益（详见下方数据说明）。
- **实验四（有效样本乘数 M_eff 的 bootstrap 估计）**：本地 B=15 曲线非单调——e=5→0.51、e=20→5.72、e=40→0.09；另一台机器独立运行得 e=5→0.72、e=20→1.50、e=40→28.3（28.3 为未截断原始值）。两机在 e=5/e=20 方向大体一致，e=40 处分歧表明 M_eff 在高 epoch 对数值稳定性极敏感。因此论文将 M_eff>1 降级为 open problem，不作为已证实的增益主张。

### 实验三消融数据说明

实验三消融数据说明：论文冻结表以三训练种子稳健评估为准（both_eq 447.1±37.7 / wm_eq_only 425.3±28.5 / policy_eq_only 426.7±26.7 / none_eq 449.8±46.7，四配置区间重叠、等变部件无显著最终回报增益）；论文定稿后另有一次两训练种子探索性重跑（s13/s14，both_eq≈630 / none_eq≈398），与冻结表口径不一致，不作为论文结论依据，仅如实记录。

## 双端互证

- 本地机：exp1–exp4 主复现（含消融三种子冻结表、M_eff B=15）
- 另一台 GPU 机：exp4 独立 bootstrap 运行 + 高 epoch 网格扩展（云端结果 JSON 存于 data/）

## 论文

- **主论文 = `paper/eqwm_acmart_v2.tex`**（acmart sigconf 双栏，匿名，6 页，含修正版方法总览图 fig:overview）；参考文献 `paper/references.bib`（24 条）。
- `paper/eqwm_main_v2.tex`（elsarticle，16 页）为期刊（Neural Networks）投稿备选版本，本地保留、不作为主版本宣传。
- 图：`paper/figs/` 下方法总览 `eqwm_fig1.pdf` 与四张结果矢量图（reward / fewshot / ablation / M_eff）。
