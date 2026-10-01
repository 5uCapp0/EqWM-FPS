# EqWM-FPS 论文优化报告（v2）

- **优化对象**：`paper/eqwm_acmart.tex`（acmart sigconf，主目标）、`paper/eqwm_main.tex`（elsarticle，次要目标）
- **产出**：`paper/eqwm_acmart_v2.tex/.pdf`、`paper/eqwm_main_v2.tex/.pdf`（原稿保持原样未动）
- **日期**：2026-10-02
- **范围**：仅英文语言 / 结构 / 标题 / 表格与图呈现 / LaTeX 规范性 / 引用格式。**实验结论、数字、方法表述一律未改**。
- **数字核对**：v2 全部数字逐项对照 `code/results/wm_comparison.json`、`code/results/fewshot.json`、`data/ablation.json`、`data/theory_verify_v2.json`、`data/theory_verify_v2_cloud.json`，一致（见末尾"数字溯源"）。

---

## 一、逐条修改清单

| ID | 类别 | 原文位置（章节 / 原行） | 问题 | 修改（v2） | 未改原因（如适用） |
|---|---|---|---|---|---|
| L1 | 语言 | Abstract，"…statistical-estimation terms, **and** a scaling law for the optimal real-to-imagination ratio." | 并列结构中 "derive bounds…, and a scaling law" 略松散，两个并列宾语的层级不够清楚 | 改为 "…statistical-estimation terms, **together with** a scaling law for the optimal real-to-imagination ratio."（两个 tex 同步） | 数字/结论不变 |
| L2 | 语言 | §6.1 Setup，"The reward is near-degenerate: the agent is **punished on death**, so roughly 99.3% of non-terminal steps carry the same **terminal-shaped offset**." | "terminal-shaped offset" 是不透明自造词，读者难以理解；"punished on death" 也可更准 | 改为 "the agent is **penalized only at death**, so roughly 99.3% of non-terminal steps carry an **identical reward offset**." | 99.3% 与"reward near-degenerate / 易过拟合"的论断保留，仅澄清措辞 |
| L3 | 语言 | §5 Theorem 3 后讨论，"…which is **absurdly** large because the constant and the **unit** of ε_WM are uncalibrated." | "absurdly" 口语化，不符合学术书面语；"the unit" 主谓一致应为复数 | 改为 "…which is **unrealistically** large because the constant and the **units** of ε_WM are uncalibrated."（两个 tex 同步） |  scaling law 定性结论（常数未校准、实践中 clip）不变 |
| TF1 | 表格与图呈现 | Table 2（wm）Note 列，PSNR/SSIM 两行写 "**std** slightly better" | 表头列名为 "Standard WM"，Note 里却用缩写 "std"，不一致 | 两行统一改为 "**standard** slightly better" | 数字不变 |
| TF2 | 表格与图呈现 | Table 3（ablation）caption，"overlap **inside** one standard deviation" | "inside one standard deviation" 搭配不地道 | 改为 "overlap **within** one standard deviation"（两个 tex 同步） | 四配置不可区分的结论不变 |
| La1 | LaTeX 规范性 | acmart 版文末 `\bibliography{references}` | **功能性 bug**：原 `eqwm_acmart.bbl` 为 0 字节；bibtex 报 "I found no \bibstyle command"，导致 acmart 版参考文献未编译出来 | 在 `\bibliography{references}` 前补 `\bibliographystyle{ACM-Reference-Format}`；v2 bbl 现为 13.5 KB，24 条文献正常排版 | elsarticle 原本已带 `\bibliographystyle{elsarticle-num}`，无需改 |
| La2 | LaTeX 规范性 | acmart 版 `\keywords{… networks , World models , …}` | 逗号前多余空格，排版不规范 | 去掉逗号前空格：`Equivariant neural networks, World models, …` | elsarticle 用 `\keyword … \sep …`，无此问题 |
| C1 | 引用格式 | 全文 `\cite` 与 `references.bib` | 需确认无幽灵引用、无列而未引 | 人工核对 + 编译验证：24 条 bib key 全部在正文被引用，所有 `\cite` 解析成功（日志 0 undefined） | 见 C2 |
| S1 | 结构 | 全文（Intro→Related→Prelim→Method→Theory→Exp→Conclusion） | 需判断是否要重排章节/贡献点 | **评估后保留原结构**：贡献三点（实现对比 / PAC-Bayes 分析 / 诚实结论）与四个实验一一对应，论证链完整；重排会改动语义归属，超出"润色"边界 | — |
| T1 | 标题 | Title | 标题含 "Sample-Efficient"，与消融空结果是否冲突（m4_review L2 已提） | **保留原标题**：样本效率收益（imagination vs 纯 PPO，order-of-2×）是实测结果，副标题 "An Empirical Study" 已标明诚实/偏负结论的定位；改标题会改动作者既定 framing | — |

---

## 二、分类覆盖小结

- **语言（L1–L3）**：3 处，均为措辞/搭配/一致性，不改任何数字与论断。
- **结构（S1）**：评估后保留原骨架（理由见上表）。
- **标题（T1）**：评估后保留（理由见上表）。
- **表格与图呈现（TF1–TF2）**：Note 列用词一致化、caption 介词地道化；**图内容与所有数字未动**。
- **LaTeX 规范性（La1–La2）**：修复 acmart 参考文献编译（0 字节 bbl）这一实质 bug；修正 keywords 标点。
- **引用格式（C1）**：核验通过；references.bib 内容按要求未改。

---

## 三、未修改项（红线 / 技术原因）

1. **全部实验数字冻结**（与 data JSON 逐项一致，见第四节），含：
   - 实验一：Reward MAE `0.104±0.025 vs 0.223±0.099`；PSNR `20.62±1.59 vs 21.78±0.66`；SSIM `0.272±0.069 vs 0.321±0.026`；逐 seed Eq=`[0.072,0.107,0.132]`、Std=`[0.089,0.256,0.324]`。
   - 实验二：~15k 想象回报 `316–508`；纯 PPO 20k–40k `284–572`；四个初值 `396/380/364/668`（均值≈452，668 为离群）；步长比 `~1.3–2.7×`（order-of-2×）。
   - 实验三：`447.1±37.7 / 425.3±28.5 / 426.7±26.7 / 449.8±46.7`（425–450 区间重叠、四配置不可区分、none-eq 数值最高）。
   - 实验四：本机 `0.51 / 5.72 / 0.09`（e=5/20/40）；云端 `0.72 / 1.50 / 28.3`（28.3 标 raw pre-clip，clip 后 2.0）；e=40 等变 replicate 发散到 0.04–0.055。
2. **诚实偏差原样保留**（不得润色成乐观说法）：
   - 实验一定位为"cross-seed stability"，并显式否定"uniform 53% more accurate predictor"；
   - 开环 open-loop MSE（拼接无关 tuple、指标有缺陷）**不重新引入**，仅保留"已省略"的说明；
   - 实验二保留 "2 seeds" 局限与 Eq/Std 互有胜负、不可区分；
   - 消融四配置不可区分如实作为局限；
   - 实验四 `M_eff>1` 降级为 open problem；全文无 "15.7"；云端 e=5/e=20 的 OCR 数据局限完整披露；28.3 标 raw pre-clip；初值 ~380–400 如实。
3. **references.bib 内容未改**：仅做引用一致性核验。bibtex 仍报一批 "empty pages/publisher/address" 提示，原因是条目把会议/arXiv 写在 `booktitle`/`note` 而非 `pages`/`volume` 字段——这是既定的著录风格，不影响编译与引用正确性；如需投稿更整洁，可日后统一补 pages/volume，但本次按红线不动。
4. **图（paper/figs/*.pdf）未重制**：所有矢量图保持原样。
5. **理论证明未补全**：m4_review T1 曾建议补 Theorem 2 / Proposition 4 完整证明到附录；本次为语言/结构润色，且补证明属内容性扩展、超出"不改方法/理论表述"边界，故保留原"proof sketch + 延后附录"写法。**建议投稿前由作者决定是否补附录证明**。

---

## 四、需作者知悉的数据偏差（未擅自改稿）

> **实验三 ablation.json 与论文表格的时间差。**
> `data/ablation.json`（文件时间 2026-10-01 21:14，对应日志 `logs_ablation_s13s14.log`，seeds s13/s14）当前 `_summary` 为
> `both_eq 630.0±52.7 / wm_eq_only 438.0±23.3 / policy_eq_only 413.3±10.7 / none_eq 398.0±6.0`，
> 与论文表格 `447.1±37.7 / 425.3±28.5 / 426.7±26.7 / 449.8±46.7` **不一致**。
>
> 判定：m4_review（17:09）记录论文表格数字"与 `ablation.json._summary` 一致"，说明论文冻结时（acmart 19:29）该文件仍是产出 447/425/427/450 的那份三-train-seed 运行；21:14 的 s13/s14 是**论文冻结之后的一次两-train-seed 探索性重跑**（其 `robust_all` 仅 2 个 train-seed 子数组，而论文写的是 3 train seeds×3 eval seeds×8 episodes）。
>
> 处理：按任务红线"数字零改动"与"消融四配置 425–450 区间重叠不可区分"的明确要求，**保留论文原表格数字，未用新文件覆盖**。新文件中 both_eq≈630 与 none_eq≈398 区间并不重叠，与论文"四配置不可区分"的诚实结论相悖，进一步说明它不是论文所报告的那次实验。
>
> **建议投稿前由作者裁定**：以三-seed 报告版为准（恢复/指明对应数据），或在文中显式说明后续 s13/s14 重跑结果；不要直接把 630/398 写进正文。

---

## 五、数字溯源（核对结论）

| 实验 | 论文数字 | 数据源 | 核对 |
|---|---|---|---|
| Exp1 PSNR/SSIM/MAE | 见表 | `code/results/wm_comparison.json.summary` | 逐位一致 |
| Exp1 逐 seed reward MAE | Eq [0.072,0.107,0.132] / Std [0.089,0.256,0.324] | `wm_comparison.json.per_seed` | 一致 |
| Exp2 回报区间 | 316–508 / 284–572 / 初值 396,380,364,668 | `code/results/fewshot.json`（eq/std_imagine@15k, real_ppo@20/40k, step0） | 一致 |
| Exp3 四配置 | 447.1/425.3/426.7/449.8 | 论文冻结版 ablation（m4_review 确认） | 一致（见第四节偏差说明） |
| Exp4 本机 | 0.51/5.72/0.09 | `data/theory_verify_v2.json`（ratio 0.5132/5.7168/0.0906） | 一致 |
| Exp4 云端 | 0.72/1.50/28.3 | `data/theory_verify_v2_cloud.json`（0.715/1.503/28.334） | 一致；OCR 局限与 e=40 全 15 replicate 披露与 JSON note 一致 |

---

## 六、图评估（Figure 1 骨架图 `eqwm_fig1.svg/png` 与论文核对）

> 评估对象：`C:\Users\86137\Desktop\科研\论文结构图\eqwm_fig1.svg/png`（骨架版，只读评估，未改动文件本身）。
> 风格规范：同目录 `风格指南.md`。对照论文 `eqwm_acmart_v2.tex` §4 Method / §5 Theory / §6 Experiments。

### 6.1 图元素 vs 论文对应描述（逐条）

| # | 骨架图元素 | 论文实际描述（§4/§5/§6） | 是否一致 | 差异 | 建议 |
|---|---|---|---|---|---|
| F1 | 顶部橙框 "λ* — optimal symmetry strength / closed-form · PAC-Bayes bound" | 论文理论（Theorem 3）给出的是 **最优 real:imagination 比 m\*/n = C·(N_eff/n)/ε_WM²**，是 scaling law，**不是"最优对称强度 λ\*"**；论文无 λ\* 概念 | **不一致** | ①论文从不把等变强度当可连续调节的 λ；②"closed-form"过强——论文明说该式未校准、非数值紧（§5 "scaling law, not a numerically tight formula"）；③**λ 已被 PPO GAE λ=0.95 占用**（Table hyper），再用 λ\* 表对称强度会冲突 | 橙框改为 "**m\*/n — optimal real:imagination ratio**" / "**scaling law · PAC-Bayes bound**"；去掉 "closed-form"，或写 "order-of-magnitude scaling law"；不要复用 λ 记号 |
| F2 | "Equivariant World Model" 框注 "group G · **parameterized symmetry**" | G = Z_2 = {e,h}（水平翻转），等变是**硬约束**：group averaging 共享单一卷积核（§4.1），不是 parameterized/soft symmetry | **不一致** | "parameterized symmetry" 暗示可学习的软对称强度，与"硬等变 group averaging"矛盾 | 改为 "**G = Z_2 · hard equivariance (group averaging)**" |
| F3 | "Equivariant World Model" 为单框，无内部子模块 | §4.1 世界模型有 **4 个子组件**：Encoder（3 strided conv, 64×64×3→8×8×32）/ 等变 ConvGRU Dynamics / Decoder（3 transposed conv, sigmoid）/ Reward head（global-avg-pool + MLP） | 部分一致 | 风格指南 §2 允许大模块内画 2–4 个子组件；骨架未展开 | 在 WM 框内补 4 个小子框：Encoder / ConvGRU Dynamics / Decoder / Reward head |
| F4 | "Observation s_t (state / visual input)" | §6.1 观测为 64×64×3 RGB 帧，frame skip 4 | 基本一致 | 仅粒度粗 | 可补 "64×64×3 RGB"（等宽字体，符合指南 §2 张量标注） |
| F5 | "Prediction ẑ_{t+1} (future / transition)" | §4.1 预测三样东西：下一 latent（Dynamics）、重建帧（Decoder）、**奖励 r̂（Reward head）**；论文最关键实测量是 reward MAE 稳定性 | 部分一致 | 只画了 latent/transition 预测，**漏了奖励头预测**，而奖励头正是论文核心发现（cross-seed stability） | 预测分支拆为 "reconstruction ẋ / next latent" + "**reward r̂**" 两路 |
| F6 | "Task head / Value (downstream objective)" | §4.2 是**等变 PPO 策略**：不变特征→forward 动作 + value 头；反对称特征 a=h_left−h_right→左/右动作 logits（ℓ_left=s+a, ℓ_right=s−a），满足 π(h·x)=h·π(x) | 部分一致 | "Task head / Value" 过于笼统；**未体现反对称左/右动作头**这一等变关键设计；也未画"在学到的模型内训练 PPO"的想象训练闭环 | 改为 "**Equivariant PPO policy**"，内列 "action logits (antisym. L/R)" + "value V"；并用回流箭头标出 imagination loop（在 ẑ 轨迹上更新 PPO） |
| F7 | 底部 "Experiments / M_eff sample-efficiency curve · symmetry ablations · few-shot · reward / value effects" | §6 四个实验：Exp1 奖励头稳定性（PSNR/SSIM/Reward MAE）、Exp2 few-shot imagination（order-of-2×）、Exp3 等变组件消融（WM/policy/both/neither）、Exp4 bootstrap M_eff 乘子 | 部分一致 | ①"M_eff sample-efficiency curve" 把 Exp4 的 bootstrap 乘子与 Exp2 的样本效率曲线**混为一谈**；②"reward/value effects" 未点明 Exp1 的"reward-head cross-seed stability"这一主发现 | 改为 "**reward-head stability · few-shot imagination · equivariance ablation · bootstrap M_eff**"，与四实验一一对应 |
| F8 | 橙色框→WM 框的虚线箭头 | Theorem 3 的 scaling law 是对 imagination ratio 的定性指导 | 方向可接受 | 虚线含义（"理论指导实践"）合理 | 保留虚线，箭头可标注 "guides m/n≈4" |

**总体结论**：骨架图的左→右数据流（观测→等变 WM→预测→任务头）与论文主线方向一致，配色符合风格指南（数据蓝/模型深蓝/贡献橙/灰阶）。但有 **2 处实质性概念错误（F1 的 λ\*、F2 的 parameterized symmetry）**、**2 处关键遗漏（F5 奖励头预测、F6 反对称动作头 + imagination 闭环）**，需在定稿前修正，否则会误导审稿人以为论文有一个"可调对称强度 λ\*"或"软对称"，与实际的硬 Z_2 等变 + imagination ratio 理论不符。

### 6.2 论文当前是否已有 Figure 1（两版均查）

- `eqwm_acmart_v2.tex` 与 `eqwm_main_v2.tex` 当前**只有 4 张结果图**：`fig:reward`（reward MAE 逐 seed）、`fig:fewshot`、`fig:ablation`、`fig:meff`。
- **没有任何方法/架构总览图**。当前论文编号下 "Figure 1" 就是 `fig:reward.pdf`（奖励 MAE 折线），不是架构图。
- 因此这张 `eqwm_fig1` 是**全新的方法总览图**，接入后会成为新的 Figure 1，并把现有 4 张结果图顺次挤为 Figure 2–5（LaTeX 自动重编号，无需手改）。

### 6.3 插入位置建议（两版通用）

| 项 | 建议 |
|---|---|
| 插入章节 | **§4 Method 开头**：`\section{Method}\label{sec:method}` 之后、`\subsection{Equivariant $\mathbb Z_2$ world model}` 之前，作为全文架构 + imagination pipeline 总览（与风格指南 §5"方法节开头"一致） |
| `\label` | `\label{fig:overview}` |
| 正文首次引用 | §4 开头加一句："Figure~\ref{fig:overview} summarizes the equivariant world model, the antisymmetric PPO policy, and the imagination pipeline."；亦可在 Intro 贡献 1（"build an equivariant ConvGRU world model and equivariant PPO policy"）处加 `(Fig.~\ref{fig:overview})` |
| 图环境 | `\begin{figure*}[t]...\end{figure*}`？否——acmart sigconf 为单栏宽版，直接用 `\begin{figure}[t]`；该图为横排 4 框 + 顶部橙框 + 底部实验条，内容宽，建议 `\includegraphics[width=0.95\textwidth]{eqwm_fig1.pdf}` |
| 宽度 | **acmart sigconf**：`width=0.95\textwidth`（sigconf 单栏，\textwidth≈\columnwidth，横排 4 框需近满宽）；**elsarticle preprint**：`width=0.9\textwidth`（preprint 版面更宽） |
| 矢量要求 | 风格指南 §4 要求论文插**矢量 PDF**。当前只有 `eqwm_fig1.svg/.png`（png 是位图）；**定稿前需把 SVG 导出为 `eqwm_fig1.pdf` 再 `\includegraphics`**（如 `inkscape eqwm_fig1.svg --export-type=pdf`），不要直接插 png |
| 图文件落点 | 导出后放 `paper/figs/eqwm_fig1.pdf`，与现有 4 张矢量图同目录（`\graphicspath{{figs/}}` 已配好） |

### 6.4 替换/接入结论

- **是否换成骨架图**：建议采用该骨架作为方法总览 Figure 1 的起点，但**先按 6.1 的 F1/F2/F5/F6/F7 修正概念与补全子组件**再定稿；当前骨架若直接插入，会引入"λ\* 最优对称强度 / parameterized symmetry"这两个论文并不存在的概念错误。
- **本次是否已改 tex**：**未改**。图文件按要求只读评估、未动；且 v2 交付目标是语言/结构润色，插入新图需作者先导出 `eqwm_fig1.pdf` 并确认 6.1 修正后再接入。本文给出接入方案与建议，不擅自新增图环境（避免在无矢量 PDF 的情况下插入位图或空引用）。
