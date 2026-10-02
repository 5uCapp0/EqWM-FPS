# 《Z₂-Equivariant World Models for Imagination Training in a First-Person Shooter: Reward-Head Stability without Return Gains》学术引用审查书

> **审查结论：** 24 条参考文献经 arXiv / Crossref / NeurIPS 官方 proceedings 逐条核验，全部真实存在；共发现 7 处题录问题（其中 1 处为严重题录配错、2 处作者列表错误、1 处作者缺失、3 处版本/格式改进项），已全部修复并同步本地双版 + Overleaf，重编译 0 错误。文内—文后引用完全对应，无孤儿条目、无缺失引用。当前版本引用体系可支持 Neural Networks 投稿。

| 审查对象 | 信息 |
|---|---|
| 论文标题 | Z₂-Equivariant World Models for Imagination Training in a First-Person Shooter: Reward-Head Stability without Return Gains |
| 稿件类型 | 期刊论文（Neural Networks, Elsevier, elsarticle） |
| 审查重点 | 真实性、题录、文内—文后对应、主张—证据匹配、版本与来源风险 |
| 审查日期 | 2026-10-02 |

---

## 一、执行摘要

- 引用体系整体可靠：24 条文献全部真实存在，经典条目（Dreamer 系列、MuZero、MBPO、G-CNN、PPO、ViZDoom 等）题录经 arXiv 官方页面与 Crossref 双重确认。
- 高风险 2025 新条目（Multi-Group Equivariant Augmentation、EMERALD、Dreamer 4、PAC-Bayesian RL）经 arXiv 官方页面确认存在且编号准确。
- 共修复 7 处题录问题，其中 **dreamerv1 条目被配成了 PlaNet（Plan from Pixels）的题录，而编号 arXiv:1912.01603 实为 Dreamer（Dream to Control）**——这是唯一影响正文论证归属的实质错误，已修正。
- 正文 24 个引用键与文后 24 条完全一一对应，无幽灵引用、无未使用条目；正文引用语境（世界模型家族、等变 RL 基线、PAC-Bayes 框架、数据集与算法）与来源内容匹配，未发现支持不足或过度推断。
- 建议：按本次修复后的版本投稿；投稿系统要求提交 bbl 时使用重编译产物。

## 二、投稿前优先行动

### 优先行动｜已完成的题录修复（7 处）

本地 `references.bib` 已修复全部 7 处问题，双版（elsarticle / acmart）重编译 0 错误、0 未定义引用，git commit `22f9fc7` 已推送；Overleaf 项目根 `references.bib` 已覆盖更新并重编译验证（Errors 0，参考文献页目检全部生效）。投稿无需额外操作。

## 三、核心发现

### 发现｜dreamerv1 条目题录配错（最严重）

**稿件位置**：`references.bib` 条目 `dreamerv1`；正文第 92 行 "The Dreamer family operationalizes this idea with a recurrent state-space model \cite{dreamerv1,dreamerv2,dreamerv3}"。

**稿件原文（修复前）**

> @inproceedings{dreamerv1, author = {Hafner, Danijar and Lillicrap, Timothy and Fischer, Ian and Villegas, Ruben and Ha, David and Lee, Honglak and Davidson, James}, title = {Learning Latent Dynamics for Planning from Pixels}, ... note = {arXiv:1912.01603}}

**专家判断**：严重题录配错。标题与作者列表是 **PlaNet**（Plan from Pixels, Hafner et al. 2018）的题录，而附注编号 arXiv:1912.01603 对应的是 **Dreamer**（"Dream to Control: Learning Behaviors by Latent Imagination", ICML 2020）。正文语境把该条目用作 Dreamer 家族成员，引用的是错误的文献身份。

**来源实际表明**：arXiv:1912.01603 官方页面显示标题 "Dream to Control: Learning Behaviors by Latent Imagination"，作者 Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi（ICML 2020 收录）。

**为什么重要**：题录与文献身份不符属于可被编辑/审稿人直接发现的硬伤；PlaNet 未被正文引用，若保留旧题录会产生幽灵引用与错误研究归属。

**建议修改**：已修正为 "Dream to Control: Learning Behaviors by Latent Imagination"，作者 Hafner, Lillicrap, Ba, Norouzi，ICML 2020。

**核验来源**：[arXiv:1912.01603](https://arxiv.org/abs/1912.01603)

### 发现｜edgi2023 作者列表错误

**稿件位置**：`references.bib` 条目 `edgi2023`；正文第 176 行 EDGI 相关句。

**稿件原文（修复前）**：作者列表含 9 人（Brehmer, Rahaman, Deac, Geng, Li, Strobelt, Chintala, Henaff, Weimer）。

**专家判断**：作者列表错误。arXiv:2303.12410 实际作者为 Johann Brehmer, Joey Bose, Pim de Haan, Taco Cohen（4 人）。

**为什么重要**：作者列表错误影响引用可信度与贡献归属。

**建议修改**：已修正为 Brehmer, Bose, de Haan, Cohen。

**核验来源**：[arXiv:2303.12410](https://arxiv.org/abs/2303.12410)

### 发现｜geodrl2020 作者姓名错误

**稿件位置**：`references.bib` 条目 `geodrl2020`。

**稿件原文（修复前）**：Mondal, Sayak Ray / Nair, Arun / Siddiqi, Biswajit。

**专家判断**：arXiv:2007.03437 实际作者为 Arnab Kumar Mondal, Pratheeksha Nair, Kaleem Siddiqi。

**建议修改**：已修正。

**核验来源**：[arXiv:2007.03437](https://arxiv.org/abs/2007.03437)

### 发现｜emerald2025 缺第二作者

**稿件位置**：`references.bib` 条目 `emerald2025`。

**专家判断**：arXiv:2507.04075 有两位作者：Maxime Burchi, Radu Timofte；原条目仅列 Burchi。

**建议修改**：已补 Timofte。

**核验来源**：[arXiv:2507.04075](https://arxiv.org/abs/2507.04075)

## 四、题录、版本与来源问题

| 引用 | 问题 | 建议 |
|---|---|---|
| dreamerv3 | 标题为 arXiv 版 "Mastering Diverse Domains through World Models"，但载体标注 Nature 2025——Nature 正式发表版标题为 "Mastering Diverse Control Tasks through World Models"（Crossref 确认） | 已对齐正式版标题，保留 note arXiv:2301.04104 |
| dreamer42025 | 标题带 "(Dreamer4)" 后缀，非精确题名 | 已改为精确标题 "Training Agents Inside of Scalable World Models"，note 标注 "Dreamer 4" |
| pacbayesrl2025 | 标注为 arXiv 预印本，但 arXiv v3 页注明 "Accepted to the 43rd ICML (2026), camera-ready version" | 已升级为 @inproceedings（ICML 2026），保留 note arXiv:2510.10544 |
| dreamerv3 / dreamer42025 | elsarticle-num 样式中 journal 后接 note 渲染缺分隔逗号（"NatureArXiv:…"） | 已给 note 加前导逗号（", arXiv:…" / ", Dreamer 4"） |

## 五、引用结构观察

- 文内 24 个引用键与文后 24 条条目完全一一对应：无幽灵条目、无未使用条目、无缺失条目（编号制自动重排，正文与参考文献一致）。
- 组合引用使用清晰：如 \cite{dreamerv1,dreamerv2,dreamerv3}、\cite{fps2017,vizdoom2016} 均指向同类来源，未发现"一引多义"式混搭。
- 来源构成合理：全部为官方 arXiv / 会议 proceedings / 出版专著，无博客、镜像站或二手转引。
- 引用密度正常：正文主张（含理论部分引用 PAC-Bayes 框架 \cite{mcallester1999}、bootstrap 方法 \cite{bootstrap1993}）均有直接来源支撑。

## 六、已确认可靠的代表性引用

- **vizdoom2016**（arXiv:1605.02097 / IEEE CIG 2016）：数据集与基准来源，题录完全匹配，Crossref + arXiv 双重确认。
- **muzero2020**（Nature 588:604–609, 2020）：Crossref 完全匹配，卷页准确。
- **multigroup2025**（arXiv:2508.11204）：2025-08-15 提交，作者 Lin/Rojas/Au 与条目一致，真实性确认（本类新条目已逐条核验）。

## 七、总体修改建议

1. 题录批量核对已在本轮完成并落地（references.bib 头部注释记录了本轮修复清单，供后续审计追溯）。
2. 若未来投稿系统要求 bbl 文件，直接使用本次重编译产物（两版均已 0 错误）。
3. 论文正文不依赖被引文献支撑任何强结论（摘要/结论已按 n=9 弱化措辞），引用角色均为方法介绍、基线与框架来源，无进一步收敛必要。

## 八、审查范围与证据边界

- 本次审查材料：`paper/eqwm_main_v3.tex`（elsarticle 版）全文引用语境、`paper/references.bib` 全部 24 条。
- 覆盖问题类型：文献真实性、题录准确性（标题/作者/年份/载体/编号）、文内—文后对应、版本状态（预印本/正式版/会议接收）、组合引用与二手转引。
- 证据层级：arXiv 官方 abs 页面（逐条）、Crossref works API（脚本批量 + 人工判读）、NeurIPS/ICML/ICLR 官方收录记录；对正文逐句主张比对仅覆盖引用语境（论文自身结果与数据未依赖被引文献证明，故未深入全文逐句核验）。
- 未覆盖：实验数据真实性、统计分析、研究伦理（不属于引用审计范围）。
- 书类条目（bootstrap1993 / suttonbarto2018）以出版记录与版本惯例为准，未逐页核验内容。

> 工作流程完成不等于论文自动"通过审计"。未发现明显问题，也不代表不存在任何引用风险。
