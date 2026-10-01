# GitHub 推送需求清单（EqWM-FPS）

> 仓库：`github.com/5uCapp0/EqWM-FPS`（Public，2026-10-01 18:0x 已推送 M4：`paper/eqwm_main.tex/.pdf` + 4 张矢量图）。
> 本清单**只评估打包，不实际 push**。本机无 GitHub 凭据、无 gh CLI、无 git clone；实际 push 由 MainAgent 稍后走浏览器 PAT 流程执行。
> 定稿策略（2026-10-02 更新）：**主版本 = acmart 双栏 `eqwm_acmart_v2`**（与 AS-Park 同款，含修正版方法总览图 fig:overview）；elsarticle `eqwm_main_v2` 降级为期刊投稿归档备选，本地保留、不作为主版本宣传、默认不推。

---

## 1. 相对基线的本地变更文件清单

基线 = 2026-10-01 18:0x 已推送的 M4 快照（`eqwm_main.tex/.pdf` + `paper/figs/fig_*.pdf`）。本次本地新增/变更：

| 文件 | 状态 | 是否建议推送 | 说明 |
|---|---|---|---|
| `paper/eqwm_acmart_v2.tex` / `.pdf` | 新（acmart sigconf 双栏，6 页，0 fatal/0 undefined） | **主版本，建议推** | 含修正版方法总览图 `fig:overview`；本仓库主论文 |
| `paper/figs/eqwm_fig1.pdf` | 新（修正版矢量方法总览，已导出） | **建议推** | 与 acmart v2 配套；m*/n scaling law 橙框、WM 4 子模块、reward 头、反对称 L/R 动作头、imagination 闭环 |
| `paper/references.bib` | 未改 | 随主版本推（基线已有则跳过） | 24 条，内容未动 |
| `paper/eqwm_main_v2.tex` / `.pdf` | 新（elsarticle，16 页） | **默认不推（归档备选）** | 期刊（Neural Networks）投稿备选，本地保留；不作为主版本宣传，不进推送集 |
| `paper/eqwm_acmart.tex` / `.pdf`（19:29 原版） | 基线未含 | 可不单独推 | 已被 `eqwm_acmart_v2` 取代；如需保留历史可并入 |
| `paper/figs/fig_*.pdf`（4 张结果图） | 未改 | 已在基线 | 无需重推 |
| `data/optimization_report.md` | 新（内部评审报告） | **不推**（.gitignore） | 内部工作/评审记录，按脱敏清单不进公开仓库 |
| `data/github_push_checklist.md` | 新（本文件） | **不推**（.gitignore） | 内部打包清单 |
| `data/m4_review.md` / `项目进度.md` | 已有，已 gitignore | 保持不推 | 基线已排除 |
| `README.md` | 已有（已按诚实口径修订） | **建议推** | 主版本=acmart v2；已清理 53%/2-3×/seed0 单值/M_eff 15.7/项目状态，并补 ablation s13/s14 诚实说明段 |
| `paper/main.tex`、`paper/main_upload.tex` | 已有（18:59 / 17:14） | **不建议单独维护** | 约为 `eqwm_main.tex` 的 Overleaf 副本/变体，陈旧重复；仓库以 `eqwm_acmart_v2` 为准 |

> 注：`data/ablation.json`（21:14，s13/s14 重跑）与论文三-seed 冻结表口径不一致，已在 `optimization_report.md` 第四节与 README「实验三消融数据说明」如实标注为探索性重跑、不作论文结论依据。是否随仓库推送该文件仍需作者裁定。

---

## 2. 建议仓库结构

```
EqWM-FPS/
├── README.md                 # 主版本=acmart v2 / 结果摘要（诚实数字）/ 论文状态
├── .gitignore
├── code/                     # env/ models/ training/ experiments/ utils/ + run_*.py
├── data/                     # 公开结果 JSON；内部 *.md 评审报告排除
├── paper/
│   ├── eqwm_acmart_v2.tex / .pdf   # 主论文（acmart 双栏，含 fig:overview）
│   ├── references.bib
│   └── figs/
│       ├── eqwm_fig1.pdf            # 修正版方法总览
│       └── fig_reward / fig_fewshot / fig_ablation / fig_meff .pdf
└── (本地，不推) eqwm_main_v2.tex/.pdf   # elsarticle 期刊投稿归档备选
```

---

## 3. .gitignore 规则（在现有基础上补两条）

现有 `.gitignore` 已覆盖：`*.log`、`code/logs/`、`__pycache__/`、`*.pyc`、`*.pt/*.pth`、`.DS_Store/Thumbs.db/.idea/.vscode/`、`paper/*.aux *.bbl *.blg *.out *.log`、`paper/.miktex/`、`项目进度.md`、`data/m4_review.md`。

**建议追加**（把本次内部报告一并排除）：
```gitignore
# 内部工作/评审记录（不进公开仓库，本地保留）
data/optimization_report.md
data/github_push_checklist.md
```
> 其余编译产物（`paper/*.aux *.bbl *.blg *.out *.log`、`paper/.miktex/`）已在规则内，v2 编译产物不会误入。
> elsarticle `eqwm_main_v2.*` 仅本地保留；若日后要推再单独处理，默认不加进推送集。

---

## 4. README.md 修订点（已完成，推送前核对）

- ~~"实验一（约降 53%）；开环 MSE 低 20%/28%"~~ → 已改为**奖励头跨种子稳定性**（0.104±0.025 vs 0.223±0.099），PSNR/SSIM 两者相当、标准略优；删除 53% 与开环 MSE。
- ~~"实验二（约 2–3× 样本效率）"~~ → 已改为 **order-of-2×（实测 ~1.3–2.7×），2 seeds、Eq/Std 不可区分**。
- ~~"实验三消融（seed0）491±187…"~~ → 已改为三-seed 冻结表 447.1±37.7 / 425.3±28.5 / 426.7±26.7 / 449.8±46.7（425–450 重叠），并补 **s13/s14 探索性重跑诚实说明段**。
- ~~"教材 ch15.7 的 M_eff≈15.7"~~ → 已删内部痕迹；改为 M_eff>1 非单调/跨机不一致、降级为 open problem，28.3 标 raw pre-clip。
- ~~"项目状态：实验中"~~ → 已改为"主版本 = acmart 双栏 eqwm_acmart_v2（含修正版方法总览图 fig:overview）；elsarticle 为期刊投稿备选，本地保留"。
- 全文已无 教材 / ch15.7 / 15.7 / 豆包 / agent / OCR 等内部痕迹。

---

## 5. 建议 commit 粒度与 message

```bash
# 1) 主论文：acmart v2（含修正版方法总览图）
git add paper/eqwm_acmart_v2.tex paper/eqwm_acmart_v2.pdf         paper/figs/eqwm_fig1.pdf paper/references.bib
git -c user.name="5uCapp0" -c user.email="5uCapp0@users.noreply.github.com"     commit -m "paper: acmart two-column main version with corrected architecture overview"

# 2) README 与 .gitignore（诚实数字 + 排除内部报告）
git add README.md .gitignore
git commit -m "docs: align README with honest findings; ignore internal review reports"
```
> elsarticle `eqwm_main_v2.*` 默认**不进上述 commit**（归档备选，本地保留）。
> commit 邮箱一律用 `5uCapp0@users.noreply.github.com`（GitHub noreply），不用真实邮箱（脱敏要求）。

---

## 6. 网络 / 代理提醒（实际 push 时）

- **本机 git 直连 github.com 被阻断**（浏览器能开、命令行连不上）。push 必须走本地代理，且用 `-c` 临时传入、**不要写进 .git/config**：
  ```
  git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 push -u origin main
  ```
  （10808 = V2Ray 本地端口，以 `research-paper-autopilot/references/github-publish.md` 踩坑表为准；如端口不同先查 `HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings` 的 ProxyServer。）
- 认证用浏览器生成的 classic PAT（勾 `repo` scope），走 `Authorization: Basic base64(user:token)`（**不要用 Bearer**）；token 只存会话变量、不落盘。
- PowerShell 会把 git 的 stderr 包成红字 `NativeCommandError`，**以 `$LASTEXITCODE -eq 0` 与 `* [new branch] main -> main` 判断成败**。
- 若 `Failed to connect to github.com port 443`：即代理未生效，重查上面的 `-c http.proxy`。

---

## 7. 本次不做 / 暂缓

- 不实际 push、不登录、不动浏览器（无凭据，按要求只出清单）。
- 方法总览 Figure 1（`eqwm_fig1.pdf`）**已完成**：SVG 已按修正清单改（删 λ*/closed-form 与 parameterized symmetry；补 m*/n scaling law 橙框、WM 4 子模块、reward 头、反对称 L/R 动作头、imagination 闭环、底部实验条四对应）、已导出矢量 PDF、已用跨栏 `figure*[t]` 插入 acmart §4 Method 开头、重编译 6 页 0 fatal/0 undefined。
- `data/ablation.json`（s13/s14）是否公开仍留作者裁定（见 `optimization_report.md` 第四节）。
- **表格栏宽修复（2026-10-02）**：acmart 双栏下三张三线表（hyper-parameters / WM comparison / ablation）已加 `\small`（hyper 表降 `\footnotesize`）、`\tabcolsep=3–4pt`、`@{}` 收边；WM comparison 表头缩写为 Equiv. WM / Std. WM，ablation 配置标签列改 `p{0.64\columnwidth}` 换行。表内数字/单位/表注一字未改、行列结构不变。表相关 Overfull \hbox 由 3 处降为 0（全文 Overfull 12→9，余 9 处为正文数学/证明行，非表格，不在本次范围）。
