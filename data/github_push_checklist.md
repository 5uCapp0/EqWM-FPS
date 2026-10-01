# GitHub 推送需求清单（EqWM-FPS）

> 仓库：`github.com/5uCapp0/EqWM-FPS`（Public，2026-10-01 18:0x 已推送 M4：`paper/eqwm_main.tex/.pdf` + 4 张矢量图）。
> 本清单**只评估打包，不实际 push**。本机无 GitHub 凭据、无 gh CLI、无 git clone；实际 push 由 MainAgent 稍后走浏览器 PAT 流程执行。
> 生成日期：2026-10-02。

---

## 1. 相对基线的本地变更文件清单

基线 = 2026-10-01 18:0x 已推送的 M4 快照（`eqwm_main.tex/.pdf` + `paper/figs/fig_*.pdf`）。本次本地新增/变更：

| 文件 | 状态 | 是否建议推送 | 说明 |
|---|---|---|---|
| `paper/eqwm_acmart.tex` / `.pdf` | 新（19:29，acmart sigconf 版，基线未含） | **建议推** | 美观预览版（Overleaf/arXiv）；与 elsarticle 版并存 |
| `paper/eqwm_acmart_v2.tex` / `.pdf` | 新（本次润色，6 页，0 fatal/0 undefined） | **建议推** | acmart 润色定稿 |
| `paper/eqwm_main_v2.tex` / `.pdf` | 新（本次润色，16 页，0 fatal/0 undefined/0 warning） | **建议推** | elsarticle 润色定稿（投稿 Neural Networks 用） |
| `paper/references.bib` | 未改 | 随 v2 一并推（若基线已有则跳过） | 24 条，内容未动 |
| `paper/figs/*.pdf`（4 张） | 未改 | 已在基线 | 无需重推 |
| `data/optimization_report.md` | 新（本次内部评审报告） | **建议不推**（加入 .gitignore） | 与 `data/m4_review.md` 同类——内部工作/评审记录，按脱敏清单不进公开仓库 |
| `data/github_push_checklist.md` | 新（本文件） | **建议不推**（加入 .gitignore） | 内部打包清单 |
| `data/m4_review.md` | 已有，已 gitignore | 保持不推 | 基线已排除 |
| `项目进度.md` | 已有，已 gitignore | 保持不推 | 基线已排除 |
| `README.md` | 已有 | **建议修订后再推** | 见第 4 节：现有 README 数字偏乐观/陈旧，需与论文诚实结论对齐 |
| `paper/main.tex`、`paper/main_upload.tex` | 已有（18:59 / 17:14） | **不建议单独维护** | 经评估：二者约为 `eqwm_main.tex` 的 Overleaf 副本/变体，属陈旧重复；建议仓库以 `eqwm_main*.tex` 为准，这两个副本删除或注明"Overleaf 镜像，勿改" |

> 注：`data/ablation.json`（21:14，s13/s14 重跑）与论文表格数字不一致，已在 `optimization_report.md` 第四节说明。**是否随仓库推送该数据文件需作者先裁定**：若推送，README/论文须能解释它与报告版三-seed 表格的关系，避免公开数据与论文表格对不上。

---

## 2. 建议仓库结构

```
EqWM-FPS/
├── README.md                 # 研究问题 / 目录 / 环境 / 复现命令 / 结果摘要（诚实数字）/ 论文状态
├── .gitignore
├── code/                     # env/ models/ training/ experiments/ utils/  + run_*.py
├── data/                     # 实验输出 JSON（公开结果数据）；内部 *.md 评审报告排除
├── figs/                     # 项目根 png 图（可选）
└── paper/
    ├── eqwm_main_v2.tex / .pdf        # elsarticle 投稿版（主）
    ├── eqwm_acmart_v2.tex / .pdf      # acmart 预览版
    ├── references.bib
    └── figs/                          # 4 张矢量图 + 未来 eqwm_fig1.pdf
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

---

## 4. README.md 需修订的点（推送前）

现有 `README.md` 的"实验结果摘要"仍为写作早期的乐观/陈旧口径，**与论文诚实结论不一致**，公开前须改：

- "实验一 ……（约降 53%）；开环 MSE@10/20 步低 20%/28%" → 论文已定位为**奖励头跨种子稳定性提升**（0.104±0.025 vs 0.223±0.099），并显式否定"53% 准确率"；开环 MSE 因指标有缺陷**已删除**。
- "实验二 ……（约 2–3× 样本效率）" → 论文口径为 **order-of-2×（实测 ~1.3–2.7×），2 seeds、曲线平坦噪声、Eq/Std 不可区分**。
- "实验三消融（seed0）both_eq 491±187 / wm_eq_only 401±100 ……" → 论文表格为**四配置 447.1±37.7 / 425.3±28.5 / 426.7±26.7 / 449.8±46.7，425–450 区间重叠不可区分、none-eq 数值反而最高**；seed0 单值口径已过时。
- "教材 ch15.7 的 M_eff≈15.7 主张不稳健" → 公开 README 不应出现"教材/ch15.7"等内部痕迹；改为"bootstrap 乘子 M_eff>1 非单调、跨机不一致，作为 open problem"。
- 顶部"项目状态：实验中（实验三/实验四进行中）" → 更新为"论文 M4 完成，v2 润色稿就绪，投稿 Neural Networks / ECAI·UAI"。

---

## 5. 建议 commit 粒度与 message

```
# 1) 论文润色定稿（acmart + elsarticle v2）
git add paper/eqwm_main_v2.tex paper/eqwm_main_v2.pdf \
        paper/eqwm_acmart_v2.tex paper/eqwm_acmart_v2.pdf \
        paper/references.bib
git -c user.name="5uCapp0" -c user.email="5uCapp0@users.noreply.github.com" \
    commit -m "paper: add polished v2 (elsarticle submission + acmart preview)"

# 2) acmart 预览版（若基线确未含 eqwm_acmart.tex）
git add paper/eqwm_acmart.tex paper/eqwm_acmart.pdf
git commit -m "paper: add acmart sigconf preview edition"

# 3) README 与 .gitignore（修订诚实数字 + 排除内部报告）
git add README.md .gitignore
git commit -m "docs: align README with honest empirical findings; ignore internal review reports"
```
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
- `data/ablation.json`（s13/s14）与论文表格的取舍留给作者裁定（见 `optimization_report.md` 第四节），本清单不替作者决定是否公开该文件。
- 新增方法总览 Figure 1（`eqwm_fig1`）需作者先把 SVG 导出为矢量 `paper/figs/eqwm_fig1.pdf` 并按 `optimization_report.md` 第六节修正"λ\*/parameterized symmetry"等概念后，再单独提一个 `paper: add architecture overview figure` commit。
