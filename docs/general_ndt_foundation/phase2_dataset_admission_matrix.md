# Phase 2A — 数据集准入矩阵 (Dataset Admission Matrix)

> 日期：2026-09-02
> 目的：对 Phase 1 registry 中待核实的五项数据做文档级准入确认。
> **原则：无法确认就保持 B/C/D/E，不得猜测升级。**
> 网络访问在本环境被阻断（WebSearch/WebFetch 403），在线核实的项均标注"待人工确认"；
> 能由仓库内既有审计（M0_public / M0_2B / M0_2C / M0_3 / phase0-1）确认的项直接落地。

---

## 〇、准入矩阵总览

| 数据集 | 模态 | 原分级 | 本轮结论 | 依据 | 升级/降级 |
|---|---|---|---|---|---|
| MDDECT | 涡流 ECT | C | **C（保持）**：分组=defect×operator；license 不明 | M0_public §7；论文 arXiv:2104.02472 | 保持 C |
| Long-term GW SHM | 导波 | B/C | **B/C（保持）**：待人工下载；NC-ND 合规待评估；单结构 | M0/phase1 registry | 保持 B/C |
| USimgAIST | 超声图像 | D | **D（保持）**：来源/许可无法确认 | phase1 registry | 保持 D |
| external_weld_ut | 超声 FMC/PAUT | 候选 B | **候选 B → 预训练可用**：license 确认 CC BY 4.0（Strathclyde Pure Portal）；4 独立试件；无标签 | M0-3 audit（exp 分支） | **有实质进展**（license 从"未知"→"已确认 CC BY 4.0"） |
| ML-NDT / NDT_ML_Flaw | 超声 | D quarantined | **D（保持）**：LGPL-3.0 对数据的授权边界不明 | M0_public §; M0-2B v2 audit | 保持 D |

---

## 一、MDDECT（涡流 ECT）→ C（保持）

**分组调查结论（基于论文 arXiv:2104.02472 已核实事 + 仓库 audit）**：
- 独立缺陷 = **18**（18 个深度档 0.3–2.0 mm，步长 0.1 mm，每深度一缺陷）；
- 48,000 次扫描是**扫描次数**，不是独立缺陷数（论文原文 "48,000 scans from 18 defects"）；
- 多人人工扫描（operator 数量待下载核实）；lift-off 有变化（档位待核实）；
- **划分必须按 defect × operator 组合**，禁止随机扫描级划分（operator 重复扫描同一缺陷 →
  同缺陷不同 operator 的扫描共享底层缺陷 → 随机划分泄漏）。

**准入判定**：
- license：arXiv 未声明；Kaggle license 字段需登录 → **未知（无法确认）**；
- 分组结构：defect × operator 组合理论可行，但 operator 数 / lift-off 档位**未核实**；
- 在线核实被阻断（本环境无网络）→ **保持 C**，不猜测升级。
- 恢复/升级路径：人工登录 Kaggle 核实 license 字段与目录结构；若 operator 分组可严格
  落实且 license 允许，可申请 C→B（迁移验证）或进一步评估。

---

## 二、Long-term GW SHM（导波）→ B/C（保持）

**下载状态**：Figshare DOI 10.6084/m9.figshare.28112504（Sci. Data 2025,
DOI 10.1038/s41597-025-05300-5）。本机此前被 Cloudflare 拦截；**仍需人工下载确认**
（与 external_weld_ut 同类的 Cloudflare 问题，后者已由浏览器人工下载成功，说明此路径可行）。

**NC-ND 许可合规评估（文档级）**：
- **NC（非商业）**：仅限非商业研究；任何商业用途禁止 → 本项目为研究用途，需明确
  不用于商业产品。
- **ND（禁改/禁派生）**：禁止分发"修改/派生"版本的数据。**模型权重是否构成"派生"**
  在学术与法务上无定论（普遍观点：权重不属于"数据"本身，但保险做法是**)不重发布数据
  修改版、不把处理后的数据当作新数据集发布**）。
- 单结构（1 块铝板，4.5 年，13 递进损伤）→ 本就只能作**预训练语料 + 单结构迁移验证**（B/C），
  进不了 A 级核心基准（多独立试件要求）。

**准入判定**：**保持 B/C**。下载需人工完成；下载后仍需：时间分段划分元数据核实 +
NC-ND 使用声明（非商业 + 不重发布数据）。

---

## 三、USimgAIST（超声图像）→ D（保持）

- IEEE Access 9:36986-36994 (2021)（Ye/Toyama）；数据集疑似**按需索取**，无公开托管；
- license 未知；已处理图像（非原始波形）→ 即使拿到也不符合"原始 NDT 信号"主线；
- 在线核实被阻断 → **保持 D**，不猜测升级。
- 升级路径：联系 Ye/Toyama 确认下载与许可；若取得且为原始波形 + 独立试板分组，可重评。

---

## 四、external_weld_ut（真实焊缝 FMC/PAUT）→ 候选 B（预训练可用）

**依据（M0-3 audit，exp 分支 `docs/M0_3_external_weld_ut_audit.md`，2026-08-26 已落地）**：

| 项 | 事实 | 状态 |
|---|---|---|
| 来源 | **Strathclyde Pure Portal**（Vedran Tunukovic 等；Cloudflare 保护，浏览器人工下载完成） | 已确认 |
| license | **CC BY 4.0** | **已确认**（此前 registry 标"未知"→ 本轮更新） |
| 独立试件 | **4**（A=316L lack-of-fusion 焊缝板 / B=Inconel 中心线裂纹 / C=304SS 3mm SDH 块 / D=PAUT 多位置） | 已确认 |
| 样本 | 690 views / 392 采集；A/B=10000 帧 FMC(128×128/45×45)、C=976 帧、D=389 PAUT B-scan | 已确认 |
| 标签 | **无下游逐位置缺陷标签**（仅整试件已知缺陷；D 为探头定位数据）→ **仅 SSL 预训练可用，不可评测** | 已确认 |
| 独立试件数 | 4 < 10 → **exploratory external pretraining source**（非 foundation-scale） | 已确认 |

**准入判定**：**候选 B（无标签预训练）**。license 从"未知"升级为"已确认 CC BY 4.0"，
这是本轮实质进展；但无标签 → 不可作任何监督评测；SSL 预训练 + 迁移验证可用。
- 待办：配套元数据（A 的 .ods、B/C 的 .xlsx）未落地，如需探头参数需向作者/论文补全
  （不影响 SSL 预训练）。

---

## 五、ML-NDT / NDT_ML_Flaw（超声，VTT）→ D quarantined（保持）

**LGPL-3.0 数据授权边界（文档级分析）**：
- LGPL-3.0 是**软件许可证**，覆盖"源代码/程序"；对**数据文件**（B-scan 图像/信号）的
  授权语义**没有明确覆盖**，且两家仓库都未单独为数据声明许可；
- 常见法律解读：仓库整体 LGPL 只约束代码复用；**数据是否授权、以何条款授权不明**；
- 保险做法（不猜测升级）：**把数据当作"授权语义不清"处理** → 保持 quarantined，
  仅限受控消融（shortcut 负对照 / 模板依赖验证），**不得用于论文主结果/工业对比/SOTA**；
- 强烈建议：任何外部使用（即使受控）前联系作者（Virkkunen / koomas）书面确认数据许可。

**准入判定**：**保持 D（quarantined）**。shortcut 证据（随机 AUC≈1.0、近重复 99.3–99.7%、
leave-template 崩塌、E2 负迁移）不因许可问题改变。

---

## 六、汇总决策

| 数据集 | 本轮决策 | 阻塞点 | 恢复/升级条件 |
|---|---|---|---|
| MDDECT | 保持 C | license 不明 + operator/lift-off 未核实 | 人工登录 Kaggle 核实 license 与分组；可严格 operator 分组且许可允许 → 评估 C→B |
| Long-term GW SHM | 保持 B/C | Cloudflare 下载 + NC-ND 合规 + 单结构 | 人工下载；非商业声明；时间分段划分；单结构 → 永不作 A |
| USimgAIST | 保持 D | 无公开托管 + 许可未知 | 联系 Ye/Toyama；若为原始波形 + 独立试板 → 重评 |
| external_weld_ut | **候选 B（license 已确认 CC BY 4.0）** | 无标签（不可评测） | 补探头元数据（.ods/.xlsx）后可作更完整预训练源 |
| ML-NDT / NDT_ML_Flaw | 保持 D quarantined | LGPL 数据授权不明 + shortcut | 作者书面确认数据许可后才可考虑扩大用途（仍需先过 sanity checks） |

> 结论：五项中仅 **external_weld_ut 的 license 由"未知"确认为 CC BY 4.0**（有实质进展）；
> 其余四项因无法在线核实保持原分级，**无任何数据被猜测性升级**。
