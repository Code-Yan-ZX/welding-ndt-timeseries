# Reproduction Audit — Pilot A: SGAN + OpenGuidedWaves

> 阶段：reproducibility audit（训练开始前）。
> 论文：Prajapati K.K., Ghosh A., Mitra M., "Semi-supervised generative adversarial network (SGAN)
> for damage detection in a composite plate using guided wave responses",
> Mechanical Systems and Signal Processing 232 (2025) 112686, DOI: 10.1016/j.ymssp.2025.112686
> 状态标记：R0 / R1 / R2 / BLOCKED（定义见 docs 阶段规范）。

## 0. 当前判定（随审计推进更新）

- **数据可得性：已确认可下载**（§1，本地直接验证 2026-10-03/04）。
- **论文全文：PAYWALLED，无任何合法公开版本**（审计 agent 已查 Unpaywall / Semantic Scholar /
  Crossref / arXiv / SSRN / ResearchGate / IIT KGP repository；
  Unpaywall `is_oa:false, has_repository_copy:false`，发表数月后仍无 deposit）。
- **官方代码：不存在**（GitHub/GitLab/PapersWithCode 检索 0 hit；
  唯一 OGW 相关 repo 是第三方个人项目，针对 dataset #2，与本论文无关）。
- **当前等级：BLOCKED（protocol 无法唯一恢复）**，除非获得全文
  （途径：① 用户校园网/订阅下载 PDF；② 邮件通讯作者索取，
  通讯作者 Prajapati 公开邮箱 ORCID 认证 `kkpkgp@gmail.com`；③ 购买单篇）。
- 对应会议前作（同作者）：AIAA SciTech 2025-1804，同样付费墙。

### 0.1 数据结构修正（以实测 + agent 验证为准）

- 66 条路径为**单向** pitch-catch（tx∈0..10, rx∈1..11，C(12,2)=66），
  **不是** 132 有向路径。
- 90 个采集目录 = **60 baseline + D1–D28（28 个）+ 2 个 multi-damage 目录
  （`D25_D28`、`D14_D25_D28`）**。此前"D28 有 3 个目录"的解读有误，
  多出的是 multi-damage 目录。
- 28 个 damage 是**板面上的 28 个损伤位置**（7 簇 × 4 位置），非 PZT 位置。
- 损伤为可逆人工缺陷：表面粘贴铝圆盘。
- 板：500×500×2 mm CFRP，12 PZT（T1–T6 @ y=470 mm，T7–T12 @ y=30 mm，x=50–450 步长 80）。

## 1. 数据来源（已验证）

### 1.1 OpenGuidedWaves 平台

- 官方站点：**https://openguidedwaves.de/**（注意：`.org` 域名已死亡 —
  `openguidedwaves.org` 在 Google 公共 DNS 下 NXDOMAIN，2026-10-03 验证；
  旧引用/论文里的 `.org` 链接全部失效，必须用 `.de`）。
- 平台论文：Moll J., Kathol J., Fritzen C.-P., et al.,
  "Open Guided Waves — Online Platform for Ultrasonic Guided Wave Measurements",
  Structural Health Monitoring 18(5–6) (2019) 1903–1914, DOI: 10.1177/1475921718817169。

### 1.2 OGW dataset #1（论文 28 damage scenarios D1–D28 对应数据）

- 下载入口：downloads 页 `https://openguidedwaves.de/downloads/` →
  "OGW dataset #1 → Guided wave basic measurement data" →
  NextCloud share `https://nextcloud.faserinstitut.de/index.php/s/UfcaKdsx1imUrTm`。
- 直链（zip 打包）：`https://nextcloud.faserinstitut.de/public.php/dav/files/UfcaKdsx1imUrTm/?accept=zip`
- 许可：平台数据公开免费使用（学术引用 dataset 论文即可）。

**目录结构（PROPFIND 实测，2026-10-03）：**

- 共 **90 个采集目录**：
  - `20180604T164628_baseline_1` … `baseline_60`：**60 个 baseline（pristine）采集**
  - `*_D1` … `*_D28`：**30 个 damage 采集目录**（D1–D28 各 1 个；**D28 出现 3 个目录**）
- 每个 damage/baseline 目录含 **12 个 HDF5 文件** `pc_f{40,60,80,...,260}kHz.h5`
  （pitch-catch，12 个激波频率 40–260 kHz，步长 20 kHz），单文件 ~13.9 MB。
- 单目录 ≈ 167 MB；整库估计 ≈ 15 GB（zip 打包下载中，完成后以实测为准）。
- 示例（`20180605T083932_D1/`）：`pc_f100kHz.h5` 13,898,976 B … `pc_f40kHz.h5` 13,958,976 B。

### 1.3 本地保存约定

- raw 只读：`data/raw/ogw/`（zip 原样保留，解压后不改写）。
- manifest：`data/manifests/ogw/`（下载 URL、时间、checksum、per-file 大小、
  scenario→目录映射、pristine/damaged 标签表）。
- 下载命令与 checksum 记录见 manifest 内 `download.json` / `checksums.sha256`。

## 2. 论文 protocol 恢复情况（A0 的 11 个问题）

> 以下待论文全文到手后填充；不可恢复项将明确标注 UNKNOWN 并枚举候选解释。

| # | 问题 | 状态 | 结论 |
|---|------|------|------|
| 1 | 用了 OGW 哪些文件/experiment | 数据侧已锁定 dataset #1（D1–D28 结构唯一匹配）；具体频率/sensor 子集 | 待全文 |
| 2 | pristine 与 damaged 如何定义 | — | 待全文 |
| 3 | 28 scenarios → binary task | — | 待全文 |
| 4 | 一个 sample 的原始 shape | **已确认**（实读 h5）：每文件 `pitchcatch/catch` (66, 13108) float64 = 66 条有序路径 × 13108 点 @ 10 MHz（1.31 ms）；另有 `pitch` (66,13108)、`channels` (66,2) actuator/sensor 定义、`signal_frequency`（激波频率）、`CTC/Temperature`+`Humidity`、`timestamp`。5 周期 tone burst | 已确认 |
| 5 | sensor/path/frequency 选择 | — | 待全文 |
| 6 | 每条 propagation path 是否独立 sample | — | 待全文 |
| 7 | 144 / 528 的推导 | — | 待全文（候选分解见 §2.1） |
| 8 | train/test 是否跨 damage location | — | 待全文 |
| 9 | normalization | — | 待全文 |
| 10 | augmentation | — | 待全文 |
| 11 | supervised/unsupervised 样本抽取 | SGAN_36/54/72 = labeled 数 → 144 = 4×36（候选） | 待全文 |

### 2.1 144 / 528 的候选分解（基于已确认数据结构枚举，**未经论文证实**）

数据侧约束：12 sensor → 66 条无向路径（C(12,2)）；`channels` 矩阵为 66×2 有序对
（实际是 actuator 固定遍历 0..11）；pitch/catch 各 (66, 13108)；
12 个频率文件/scenario；60 baseline 采集 + 30 damage 采集（D1–D28，D28 有 3 目录）。

- **144 的候选**：
  1. `12×12 = 144`：单频率下单 scenario 的全部 actuator×sensor 组合（含对角自收发）——与 12 sensor 结构强吻合；
  2. `12 频率 × 12`：单路径跨全部 12 个频率；
  3. `24 scenarios × 6`：无结构支持，弱。
- **528 的候选**：
  1. `132 × 4 = 528`：12×11 有向路径 × 4 个 damage scenario；
  2. `66 × 8 = 528`：66 无向路径 × 8 个 damage scenario；
  3. `12 × 44`：无结构支持，弱；
  4. `528 = 4 × 132` 与候选 1 等价的 pitch+catch 双向变体：`4 × 66 × 2`。
- 注意 `528/28 ≈ 18.86` 非整数 → 若 528 全部来自 damage 侧，则**不可能**是
  28 scenarios 全部等贡献；要么只用部分 scenarios，要么 train 侧也含 damage，
  要么按 baseline+damage 混合计数。这一点必须由论文表格裁决，不得猜测。
- SGAN_36/54/72：labeled 数（论文已述）。`36 = 144/4`，`72 = 144/2`，
  与"从 144 个训练样本中抽取 36/54/72 个做 supervised"自洽（候选解释）。

## 3. 官方代码检索（已完成）

- **不存在官方代码**。GitHub API 检索 `OpenGuidedWaves` / `SGAN guided wave` 0 hit；
  `open guided waves` 唯一相关 repo 为
  [2HA-Dev/ogw-guided-waves-damage-detection](https://github.com/2HA-Dev/ogw-guided-waves-damage-detection)，
  是第三方个人项目（dataset #2 温度变化任务），与本论文无关。
- OGW 官方仅提供 MATLAB 读取器（`h5view`、`Analysis_Differential_Signal`，NextCloud 分发）。
- **结论：R1 paper-guided reimplementation**（若全文可获）。
  所有报告使用 "reimplementation"，不得写 "official reproduction"。

## 3.1 论文全文检索（已完成）

- Unpaywall：`is_oa:false`、`has_repository_copy:false`、0 个 OA location（2026-10 查询）。
- Semantic Scholar：`isOpenAccess:false`，无 open PDF。
- Crossref：无 abstract、无 funding、仅 TDM 链接（需 Elsevier API key）。
- arXiv / SSRN / ResearchGate / IIT KGP 机构库：无。
- 会议前作（同作者，AIAA SciTech 2025-1804）：同样付费墙。
- 二手信息可恢复的事实：SGAN 训练在 OGW benchmark；SGAN_36/54/72 三个监督样本量；
  最优 SGAN 比最优 CNN 准确率高 **24.62%**、比 ResNet-AE 迁移学习高 **10.96%**。

## 3.2 作者联系（备用途径）

- 通讯作者 Kamal Kishor Prajapati：ORCID 0009-0006-0381-794X，
  公开邮箱 **kkpkgp@gmail.com**（ORCID API 验证）。
- Anup Ghosh：ORCID 0000-0002-1593-002X，anup@aero.iitkgp.ac.in（二手来源）。

## 4. 泄漏风险预检（基于数据结构，最终以论文 protocol 为准）

- OGW-1 的 60 个 baseline 采集贯穿整个采集序列（D1 前后都有），
  **同一 pristine 状态多次独立采集** → 若论文把 baseline 全部当 train 且
  damage 测试只用 damage 后采集，需检查 acquisition-time drift leakage。
- 28 个 damage 场景互相独立（不同位置/尺寸的人工缺陷）→
  若 train/test 混合场景，需检查 scenario leakage。
- 12×12 通道矩阵中同一物理测量的不同 path 若分别作为 sample，
  **path 间高度相关** → 若按 path 切分 train/test 会产生 path leakage。
  论文若这样做，先严格复现，再在报告中记录风险（不擅自修正）。

## 5. 私有数据部分（预期 NOT REPRODUCIBLE）

- 论文另有作者自建 aluminium plate 数据集做 generalization。
- 预期私有 → 该部分标记 **PRIVATE / NOT REPRODUCIBLE**，不复现、不比较。
- 待全文确认是否公开。

## 6. 参考文献（数据侧）

- Moll et al., SHM 18(5-6) 2019, DOI: 10.1177/1475921718817169（平台论文）
- openguidedwaves.de downloads 页（archive: 本文件 1.1–1.2 节已固化 URL）
