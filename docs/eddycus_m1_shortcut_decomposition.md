# M1 — Shortcut Decomposition：EddyCus 跨条件泛化失败的机制分解

> 日期：2026-10-03　分支：`research/general-ndt-foundation`
> 前置：NDT 公开 benchmark pilot（commit 618e5d0）——P0/P1 AUC 0.93-0.99 vs P3
> material-held-out 0.55-0.73，sensor-ID probe 95-98%，判定 GO。
> 代码：`scripts/m1_eddycus_*`　结果：`experiments/results/eddycus_m1/`
>
> **M1 只回答一个问题**：跨 sensor / material generalization failure，究竟主要来自
> 简单的 **amplitude shortcut**，还是来自更深层的 **sensor/material-dependent
> representation**？（目标不是提出新方法。）

---

## 0. Executive Summary

1. **Gate A 触发（结构性 label confound，实锤）**：完全不读取信号、仅用采集元数据
   （传感器/频率组合/栅格尺寸/板温/采集月份等 62 个标量）训练的 RF/LR，
   P1 AUC = **1.000**、P0 = 0.99、P3 material-held-out = **0.92-0.94**——
   不读任何信号就打平甚至超过 pilot 全部 7 个信号方法（后者 P3 仅 0.55-0.73）。
   频率组合单变量在 P1 上已达 0.98。
2. **Amplitude-only 同样覆盖 P1**：每扫描仅 96 个全局幅值统计量（无空间纹理），
   RF 在 P1 达 **0.997**、P3 坍塌到 0.49-0.74——与 pilot CNN 的完整剖面
   （P1 0.93-0.99 / P3 0.55-0.73）几乎相同。**pilot 的 CNN 在这个任务上没有学到
   任何超越"幅值直方图"的东西。**
3. **但 P3 坍塌不是 amplitude shortcut（Gate B 不触发）**：7 种逐扫描归一化
   （N1 z-score / N2 robust / N3 unit-energy / N4 mag-norm complex / N5 rank /
   N6 phase-only）**没有任何一种改善 P3**——多数反而更差（N2 最差 0.44-0.45，
   N0 参照 0.74/0.68 最好）；同时部分归一化把 sensor-ID probe（balanced）从
   0.64 压到 0.34-0.36，P3 却不动。**"sensor 可读性"与"held-out 泛化"被解耦了。**
4. **Gate C 触发 → GO**：归一化后 sensor/material probe 仍明显高于 chance
   （raw acc 0.84-0.99，majority 0.96；bal-acc 部分 arm 0.64-0.79），
   P3 gap 保持。失败机制是 sensor/material-dependent 的**空间/物理表征**
   （背景纹理、频率响应、栅格/提离差异），幅值只是其中最表层、且**不是**
   P3 gap 的主因。
5. **Gate D 触发（conditional）**：发现采集端显式记录的跨传感器 campaign
   （`GB_170629-170705`：同一 gap 试件配置 × 最多 6 传感器 × 3 方向；70 个
   confirmed-by-session 跨 sensor 扫描对），但**全部为 defect 扫描，clean 零配对**。
6. **Binary task 判定**：全局 normal-vs-defect 二分类应从主 benchmark 降级。
   推荐主 benchmark = **restricted binary（clean 所在的 S13131 生态位内
   leave-material-out）** + 跨条件表征 benchmark（GB campaign 配对检索）；
   defect-type 分类因类型-材料-采集 campaign 强耦合而被否决。

---

## 1. Metadata-only shortcut（不读取任何信号）

`scripts/m1_eddycus_metadata_shortcut.py` → `metadata_shortcut.csv/_summary.json`

特征组（A-E，全部排除 label-derived 字段 description / defect_depth /
defect_size——clean 的 depth/size 恒为 0，属直接标签泄漏）；模型 LR / RF
（AUC 二者一致，RF 有并列分数）；协议 P0 / P1 / P3 二分类 fold。

| 特征组 | n_feat | P0 | P1 | P3::HP-U300 | P3::Kohlegelege |
|---|---|---|---|---|---|
| A sensor | 7 | 0.536 | 0.522 | 0.500 | 0.500 |
| B material | 6 | 0.786 | 0.891 | 0.500 | 0.500 |
| C frequency（组合+MHz+增益 db_ac） | 16 | 0.888 | **0.981** | 0.851 | 0.565 |
| D sensor+material | 13 | 0.786 | 0.891 | 0.500 | 0.500 |
| **E all-acquisition**（+fiber/layup/厚度/方向/栅格/点数/年月/板温/trigger） | 62 | **0.994** | **1.000** | **0.916/0.932** | **0.940/0.921** |

（P3 两列为 LR / RF；A-D 行为 LR，RF 与 LR 完全一致）

要点：

- **E 组在所有协议上 ≥ pilot 最好的信号方法**（pilot P1 最佳 0.995、P3 最佳
  0.731 vs 这里 1.000 / 0.93-0.94）。
- 6 类缺陷分类（gap/mis_orientation/ptfe/copper_foil/copper_roving/clean）：
  E 组 P0 bal-acc **0.854** / P1 **0.812**（majority 0.798 / 0.890）——
  **defect type 本身也大部分由采集/材料侧决定**（类型↔材料↔campaign 耦合）。
- RF feature importance 前列：board_temperature、n_points、layup、acq_month、
  频率组合、f4 增益——**全部是采集侧变量，无一涉及信号**。
- 机制：clean 扫描只存在于特定"采集生态位"（主传感器 + 主频组合 + 特定
  session/参数组合），metadata 直接把 clean 生态位从 defect 里分离出来。

**结论（Gate A）：STRUCTURAL LABEL CONFOUND 成立。** random/specimen-disjoint
的 binary AUC 不得再作为方法有效性的核心证据；pilot 的 P0/P1 高分应重新解读为
"任务含大量可由采集侧恢复的结构"。

## 2. Amplitude-only shortcut（禁空间纹理）

`scripts/m1_eddycus_amplitude_shortcut.py` → `amplitude_shortcut.csv/_summary.json`

每扫描（valid native grid，无 resize、无逐像素输入）提取 Real/Imag/Magnitude 的
mean/std/median/IQR/min/max/ptp/RMS/q05-q95/dynamic-range + Phase 圆统计，
共 172 个标量（另有 amp+FFT 对照 arm，FFT 全局统计无增益，见 CSV）。

| arm | 模型 | P0 | P1 | P3::HP-U300 | P3::Kohlegelege |
|---|---|---|---|---|---|
| amp_only | LR | 0.954 | 0.988 | 0.736 | 0.675 |
| **amp_only** | **RF** | **0.996** | **0.997** | 0.635 | 0.492 |
| amp_plus_fft | RF | 0.989 | 0.996 | 0.643 | 0.514 |

对照 pilot（同协议，8ch CNN 系列）：P0 0.927-0.985 / P1 0.911-0.995 /
P3 0.385-0.731。

**判读**：

1. 仅幅值直方图就完整复现了 CNN 的"random-split 高 / material-held-out 坍塌"
   剖面——pilot CNN 的 P1 成绩没有超越幅值统计的成分；
2. FFT 纹理统计无增益 → 弱空间全局信息不增加判别力；
3. amplitude-only 的 P3（0.49-0.74）与 CNN（0.55-0.73）同域 → **P3 gap 同样
   存在于幅值层**，但注意这与 §3 的结果一起读：归一化消除幅值尺度并不修复 P3，
   说明 P3 gap 的载体是"幅值分布形状/背景纹理"这类与材料/传感器绑定且
   per-scan 归一化保留下来的成分，而非整体 scale。

## 3. Conditional normalization ablation（核心消融）

`scripts/m1_eddycus_norm_ablation.py` → `norm_ablation.csv / norm_ablation_core_table.csv /
norm_ablation_probes.json / embedding_N*_P1_s42.npy`

单一 backbone：**ResNet18 scratch**（8ch，与 pilot cnn-scratch-RN18 完全同超参；
N0/P1/seed42 复现校验 AUC 0.9747 ≈ pilot 0.974±0.01）。3 seeds (42/43/44)，
val 调阈值；probe = P1 训练模型(seed42) penultimate(512d) + logistic
（P1 train/test；**majority baseline：sensor 0.96，material 0.88**）。

归一化 arm（全部逐扫描独立统计，无 test-population 信息）：
N0 RAW（pilot 参照）｜N1 per-scan z-score｜N2 (median/IQR)｜N3 unit-energy｜
N4 magnitude-normalized complex（单尺度 α 后 asinh，保留 I/Q 相对结构）｜
N5 Gaussian rank｜N6 phase-only (cosφ, sinφ)。

**核心表**（3-seed 均值 ± std 见 CSV；AUC）：

| Normalization | P1 defect | P3::HP-U300 | P3::Kohlegelege | Sensor probe (bal) | Material probe (bal) |
|---|---|---|---|---|---|
| N0 RAW（参照） | 0.974 | **0.740** | **0.682** | 0.643 | 0.391 |
| N1 per-scan z-score | 0.924 | 0.545 | 0.490 | 0.714 | 0.610 |
| N2 robust (med/IQR) | 0.915 | 0.447 | 0.440 | 0.337 | 0.302 |
| N3 unit-energy | 0.942 | 0.615 | 0.502 | 0.356 | 0.424 |
| N4 mag-norm complex | 0.932 | 0.696 | 0.626 | **0.786** | 0.447 |
| N5 rank-gauss | 0.972 | 0.635 | 0.589 | 0.357 | 0.519 |
| N6 phase-only | 0.980 | 0.643 | 0.593 | **0.786** | 0.418 |

（P0 参照列：0.86-0.98，全 arm 随 P1 一起饱和；sensor/material probe raw-acc
全部 0.84-0.99，majority baseline 0.96/0.88——**读 bal-acc 才有意义**）

**三个决定性观察**：

1. **没有任何归一化改善 P3**（N0 0.740/0.682 是全场最好；最激进的 N2 反而最差
   0.447/0.440）。→ 简单归一化不足以修复跨材料泛化（Gate B 不触发）。
2. **sensor-probe 与 P3 解耦**：N2/N3/N5 把 sensor probe bal-acc 从 0.64 压到
   0.34-0.36，P3 却同样坍塌；N4/N6 的 sensor probe 反而最高（0.786）而 P3 中游。
   → "模型能读出 sensor"不是 P3 失败的直接原因；**幅值/尺度可读性 ≠ 泛化瓶颈**。
3. **P1 在所有 arm（含完全去幅值的 N5 rank 0.972、N6 phase-only 0.980）保持
   0.92-0.98** → 扫描内部的相对空间结构本身携带足够 defect 证据；
   而 P3 依然坍塌 → 跨材料失败发生在**空间/纹理表征层**，
   且该表征是 sensor/material-dependent 的。

## 4. Sensor/material leakage probes（对比 pilot）

pilot（RAW 输入，inet-RN18 微调 embedding）：sensor-ID probe acc **98%** / bal 0.914；
material acc 94% / bal 0.632。M1（scratch RN18, N0）：sensor acc 98.7% /
**bal 0.643**，material acc 94% / bal 0.391。

- raw-acc ~0.94-0.99 的表象主要是 **majority-class（S13131 占 93%）**；
  bal-acc 才是有效读数——N0 下 sensor bal 0.64、material bal 0.39，
  显著高于 chance 但远低于表象。
- 归一化对 sensor bal-acc 的压缩（0.64→0.34）与 P3 的不响应（§3），
  是 M1 最重要的单条证据。

## 5. Physical paired-data audit（Gate D）

`scripts/m1_eddycus_paired_audit.py` → `paired_audit.json / paired_groups.csv /
paired_cross_sensor_pairs.csv`

物理配置组 = (material, fiber, layup, description, depth, size, thickness) 哈希
（**数据集无显式试件 ID，全部为推断**）。本次新增：从
`measurement_metadata.scan_parameter_comment` 提取显式 session 标签
（`GB_YYMMDD*` / `YYMMDD_Jan`）作为配对置信度证据。

| 统计 | 值 |
|---|---|
| 物理配置组 | 146（695 扫描） |
| 组内 ≥2 sensor | **7 组**（且全部 ≥3 sensor；组均 4.43 个 sensor） |
| 跨 sensor 扫描对 | **70 对，全部 confidence=confirmed-by-session** |
| 多频率组合（跨文件） | 7 组；非主频组合扫描 239/695 |
| clean 组 | 18 组（17 组跨多日期重扫，但 **0 组跨 sensor**） |
| 配对缺陷类型 | 全部为 **gap**（GB campaign）+ 少量 ondulation/fuzz_ball/gap（Jan campaign） |

session 结构：`GB_170629/170630/170703/170705`（2017-06/07，Fabric-524 gap 试件，
0/0.5/1.0 mm 三深度 × 0°/90°/45° 方向，**最多 6 个传感器复扫同一配置**）；
`160713-160809_Jan`（2016-07/08，S13132/S15152，ondulation/fuzz_ball/gap）。

**置信度分级**：confirmed-by-session = 组内 ≥2 sensor 且共享显式 session 注释
（采集端记录的多传感器复扫活动）；其余 inferred-metadata（可能是同型试件的
不同物理个体，标记 uncertain）。**注意：即使 confirmed 组，"同一物理试件"仍
是强推断而非数据集保证。**

**判定（Gate D）：HIGH-VALUE STRUCTURE（conditional）成立**——存在天然
cross-sensor correspondence（70 对），但有两个硬约束：
(a) 全部配对为 defect 扫描，**clean 零配对**（与"clean 只被 1 sensor 扫过"互为
表里）；(b) 覆盖面窄（7/146 组，仅 gap 等少数类型）。

## 6. Binary task 有效性判定

任务书三个选项的裁决：

- **A. defect-type classification under unseen sensor/material：否决。**
  defect type 与材料/采集 campaign 强耦合（metadata-only 6 类 bal-acc 0.81-0.85；
  非主传感器数据只覆盖 gap/ondulation/fuzz_ball 两三个类型）——
  类型预测在 held-out 条件下仍是"预测 campaign"，不是"识别缺陷物理"。
- **B. restricted binary（仅在同 sensor/material stratum 内含正负类时比较）：
  采纳为主 benchmark。** 由于全部 63 个 clean 扫描来自 S13131，任何二分类
  benchmark 在结构上必然是 S13131 内的 benchmark——不如显式承认并把协议定义为
  **"same-sensor, leave-material-out defect detection"**（即现有 P3 二分类 fold，
  HP-U300 545def/27clean 与 Kohlegelege 35def/36clean），明示 scope。
- **C. cross-condition representation benchmark：采纳为第二 benchmark（M2 起）。**
  利用 GB campaign 的 70 个 confirmed 跨 sensor 对做 pair-retrieval / alignment
  指标（sensor-invariance），与 restricted binary 互补。
- **P0/P1 全局 random binary：降级为 sanity check**，不再作为方法证据
  （Gate A：metadata-only 已 1.000）。

## 7. Final mechanism judgment

对 M1 核心问题的回答：

> 跨 sensor/material generalization failure **不是**主要来自简单的 amplitude
> shortcut；它来自 (i) **结构性 label confound**（clean 只存在于一个采集生态位，
> 连 metadata 都能高精度恢复标签）+ (ii) **sensor/material-dependent 的空间/物理
> 表征**（背景纹理与频率响应绑定采集条件，per-scan 归一化无法消除）。

证据链：
1. metadata-only P1=1.000 / P3=0.92-0.94（§1）→ 任务标签可由采集侧恢复，
   P0/P1 高分含大量结构水份；
2. amplitude-only 完整复现 CNN 剖面（§2）→ CNN 未超越幅值统计；
3. 7 种归一化无一改善 P3，且 sensor-probe 可读性与 P3 解耦（§3）→
   幅值 scale 不是瓶颈；剩余 gap 在空间/物理表征层；
4. clean 生态位 + 70 个 defect-only 跨 sensor 对（§5）→ 失败与可利用结构
   同源：都是"采集条件 ↔ 标签"耦合的表现。

## 8. Decision Gates

| Gate | 条件 | 判定 |
|---|---|---|
| **A** | metadata-only 高度预测 defect | **TRIGGERED**（P1 1.000 / P3 0.92-0.94）→ 禁止用 random binary 作为方法证据 |
| **B** | 简单归一化显著提高 held-out 且 probe 大降 | **NOT triggered**（P3 无改善，多 arm 更差；且 P3 改善应先怀疑与归一化无关的结构因素）→ 方法创新价值不在归一化侧 |
| **C** | 归一化后 probe 仍高于 chance 且 gap 保持 | **TRIGGERED → GO**（sensor bal-acc 部分arm 0.64-0.79、raw-acc≈0.96+；P3 gap 保持）|
| **D** | 大量可信 same-defect-different-sensor 配对 | **TRIGGERED（conditional）**：70 对 confirmed-by-session；约束：defect-only、clean 零配对、覆盖窄 |

## 9. GO / NO-GO 与 M2 最小 hypothesis

**判定：GO（条件性）。** 但 M2 的问题定义必须同时面对两件事：
(i) 表征层面存在真实、非幅值性的跨条件泛化失败（Gate C）；
(ii) 任务层面 global binary 已被证伪（Gate A），任何方法改进只能在
restricted binary + cross-condition representation benchmark 上主张。

**M2 最小 hypothesis**（Gate C + D 均成立后的最小可验证命题）：

> 用 GB campaign 的 70 个 confirmed 跨 sensor 配对（同物理 gap 配置、不同传感器）
> 做 **defect-side cross-sensor paired alignment**（不引入对抗技巧、不建大网络：
> 最小实现 = 现有 scratch RN18 encoder + 成对对齐损失，clean 侧不参与对齐），
> 是否能在不升高 sensor-ID probe 的前提下，改善
> (a) restricted binary P3（same-sensor leave-material-out AUC）
> 与 (b) 跨 sensor pair-retrieval 指标？

关键风险（先于实现声明）：
- 配对只覆盖 defect（clean 无法配对）→ 对齐可能只学"defect 表征传感器不变"，
  而 restricted binary 还需要 normal 背景跨材料迁移——这正是 pilot 中
  clean 生态位问题的另一面，M2 结果可能为负；
- 7 组/70 对样本极小，任何正结果都需要 per-fold 严格协议 + 多 seed
  （沿用 [[feedback-eval-rigor]]：val-test gap 是真假单一最可靠指示器）；
- 若 M2 为负，则结论收敛为"该公开数据上不存在可修复的跨条件表征问题，
  benchmark 问题大于方法问题"，同样是可发表的诚实结果。

---

## 10. 复现

```bash
# 1 metadata-only（~2 min CPU；LR/RF，label 字段已排除）
.venv/bin/python scripts/m1_eddycus_metadata_shortcut.py
# 2 amplitude-only（~3 min CPU；valid native grid 全局统计）
.venv/bin/python scripts/m1_eddycus_amplitude_shortcut.py
# 3 归一化消融（~11 min GPU 4090D, CUDA_VISIBLE_DEVICES=1）
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m1_eddycus_norm_ablation.py
# 5 paired audit（~1 min CPU）
.venv/bin/python scripts/m1_eddycus_paired_audit.py
```

| 产物 | 路径 |
|---|---|
| metadata 结果 | `experiments/results/eddycus_m1/metadata_shortcut{.csv,_summary.json}` |
| amplitude 结果 | `experiments/results/eddycus_m1/amplitude_shortcut{.csv,_summary.json}` |
| 消融逐 run / 核心表 | `experiments/results/eddycus_m1/norm_ablation{.csv,_core_table.csv,_meta.json,_probes.json}` |
| 各 arm P1 embedding (fp16) | `experiments/results/eddycus_m1/embedding_N*_P1_s42.npy` |
| paired audit | `experiments/results/eddycus_m1/paired_audit.json / paired_groups.csv / paired_cross_sensor_pairs.csv` |

约定：split seed=42（沿用 pilot manifests，未重新划分）；torch seeds 42/43/44；
probe seed=42；classical 单 seed=42。消融输入基于 pilot 缓存
`grid_96x384.npy`（raw I/Q，空洞逐通道中位数填充，等比缩放进 96×384 盒），
逐扫描统计在填充后图像上计算（各 arm 一致，不影响 arm 间可比性）。
