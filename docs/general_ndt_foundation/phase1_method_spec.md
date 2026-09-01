# Phase 1：第一版方法设计（方法规格 V0）

> 阶段：General NDT Foundation / Physics-Aware SSL — Phase 1 方法设计
> 分支：`research/general-ndt-foundation`
> 日期：2026-09-01
>
> **工作题目（非最终方法，未宣称 novel）**：
> *Physics-Aware Self-Supervised Representation Learning for General-Purpose NDT Signals*
>
> 研究问题：能否通过 **模态适配 + 物理感知掩码 + 时域—时频联合建模 + 多源自监督学习**，
> 在不同信号型 NDT 数据（超声/PAUT、导波、涡流、声发射）之间学习**可迁移的通用表征**，
> 并改善**少样本、跨试件和跨域泛化**？
>
> 本规格只定义 **第一版可实际实现的最小结构**，不堆砌全部模块；novelty 审计列为待办。

---

## 〇、设计约束（来自 Phase 0/1 审计，必须正面回应）

1. **已知负面证据**：普通单编码器 + 单一重建目标跨物理模态迁移 → 负迁移
   （M0-2B E2−E0 = −0.0075；M0-2C E→P→E 顺序 SSL 保持 PAUT 灾难性遗忘 −0.0606）。
   → 方法必须包含**显式模态对齐机制**，且**多源必须联合训练**（不允许顺序式灾难性遗忘路径）。
2. **可用真实数据**：PENELOPE（超声，5 coupon，3000 位置）、EddyCus（涡流，148 组，738 扫描）
   已本地且可严格评测；VTT 超声与合成超声只作预训练语料（严格防泄漏分组）。
3. **统一 token 空间**：现有 `UltrasoundMAE` 是 2D patch Transformer MAE（可复用），
   但 batch 内不支持混合 shape → 设计需在 **adapter 层把各模态统一到固定 token 网格**。
4. **严格评测纪律**：同试件切片/增强副本/同缺陷重复采样绝不随机跨 train/test；
   每折均值±std；负迁移审计。

---

## 一、总体结构（图）

```
                    ┌──────────────────────────────────────────────┐
                    │            共享 Backbone（单个轻量 Patch Transformer）│
                    │         E: ℝ^{N×d} → ℝ^{N×d}  (N = 统一 token 序列)     │
                    └──────▲───────────────▲───────────────▲──────────┘
                           │               │               │
                 ┌─────────┴──┐   ┌───────┴────┐   ┌──────┴────────┐
                 │ 1D Stem    │   │ 2D Stem    │   │ 时频 Stem      │
                 │(channel×time)│  │(time×space)│   │(STFT 谱图)     │
                 └─────────┬──┘   └───────┬────┘   └──────┬────────┘
                           │               │               │
          ┌────────────────┴── 模态适配器(含 metadata embedding) ──┴──────────┐
          │  X_raw:  ℝ^{C×T}  (通道×时间)     X_spatial: ℝ^{T×S}   S=STFT(X)  │
          │  (AE/导波/ECT曲线)               (B-scan/C-scan 栅格)   (log-mag) │
          └───────────────────────────────────────────────────────────────────┘
                              ▲
              ┌───────────────┴───────────────┐
              │ 物理感知掩码控制器 (mask controller)  │
              │ random / time-segment / freq-band / │
              │ sensor-channel / spatial-region     │
              └─────────────────────────────────────┘

  自监督目标（2~3 项）:
    L = λ₁·L_recon (masked reconstruction)
      + λ₂·L_tf    (raw ↔ time-frequency view consistency)
      (+ λ₃·L_inv  (cross-sensor invariance, 仅当有配对多传感器数据时启用))
```

## 二、输入 / 输出

**输入**（统一样本结构，见 Phase 1 代码部分 `GeneralNDTSample`）：

```yaml
signal:            # 原始/轻度预处理信号, 3 种形态之一
  - {channels: C, time: T}      # 1D: 通道×时间 (AE 波形 / 导波 A 扫 / ECT 曲线)
  - {time: T, spatial: S}       # 2D: 时间×空间 (PAUT B-scan / ECT C-scan 栅格)
mask:              # 该样本是否带 self-supervised mask 计划
modality:          # ultrasonic / guided_wave / eddy_current / acoustic_emission
sample_id / specimen_id / sensor_id / sampling_rate / spatial_coordinates
label:             # 可选 (0/1, 缺陷类型, 深度/尺寸, 或无标签)
metadata:          # 数据集名、license、split_group、data_origin...
```

**输出**：
- 预训练产物：冻结 encoder `E`（每模态/每域 token 序列 → 池化表征 `z ∈ ℝ^d`）。
- 下游产物：linear probe / fine-tune 分类头或回归头。

## 三、模块结构

### 3.1 模态适配器（Modal Adapter）

| 组件 | 作用 | 输入→输出 |
|---|---|---|
| 1D Stem | 将 channel×time 信号切成 patch 并映射到 token | `ℝ^{C×T}` → patchify(conv1d, kernel=τ, stride=τ) → `ℝ^{N₁×d}` |
| 2D Stem | 将 time×space 栅格切成 patch 网格 | `ℝ^{T×S}` → patchify(conv2d, p×p) → flatten → `ℝ^{N₂×d}` |
| 时频 Stem | 将 STFT 谱图当作 2D 图像切 patch | `ℝ^{F×T_f}` → patchify(conv2d) → `ℝ^{N₃×d}` |
| Metadata 嵌入 | 把 modality / sensor / sampling-rate / 空间坐标编码进 token | 加法式 token embedding |

**要点**：
- 三个 stem 都很轻（单层 conv + LayerNorm + 激活），**骨干共享**；stem 是唯一按模态区分的部分。
- 统一 token 序列长度通过 `pad_to + mask`（valid token 掩码）对齐，支持 batch 内混合 shape。
- **采样率**不强制重采样：位置编码按 `t * sampling_rate` 缩放（连续位置编码），
  时间物理尺度进入 token，而不是把不同采样率数据粗暴对齐。
- modality / sensor 用可学习 embedding 加性注入；spatial_coordinates 作为可选位置编码叠加。

### 3.2 时域—时频双视图（Dual-View: raw ↔ time-frequency）

- **视图 A（原始波形/栅格）**：`X` 经对应 stem → token → 共享 encoder → 池化 `z_raw`。
- **视图 B（时频）**：`S = log|STFT(X)|`（第一版只引入 STFT，固定窗长 128、hop 64、Hann 窗；
  CWT 作为扩展）→ 时频 Stem → 共享 encoder → 池化 `z_tf`。
- **一致性目标**：InfoNCE（同一样本两视图为正对，batch 内其余为负对），温度 τ。

### 3.3 物理感知掩码（Physics-Aware Masking）

与普通随机掩码对比的 4 种物理掩码模式：

| 模式 | 掩码对象 | 物理动机 | 目标域示例 |
|---|---|---|---|
| `random` | 随机 token | baseline | 所有 |
| `time_segment` | 连续时间段的全部 token | 强制从其余时间恢复被遮挡时间段（回波时序结构） | 超声 A 扫、导波、AE |
| `freq_band` | 时频视图连续频带 | 强制从其它频带推断被遮挡频带（频散/共振结构） | 导波、超声、涡流多频 |
| `sensor_channel` | 掩掉整个传感器/波束通道 | 强制跨传感器冗余恢复（多传感器鲁棒性） | EddyCus 8 传感器、PAUT 波束 |
| `spatial_region` | 栅格上连续空间区域 | 强制从相邻空间位置恢复（缺陷空间连续性） | B-scan/C-scan 栅格 |

**第一版策略**：掩码控制器按数据集/样本采样掩码模式（每种模式一个可配置概率），
mask_ratio 沿用 0.3–0.5；在**消融实验**中逐一对比 `random` vs 各物理模式 vs 混合。

### 3.4 共享骨干（Shared Backbone）

- **第一版只选一个 backbone**：轻量 Patch Transformer（2D patch 风格，与仓库 `UltrasoundMAE`
  同构但简化），4–6 层、d_model=128、MLP 4×、GELU、可加性 position embedding。
- 选择理由：与已验证的仓库 MAE 结构一致（CPU 可训、风险低）、token 长度可变、
  易于 1D/2D 统一；Mamba 列为**后续扩展**（不在第一版同时实现两个复杂骨干）。

### 3.5 自监督目标（2–3 项）

```
L = λ₁·L_recon + λ₂·L_tf [+ λ₃·L_inv]

L_recon = 1/|M| · Σ_{i∈M} || (x̂_i − x_i) ⊙ v_i ||₂²      # MAE 式, 只算 masked∩valid
L_tf    = −log[ exp(z_raw·z_tf/τ) / Σ_j exp(z_raw·z_j/τ) ]  # InfoNCE 双视图
L_inv   = 1/|P| · Σ_{(a,b)∈P} || E(x_a^l) − E(x_b^l) ||₂²    # 同物理位置 l 的传感器 a,b (可选)
```

- **L_recon**：每模态一个轻量重建头（解卷积/linear），把 masked token 还原到原始 patch
  （1D 波形 patch 或 2D 栅格 patch）；只用 `valid` token 计算（batch 内混合 shape 时关键）。
- **L_tf**：原始视图与时频视图的一致性；作用是强制 encoder 学到"时间"与"频率"两个互补视角
  共享的物理表征，防止只拟合单一视图捷径。
- **L_inv**（**条件启用**）：当某数据集存在"同物理位置 × 多传感器/多频"配对时启用
  （EddyCus：同配置多传感器；PENELOPE：90/270 族）。无配对数据时 λ₃=0。
  作用：显式注入跨传感器不变性（物理量不变、传感器响应不同）。

### 3.6 下游任务（Evaluation）

| 任务 | 数据集候选 | 指标 |
|---|---|---|
| 缺陷/健康二分类 | PENELOPE（5 coupon LOOCV）、EddyCus（cross-config） | AUROC、balanced acc、Macro-F1 |
| 多类缺陷分类 | EddyCus（8 类）、VTT 超声 | Macro-F1、confusion |
| 严重度/深度回归 | ML-NDT（eq. size）、MDDECT（深度 18 档）、Long-term GW（13 级） | MAE/RMSE、秩相关 |
| 正常样本训练的异常检测 | PENELOPE clean 位置、EddyCus clean(84)、Pipeline UGW healthy(207) | AUROC/AUPRC（留一结构） |

## 四、数学目标汇总

见 §3.5。总体目标：`L = λ₁L_recon + λ₂L_tf + λ₃L_inv`，其中 λ 第一版取 λ₁=1.0、λ₂=0.1、
λ₃=0.0（条件 1.0）。预训练后冻结 encoder，下游用 **linear probe**（规范化头协议：
lr=1e-3 / 80ep，沿用仓库规范）与 **全量 fine-tune** 两种。

## 五、与普通 MAE / 对比学习 / 单模态 SSL 的区别

| 维度 | 普通 MAE | 普通对比学习 (SimCLR 式) | 单模态 SSL | **本方法 V0** |
|---|---|---|---|---|
| 掩码 | 随机 | 无（用增强） | 随机/领域特定 | **物理感知掩码（时间/频带/传感器/空间区域）+ 随机对照** |
| 视图 | 单视图重建 | 增强视图对比 | 单视图 | **原始 × 时频 双物理视图**（非合成增强） |
| 模态 | 单域 | 单域 | 单域 | **多源联合：模态适配 + 共享 token 空间 + metadata 嵌入** |
| 对齐 | 无 | 无 | 无 | **跨传感器不变性（条件启用）+ 视图一致性** |
| 主目标 | 重建 | 对比 | 重建/对比其一 | **重建为主 + 视图一致性为辅（2 目标 MVP）** |

## 六、最小可行版本（MVP）

- 模态：**2 种**（超声 PENELOPE/合成超声 + 涡流 EddyCus）——均已本地、可严格评测。
- 目标：**L_recon + L_tf**（跳过 L_inv）。
- 骨干：4 层 Patch Transformer，d_model=128，单卡 CPU/低显存可训。
- 掩码：`random` + `time_segment` 两种（2D 栅格加 `spatial_region`），mask_ratio=0.3。
- STFT：固定窗 128 / hop 64 / Hann / log-mag。
- 预期产出：冻结 encoder + linear probe 的 PENELOPE LOOCV 与 EddyCus cross-config 结果，
  与 scratch / vanilla MAE / target-only SSL / multi-source SSL 对照（见实验协议）。

## 七、后续扩展（非第一版）

1. 骨干升级：Mamba / 混合 1D-2D backbone（在多模态 token 序列上）。
2. 时频扩展：CWT 或可学习的 Gabor 滤波器组；多尺度 STFT。
3. 掩码扩展：可学习掩码策略 / 强化选择物理掩码模式。
4. 目标扩展：跨模态对齐（超声↔涡流同物理语义，需配对数据）；物理量预测（声速、衰减）。
5. 数据扩展：导波（Long-term GW / Pipeline UGW）、AE（NASA SiC/SiC）接入。
6. 下游扩展：缺陷定位（FMC/TFM 成像）、分割。

## 八、可能失败的原因（提前声明，实验协议中监测）

1. **模态不对齐**：token 空间异构 → adapter 无法统一 → 负迁移依旧。
   → 监测：按模态/数据集逐折指标；负迁移审计。
2. **单域主导**：合成超声 10 万样本淹没真实数据 → 学到的表征偏合成分布。
   → 多源采样比例控制（按数据量/模态平衡采样）。
3. **视图一致性坍缩**：InfoNCE 使所有表征趋同 → 判别信息丢失。
   → 温度 τ 调参、projection head、stop-gradient 方案。
4. **掩码太易/太难**：time_segment 在长信号上恢复过易（相邻信息充足）或过难（无交叉信息）。
   → 按掩码模式分别报告重构损失与下游指标。
5. **严格评测数据稀缺**：只有 PENELOPE（5 coupon）与 EddyCus（148 组）可严格评测，
   单结构导波/AE 数据无法验证跨试件泛化。
   → 结论范围限定；导波/AE 仅作迁移验证与预训练。
6. **物理感知掩码不带来增益**：即"物理掩码 ≈ 随机掩码"（可能被证伪——这本身是贡献）。

## 九、Novelty 审计待办（不提前宣称 novel）

- [ ] 检索"physics-aware mask / time-frequency contrastive SSL / NDT foundation model"
      最新文献（2024–2026），尤其：NDT 领域基础模型、导波/AE 自监督、超声表征学习。
- [ ] 与已知时序/信号基础模型逐一对比并声明差异：Moirai、MOMENT、TimesFM、TOTEM、StorSeismic、
      USFM（超声传感基础模型，如有）、PENELOPE 相关工作。
- [ ] 与仓库既往负面结果的关系定位：本方法 ≠ 普通单编码器单目标；须在报告里明确"挑战而非
      否定"既有结论。
- [ ] 若发现高度重叠工作 → 明确差异化（数据、掩码、下游协议），或调整定位。

## 十、输出物索引

- 方法规格：本文档
- 实验协议：`docs/general_ndt_foundation/phase1_experiment_protocol.md`
- 数据集 registry：`configs/general_ndt_datasets.yaml`
- 代码骨架：`src/general_ndt/`（Phase 1 代码部分）
