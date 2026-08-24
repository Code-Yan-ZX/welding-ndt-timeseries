# M0-2C 本地涡流 ECT 数据盘点与接入方案

> 阶段：M0-2C 第一步（只盘点，不下载 / 不训练 / 不改 checkpoint）
> 日期：2026-08-24
> 目标：为 **PAUT SSL encoder → ECT continued SSL pretraining** 确定输入方案，
> 回答"本机有哪些 ECT 数据、能否支撑 encoder 迁移"。
> 合规：本次未下载任何新数据、未运行任何训练、未覆盖任何 checkpoint、未修改历史结果。

---

## 0. 结论先行（TL;DR）

1. **本机当前 ECT 数据数量 = 0。** 全盘搜索（项目 data/raw、data/processed、
   manifests、configs、`/home/Datasets`、`/home/yzx` 全目录、`/opt`、`/srv`、
   `/media`、归档文件、HF cache、git 历史）**没有发现任何涡流 ECT 文件**。
   项目内已落地的三个外部数据集（PENELOPE / ML-NDT / NDT_ML_Flaw）均为**超声
   （B-scan）**，manifest 的 `primary_modality` 全部为 `ultrasonic`。
2. 审计（`docs/M0_public_ndt_dataset_audit.md` §7–8）已确认两个候选 ECT 数据集，
   **均未下载**：
   - **MDDECT**：真实金属 ECT（304 不锈钢平面试件，18 深度档），但为 **1D I/Q
     曲线**，且需 Kaggle 登录、license 不明；
   - **EddyCus-HDF5**：CFRP 多传感器多频 ECT，**2D 空间网格 + I/Q（复数阻抗）**，
     CC BY 4.0、免登录、3.7 GB，与本机 PAUT 2D 卷积 encoder 结构最匹配。
3. **本机没有可继续 SSL 的 ECT 语料 → 必须先落地一个 ECT 数据集才能做
   "PAUT encoder → ECT continued SSL"。** 若只以"输入方案"为目的，首选
   **EddyCus-HDF5**（结构匹配 + 许可清晰 + 免登录），MDDECT 作为独立 1D 基线，
   不适合直接迁移 PAUT 2D encoder。
4. 推荐的标准化 tensor：**单频点、单传感器 → `(I, Q) 双通道 2D 网格`，resize/pad
   到与 PAUT 一致的 `(2, 49, 512)`**（方案 B，见 §6）。物理上最合理、改动最小。
5. 推荐的 SSL 任务：**空间块掩码重建（MAE 风格，2D 网格上掩码）**，**不能沿用
   PAUT 的 beam 掩码**；**必须新建 ECT decoder**（PAUT decoder 重建目标是
   (49,512) 波束结构，物理语义不通用）。
6. **可以加载** P4a/P1 PAUT SSL encoder 的 **22/23 个权重**（方案 B 下仅
   `conv.0` 首层输入通道 1→2 需重建）；是否**值得**加载需用 M0-2B 同款
   **E2−E0 停止判据**（≥+0.01 且 2/3 seed 为正）判断——超声→涡流跨模态迁移比
   M0-2B 的超声→超声迁移**更激进，先验上更不乐观**，必须带对照组。

---

## 1. 搜索方法与覆盖范围（证明"本地无 ECT"这一结论可靠）

搜索范围与命令见附录 A。关键点：

| 搜索位置 | 结果 |
|---|---|
| `data/raw/{saw,ML-NDT,NDT_ML_Flaw}`（含 .gitignore 忽略区） | 全部为超声/工艺数据，无 ECT |
| `data/processed/{paut,saw,synth_ut,synth_ut_50x2k}` | 全部为超声处理产物 |
| `data/manifests/*/dataset_card.json` | 3 个数据集 `primary_modality` 全为 `ultrasonic` |
| `configs/*.yaml` | 无 ECT 路径/配置 |
| `src/wndt/data/adapters/` | 只有 3 个超声 adapter（penelope / ml_ndt / ndt_ml_flaw），无 ECT adapter |
| `/home/Datasets`（共享数据盘） | 仅 nuScenes / VOC（自动驾驶），无 NDT 数据 |
| `/home/yzx` 全目录（含隐藏目录、cache） | 无 ECT 文件；HF cache 仅 MOMENT 模型权重 |
| `/opt`、`/srv`、`/media`、`/mnt`、`/tmp` | 无 NDT 数据（/tmp 为他人机器人实验临时文件） |
| 全盘归档（zip/tar/tgz/7z）与 `.mat/.npy/.npz/.h5/.hdf5/.lzma/.xz` | 项目外零命中 |
| 文件名/目录名关键字（`ect`/`eddy`/`eddycurrent`/`eddycus`/`mddect`） | 零命中 |
| git 历史（`git log --all`） | 无任何 ECT 相关 commit（最近提交均为 M0-2A/M0-2B 超声数据落地） |

> ⚠ 边界说明：本服务器为项目迁移后的新服务器（见 CLAUDE.md 环境备注）。
> 若 ECT 数据曾在**旧服务器**上下载，则不在本机，需人工确认旧机或重新下载。

---

## 2. 本机全部 NDT 相关数据清单（用于证明"没有涡流"）

### 2.1 已落地的外部数据集（data/raw，全部为超声）

| 数据集 | 模态 | 本地路径 | 文件数 | 大小 | 完整性 | README/license |
|---|---|---|---|---|---|---|
| PENELOPE / SAW | 超声 PAUT `.nde`(HDF5) + 焊接工艺 `.hdf5` + 图片 | `data/raw/saw/ZENODO_Penelope/` | 690 | 69 GB（zip 12.7 GB 已解压） | 完整（5 coupons：PP3–PP7 + Coupon1/2，MD5 校验过） | `README.txt`/coupon + Zenodo DOI 10.5281/zenodo.15083865 |
| ML-NDT (VTT) | 超声 B-scan 图像 `.bins`(uint16, 100×256×256) | `data/raw/ML-NDT/` | 809 | 2.5 GB | 完整（**201/201** 容器） | `README.md` + LGPL-3.0 |
| NDT_ML_Flaw (VTT) | 超声 B-scan 条带 `.xz`/`.lzma`(uint16 480×7168) | `data/raw/NDT_ML_Flaw/` | 37 | 227 MB（解压 ~117 GB） | 完整（**17/17**：7 真实 + 10 模拟） | `README.md` + LGPL-3.0 |

### 2.2 处理产物（data/processed，全部为超声）

| 路径 | 内容 | 大小 |
|---|---|---|
| `data/processed/paut/` | `ascans_mv.npy` (2995,4,49,512) float32 + 标签/坐标/norm 统计（PAUT 主线数据） | 3.0 GB |
| `data/processed/saw/` | SAW 工艺信号窗口化 | 4.0 GB |
| `data/processed/synth_ut/`、`synth_ut_50x2k/` | 程序生成超声（P7） | 1.2 + 9.4 GB |

### 2.3 共享数据盘 /home/Datasets

仅 `nuscenes`、`nuscenes_trainval`、`VOC`（自动驾驶/视觉），**无任何 NDT 数据**。

> 结论：**本机 NDT 相关数据 100% 为超声模态，无涡流（ECT）数据。** 任务 3–5
> 的"抽样读取"无法执行——本地没有可抽样的 ECT 文件。以下 §4–5 基于
> `docs/M0_public_ndt_dataset_audit.md` 已确认的文档事实，凡"待下载核实"项均明确标注。

---

## 3. PAUT SSL encoder 现状（迁移的"源"）

### 3.1 最佳 checkpoint（PP3–PP7，非PP4 逐折均值口径）

| checkpoint | 来源 | 掩码策略 | 非PP4 逐折均值 | 备注 |
|---|---|---|---|---|
| `experiments/runs/ssl_ae/encoder.pt` | P1 | beam 掩码 (mask_ratio 0.3) | **0.5713** (s42) | 单种子最佳 |
| `experiments/runs/ssl_ae_both/encoder.pt` | P4a | beam+depth 掩码 | 0.5658 / 0.5669 / 0.5666 (s42/43/44) | **三种子稳定，推荐作为迁移源** |
| `experiments/runs/ssl_p6_base/encoder.pt` | P6 | 同上类 | 0.5560 (s42) | 对照 |

（数值来源：`experiments/results/paut_p4a_baseline_*_full.json`；baseline 无 SSL = 0.5106。）

> 注意：M0-2B 的 `experiments/runs/m0_2b/pretrain/*.pt` 是**另一种架构**
> （统一 MAE：`patch_embed(128,1,16,16)` + 4 层 Transformer），**不是**本节
> 讨论的 PAUT MaskedAE 卷积 encoder，迁移 ECT 时应以 `ssl_ae_both` 为源。

### 3.2 PAUT MaskedAE encoder 结构（`src/wndt/models/ssl_ae.py`）

```
MAEEncoder: (B,1,49,512)
  conv: Conv2d(1,32,(3,7),pad(1,3)) → BN → GELU → MaxPool(2)
        Conv2d(32,64,(3,7)) → BN → GELU → MaxPool(2)
        Conv2d(64,128,(3,7)) → BN → GELU → MaxPool(2)
  pool: AdaptiveAvgPool2d((1,1))
  proj: Flatten → Dropout → Linear(128, 128)   # d_model=128
```

实际 checkpoint 状态键（已用 `torch.load` 核验，共 23 键）：

| 键 | shape | 含义 |
|---|---|---|
| `conv.0.weight/bias` | (32, **1**, 3, 7) / (32,) | **首层，输入通道=1** |
| `conv.1.*`（BN32）| (32,) ×4 | 批归一化 |
| `conv.4.*` | (64, 32, 3, 7) | 第 2 层 |
| `conv.5.*`（BN64）| (64,) ×4 | |
| `conv.8.*` | (128, 64, 3, 7) | 第 3 层 |
| `conv.9.*`（BN128）| (128,) ×4 | |
| `proj.2.weight/bias` | (128, 128) / (128,) | 输出投影 |

输入数据统计（PAUT）：`ascans_mv.npy` (2995,4,49,512) float32，原始幅值范围
≈[3, 7120]，均值≈111.8、std≈146.1（原始 uint16 幅值，训练前做 per-timestep
标准化）。49 = 波束（beam，扫描位置），512 = 深度/时间采样。

---

## 4. 候选 ECT 数据集（审计已确认，本机未下载）

### 4.1 EddyCus-HDF5（首选候选）

| 项 | 事实 | 来源 |
|---|---|---|
| Zenodo | record 19251759，DOI 10.5281/zenodo.19251759（2026-03-27 v1.0） | audit §8 |
| 数据 | `eddy_current_data.zip` 3.7 GB，**738 次多频扫描** | audit §8 |
| 文件格式 | **HDF5**；单文件 50–500 KB；gzip L6+shuffle | audit §8 |
| schema | 4 层组：`measurement_metadata/`（frequencies+sample_properties）、`spatial_data/`（**x/y/z mm**）、`signal_data/fN/`（**real/imaginary/complex_impedance**，复合 dtype）、`analysis_results/fN/`（magnitude/phase） | audit §8 |
| 传感器 | **8 个 Fraunhofer IKTS 传感器**（3 绝对式 + 5 差分半透射），6.1–24.3 MHz | audit §8 |
| 材料 | 3 种 CFRP 无屈曲织物（非金属焊缝） | audit §8 |
| 缺陷 | 8 类：Gap 492 / 无缺陷参考 84 / 错铺层 80 / PTFE 膜 24 / 铜膜 24 / 镀铜丝束 24 / 波纹 6 / 毛球 4 | audit §8 |
| license | **CC BY 4.0**（数据）；转换软件 MIT | audit §8 |
| 下载 | **免登录** | audit §8 |
| 待核实 | 独立试件数、单扫描网格尺寸（x/y 点数）、频率数 | audit §8 |

**为何结构上匹配 PAUT encoder**：数据含 2D 空间坐标（x/y mm）与每频点
real/imaginary（I/Q），可天然排布为 **`(I,Q) 双通道 2D 网格`**，与 2D 卷积
encoder 的输入形态（单通道 2D 图）同构——只需把通道数 1→2。多频/多传感器
可作额外通道或独立样本。

### 4.2 MDDECT（备选，结构不匹配）

| 项 | 事实 | 来源 |
|---|---|---|
| 论文 | *Depth Evaluation for Metal Surface Defects by ECT using DRNN*, arXiv:2104.02472 | audit §7 |
| 数据 | 48,000 次扫描（=18 深度档 × 多 operator × lift-off 变化，**非独立缺陷**）；独立缺陷 18（深度 0.3–2.0 mm，步长 0.1 mm） | audit §7 |
| 形态 | **1D I/Q 曲线**（Zynq-7020 采集 + I/Q 解调） | audit §7 |
| 材料 | 304 不锈钢薄板平面试件（非焊缝几何） | audit §7 |
| 任务 | 缺陷深度档分类（论文 1D ResNeXt-38 93.58%） | audit §7 |
| 下载 | **Kaggle 需登录**；license **未知**；operator/lift-off 分组待核实 | audit §7 |
| 与本目标关系 | 语义接近焊缝 ECT，但 **1D 信号与 PAUT 2D encoder 形态不匹配**，不能直接迁移；只适合作为独立 1D 深度分类基线 | 评估 |

### 4.3 其它审计项

- **EddyNet**：涡流逆问题仿真代码，无 license → Reject（audit §9）。
- **UT+ECT 融合（WAAM, NDT&E 2026）**：仅论文公开、数据需联系作者 → 当前 Reject
  （audit §10；与项目红线一致：无同试件同坐标配对数据不做融合）。

---

## 5. 数据独立性 / 标签分析（基于审计文档；标注待核实）

> 本地无 ECT 文件，无法做真实抽样统计。以下为 EddyCus / MDDECT 的**文档已知**事实，
> 与本项目已建立的"记录数 ≠ 独立物理单元数"教训（M0-2B VTT 审计）直接相关。

| 维度 | EddyCus（文档已知） | MDDECT（文档已知） |
|---|---|---|
| specimen 数 | **待下载核实**（zip 内 sample_properties） | 1 试件（304 薄板） |
| defect_instance 数 | 8 类缺陷，数量已知（见 §4.1） | **18**（=18 深度档） |
| acquisition/重复扫描 | 738 次多频扫描；同一试件/缺陷多次扫描待核实 | 48,000 扫描 = 18 缺陷 × operator × lift-off |
| operator 数 | 待核实 | 多人人工扫描，**数量待核实** |
| sensor/probe 数 | **8**（3 绝对 + 5 差分） | 待核实 |
| lift-off 档位 | 待核实 | 存在变化，档位待核实 |
| material 数 | 3 种 CFRP 织物 | 1 |
| clean 与 flaw | 84 clean / 654 flaw（文档口径） | 深度档即缺陷，无 clean 概念 |
| 标签位置 | HDF5 内 metadata + 文件名/目录 | Kaggle 表格/文件名，待核实 |

**关键提醒（沿用 M0-2B 教训）**：EddyCus 的 738"扫描"若含同一试件/传感器/频率
的重复，**有效独立单元会远小于 738**；分组划分必须按 specimen / defect_instance
（必要时 + sensor / material），**禁止随机扫描级划分**，否则跨组泛化被高估
（与 `docs/M0_2B_VTT_virtual_flaw_data_audit_v2.md` 的结论同构）。MDDECT 的
48,000 扫描更是明确的"扫描次数≠独立缺陷"。

---

## 6. 三种接入方式分析（按"实际数据形状"）

> 因本地无 ECT 文件，本分析基于 **EddyCus 的文档化结构**（2D 网格 + I/Q），
> 并给出对 MDDECT（1D）的独立说明。PAUT encoder 输入为 (B,1,49,512)。

### 方案 A：单通道二维 ECT tensor，直接加载原 PAUT encoder

- 形状：`(1, H, W)`，取 **magnitude（幅值）** 单通道；H×W = 单频点单传感器
  的空间网格，resize/pad 到 (49,512)。
- 代码改动：**最少（≈0 行架构改动）**。22/23 权重全部可加载（`conv.0` 输入
  通道仍为 1）。
- 缺点：**丢弃相位（I/Q 中 Q 通道）**。ECT 缺陷检测相位信息是关键
  （相位与缺陷深度/取向强相关），纯幅值会损失信息；且 MDDECT 这类 I/Q 数据
  无法用本方案（幅值也 OK，但仍是 1D→2D 强排布）。

### 方案 B：I/Q 或 amplitude/phase 双通道，改第一层输入通道后加载

- 形状：`(2, H, W)`，通道 = (real, imag) 或 (amplitude, phase)；resize/pad 到
  (2,49,512)。
- 代码改动：**只改 `conv.0` in_channels=1→2**（约 672 参数）。初始化建议：
  将原 (32,1,3,7) 权重**复制到两通道**（或取平均），其余 conv.4/conv.8/BN/proj
  全部原样加载（22/23 键）。
- **物理上最合理**：保留 I/Q 相位信息，同时 2D 网格与 PAUT 卷积同构。

### 方案 C：ECT 专属 stem + 共享 encoder

- 结构：`ECT stem（如 2→32ch 卷积）→ 加载 conv.4+conv.8+proj 的共享 encoder`；
  或 stem 直接产出 32 通道特征图再进入原 conv 栈。
- 代码改动：**最多**（新增 stem 模块 + 加载逻辑）。灵活性最高（可把多频/
  多传感器堆叠进通道，如 2×n_freq 通道）。
- 缺点：加载的权重只剩后 2 层 conv + proj（约 2/3 参数），且 stem 需从头训练，
  迁移收益更依赖 ECT 侧数据量。

### 对比表

| 维度 | A（幅值单通道） | B（I/Q 双通道） | C（ECT stem） |
|---|---|---|---|
| 代码改动 | 最少 | 很小（仅 conv.0） | 最多 |
| 物理合理性 | 中（丢相位） | **最高**（保相位） | 高（可多频/多传感器） |
| 可加载权重 | 23/23 | **22/23** | 约 2/3 |
| 新建模块 | decoder + masking | conv.0 + decoder + masking | stem + decoder + masking |
| 对 MDDECT(1D) 适用性 | 差（1D→2D 强排布） | 差（同上） | 差（1D 应走 1D 基线） |

> **推荐：方案 B。** 它同时满足"物理上最合理"与"改动最小"，且保留 ECT 最关键
> 的相位信息；若后续发现多频堆叠更优，可平滑升级为 C（stem 输入通道扩展）。

---

## 7. 权重加载与必须新建的模块

### 7.1 哪些权重能加载（以 `ssl_ae_both/encoder.pt` 为准，已核验键名）

| 模块 | 方案 A | 方案 B | 方案 C |
|---|---|---|---|
| `conv.0`（1→32） | ✅ 原样 | ⚠ 通道复制初始化（1→2） | ❌ 由 ECT stem 取代 |
| `conv.4`（32→64）、BN | ✅ | ✅ | ✅（stem 输出对齐 32ch） |
| `conv.8`（64→128）、BN | ✅ | ✅ | ✅ |
| `proj`（128→128） | ✅ | ✅ | ✅ |

### 7.2 必须新建

1. **ECT decoder**：PAUT 的 `MAEDecoder` 把 z 上采样回 **(49,512) 波束结构**
   （mid 7×64 + bilinear 到 49×512），其语义是"重建被掩码的波束"。ECT 重建
   目标是 2D 空间网格，**物理语义不同** → 新建（或在 ECT 网格 pad 到 49×512
   后复用形状，但仍是"形状复用、语义新训"）。
2. **masking 策略**：PAUT 是 **beam 掩码（整行置零）** +（P4a 加 depth 块掩码）。
   ECT 应为 **2D 空间块掩码（MAE 风格随机块）**，不宜沿用"掩码整行波束"
   （ECT 网格无"波束"概念）。
3. 方案 B 的 `conv.0` 输入通道、方案 C 的 stem。

---

## 8. 训练样本量与显存估算（预期）

- **样本量（EddyCus，若选定）**：738 扫描 × 8 传感器 × n 频点 可切片出数千个
  `(2,H,W)` 帧，但**独立物理单元 = 试件/缺陷级，远小于帧数**（沿用 VTT 审计
  口径）。若把"频点 × 传感器"当独立样本，相当于 M0-2B 中"ML-NDT 变帧数"的
  虚高陷阱 → **必须按 specimen/defect 划分，帧级仅作增强**。
- **显存**：MAEEncoder 参数量级 ~20 万，d=128、batch 128–256、输入 (2,49,512)
  时激活 <1 GB（任意单卡可跑）。**显存不是约束，数据独立单元才是。**
- 若走 MDDECT：48,000 帧 1D I/Q，但独立缺陷仅 18 → 同样受限于独立单元数。

---

## 9. 明确结论

1. **本地 ECT 数据：无。** 本机全部 NDT 数据为超声模态；要做
   "PAUT SSL encoder → ECT continued SSL"，**必须先落地一个 ECT 数据集**。
2. **首选数据集：EddyCus-HDF5**（Zenodo 19251759，3.7 GB，CC BY 4.0，免登录）。
   它是唯一在结构（2D 网格 + I/Q）、许可、下载便利性上都满足本目标的外部 ECT
   数据。MDDECT 作为独立 1D 深度分类基线（不用于 encoder 迁移）。
3. **推荐 tensor shape：`(2, 49, 512)`**——单频点单传感器的 (real, imag)
   双通道网格，resize/pad 到与 PAUT 相同空间尺寸（方案 B）。
4. **推荐 SSL 任务：2D 空间块掩码重建（MAE 风格）+ 新 ECT decoder**；
   **不沿用 PAUT beam 掩码与 decoder**。损失沿用 Huber（smooth_l1）可。
5. **是否适合加载 P4a/P1 PAUT SSL encoder：可以做，但必须带对照组**——
   22/23 权重可加载（conv.0 双通道复制初始化）。判据沿用 M0-2B：
   **E2（PAUT→ECT 续训）− E0（ECT scratch）≥ +0.01 且 ≥2/3 seed 为正**。
   ⚠ 超声→涡流是**跨模态**迁移，比 M0-2B 的超声→超声更激进，先验不乐观；
   E1（纯 ECT 目标域 SSL）在 PAUT 上系统性为负的历史也提示**续训未必有益**，
   本实验的意义正是**用最小成本证伪/证实**这一点。
6. **下一步最小实验矩阵**（下载数据后方可执行，本轮不执行）：

   | 步骤 | 内容 | 产出 |
   |---|---|---|
   | S0 | 下载 EddyCus 3.7 GB + 解包校验 | 核实 specimen 数、单扫描网格尺寸、频率数、标签表 |
   | S1 | 按 `M0_unified_ndt_schema.md` §涡流 写 manifest + adapter | `data/manifests/eddycus/`、`src/wndt/data/adapters/eddycus.py` |
   | S2 | **E0**：ECT scratch MAE SSL（方案 B 形状）→ 规范头（lr 1e-3/80ep）冻结探针 | 基线 AUC/acc（按 specimen/defect 分组 LOOCV 或 cross-material/sensor 划分） |
   | S3 | **E2**：加载 `ssl_ae_both/encoder.pt`（conv.0 双通道复制）→ ECT 续训 MAE → 同头协议 | E2−E0 差值（3 seed，判据 ±0.01 & 2/3 正） |
   | S4 | （可选）**冻结零样本**：不续训，直接 PAUT encoder 提取 ECT 特征探针 | 判断"直接迁移"与"续训迁移"的差别 |
   | S5 | 汇总进 README 实验日志 + `scripts/make_table.py` 口径 | 结论入档 |

   > 全程遵守项目红线：不做 UT+ECT 融合（无配对数据）；不扩大 10 GB 以上下载；
   > 主指标 = 按物理单元分组的逐折均值（非 pooled）。

---

## 附录 A：本报告使用的核验命令（证据链）

```bash
# 1) 项目内全部数据目录
find data -type f | wc -l; du -sh data/raw/* data/processed/* data/manifests/*

# 2) 全盘 ECT 关键字（文件名/目录名）
find /home /opt /srv /media /mnt -xdev \( -iname "*ect*" -o -iname "*eddy*" \
  -o -iname "*eddycus*" -o -iname "*mddect*" -o -iname "*eddy_current*" \) \
  -not -path "*/.venv*/*" -not -path "*/.cache/*" -not -path "*/anaconda3/*"  # 0 命中

# 3) 全盘数据文件格式
find /home /opt /srv /media -xdev \( -iname "*.mat" -o -iname "*.npy" -o \
  -iname "*.npz" -o -iname "*.h5" -o -iname "*.hdf5" -o -iname "*.lzma" -o \
  -iname "*.xz" \) -not -path "*/welding-ndt-timeseries/*" ...  # 项目外 0 命中

# 4) 归档文件
find /home /opt /srv /media -xdev \( -iname "*.zip" -o -iname "*.tar*" -o \
  -iname "*.7z" \) -not -path "*/welding-ndt-timeseries/*" -not -path "*/nuscenes*/*"

# 5) 项目内 ECT 引用
grep -rniE "ect|eddy|涡流" --include="*.py" --include="*.yaml" --include="*.md" \
  --include="*.json" .   # 全部为超声文档/审计文档/计划文档中的"未来 ECT"提及

# 6) checkpoint 权重核验
torch.load("experiments/runs/ssl_ae_both/encoder.pt")  # 23 键，conv.0=(32,1,3,7)

# 7) manifest 模态核验
cat data/manifests/{penelope,ml_ndt,ndt_ml_flaw}/dataset_card.json  # 全 ultrasonic
```

## 附录 B：本任务合规声明

- ❌ 未下载任何新数据（含 EddyCus / MDDECT）；
- ❌ 未运行任何训练/预训练/微调；
- ❌ 未覆盖/删除任何 checkpoint 与历史结果；
- ✅ 仅只读盘点 + 本报告。
