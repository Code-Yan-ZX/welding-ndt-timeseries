# M0-2C EddyCus-HDF5 真实数据审计与接入设计

> 阶段：M0-2C（下载 + 真实数据审计 + 接入设计 + adapter/manifest/smoke；**不训练**）
> 日期：2026-08-24
> 目标：为 **PAUT SSL encoder（`ssl_ae/encoder.pt`，P1 beam-mask，非PP4 0.579±0.007）
> → ECT continued SSL pretraining** 确定输入方案与实验设计。
> 合规：已下载并校验官方数据；未运行任何训练；未覆盖任何历史 checkpoint；
> 原始数据不提交（`data/raw/` 在 .gitignore）。

---

## 0. 结论先行（TL;DR）

1. **EddyCus-HDF5 已完整落地**：Zenodo 19251759，`eddy_current_data.zip`
   3,657,641,862 字节，**md5 = `814f496342d77eb2eeabb1e0d34645c3` 校验通过**；
   解压 `data/raw/EddyCus-HDF5/output/` 共 **738 个 `scan_XXXXX.h5`**（3.74 GB）。
2. **数据结构**：每文件 = **1 次扫描**（1 传感器 × 1 样品的多频 2D 栅格 C-scan），
   **4 个频率**（f1–f4，MHz 值随传感器而异），每频率 1D I/Q 信号（real/imaginary
   float64 + magnitude/phase），`spatial_data/{track_number, sample_number, x_mm,
   y_mm, z_mm}` 可重建 **2D 栅格**（主流 101×451，另有 51×451 / 202×1067 /
   501×560 等）。
3. **独立性**：738 文件 = 738 次扫描（`sample_properties.id` 是**扫描序号**，
   **不是试件号**）；元数据无显式试件 ID。最细物理配置组（material, fiber,
   layup, description, defect_depth, defect_size, thickness）= **148 组**；
   缺陷组（description+depth+size）= **133 组**；**8 类缺陷**
   （gap 492 / mis-orientation 80 / clean 84 / Cu foil 24 / Cu roving 24 /
   PTFE 24 / ondulation 6 / fuzz ball 4）。**sensor×frequency 绝不是独立物理
   样本**（8 传感器 × 4 频率 × 148 配置 ≠ 4736 独立单元）。
4. **数据质量**：695/738（94.2%）有信号；**43 文件（5.8%）仅元数据无信号**
   （2022-11 批次，scan_00460 附近）；**36 文件（4.9%）有信号但 x/y/z mm 为 NaN**
   （可用 track/sample 编号重建网格）。
5. **输入方案（推荐）**：每（扫描, 频率）→ **I/Q 双通道 `(2, H, W)`**，
   H×W = 原生栅格（H=track 数，W=每 track 采样数，≈450–451）；**优先保留原生
   网格**，跨扫描尺寸不一致时 **batch 内 padding**（栅格本身近规则 ±1 点），
   仅超大网格（501×560 / 202×1067）必要时才**等比例下采样**；**不强制 49×512**
   （encoder 有 AdaptiveAvgPool2d，任意 H×W 可用）。
6. **第一层迁移**：`Conv2d(1,32,3×7) → Conv2d(2,32,3×7)`，
   `new_weight = old_weight.repeat(1,2,1,1)/2`（已数值验证：双通道拷贝输入时输出
   与原单通道一致，diff<1e-4；**22/23 权重可加载**）。
7. **实验设计**：E（scratch）vs **P→E**（加载 `ssl_ae/encoder.pt` 续训），
   同数据 / 同 steps / 同新 ECT decoder / 同 head / 同 seed；评估含 **ECT 下游**
   （P→E vs E，按物理单元 cross-material / cross-sensor 划分）与 **PP3–PP7 回测**
   （P→E 的 encoder 回 PAUT 任务，检查**灾难性遗忘** vs 原 0.579±0.007）。
   判据沿用 M0-2B：**P→E − E ≥ +0.01 且 ≥2/3 seed 为正**。

---

## 1. 下载与校验

| 项 | 值 |
|---|---|
| 来源 | Zenodo record 19251759，DOI 10.5281/zenodo.19251759 |
| 文件 | `eddy_current_data.zip` |
| 期望大小 | 3,657,641,862 字节（HEAD content-length 核对一致） |
| 官方 md5 | `814f496342d77eb2eeabb1e0d34645c3` |
| 本机 md5 | `814f496342d77eb2eeabb1e0d34645c3` ✅ |
| 解压 | `data/raw/EddyCus-HDF5/output/`，**738 个 `scan_XXXXX.h5`**，3.74 GB |
| zip 内容 | 739 条目 = `output/` 目录 + 738 h5（**无 README**，文档见 Zenodo 页） |
| 原始数据 | 不入库（`data/raw/` 已 gitignore） |

> 下载方式备注：本机 127.0.0.1:7890 代理当前 TLS 故障，改为**直连** + aria2c
> 16 连接续传（`-c -x16 -s16`），数分钟内完成；全程未走代理。

---

## 2. 文件结构（HDF5 schema，实测 scan_00001）

```
scan_XXXXX.h5
├── measurement_metadata/            (attrs: sensor_type / scan_parameter_comment /
│   │                                   measurement_datetime / trigger_rate_hz /
│   │                                   trigger_mode / encoder / normal_vector /
│   │                                   preamplifier / roboter_type / adc_rate...)
│   ├── frequencies/f1..f4            (attrs: frequency_mhz / db_ac / id / phase_deg /
│   │                                   x_offset / y_offset / x_y_ratio)
│   └── sample_properties             (attrs: id / material_type / fiber_type /
│                                       manufacturer / layup_sequence /
│                                       layer_thickness_mm / thickness_mm /
│                                       defect_depth_mm / defect_size_mm /
│                                       description / sensor_orientation_degree /
│                                       custom_custom_sensor_type)
├── spatial_data/                     track_number / sample_number / x_mm / y_mm / z_mm
├── signal_data/f1..f4/               complex_impedance (structured
│                                     [('real','<f8'),('imaginary','<f8>')]) /
│                                     real (float64) / imaginary (float64)
└── analysis_results/f1..f4/          magnitude / phase_degrees / phase_radians (float64)
```

标签全部在 **HDF5 attrs**（`sample_properties`），不在文件名/目录名。

---

## 3. 全量统计（738 文件，脚本 `scripts/m0_2c_eddycus_audit.py`）

### 3.1 文件 / 频率 / 传感器 / 材料

| 维度 | 数量 | 明细 |
|---|---|---|
| HDF5 文件 | **738** | `scan_00001.h5`–`scan_00738.h5` |
| 每文件频率数 | **4**（f1–f4） | 全部 738 文件一致 |
| 传感器（去重字符串 9 → 实际 8 种） | **8** | S13131 7.3MHz（650+42 带引号变体，同传感器）、S15152 24.3MHz×10、S15172 8.5MHz×9、S14150 9.2MHz×6、S14152 24.3MHz×6、S17257 17.0MHz×6、S13132 6.1MHz×9 |
| 材料（material_type 字符串） | **5** | HP-U300/122C（HT 50K）、Kohlegelege ST 50g（ZOLTEK PX35 50K）、0/90° 524 g/m²（Toho HTS40 12K）、0/90 565 g/m²（Toho HTS45-E23 12K）、0/90/45/-45° 1013 g/m² |
| 材料×纤维×铺层配置 | 25 | 如 [0/90]×61、[90/90]×69、[0/90/90/0]×71、ST50g [0]×30 … |
| 每文件频率集 | 随传感器 | S13131: {4,7,8,12}MHz×456 或 {4,6,8,12}MHz×188（42 个带引号文件 freq=0.0 为导出占位） |

### 3.2 缺陷（标签位于 sample_properties.description + defect_depth/size）

| 缺陷类 | 数量 | 判定关键词 |
|---|---|---|
| gap（纤维间隙） | **492** | "gap"（含 10/15/20/25/30mm，第一/二层，0/90、45/90、90/0、90/45、90/90 铺层） |
| clean（无缺陷参考） | **84** | "reference without defect" / "Reference sample" |
| mis-orientation（错铺层） | **80** | "mis-orientation in the upper 90° stack" |
| copper foil（铜膜 10×10mm） | **24** | "copper film inserted" |
| copper coated roving（镀铜丝束） | **24** | "copper coated roving inserted" |
| PTFE 膜 | **24** | "PTFE" |
| ondulation（波纹，5/10/15mm 幅值） | **6** | "ondulation" |
| fuzz ball（毛球 60/120/180mg） | **4** | "fuzzy ball of carbon fibers" |
| **合计** | **738** | 与审计文档 §8 完全一致 ✅ |

### 3.3 I/Q 与幅值/相位统计（float64）

| 量 | 全文件范围（f1–f4 聚合） | 单文件示例（scan_00001） |
|---|---|---|
| real | [-32835, 85886]（f 依赖），文件内 min/max/mean/std 各异 | f1: [-917,-212] μ=-578 σ=88 |
| imaginary | [-82755, 50510] | f1: [629,1401] μ=969 σ=98 |
| complex_impedance | structured `[('real','<f8'),('imaginary','<f8>')]`，**非 numpy complex** | 同 real/imaginary |
| magnitude | [0, 111026]（Arbitrary Units） | f1: [789,1487] μ=1133 σ=89 |
| phase_degrees | [-180, 180] | f1: [99.4,136.7] μ=120.9 σ=4.9 |
| NaN/Inf | 信号数据 **0 NaN / 0 Inf**（659 完整文件） | — |

每文件每频率信号为**一维数组**（长度 22,949–280,124，随扫描范围/传感器变化），
`spatial_data` 提供栅格坐标将其重构为 2D。

### 3.4 空间坐标与网格（H×W）

| 网格（track × samples） | 文件数 | 说明 |
|---|---|---|
| **101 × 451** | 442 | 主流；x∈[0,100]mm，y∈[1,101]mm（track 编号即 y），dx≈0.23mm |
| 51 × 451 | 184 | 半宽扫描 |
| 202 × 1067 | 34 | 高分辨 |
| 501 × 560 | 9 | 大扫描 |
| 101×450 / 37×451 / 203–204×1067 / 501×564 等 | 零星 | 边缘情况 |

- **网格规则性**：栅格**近似规则**——每 track 采样数 450–451（±1 点/末尾丢触发），
  101×451 型文件总点 45,523 ≈ 101×451 − 28。`x_mm` 呈锯齿（每 track 回程跳变）。
- **37 文件（4.9%）x/y/z mm 全 NaN**（description 含 "missing"，多为特定传感器
  朝向参考扫描），但 `track_number/sample_number` 仍存在 → **仍可重建网格**。
- 每文件 4 频率共享同一空间栅格（同一扫描）。

---

## 4. 独立性分析（738 scan 与 specimen/defect 的关系）

### 4.1 硬事实

- `sample_properties.id` 1..738 **每文件唯一** → 它是**扫描序号**，不是试件号；
- 元数据**无显式 specimen ID**；
- 可验证的分组：
  - **(material_type, fiber_type, layup_sequence, description, defect_depth_mm,
    defect_size_mm, thickness_mm) = 148 组**（本仓 manifest 的 `specimen_id` 代理）；
  - 缺陷组（description+depth+size）= 133 组（127 组有信号）；
  - 8 传感器、5 材料、8 缺陷类。
- **重复扫描**：同配置组内 2–47 次扫描（如 gap 组 9 次 = 3 批次 × 3 次，批次间
  相隔数日、传感器朝向 0°/90° 不同）；但**同配置组内信号长度可差 2–3 倍**
  （如 scan_00001=45523 vs scan_00002=16588，同为 [0/90] Reference 0.6mm）→
  "配置组"**不能**安全等同为"同一物理样品"，真实独立样品数应界于
  **148（下限）与 738（上限）之间**。

### 4.2 结论（禁止事项）

- **禁止把 sensor×frequency 当独立物理样本**（8×4×148 = 4736 ≫ 148）；
- **禁止扫描级随机划分**（同缺陷组/同配置组的重复扫描必须同 split）；
- 推荐划分单元（按优先级）：
  1. **cross-material**：test = 未见过的材料（5 材料，EddyCus 官方推荐 domain）；
  2. **cross-sensor**：test = 未见过的传感器（8 传感器，S13131 之外的传感器单组
     仅 3–10 文件，评估时需聚合或仅用 S13131 vs 其余）；
  3. **defect_group / 配置组**：组内重复扫描同 split（133/148 组，组样本 1–47）。

### 4.3 标签与 specimen/sensor/material 的绑定

- 标签（description/depth/size）位于 `sample_properties` attrs，与**材料+铺层**
  强绑定（同一描述总是同材料族）；
- **sensor 与标签不绑定**：同缺陷组可被多个传感器扫描（39 配置组跨 2+ 传感器），
  因此 **cross-sensor 划分可测"未见传感器上的缺陷泛化"**（物理上最有价值的
  协议）；
- **material 与标签部分绑定**（如 gap 只存在于 HP-U300/122C 与 ST 50g），
  cross-material 时部分类会缺失 → 评估需注意类别不平衡/缺失类处理。

---

## 5. 输入方案（据真实 shape 决定）

### 5.1 推荐：I/Q 双通道，保留原生 2D 网格

```
每条 SSL 样本 = (scan, frequency) 视图：
  1D I/Q (N,) → 栅格重构 (H, W)，H=track 数（~101），W=每 track 采样（~450）
  → tensor (2, H, W)，通道 0 = real(I)，通道 1 = imaginary(Q)
```
- **首选 I/Q 双通道**（保留相位；magnitude/phase 可由 I/Q 导出，不另设通道）；
- **优先保留原生网格**（101×451 型直接 (2,101,451)；不强制 49×512）；
- 栅格近规则 ±1 点：**按 track 裁齐到中位 W**（每 track 取前 450 或补 1），
  损失 ≤1 点/track，无需插值；
- **尺寸不一致**（51/101/202/501 等家族）：
  - 同 batch 内 **padding 到该批最大 H×W**（encoder 末端 AdaptiveAvgPool2d，
    批内统一尺寸即可，跨批可不同）；
  - **仅当网格过大**（202×1067=215k px、501×560=280k px）且显存/算力受限时
    才**等比例下采样**（如最近邻 2×2 到 101×534 / 251×280）；
  - **禁止**无依据强制变形为 49×512。
- 每频率独立成样本（4 频率 × 有信号 695 扫描 = **2780 个 (2,H,W) 视图**），
  但 split 按 4.2 的物理单元，频率只作批内多样性。

### 5.2 第一层权重迁移（方案 B）

```
old: Conv2d(in=1, out=32, kernel=(3,7), pad=(1,3))
new: Conv2d(in=2, out=32, kernel=(3,7), pad=(1,3))
init: new.weight = old.weight.repeat(1, 2, 1, 1) / 2     # (32,1,3,7) -> (32,2,3,7)
      new.bias   = old.bias
```
- 数值验证（`tests/test_m0_2c.py::test_first_layer_transfer`）：对双通道拷贝输入，
  new 输出与原单通道输出一致（diff<1e-4）；
- 其余权重 **22/23 键**（conv.4 / conv.8 / BN×3 / proj）从
  `experiments/runs/ssl_ae/encoder.pt`（`encoder_state`）原样加载；
- 可选消融：`ssl_ae_both`（P4b beam+depth，~0.566）不作为主源。

---

## 6. 实验设计（本轮只设计，不运行）

### 6.1 两个条件（除初始化外完全一致）

| | E（scratch） | P→E（迁移续训） |
|---|---|---|
| encoder | `MAEEncoder(in_channels=2)` 从零初始化 | 加载 `ssl_ae/encoder.pt`，conv.0 用 `repeat(1,2,1,1)/2`，其余原样 |
| 数据 | 有信号 695 扫描 × 4 频率 = 2780 个 (2,H,W) 视图（无标签） | **同左** |
| 任务 | **2D 空间块掩码重建**（MAE 风格，掩码块 16×16，mask_ratio 0.3 起步） | 同左 |
| decoder | **新建 ECT decoder**（输出 = 原生网格 H×W，不沿用 PAUT (49,512) beam decoder） | 同左 |
| 损失 | Huber（smooth_l1，同 PAUT P1） | 同左 |
| steps / lr / batch | 相同（如 10,000 steps / lr 1e-3 / batch 32） | 同左 |
| seeds | 3×（model_seed 42/43/44；split_seed/data_seed 职责分离） | 同左 |
| 归一化 | 全局统计只在 SSL train 视图上计算（无泄漏） | 同左 |

### 6.2 评估（双下游）

1. **ECT 下游：P→E vs E**
   - 冻结 encoder + 规范头（lr 1e-3 / 80ep / batch 128 / 加权采样，同
     `configs/m0_2b_ultrasound_mae.yaml` head 协议）；
   - 任务：缺陷分类（8 类 / 或 clean vs flaw 二分类）；
   - 划分：**cross-material**（主，5 材料）与 **cross-sensor**（辅，8 传感器），
     按 4.2 物理单元，重复扫描同 split；
   - 指标：逐折均值 AUC（沿用"非PP4 逐折均值"口径；EddyCus 无 PP4，用
     cross-material/cross-sensor 的逐折均值）。
2. **PP3–PP7 回测（灾难性遗忘）：P→E vs 原 `ssl_ae`**
   - 把 P→E 续训后的 encoder 冻结，回 PAUT 任务（`scripts/paut_loocv.py` SSL
     变体，规范头 3 seed）；
   - 对照 = 原 `ssl_ae` checkpoint 的 0.579±0.007（fold）/0.616±0.016（pooled）；
   - 若回测显著回落（> ~0.01，seed 噪声级）→ 报告**灾难性遗忘**，P→E 的 ECT
     增益需与 PAUT 损失权衡。

### 6.3 停止判据（沿用 M0-2B det_v2）

- **ECT 下游：P→E − E ≥ +0.01 且 ≥2/3 seed 为正** 才算迁移有效；
- **PAUT 回测**：P→E 的 PAUT 非PP4 与 0.579±0.007 的差距进入结论；
- ⚠ 跨模态（超声→涡流）迁移比 M0-2B 的超声→超声更激进，先验不乐观；本实验的
  意义就是**最小成本证伪/证实**（E/P→E 均用小 encoder，GPU 分钟级）。

---

## 7. 交付物（本轮已完成）

| 交付物 | 路径 |
|---|---|
| 原始数据 | `data/raw/EddyCus-HDF5/`（zip + output/738 h5；**不入库**） |
| 审计脚本 | `scripts/m0_2c_eddycus_audit.py`（--probe / --full） |
| 审计摘要 | `experiments/results/m0_2c_eddycus_audit.json`（入库） |
| 全量转储 | `experiments/results/m0_2c_eddycus_audit_full.json`（gitignore，17MB） |
| manifest | `data/manifests/eddycus/dataset_card.json` + `records.parquet`（738 记录 / 148 配置组 / 127 缺陷组 / 8 类） |
| adapter | `src/wndt/data/adapters/eddycus.py`（懒加载 I/Q，`read_frequency`，`split_indices` 按物理单元） |
| stem | `src/wndt/models/multimodal/dataset_stems.py::EddyCusStem`（1D→2D→32 token，任意网格） |
| 注册 | `adapters/__init__.py`、`adapters/unified.py::ADAPTERS`、`scripts/m0_inspect_dataset.py` |
| smoke 测试 | `tests/test_m0_2c.py`（8 项全过，含第一层迁移数值验证） |
| 统一检查 | `python scripts/m0_inspect_dataset.py eddycus`（stats/read/stem/split 全过） |

**下一步（不在本轮）**：`scripts/m0_2c_ect_pretrain.py`（E / P→E 预训练 +
LOOCV 评估 + PP3–PP7 回测），按 §6 设计实现后运行。

---

## 附录 A：关键命令

```bash
# 下载（直连，绕过故障代理；aria2c 16 连接续传）
cd data/raw/EddyCus-HDF5
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  aria2c -c -x 16 -s 16 -o eddy_current_data.zip \
  "https://zenodo.org/records/19251759/files/eddy_current_data.zip?download=1"

# 校验 + 解压
md5sum eddy_current_data.zip        # 814f496342d77eb2eeabb1e0d34645c3
unzip -q eddy_current_data.zip      # -> output/scan_00001..00738.h5

# 审计
python scripts/m0_2c_eddycus_audit.py --probe data/raw/EddyCus-HDF5/output/scan_00001.h5
python scripts/m0_2c_eddycus_audit.py --full --root data/raw/EddyCus-HDF5/output

# manifest + 冒烟
python -m wndt.data.adapters.eddycus          # build_eddycus_manifest()
python scripts/m0_inspect_dataset.py eddycus --n 3
python tests/test_m0_2c.py
```
