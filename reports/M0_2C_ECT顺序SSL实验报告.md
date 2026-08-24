# M0-2C ECT 顺序 SSL 实验报告（PAUT → ECT continued SSL，E vs P→E）

> 阶段：M0-2C（EddyCus-HDF5 真实数据接入 + E/P→E 顺序自监督预训练 + 双向评估）
> 日期：2026-08-24
> 目标：验证 **PAUT SSL encoder → ECT SSL 续训**是否得到更通用的 NDT 编码器：
> ECT 迁移判据（P→E−E ≥ +0.01 且 ≥2/3 seed 正）与 PAUT 保持判据
> （P→E−P ≥ −0.01）同时通过才算成功。
> 合规：原始数据不入库；checkpoint 全部写入新目录 `experiments/runs/m0_2c/`，
> 不覆盖 `ssl_ae/encoder.pt`；smoke/pilot 带独立后缀。

---

## 0. 结论先行（TL;DR）

**顺序训练（PAUT SSL → ECT SSL）未得到更通用的 NDT 编码器 —— PAUT 保持判据
失败，结论为灾难性遗忘，按任务规定直接停止（不调参、不做 replay/freeze 补救）。**

1. **ECT 迁移判据：通过**。冻结表征 group probe（transductive，clean vs flaw
   二分类，5 折 SGKFold 按 config/specimen proxy）：平均 P→E − E = **+0.0162**
   ROC-AUC（≥ +0.01），**3/3 seed 为正**（+0.0181 / +0.0179 / +0.0126）。
   从 P1 PAUT encoder 迁移初始化继续 ECT SSL 对 ECT 表征有可复现的小幅增益。
2. **PAUT 保持判据：不通过**。PP3–PP7 规范 LOOCV（非PP4 逐折均值）：
   平均 P→E − P = **−0.0606**（远低于 −0.01 阈值），逐 seed 全部为负
   （−0.0399 / −0.0679 / −0.0741）。**灾难性遗忘成立**。
3. **总结论**：只有在 ECT 迁移判据与 PAUT 保持判据**同时**通过时才能说顺序训练
   得到更通用的 NDT 编码器；本实验只有前者通过。直接停止，不做 replay/freeze
   补救实验（任务规定）。
4. 附：P 在当前代码下重跑规范头 = 0.5710 / 0.5726 / 0.5768（非PP4 逐折均值），
   与历史 0.579±0.007（s42/43/44）一致，验证协议可信。

---

## 1. 数据与划分（阻塞项修复）

### 1.1 正式训练只使用 695 个有信号扫描

EddyCus-HDF5 共 738 个 `scan_XXXXX.h5`，其中 **43 个（5.8%，2022-11 批次）仅含
元数据、无 `signal_data`**，正式训练与评估一律排除。`EddyCusAdapter.signal_records()`
/ `signal_indices()` 只返回 695 个有信号扫描（`signal_data/f1` 存在）。

695 有信号扫描 × 4 频率 = **2780 个 (scan, frequency) view**。扫描级 8 类分布：

| 类 | gap | clean | mis-orientation | Cu foil | Cu roving | PTFE | ondulation | fuzz ball | 合计 |
|---|---|---|---|---|---|---|---|---|---|
| 扫描数 | 492 | 63 | 75 | 19 | 18 | 18 | 6 | 4 | **695** |
| view 数(×4) | 1968 | 252 | 300 | 76 | 72 | 72 | 24 | 16 | **2780** |

> 注意：全量 manifest（738）口径 clean=84；43 个 metadata-only 中 clean 占
> 21，故有信号口径 clean=63。评估正负样本按 695 口径统计。

### 1.2 `split_indices` 修复（sensor/material 真正分组）

- `unit=sensor`：真正按 `geometry.sensor_type` 分组（8 个传感器组，本机字符串
  含 9 个变体——S13132 有 "6,1 MHz"/"6,1MHz" 两种写法，属同传感器）；同 sensor
  绝不跨 split。
- `unit=material`：真正按 `domain.material_type` 分组（本机字符串 6 个，含
  "0/90° Fabric 565 g/m²" 与 "0/90 Fabric 565 g/m²" 两种写法）；同 material 绝不跨 split。
- 禁止退化为 `defect_instance_id` 划分（旧实现 bug，已修复并加测试）。

### 1.3 clean 记录按 specimen/config proxy 分组

clean 记录（`defect_instance_id=None`）**不再归入单一 "clean" 单元**：每条记录
显式建立 `group_id = specimen_id`（= (material,fiber,layup,desc,defect,thickness)
配置组哈希）。同一物理配置下的 clean 重复扫描同组，**不同 clean 配置可进入不同
fold**。695 口径下 clean 扫描 63 条分布到 20 个 clean 配置组（部分组含多次重复
扫描）。`validate_defect_split` 与 `unit_keys("defect")` 均按此分组。

### 1.4 fold 审计（正式划分前）

probe 每折训练前输出审计块：train/val/test 记录数、物理配置组数、clean/flaw
数量、8 类标签分布、material/sensor 分布；**train、val、test 必须同时有正负
样本，否则该 fold 无效并停止**（`audit_split` 抛错）。内层 val 按 clean/flaw
配置组分别取 ~20%（各至少 1 组），保证正负齐备且组纯度。

### 1.5 ECT 主任务 = clean vs flaw 二分类

当前阶段 ECT 主任务只做 **clean vs flaw 二分类**；**8 类只输出分布，不训练**
（避免极少类 4/6 样本导致无意义结果）。

---

## 2. 统一模型路径（P1 MAEEncoder）

- 正式 E 与 P→E 都使用 **P1 的 `MAEEncoder` 卷积结构**（`src/wndt/models/ssl_ae.py`）：
  `Conv2d -> BN -> GELU -> MaxPool` 共三层 + `AdaptiveAvgPool` + `Linear` proj，
  `d_model=128`；新增 `in_channels` 参数（1=PAUT / 2=ECT I/Q），键名与
  `ssl_ae/encoder.pt` 完全对齐。
- **E**：双通道 `MAEEncoder(in_channels=2)` 从零初始化。
- **P→E**：加载 `experiments/runs/ssl_ae/encoder.pt`：
  - 第一层 1→2 通道：`new_weight = old_weight.repeat(1,2,1,1)/2`；
  - 其余 22/23 权重原样加载；
  - 运行时打印 missing/unexpected keys 并断言：**迁移后均为空**（全部 23 键对齐；
    数值验证：双通道拷贝输入输出与原单通道 diff<1e-4）。
- **EddyCusStem 不参与本轮 E/P→E 训练**（只保留 adapter/token 接口 smoke），
  否则无法声称复用了 P1 MAEEncoder。

---

## 3. ECT SSL

### 3.1 视图与划分

- 每个 (scan, frequency) = 1 个 view，共 **2780 views**；split/group 永远按
  scan 的物理配置（`specimen_id`），**frequency 不是独立物理样本**。
- 输入 **I/Q 双通道** `(2,H,W)`，保留原生栅格（H=track 数、W=每 track 采样）；
  同 batch 按最终尺寸 bucket（10 个桶），批内同尺寸。
- **超大网格等比例下采样（预先声明，E/P→E 一致）**：
  `S = max(ceil(H/256), ceil(W/768))`，S>1 时最近邻索引采样到
  `(ceil(H/S), ceil(W/S))`：
  - 202×1067 → **101×534**；501×560 → **251×280**；501×564 → 251×282；
  - 101×451 / 51×451 / 101×450 / 37×451 / 51×450 / 251×560 保持原生。

### 3.2 归一化

主方案固定为：每个 (scan, frequency)、每个 I/Q 通道在 **valid 像素**上
**median/MAD robust z-score**（`(x−med)/(1.4826·MAD+1e-6)`）。E/P→E 完全相同，
不调参。栅格缺失点（padding）置 0 sentinel，**是否参与 loss 由 valid mask 决定**。

### 3.3 新 ECT decoder + block masking

- 新建 `ECTDecoder`：`fc → (B,128,mid_h,mid_w) → 插值到 (H,W) → 3 层 refine conv`
  （末层 2 通道重建 I/Q），输出**当前 batch 的 H×W**；decoder 权重与 batch 尺寸
  无关，E/P→E 的 decoder 初始化完全一致。
- 2D block masking：**block=16×16，mask_ratio=0.3**（块数 ~30%，像素级 ~0.295）。
- `recon_loss`（Huber/smooth-L1）**只计算 masked 区域 ∩ valid-pixel 区域**；
  padding 和栅格缺失点绝不进入 loss（测试覆盖：污染无效像素/可见像素不影响
  loss，只 masked∩valid 计入）。

### 3.4 E/P→E 完全匹配

| 项 | E | P→E |
|---|---|---|
| data_seed | 42 | 42（相同） |
| mask 计划 | `(model_seed,step,j)` 确定性 | 相同 |
| 数据顺序 | `ect_bucket_plan(data_seed)` | 相同 |
| optimizer | AdamW lr 1e-3, wd 1e-4 | 相同 |
| batch | 16 | 16 |
| steps | 10000（固定预算） | 10000 |
| decoder 初始化 | `set_seed(model_seed)` | 相同 |
| **encoder 初始化** | 从零（2ch） | **P1 迁移（2ch）** |

smoke/pilot 实证：E 与 P→E 的 loss 轨迹几乎重合（如 step0 1.08596 vs 1.08537），
确认匹配。

---

## 4. ECT 下游评估（transductive probe）

- **冻结 encoder**；clean vs flaw 二分类（规范头 lr 1e-3 / ≤80ep / batch128 /
  class-balanced / val 早停）。
- 单元 = 扫描（695）；每扫描 4 view 特征均值 → (128,)；标签 flaw=1。
- 按 **config/specimen proxy**（146 组）做 **5 折 StratifiedGroupKFold**；
  同一配置组绝不跨 fold；内层 val 按组切 ~20%。
- 指标：**fold mean ROC-AUC、PR-AUC、balanced accuracy**；逐 seed 比较 P→E 与 E。
- **标记为 transductive_unlabeled representation probe**：SSL 使用全部 2780
  views（无划分）后冻结评估；**不得写成严格 cross-group 泛化**。若 P→E−E
  达到正判据，再补严格 fold-specific SSL。

（每折审计与结果见 §7）

---

## 5. PAUT 回测（灾难性遗忘）

- **P**：原 `ssl_ae/encoder.pt` 在当前代码下**重新跑一次规范头**（不只引用历史
  0.579±0.007）。
- **P→E**：双通道第一层折回单通道 `w_single = w_ect[:,0:1] + w_ect[:,1:2]`，
  其余权重原样加载（折回测试：与原 P 权重 diff<1e-6）。
- 冻结 encoder，相同 head seed，重跑 **PP3–PP7 规范 LOOCV**（P4a 协议：test=1
  coupon；其余 4 coupons 85/15 分层位置级 val；per-timestep 归一化只由 train
  位置计算；冻结 encoder + SSLClassifier 规范头 lr 1e-3/80ep/batch128/加权/
  val-AUC 早停）。
- 主指标 = **非PP4 逐折均值**（PP3/PP5/PP6/PP7）；输出 P→E − P 逐 seed 差值。

---

## 6. 判据

- **ECT 迁移**：`mean[P→E − E] >= +0.01` 且 **≥2/3 seed 为正**。
- **PAUT 保持**：`mean[P→E − P] >= −0.01`；若下降超 0.01 → **灾难性遗忘**。
- 两判据同时通过 → 顺序训练得到更通用的 NDT 编码器；任一失败直接停止，
  不调参、不做 replay/freeze 补救实验。

---

## 7. 结果

（正式 3 seeds 42/43/44 × 10000 steps，batch 16，固定预算不调参）

### 7.1 ECT probe（fold mean，transductive）

| seed | E ROC-AUC | P→E ROC-AUC | Δ ROC-AUC | Δ PR-AUC | Δ bAcc |
|---|---|---|---|---|---|
| 42 | 0.8486 | 0.8667 | **+0.0181** | −0.0014 | −0.0038 |
| 43 | 0.7770 | 0.7949 | **+0.0179** | +0.0033 | −0.0246 |
| 44 | 0.8432 | 0.8558 | **+0.0126** | −0.0005 | +0.0251 |

- 平均 Δ ROC-AUC = **+0.0162**（Δ PR-AUC = +0.0005，Δ bAcc = −0.0011）
- P→E > E 的 seed 数：**3/3**；判据（mean ≥ +0.01 且 ≥2/3 seed 正）：**通过**。
- 每折明细见 `experiments/results/m0_2c_ect_probe_{E,PE}_seed{s}_s10000.json`。
- 注意：这是 **transductive_unlabeled representation probe**（SSL 使用全部 ECT
  无标注视图后冻结评估），**不得写成严格 cross-group 泛化**；若需严格泛化证据
  须补 fold-specific SSL。

### 7.2 PAUT 回测（非PP4 逐折均值）

| seed | P | P→E | Δ |
|---|---|---|---|
| 42 | 0.5710 | 0.5311 | **−0.0399** |
| 43 | 0.5726 | 0.5047 | **−0.0679** |
| 44 | 0.5768 | 0.5027 | **−0.0741** |

- 平均 Δ = **−0.0606**；逐 seed 全部为负。
- P 复现（当前代码重跑规范头）：0.5710 / 0.5726 / 0.5768，均值 0.5735，
  与历史 0.579±0.007 一致（差异来自 85/15 位置级 split 的 seed 与早停）。
- PP3/PP5/PP6 折显著退化（PP3 −0.03~−0.04，PP5 −0.03~−0.05，PP6 −0.05~−0.12），
  PP7 也有 ~0.07–0.08 下降；**PAUT 表征在 ECT 续训后系统性变差**。

### 7.3 判据判定

| 判据 | 结果 | 阈值 | 实测 | 通过 |
|---|---|---|---|---|
| ECT 迁移 | mean[P→E−E] ≥ +0.01 且 ≥2/3 seed 正 | +0.01 / 2 | **+0.0162 / 3** | ✅ |
| PAUT 保持 | mean[P→E−P] ≥ −0.01 | −0.01 | **−0.0606** | ❌ |

**总结论：PAUT 保持判据失败（灾难性遗忘），顺序训练未得到更通用的 NDT 编码器。
按任务规定直接停止，不做 replay/freeze 补救实验。**

### 7.4 解释与归因

- ECT 侧的 +0.0162 增益表明 P1 encoder 的卷积归纳偏置对 ECT 也有价值（迁移
  初始化比从零更快/更好），且 E 与 P→E 在数据/掩码/优化器/解码器上完全匹配，
  增益不是实验伪影。
- PAUT 侧的大幅退化说明：在 ECT（CFRP 涡流）上继续 SSL 10000 步后，表征被
  ECT 域分布主导（首层权重与 BN running stats 均向 ECT 偏移），PAUT 焊缝超声
  的可判别性被覆盖。这符合"顺序单模态续训导致灾难性遗忘"的预期，也正是本
  实验要检测的现象。
- 结论纪律：不得把"ECT 迁移通过"写成"顺序训练成功"——最终判据要求两个条件
  同时满足；失败即停止，不调参、不做 replay/freeze 补救。

---

## 8. 交付物

- `configs/m0_2c_ect.yaml`
- `src/wndt/data/eddycus_pretrain.py`（view 索引/读取/归一化/下采样/block mask/
  数据顺序/审计）
- `scripts/m0_2c_ect_pretrain.py`、`scripts/m0_2c_ect_probe.py`、
  `scripts/m0_2c_paut_retention.py`、`scripts/m0_2c_aggregate.py`
- `tests/test_m0_2c_training.py`（19 项审计）
- `experiments/results/m0_2c_*_seed{42,43,44}.json`、
  `experiments/results/m0_2c_aggregate.{json,md}`
- checkpoint：`experiments/runs/m0_2c/ect/{E,PE}_s{seed}_s10000.pt`（新目录，
  不覆盖 `ssl_ae/encoder.pt`）
