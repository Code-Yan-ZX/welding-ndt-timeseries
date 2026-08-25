# M0-3 真实焊缝多源超声 SSL 实验报告（P-long vs W→P）

> 日期：2026-08-25
> 分支：`exp/m0-3-multisource-weld-ut`
> 状态：**基础设施与全链路 smoke 完成；真实数据下载被 Cloudflare challenge
> 阻止，pilot 尚未运行（等数据落地后执行）**

## 1. 目标与假设

**主问题**：在外部**真实焊缝**超声数据（FMC/PAUT）上预训练，是否能改善
PP3–PP7 严格跨试件 PAUT 表征（相对等预算的 PAUT-only SSL）？

- 只研究超声内部迁移（本轮不混入 ECT/CFRP/岩石超声）；
- 与 M0-2B（外部超声但为 ML-NDT/NDT_ML_Flaw 单试件虚拟缺陷语料）的区别：
  使用**真实焊缝多源**数据（316L / Inconel 82/182 / 304SS MMA / PAUT 定位）。

## 2. 实验设计（Protocol V2 + 等预算）

- **outer test**：一个完整 coupon；**inner validation**：剩余 coupons 中一个
  完整 coupon；**train**：其余完整 coupons；
- PAUT SSL、归一化、分类头训练**只能使用本折 train coupons**；validation
  coupon 只用于模型选择；test coupon 全程不可见；不使用历史 P4a 随机位置级
  validation 作为正式协议；
- **主指标**：非PP4 逐折 mean ROC-AUC（PP3/PP5/PP6/PP7）；同时报告逐折、
  pooled、PR-AUC、bAcc。

**两个等计算预算条件**（唯一差别 = 阶段 1 的数据源）：

| | 阶段 1 | 阶段 2 | 总 optimizer steps |
|---|---|---|---|
| **P-long** | 本折 PAUT train coupons SSL × ext_steps | PAUT SSL × tgt_steps | ext_steps + tgt_steps |
| **W→P** | 外部真实焊缝 FMC SSL × ext_steps（fold 无关，一次复用 5 折） | PAUT SSL × tgt_steps（**新建 PAUT decoder**，只加载阶段 1 encoder） | ext_steps + tgt_steps |

- mask 计划 / batch / 数据顺序 / 优化器 / 头协议 / seed **完全一致**；
- 阶段边界都重建 optimizer（lr 计划每阶段重启动）；
- **外部 decoder 与 PAUT decoder 分离，不迁移 decoder**；
- 结构：P1 `MAEEncoder`（共享） + 数据源专用 decoder（`FlexDecoder`）；
  block=16×16 mask，mask_ratio=0.3，**recon loss 只算 masked∩valid**；
- 禁止用缺陷标签做 SSL 采样捷径。

**执行闸门**：
- Stage A smoke（已通过，见 §6）；
- Stage B pilot（seed 42，外部 2000 + PAUT 2000 steps，全 folds）——**待数据**；
- Stage C 正式 3 seeds（仅 pilot 通过后，固定预算 ext 5000 + tgt 5000）。

**pilot GO 判据**：W→P − P-long 非PP4 mean ROC-AUC ≥ +0.01；PP3/5/6/7 至少
3 折不下降；无单折下降 > 0.05；结果不由 PP4 / pooled 驱动。任一不满足 → 立即
停止、不调参、不跑 3 seeds，结论写"少量外部真实焊缝 FMC 未带来稳定迁移"。

## 3. 数据获取（被 Cloudflare 阻止，需人工下载）

目标数据源（全部 Strathclyde Pure Portal，CC BY 4.0）：

| id | 内容 | 大小 | DOI |
|---|---|---|---|
| A | 316L 焊缝 lack-of-fusion FMC | 260MB + 53KB(.ods) | 10.15129/086404bd-eb69-429b-978c-2c35cdbfcf87 |
| B | Inconel 82/182 中心线裂纹 FMC | 33.1MB + 8.9KB(.xlsx) | 10.15129/179e1b38-e701-443d-b995-a4449851330c |
| C | 304SS MMA 3mm SDH FMC | 44.9MB + 10.1KB(.xlsx) | 10.15129/60b6a5b8-e78e-4742-8414-aaba9399a9c8 |
| D | PAUT 探头定位 | 98.9MB(zip) | 10.15129/bfb5a77d-dabe-4be4-82c9-b10e8c237dea |

**失败记录**（`data/manifests/external_weld_ut/download_manifest.json`）：
Pure Portal `/files/` 由 Cloudflare managed challenge 保护，curl / WebFetch /
Firefox(snap) / Playwright Chromium headless 全部失败（403 / 挂死 / 挑战不通过 +
EPIPE 崩溃）；按项目纪律不做绕过。**人工下载步骤**：浏览器打开各 page 点
Download（无需登录），放入 `data/raw/external_weld_ut/{A,B,C,D}/`，
再跑 `scripts/m0_3_external_weld_ut_audit.py --full` + `build_manifest`。

## 4. 数据审计状态

- 网页级信息已记录（材料/焊缝/缺陷/DOI/许可证），见 `docs/M0_3_external_weld_ut_audit.md`；
- 审计脚本就绪：`scripts/m0_3_external_weld_ut_audit.py`（MATLAB v5/v7.3、
  xlsx/ods、zip 自动探测；NaN/Inf、维度语义、独立试件数、group_id、哈希/近重复、
  每源标记、<10 试件 exploratory 标注）；
- **独立性纪律**：同一试件的 Tx×Rx、scan position、重复扫查共用 group_id；
  "多个信号/通道 ≠ 多个独立试件"；真实独立焊缝试件大概率 < 10（A/B/C 各 1
  个 .mat ≈ 1 个试件）→ 结论必须标注 **exploratory external pretraining
  source**，不称为 foundation-scale dataset。

## 5. 交付物

- `configs/m0_3_weld_ut.yaml`（预算/seed/头协议/判据）
- `src/wndt/models/ssl_ae.py`（+`FlexDecoder` / `ExternalUTMaskedAE`）
- `src/wndt/data/adapters/external_weld_ut.py`（adapter + manifest 生成）
- `src/wndt/data/external_weld_ut_pretrain.py`（view/bucket/mask/PAUT 计划）
- `scripts/m0_3_weld_ut_pretrain.py`（P-long / W→P 顺序 SSL）
- `scripts/m0_3_loocv.py`（Protocol V2 LOOCV）
- `scripts/m0_3_aggregate.py`（pilot/正式 GO 判据 + aggregate.json/md）
- `scripts/m0_3_external_weld_ut_audit.py`、`scripts/m0_3_download_external_weld_ut.sh`、
  `scripts/m0_3_download_firefox.py`（浏览器下载工具）
- `docs/M0_3_external_weld_ut_audit.md`、`data/manifests/external_weld_ut/`
- `tests/test_m0_3.py`（9 项：adapter/group 独立性/变长输入/valid-mask loss/
  模型加载/mask 与采样确定性）
- checkpoint 目录：`experiments/runs/m0_3/`（全新，不覆盖既有）

## 6. Stage A smoke（已通过）

用**合成 FMC .mat**（3 源 × (Tx×Rx×T)）验证全链路（20 steps + 1 head epoch）：

- 外部 SSL（W→P 阶段 1）：loss 收敛、无 NaN/Inf、valid-mask loss 正常；
- P-long / W→P 阶段 2：encoder 加载 missing/unexpected 全空，PAUT decoder 新建；
- Protocol V2 LOOCV：5 折全跑通（逐折 AUC/PR-AUC/bAcc/thr），协议字段完整；
- 聚合判据脚本：正确输出 pilot GO 判定（合成数据为"不通过"，属预期）；
- `pytest tests/`：116 passed / 2 failed（2 个为既有 `test_models.py` 需
  HuggingFace hub 访问，与 M0-3 无关）。

⚠ smoke 使用的合成数据**不是**真实审计结果；真实数据落地后重跑 audit +
manifest，再做 Stage B。

## 7. 当前结论与下一步

- **基础设施 100% 就绪**：数据一旦落地（人工下载约 10 分钟），即可顺序执行
  审计 → manifest → Stage A smoke（真实数据）→ Stage B pilot（seed 42,
  2000+2000 steps）→ 判据 →（通过才）Stage C 3 seeds。
- **当前阻塞点唯一**：Strathclyde 数据下载（Cloudflare challenge）。
- **预期结论口径**：独立试件 < 10 → exploratory；pilot 判据不过即停止并写
  "少量外部真实焊缝 FMC 未带来稳定迁移"。
