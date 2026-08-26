# M0-3 真实焊缝多源超声 SSL 实验报告（P-long vs W→P）

> 日期：2026-08-26
> 分支：`exp/m0-3-multisource-weld-ut`
> 状态：**pilot 完成，GO 判据未通过 → 按纪律结束，不调参、不跑 3 seeds**

## 1. 结论（TL;DR）

**少量外部真实焊缝 FMC/PAUT（4 个独立试件，exploratory）未带来稳定迁移**：
pilot（seed 42，等预算 ext 2000 + tgt 2000）非PP4 逐折 mean ROC-AUC
W→P − P-long = **−0.0138**（需 ≥ +0.01），仅 1/4 非PP4 折未下降（需 ≥3）。
外部预训练既未提升平均表现，也未改善多数折 —— 按协议立即停止，不做超参搜索，
不跑正式 3 seeds。

## 2. 数据（真实焊缝，多源，全部已落地并审计）

| 源 | 内容 | 结构（规范化） | 独立试件 | 采集 |
|---|---|---|---|---|
| A | 316L lack-of-fusion FMC | (Tx=128, Rx=128, T=10000) int32 | 1 | 1 |
| B | Inconel 82/182 中心线裂纹 FMC | (Tx=45, Rx=45, T=10000) int32 | 1 | 1 |
| C | 304SS MMA 3mm SDH FMC | (Tx=128, Rx=128, T=976) float64 | 1 | 1 |
| D | PAUT 探头定位 B-scan | 389 × (beam=401, time=762) | 1 | 389 |
| **合计** | | **690 views / 4 groups** | **4（<10 → exploratory）** | 392 |

- 下载：Cloudflare managed challenge 阻止自动下载（curl/WebFetch/Firefox snap/
  Playwright Chromium 全部失败），最终**浏览器人工下载**；校验和与失败记录见
  `data/manifests/external_weld_ut/download_manifest.json` 与
  `data/raw/external_weld_ut/checksums.txt`。
- 审计：`docs/M0_3_external_weld_ut_audit.md` + `scripts/m0_3_external_weld_ut_audit.py`
  + `experiments/results/m0_3_external_weld_ut_audit_full.json`。
- 每源标记：全部 `ssl_pretrain_usable`；**均无逐位置缺陷标签** →
  `downstream_label_usable = false`（外部只做无监督 SSL）。

## 3. 实验设计（Protocol V2，等预算）

- **P-long**：阶段1 本折 PAUT(train coupons) SSL × 2000 + 阶段2 PAUT SSL × 2000；
- **W→P**：阶段1 外部真实焊缝 FMC SSL × 2000（fold 无关，一次复用 5 折）+
  阶段2 PAUT SSL × 2000（**新建 PAUT decoder，只加载阶段1 encoder，不迁移 decoder**）；
- 总 optimizer steps 完全相等（4000）；mask/batch/数据顺序/优化器/头协议/seed 一致；
- 主指标 = **非PP4 逐折 mean ROC-AUC**（PP3/PP5/PP6/PP7）；pooled/PP4 仅参考。

## 4. pilot 结果（seed 42，2026-08-26）

| 折 | P-long | W→P | Δ (W→P − P-long) |
|---|---|---|---|
| PP3 | 0.4928 | 0.4749 | **−0.0179** |
| PP5 | 0.5484 | 0.5116 | **−0.0368** |
| PP6 | 0.5198 | 0.5169 | **−0.0029** |
| PP7 | 0.6594 | 0.6616 | **+0.0022** |
| **非PP4 mean** | **0.5551** | **0.5413** | **−0.0138** |
| PP4（参考） | 0.3757 | 0.5362 | +0.1605 |
| pooled（参考） | 0.5772 | 0.5827 | +0.0055 |
| PR-AUC 非PP4 mean | — | — | −0.007（PP3 −0.007/PP5 −0.003/PP6 −0.002/PP7 −0.050） |
| bAcc 非PP4 mean | 0.5210 | 0.5074 | −0.0136 |

### GO 判据逐项

| 判据 | 要求 | 实测 | 通过 |
|---|---|---|---|
| 1. W→P−P-long 非PP4 mean | ≥ +0.01 | **−0.0138** | ✗ |
| 2. 非PP4 至少 3 折不下降 | ≥ 3 | **1/4**（仅 PP7） | ✗ |
| 3. 无单折下降 > 0.05 | > −0.05 | −0.0368（PP5） | ✓ |
| 4. 结果非 PP4/pooled 驱动 | — | WP 仅 PP4（+0.16）与 pooled（+0.005）"看似"更好，均被排除 | ✓（说明：主指标不包含它们） |

**pilot GO 判据不通过 → 立即停止；不调参；不跑 3 seeds。**

## 5. 分析与解读

- **外部预训练没有带来稳定迁移**：4 个非PP4 折中 3 折下降、1 折微升（+0.002），
  平均 −0.0138。即使"受益"的折（PP7）增益也远小于判据阈值。
- **PP4 与 pooled 的"改善"是假象**：W→P 在 PP4（近零缺陷试件，仅 3 正样本）
  高 +0.16，pooled +0.005 —— 按协议这两者仅参考，且 PP4 的高波动已知会虚高
  指标（见 [[project-pp4-verified-clean]]）。主指标（非PP4 逐折均值）明确为负。
- **等预算对照是干净的**：P-long 与 W→P 总 steps、mask、batch、优化器、头协议、
  seed 完全一致，唯一差别 = 阶段 1 数据源。差异只能归因于外部 FMC 预训练。
- **P-long 绝对水平 0.555 < P1 历史 0.579**：符合预期 —— P-long 是严格 per-fold
  （SSL 只读本折 train coupons）+ 4000 steps 短预算；P1 用全量数据（transductive）
  预训练。这不是本实验的对照对象，P-long 只是等预算控制。
- **数据规模限制（exploratory）**：仅 4 个独立真实焊缝试件（A/B/C 各 1 个 FMC +
  D 的 1 个焊缝多位置 PAUT），远小于 foundation-scale。结论只适用于"少量外部
  真实焊缝 FMC 预训练"，不能外推到大规模预训练语料。

## 6. 交付物

- 数据与审计：`download_manifest.json`（下载记录+校验和）、`docs/M0_3_external_weld_ut_audit.md`、
  `experiments/results/m0_3_external_weld_ut_audit_full.json`、
  `data/manifests/external_weld_ut/{dataset_card.json,records.parquet}`；
- 代码：`scripts/m0_3_external_weld_ut_audit.py`、`scripts/m0_3_weld_ut_pretrain.py`、
  `scripts/m0_3_loocv.py`、`scripts/m0_3_aggregate.py`、`src/wndt/data/adapters/external_weld_ut.py`、
  `src/wndt/data/external_weld_ut_pretrain.py`、`src/wndt/models/ssl_ae.py`（FlexDecoder/
  ExternalUTMaskedAE）、`configs/m0_3_weld_ut.yaml`、`tests/test_m0_3.py`（9 项全过）；
- 结果：`experiments/results/m0_3_loocv_{P-long,WP}_seed42_e2000_t2000.json` +
  `m0_3_aggregate.{json,md}`；checkpoint 在 `experiments/runs/m0_3/pretrain/`（pilot）。

## 7. 复现

```bash
# 数据已落地（data/raw/external_weld_ut/）
.venv/bin/python scripts/m0_3_external_weld_ut_audit.py --full
# 预训练（等预算：P-long 4000 PAUT；W→P 2000 外部 + 2000 PAUT）
.venv/bin/python scripts/m0_3_weld_ut_pretrain.py --cond WP  --seed 42          # 外部阶段
.venv/bin/python scripts/m0_3_weld_ut_pretrain.py --cond P-long --fold PP3 --seed 42
# ... 5 折 × 2 条件
.venv/bin/python scripts/m0_3_loocv.py --cond P-long --seed 42
.venv/bin/python scripts/m0_3_loocv.py --cond WP --seed 42
.venv/bin/python scripts/m0_3_aggregate.py
```
