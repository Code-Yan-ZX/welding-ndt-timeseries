# M0-1.5：评估协议 V2（Evaluation Protocol V2）

> 阶段：M0-1.5 协议与数据底座修正（不训练、不下载）
> 日期：2026-08-18
> 适用范围：本仓库全部 PAUT / SAW / 融合 / 合成数据的**位置级判别**评估。
> 配套实现：`scripts/paut_p7_synth_to_real.py`、`tests/test_eval_protocol.py`、
> `docs/M0_unified_ndt_schema.md`、`docs/partner_data_spec_v0.1.md`

## 0. 为什么需要 Protocol V2

M0-1 审计与 P0–P7 实验暴露出几类必须立即修正的问题：

1. **泄漏 / transductive 混用**：P7 早期脚本在预训练阶段把全部合成 + 真实数据
   一起喂进编码器（`--mix-real` 直接 `concatenate`），未区分 test coupon 的
   信号是否进入预训练 —— 属于 transductive 设定，却被当作跨试件泛化证据。
2. **validation 未按物理单元分组**：`loocv_real` 里对 test coupon 之外的
   "rest" 做**随机位置级** 15% 切分，同一 coupon 的位置横跨 train/val，
   物理独立单元被切开，val 不能代表"未见 coupon"。
3. **`def_rate` 不控制标签**：`synth_ultrasound.py` 的 `coupon_cfg.def_rate`
   只进 meta，不参与 `make_coupon` 的缺陷区段生成，标签实际由 `n_def`/
   `def_len` 隐式决定 → 目标缺陷率与真实缺陷率脱节。
4. **≥50 mm 贯穿大裂纹被当作背景负样本**：位置级判别把它们当背景，实际是
   另一类物理对象，会污染负样本分布。
5. **指标单一**：只有 position AUC，缺少 PR-AUC 与 defect-level /
   false-positive-per-meter 接口。

Protocol V2 就是为这些修正立规矩。**凡与本协议冲突的历史结果不删除**，但
必须按 §6 重新标注，且**不得**把 transductive 与 strict inductive 结果混成
同一个主指标。

## 1. 协议定义

### 1.1 `strict_inductive`（主协议）

test coupon 的一切信息（信号、统计量、无标签数据）**均不得**进入：

- 预训练（含 SSL 预训练、联合预训练、数据扩充）；
- normalization（均值/标准差只能在 train coupon 上计算）；
- validation；
- 模型选择（含早停、超参选择、head 协议选择）。

即：每个 outer fold 的完整流程（预训练 → 冻结编码器 → head 训练 → 早停 →
评估）只能使用**非 test coupon** 的真实数据；test coupon 仅在最终评估时出现。

> 对位置级判别：test coupon 的**信号**完全不可见，连无标签形式也不行。

### 1.2 `transductive_unlabeled`（仅诊断 / 单独报告）

允许使用 test coupon 的**无标签信号**（例如 SSL-MAE 在 test coupon 上做
表征级适配、TTT、或利用 test coupon 统计量做 normalization）。

**限制与标注**：
- 只能作为诊断/探索单独报告，**不得**与 strict inductive 结果合并成一个主指标；
- 每个实验 JSON 的 `protocol` 字段必须显式为 `transductive_unlabeled`；
- 报告与汇总表中必须单独一节，明确"transductive：允许使用 test 无标签信号"，
  不得声称是"严格跨试件"结论。

### 1.3 为什么不设 `transductive_labeled`

test coupon 的**有标签**数据进入训练即训练-测试重叠（切题评估泄漏），任何
协议都不允许。

## 2. PAUT：outer leave-one-coupon-out（LOOCO）

PAUT 真实数据（PP3/PP4/PP5/PP6/PP7，位置级 0/1 标签）使用 **5 折
leave-one-coupon-out**：

- 每个 outer fold：`test_coupon = PPx`，其余 4 个 coupon 为
  `train+val`（按 coupon 分组，见 §3）；
- 主指标 = **非PP4 逐折均值**（PP4 为近零缺陷试件，test 折 AUC 为纯噪声，
  见 README 评价指标统一规则）；pooled 仅参考；
- 每次评估的 head 协议统一为规范头（冻结编码器 + 分类头 lr=1e-3/80ep/batch=128），
  与 P4a 同口径。

## 3. Validation 必须按完整 coupon 分组

**禁止随机位置级 validation**：同一 coupon 的所有位置必须全部落在 train 或
全部落在 val，绝不允许同一 coupon 的位置横跨 train/val。

实现约定（`paut_p7_synth_to_real.py` 的 `loocv_real`）：

```
rest = 非 test coupon 的 coupon 集合（4 个）
随机 shuffle rest（seed 固定）
n_val_coupons = max(1, 每折取 1 个 coupon 作 val)
va_coupons = rest[:n_val_coupons]
tr_coupons = rest[n_val_coupons:]
va = 全部 va_coupons 的位置；tr = 全部 tr_coupons 的位置
```

这样 val 由**完整 coupon** 组成，val AUC 反映"未见 coupon"的可判别性，
早停不会因位置级混洗而虚高。

## 4. 实验 JSON 必须记录的字段

每个位置级评估实验，结果 JSON 顶层必须包含：

| 字段 | 说明 | 必填 |
|---|---|---|
| `protocol` | `strict_inductive` / `transductive_unlabeled` / `smoke` | ✓ |
| `pretrain_coupons` | 参与预训练的 coupon 集合（strict 下 = 非 test coupon） | ✓ |
| `train_coupons` | 每折 head 训练的 coupon 集合 | ✓ |
| `val_coupons` | 每折 validation 的 coupon 集合（完整 coupon） | ✓ |
| `test_coupon` | 每折 test 的 coupon | ✓ |
| `normalization_scope` | `train_coupons`（strict 必填）/ `all_coupons`（transductive） | ✓ |
| `seed` | 随机种子 | ✓ |
| `code_commit` | 产生该结果的 git commit hash | ✓ |
| `run_type` | `smoke` / `full` | ✓ |

> `code_commit` 在脚本运行期由 `git rev-parse HEAD` 自动记录；若运行时代码
> 有未提交改动，额外记录 `code_dirty: true`。

## 5. 缺陷标签与任务口径

### 5.1 ≥50 mm 贯穿型大裂纹 → ignore，不作为背景负样本

PENELOPE 的贯穿型大裂纹（轴向 ≥ 50 mm）是**另一类物理对象**（整段贯穿、
与局部位点缺陷判别的目标不同）。Protocol V2 起：

- 位置级判别只以**局部缺陷**（轴向 < 50 mm）为正样本；
- ≥50 mm 大裂纹位置标为 `ignore`（不参与 0/1 训练与 AUC 计算），
  **不得**作为背景负样本；
- `label_status = "ignore"` 的记录在训练/评估中被显式排除（见
  `docs/M0_unified_ndt_schema.md` 的 `label_status` 字段）。

### 5.2 指标

位置级判别主指标：

- **position AUC**（ROC-AUC，非PP4 逐折均值）；
- **position PR-AUC**（P-R 曲线下面积，对缺陷稀疏场景更诚实）。

预留接口（M0-2 起，当前只留占位）：

- `defect_level_detection`：以独立缺陷实例为单位的检测（如最大回波阈值法、
  聚类后判定）；
- `false_positives_per_meter`（FP/m）：沿焊缝长度归一化的误报率。

评估脚本输出结构预留这些键（无数据时可空列表 / `null`），不得因缺失而崩。

## 6. 历史结果处置

- **不删除**任何历史结果文件；
- 被判定为 transductive / 泄漏 / smoke 的结果，JSON 内 `protocol` 改为对应值
  （如 `transductive_unlabeled` / `smoke`），并在报告里重新标注；
- `paut_p7_synth_ssl_s42_full.json`：实际为 **1024 样本、2 epoch** 的 smoke
  运行（见 §7），改名为 `_smoke` 后缀并标注 `run_type: smoke`；
- 任何"联合预训练有效"类结论，若无 strict 逐 coupon 预训练证据，必须改为
  "transductive 探索性证据 / 尚待严格验证"，不得作为严格跨试件结论。

## 7. Smoke 与 Full 的运行纪律

- `--smoke` 输出必须带 `_smoke` 后缀，**禁止覆盖** `_full` 结果；
- smoke 运行（1024 样本、2 epoch 预训练、4 epoch head）只用于管线连通性，
  数字无科学意义；
- full 运行必须满足 §4 的完整字段记录。

## 8. 自动防泄漏断言

`tests/test_eval_protocol.py` 提供：

1. **strict 预训练隔离断言**：给定 `(pretrain_coupons, test_coupon)`，
   断言 `test_coupon ∉ pretrain_coupons`；
2. **val 完整 coupon 断言**：给定 `(train_coupons, val_coupons)`，
   断言 `train_coupons ∩ val_coupons = ∅` 且两者都是 coupon 级集合（非位置）；
3. **normalization scope 断言**：strict 下 `normalization_scope` 必须是
   `train_coupons`；
4. **smoke/full 隔离断言**：smoke 输出文件名含 `_smoke`，且不与任何 `_full`
   文件重名。

新脚本接入时必须在 CI 中运行这些断言（见 `pyproject.toml` 的
`test` optional dependencies 与 GitHub Actions 配置）。
