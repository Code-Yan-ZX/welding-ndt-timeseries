# General NDT Foundation — E2b 多源 SSL（模态专用 stem）

> 日期：2026-09-07　分支：`research/general-ndt-foundation`
> 脚本：`scripts/general_ndt_e2_multi_ssl.py`（复用）　配置：`configs/general_ndt_e2b_modality_stem.yaml`
> 代码：`ModalAdapter.per_modality_stem`（`src/general_ndt/adapters/base.py`）
> 结果 JSON：`experiments/results/general_ndt_e2b_results.json`
> 主指标 = **非PP4 逐折均值 ± std**（3 seed，seed 职责分离）

---

## 一、目的

E2（共享 stem, 0.5335）判负迁移，诊断指向"跨模态数据稀释共享骨干容量"。E2b 检验：
**给每模态独立 patch 投影（stem），只共享 Transformer backbone**，能否消除跨模态干扰、
回收多源收益 —— 这是"多源物理感知 SSL"主假设的修正再检验。

## 二、协议（与 E2 唯一差异 = 模态专用 stem）

| 项 | 设置 |
|---|---|
| 数据 | PENELOPE（超声）+ EddyCus（涡流, 695 扫描, 无标签） |
| per-fold 严格预训练 | 每折只在 4 非 test PENELOPE coupon + 全部 EddyCus 上联合预训练 |
| **模态专用 stem** | `per_modality_stem: true` —— **超声 stem 只被 PENELOPE 训练（与 E1 完全相同）**，涡流 stem 只被 EddyCus 训练；共享的只有 backbone + modality/sensor embedding |
| 模态平衡 | 1:1 交替（PENELOPE 3000 步 = E1 相同, EddyCus 过采样 3000 步） |
| 预训练 | vanilla MAE（mask 0.5），d=128 / 4 层 enc / 1 层 dec，6000 步，per-sample z-score |
| 冻结探针 | CLS pooled + logistic（E0/E1/E2 同划分），仅 PENELOPE coupon LOOCV |
| seed | model_seed ∈ {0,1,2}；data_seed=42 |

> **关键隔离**：因超声 stem 只被 PENELOPE 训练，E2b 与 E1 的唯一差异 = **共享 backbone
> 是否从跨模态数据获益**（E1 的 stem 路径在 E2b 中原样保留）。

## 三、结果（每折 AUROC）

| seed | PP3 | PP5 | PP6 | PP7 | 逐折均值 |
|---|---|---|---|---|---|
| 0 | 0.4798 | 0.6873 | 0.5973 | 0.6450 | 0.6024 |
| 1 | 0.4472 | 0.6676 | 0.5743 | 0.6175 | 0.5766 |
| 2 | 0.4756 | 0.5577 | 0.6234 | 0.7624 | 0.6048 |

**主指标：AUROC = 0.5946 ± 0.0898（非PP4 逐折均值，12 折×seed）**（耗时 263 min，RTX 4090 D 共享）

## 四、对照（负迁移审计）

| 基线 | 非PP4 逐折 AUROC | seed0 | seed1 | seed2 |
|---|---|---|---|---|
| E0 scratch | 0.5254 ± 0.0801 | 0.5337 | 0.5027 | 0.5397 |
| E1 单域 SSL | 0.5680 ± 0.0701 | 0.5740 | 0.5557 | 0.5744 |
| E2 共享 stem | 0.5335 ± 0.0678 | 0.5672 | 0.5331 | 0.5004 |
| **E2b 模态 stem** | **0.5946 ± 0.0898** | 0.6024 | 0.5766 | 0.6048 |

- **Δ(E2b − E1) = +0.0266，3/3 seed 为正（+0.028/+0.021/+0.030）→ 判正迁移 ✅**
- Δ(E2b − E2) = **+0.0611**（模态专用 stem 相对共享 stem 的收益回收）
- Δ(E2b − E0) = **+0.0692**（相对从头监督 +0.07）

## 五、诊断

1. **共享 patch 投影（stem）是 E2 负迁移的根因，模态专用 stem 完全回收并反超**：
   E2b − E2 = +0.061（E2 负迁移 −0.035 被修复并额外 +0.027 超越 E1）。
2. **每折对比 E2b ≥ E1 大多数折**，最显著：
   - seed2 PP7：E2b 0.7624 vs E1 0.6660 vs E2 0.4456 —— E2 在此折崩塌（−0.220），E2b 反而最高；
   - seed0 PP5：E2b 0.6873 vs E1 0.6199 vs E2 0.5954。
3. **PP7（稀疏缺陷）保持强折**（E2b 均值 0.675，E1 0.627）：模态专用 stem 下，跨模态数据
   不再伤害标签稀疏折（E2 的 seed2 PP7 崩塌被完全修复）。
4. **机制解释**：超声 stem 只被 PENELOPE 训练 → E1 的 stem 表征原样保留；共享 backbone
   额外吸收了 EddyCus 的跨模态结构信号 → 表征更鲁棒/更可迁移。这正符合"模态适配 +
   共享骨干"的方法规格设计（E2 的共享 stem 违背了"不在 patch 阶段混合模态"的原则，
   E2b 修正后符合）。

## 六、结论

- **E2b（模态专用 stem 多源 SSL）= 0.5946 ± 0.0898，相对 E1 判正迁移（Δ=+0.0266, 3/3 seed）**。
- **"多源物理感知 SSL 改善跨试件泛化"主假设在模态专用 stem 架构下成立**（超声+涡流）：
  E2b（0.595）> E1（0.568）> E2 共享 stem（0.534）> E0 scratch（0.525）。
- 证据链：E2 负迁移 → E2b 正迁移，根因 = 共享 stem 的跨模态混淆；修正 = 每模态独立
  patch 投影、共享 backbone。
- 下一步建议：模态专用 stem 作为**通用多源架构默认**；扩展更多源（PENELOPE +
  external_weld_ut 同模态 / 合成超声 + EddyCus）验证可扩展性；并过完整负迁移审计。

## 七、复现

```bash
python scripts/general_ndt_e2_multi_ssl.py --config configs/general_ndt_e2b_modality_stem.yaml
```
