# General NDT Foundation — E1 单域 SSL（PENELOPE vanilla MAE + 冻结探针）

> 日期：2026-09-02　分支：`research/general-ndt-foundation`
> 脚本：`scripts/general_ndt_e1_single_ssl.py`　配置：`configs/general_ndt_e1_single_ssl.yaml`
> 结果 JSON：`experiments/results/general_ndt_e1_results.json`（逐折 ckpt 在 `experiments/runs/general_ndt_e1/ckpts/`）
> 主指标 = **非PP4 逐折均值 ± std**（3 seed，seed 职责分离）

---

## 一、目标

对照 E0 scratch-supervised（AUROC 0.5254 ± 0.0801）：**单域 vanilla MAE 自监督预训练**
能否在 general_ndt 骨干上超过从头监督 —— E1 是 E2（多源 SSL）的前置验证。

## 二、协议（严格 per-fold pretrain）

| 项 | 设置 |
|---|---|
| 数据 | PENELOPE 3000 位置级 B-scan（49×512），0/1 标签 |
| **per-fold 严格预训练** | 对每个 test coupon T（非PP4），MAE 只在其余 4 coupon（**无标签**）上预训练 —— **test coupon 的任何信号都不得进入预训练**（无 transductive 泄漏，对齐项目"评估严谨性"纪律） |
| 预训练 | vanilla MAE（random mask 0.5），d=128 / 4 层 enc / 1 层 dec / patch=16，AdamW lr=1e-3 / 3000 步（≈40 epoch，对齐 P1）/ per-sample z-score |
| 冻结探针 | CLS pooled 特征 + logistic（class_weight balanced），在 **E0 同划分**（rest 85/15 分层 train/val，test=coupon）上训练 head → 评估 test coupon |
| seed | model_seed ∈ {0,1,2}（预训练初始化/训练随机性）；data_seed=42（划分/采样）—— 职责分离 |
| 主指标 | 非PP4 逐折均值 ± std（跨 4 折 × 3 seed = 12 项） |

## 三、结果（每折 AUROC，test coupon 完全隔离 + 预训练隔离）

| seed | PP3 | PP5 | PP6 | PP7 | 逐折均值 |
|---|---|---|---|---|---|
| 0 | 0.4578 | 0.6199 | 0.5745 | 0.6436 | 0.5740 |
| 1 | 0.4649 | 0.5891 | 0.5965 | 0.5725 | 0.5557 |
| 2 | 0.4435 | 0.5894 | 0.5987 | 0.6660 | 0.5744 |

**主指标：AUROC = 0.5680 ± 0.0701（非PP4 逐折均值，12 折×seed）**（耗时 95 min，RTX 4090 D）

## 四、对照 E0（负迁移审计）

| 基线 | 非PP4 逐折 AUROC | seed0 Δ | seed1 Δ | seed2 Δ |
|---|---|---|---|---|
| E0 scratch-supervised | 0.5254 ± 0.0801 | — | — | — |
| **E1 单域 SSL（本轮）** | **0.5680 ± 0.0701** | **+0.0403** | **+0.0530** | **+0.0347** |

**Δ = E1 − E0 = +0.0427 ≥ +0.01，且 3/3 seed 为正 → E1 判正迁移（positive transfer）✅**

## 五、诊断

1. **单域 vanilla MAE 预训练显著超过 scratch 监督**（+0.043，3/3 seed 正）：与 PAUT 既有结论
   （P1 域内 SSL 是唯一有效路线，SSL ≥ VLM）在 general_ndt 骨干上复现。
2. **PP7（稀疏缺陷，正率 0.138）从 E0 最弱折（0.39–0.53）变为 E1 最强折（0.57–0.67）**：
   SSL 预训练在标签稀疏时收益最大 —— 无监督结构学习弥补了正样本稀缺，冻结表征跨 coupon
   迁移性更好。这是最显著的可解释信号。
3. **PP5/PP6 折中等提升**（+0.01~+0.03）：缺陷率中等/高的折本身可判别性较强，SSL 增益收窄。
4. **val-test 鸿沟**：E1 的 test AUROC 跨度（0.44–0.67）仍大，跨 coupon 泛化仍是瓶颈，
   但 SSL 表征整体抬升了分布。

## 六、结论

- general_ndt 骨干的 **E1 单域 vanilla MAE SSL = 0.5680 ± 0.0701**（非PP4 逐折均值，3 seed）。
- **相对 E0 scratch（0.5254）判正迁移（Δ=+0.0427，3/3 seed）** → 单域 SSL 是有效路线，
  E2 多源 SSL 的正当性得到支撑。
- 下一步：E2 多源 SSL（PENELOPE + EddyCus 等无标签预训练）对照 E1；若 E2 相对 E1 再提升
  且过负迁移审计，则多源假设成立。

## 七、复现

```bash
python scripts/general_ndt_e1_single_ssl.py --config configs/general_ndt_e1_single_ssl.yaml
```
