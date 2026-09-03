# General NDT Foundation — E2 多源 SSL（PENELOPE + EddyCus 联合预训练）

> 日期：2026-09-04　分支：`research/general-ndt-foundation`
> 脚本：`scripts/general_ndt_e2_multi_ssl.py`　配置：`configs/general_ndt_e2_multi_ssl.yaml`
> 结果 JSON：`experiments/results/general_ndt_e2_results.json`（逐折 ckpt 在 `experiments/runs/general_ndt_e2/ckpts/`）
> 主指标 = **非PP4 逐折均值 ± std**（3 seed，seed 职责分离）

---

## 一、目标

对照 E1 单域 SSL（0.5680 ± 0.0701）：**加入跨模态无标签数据（EddyCus 涡流）联合预训练**
能否在 PENELOPE 上进一步改善跨 coupon 泛化 —— E2 是"多源物理感知 SSL"主假设的直接检验。

## 二、协议（严格 per-fold pretrain + 模态平衡）

| 项 | 设置 |
|---|---|
| 数据 | PENELOPE（超声, 4 非 test coupon, 无标签）+ EddyCus（涡流, 695 扫描 exploratory_flat_1d, 无标签） |
| **per-fold 严格预训练** | 每折只在 4 非 test PENELOPE coupon + 全部 EddyCus 上联合预训练 —— test coupon 信号不进预训练（无 transductive 泄漏） |
| **模态平衡采样** | 1:1 确定性交替（`train_multi`）：**PENELOPE 曝光 3000 步 = E1 相同**，EddyCus 小源过采样 3000 步 → 干净隔离"多源"效应（唯一差异 = 额外跨模态数据） |
| 预训练 | vanilla MAE（random mask 0.5），d=128 / 4 层 enc / 1 层 dec，AdamW lr=1e-3 / 6000 步，per-sample z-score，共享 backbone（modality embedding 区分超声/涡流） |
| 冻结探针 | CLS pooled 特征 + logistic（class_weight balanced），E0/E1 同划分（rest 85/15 分层 + test=coupon），仅评估 PENELOPE |
| seed | model_seed ∈ {0,1,2}；data_seed=42 —— 职责分离 |
| 主指标 | 非PP4 逐折均值 ± std（4 折 × 3 seed = 12 项） |

## 三、结果（每折 AUROC，test coupon 完全隔离 + 预训练隔离）

| seed | PP3 | PP5 | PP6 | PP7 | 逐折均值 |
|---|---|---|---|---|---|
| 0 | 0.4628 | 0.5954 | 0.5565 | 0.6540 | 0.5672 |
| 1 | 0.4438 | 0.6058 | 0.5253 | 0.5576 | 0.5331 |
| 2 | 0.4525 | 0.5175 | 0.5859 | 0.4456 | 0.5004 |

**主指标：AUROC = 0.5335 ± 0.0678（非PP4 逐折均值，12 折×seed）**（耗时 220 min，RTX 4090 D）

## 四、对照 E0 / E1（负迁移审计）

| 基线 | 非PP4 逐折 AUROC | seed0 | seed1 | seed2 |
|---|---|---|---|---|
| E0 scratch | 0.5254 ± 0.0801 | 0.5337 | 0.5027 | 0.5397 |
| E1 单域 SSL | 0.5680 ± 0.0701 | 0.5740 | 0.5557 | 0.5744 |
| **E2 多源 SSL** | **0.5335 ± 0.0678** | 0.5672 | 0.5331 | 0.5004 |

- **Δ(E2 − E1) = −0.0345，3/3 seed 为负（−0.007/−0.023/−0.074）→ 触发负迁移判据（Δ ≤ −0.01 且全 seed 负）→ 判负迁移，按协议停止该方向。**
- Δ(E2 − E0) = +0.0082（< +0.01 门槛）→ 相对 scratch 也无正迁移（仅微弱高于随机初始化监督）。

## 五、诊断

1. **加入跨模态 EddyCus 干扰了 PENELOPE 表征学习**：E2 落在 E0 与 E1 之间
   （0.5335 vs 0.5254/0.5680）——共享骨干容量被跨模态数据稀释，交替优化两种差异很大的
   重建任务（超声 49×512 稀疏 A-scan vs 涡流 8×2048 阻抗轨迹）损害了目标域特定表征。
2. **最显著单折崩塌：seed2 PP7（Δ=−0.2205，E2 0.4456 vs E1 0.6660）**：稀疏缺陷 coupon
   （正率 0.138）在跨模态干扰下判别信号被进一步稀释（呼应 E1 发现：SSL 在标签稀疏时
   收益最大 → 反向，跨模态干扰在标签稀疏时伤害也最大）。
3. **PP7 总体下降**（E1 0.627 → E2 0.552）：跨模态预训练对 E1 最有价值的一折伤害最大。
4. **与仓库历史互证**：M0-2B/2C 外部超声/涡流迁移对 PAUT 均为负迁移；E2 表明即使换成
   general_ndt 多源 MAE 框架（模态平衡 + per-fold 严格），**跨模态（超声+涡流）联合
   预训练仍无法翻盘** —— "多源物理感知 SSL 正面改善跨试件泛化"这一主假设在
   超声+涡流组合上被证伪（至少在当前骨干/规模/训练量下）。

## 六、结论

- **E2 多源（超声+涡流）联合预训练对 PENELOPE = 负迁移（Δ=−0.0345，3/3 seed 负）**。
- E1（单域 vanilla MAE）仍是当前最佳（0.5680），且显著优于 E0 scratch（0.5254）。
- **按协议停止"超声+涡流跨模态联合预训练"方向**；多源假设若要再试，需换组合/架构
  （如同模态多源、模态专用 stem、或更大的目标域数据占比），并重新过负迁移审计。

## 七、复现

```bash
python scripts/general_ndt_e2_multi_ssl.py --config configs/general_ndt_e2_multi_ssl.yaml
```
