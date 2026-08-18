# M0：后续实验路线图（M0-2 规划，本轮不执行）

> 阶段：M0-1 公开数据集审计与统一数据架构设计
> 日期：2026-08-18
> **本路线图只规划，不执行。** 具体数据接入优先级以
> `docs/M0_public_ndt_dataset_audit.md` 为准；本仓库 P0–P6 已证明
> 5 试件 PAUT 的表征级天花板（非PP4 逐折 ~0.58），翻盘必须靠**新数据/更大
> 数据底座**，这正是 M0-2 的动机。

## 一、M0-2 目标

按审计优先级接入公开 NDT 数据集（超声源域 + 涡流），建立：

1. 每个接入数据集的 **manifest + adapter**（统一 schema，见
   `docs/M0_unified_ndt_schema.md`）；
2. **单模态基线**（超声、涡流各自独立）；
3. **跨数据集评估协议**（in-domain / cross-specimen / cross-sensor /
   cross-material）；
4. **融合接口保持就绪但不训练**——除非获得可验证的同试件、同坐标、成对
   UT+ECT 公共数据。

## 二、M0-2 最小实验矩阵（候选基线）

### 超声（源域预研，为未来合作单位数据做技术准备）

| 基线族 | 方法 | 输入 | 预期作用 |
|---|---|---|---|
| 经典 ML | handcrafted（包络/谱特征）+ RF / XGBoost | A-scan/B-scan 特征 | 廉价上限参照（复用 `wndt/features/handcrafted.py` 思路） |
| 2D CNN | 2D 卷积（B-scan 视为图像） | B-scan | 简单 CNN 上限 |
| 2.5D/3D CNN | 2.5D/3D 卷积（体积） | B-scan 堆叠/体积 | 更强空间建模 |
| SSL | MAE / 掩码自监督预训练 → 下游 | B-scan | 域内自监督（P1 已验证有效路线） |
| 源监督预训练 | 在源域监督训练 → 迁移 PAUT | B-scan | 跨数据集迁移（MOMENT 冻结已失败，改为域内/源域预训练） |

### 涡流

| 基线族 | 方法 | 输入 | 预期作用 |
|---|---|---|---|
| 经典 ML | handcrafted + RF / XGBoost | I/Q 曲线特征 | 廉价上限参照 |
| 1D ResNet/ResNeXt | 1D 深度残差 | I/Q 曲线 | MDDECT 深度分类主线（论文 ResNeXt-38 93.58%） |
| 时频 2D CNN | STFT/CWT 谱图 + 2D CNN | 时频图 | 频域表征 |
| 多频/多传感器注意力 | 跨频率/通道注意力 | 多频 I/Q | EddyCus 多传感器、多频 |
| 深度分类/回归 | 缺陷深度档位分类或回归 | I/Q | MDDECT 核心任务 |
| lift-off 鲁棒性 | lift-off 域泛化评估 | I/Q | MDDECT 含 lift-off 变化 |

### 融合（只有取得配对数据后才训练）

- **unimodal baselines**（先各自训好）
- **score-level late fusion**（可先在单模态上独立训练再融合，不违反配对约束）
- **feature-level gated fusion**（`GatedFusionHead` 接口已就绪）
- **missing-modality fusion**（`availability mask` 接口已就绪）
- **uncertainty-aware fusion**（待定）

> ⚠ 红线：没有可验证的同试件、同坐标、成对 UT+ECT 公共数据前，**不做真正的
> 融合训练**；严禁把不同试件/材料/任务的 UT 与 ECT 强行拼接后称为融合。

## 三、M0-2 执行顺序建议

1. **优先接入**（audit 后按 P0/P1 优先级）：
   - 超声源域：NDT_ML_Flaw（真实焊缝缺陷优先）、ML-NDT（PAUT 体积 + 缺陷尺寸）；
   - 涡流：MDDECT（真实金属 ECT，但需先登录 Kaggle 核实 license 与分组结构）、
     EddyCus-HDF5（结构化完整、CC BY 4.0，但对象是 CFRP）。
2. 每个数据集：写 manifest → 写 adapter → 跑单模态基线 → 记入 README 实验日志。
3. 评估协议统一：**主指标 = 非PP4 逐折均值**（沿用本仓库口径），跨数据集时按各自
   manifest 的 split_group 物理单元划分；pooled 仅作参考。
4. 融合仅在配对数据到手后启动。

## 四、不做的事（M0-2 边界）

- 不做超过 10 GB 的新下载（EddyCus 3.7 GB 可下；PENELOPE 12.7 GB 已在本机）；
- 不自动下载全部 LOTSA/UTSD；
- 不跑长时 GPU 大规模训练（单模态基线可，融合/预训练大规模任务需用户确认）；
- 不把公开小数据实验描述为最终课题结论；
- 不宣称已实现超声—涡流融合。

## 五、完成标准（M0-2 结束）

- [ ] ≥2 个超声源域数据集完成 manifest + adapter + 单模态基线
- [ ] ≥1 个涡流数据集完成 manifest + adapter + 单模态基线
- [ ] 跨数据集/跨域协议至少各一种跑通并出结果
- [ ] 融合接口经单测验证可运行（已具备），融合训练仍保持"待配对数据"
- [ ] README 实验日志新增 M0-2 条目
