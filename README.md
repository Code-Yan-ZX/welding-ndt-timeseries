# welding-ndt-timeseries

大模型 + 焊缝无损检测（NDT）--智能无损检测的时序基础模型研究仓库。

## 项目背景与长期目标

本项目探索**时序基础模型**与**时序–语言多模态模型**在焊接无损检测质量评估中的应用，长期目标是构建面向 NDT 信号的智能检测方法。

- **当前输入**：GMAW（熔化极气体保护电弧焊）工艺过程中的电压 / 电流信号--本质属于**工艺参数 / 电信号**。第一阶段的 ITFormer 基线实验即基于此（见下文）。
- **方向调整**：后续工作将把输入从“工艺参数”转向**相控阵（phased array）超声信号**或其它真正意义上的无损检测信号。现有 GMAW 电信号基线作为对照保留。
- **参考论文**：
  1. **ITFormer**: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset, ICML 2025，[arXiv:2506.20093](https://arxiv.org/abs/2506.20093)；代码 [Pandalin98/ITFormer-ICML25](https://github.com/Pandalin98/ITFormer-ICML25)。
  2. **Alliance**: All-in-One Spectral-Spatial-Frequency Awareness Foundation Model（谱-空-频联合感知基础模型，相控阵信号建模的重要参考）。

## 当前实验：ITFormer 基线

将 ITFormer（ICML 2025）适配到焊接质量二分类任务作为时序基础模型方向的 baseline，并与经典机器学习、简单深度学习、近期时序模型、文献 SOTA 官方复现四族方法系统对比。**详细实验报告见 [`reports/实验报告.md`](reports/实验报告.md)。**

### 任务

单周期焊接质量二分类：

- 输入：一个焊接周期 = 200 个电压采样 + 200 个电流采样（100 kHz，GMAW）
- 目标：`labels` ∈ {0 = 质量差, 1 = 质量好}；`-1` = 无标签（全部排除）
- 划分：官方 `(experiment, welding_run)` 实验对（论文惯例）
  - val：(3,32),(3,18),(1,27),(3,19),(3,17),(2,21),(1,20),(1,11)
  - test：(3,3),(2,10),(1,24),(3,24),(1,32),(2,1),(1,10),(1,16)（T 型接头 -> 刻意的 train->test 分布漂移）
  - train：其余全部实验对
- 数据集 v2 有标签行数：train 74,732 / val 10,614 / test 11,062

⚠ 官方 `tmdt-buw/VQ-VAE-Transformer-Arc-Welding` 代码中这两组实验对的变量名与论文**互换**（`dataloader/utils.py::get_val_test_ids`）；本仓库一律按 (exp, run) **对集合 + 行数断言**对齐，不照抄官方变量名。

### ITFormer 适配（QA 式似然打分）

- PatchTST 风格编码器（patch 20 -> 每通道 10 个 patch，d_model 512，4 层）
- ITFormer 桥接：25 个可学习指令 token + 两级 Instruct Time Attention（先通道维、后时间维），2 层
- 冻结本地 **Qwen3** LLM（项目要求用 Qwen3；论文原用 Qwen2.5）：25 个融合 token 替换 prompt 中的占位 token
- 训练：仅在答案 token 位置计算交叉熵（可训练 ≈ 34M 参数）
- 评测：单次前向，分数 = logit(good) − logit(bad)；**不做自由生成**（base 模型采样输出不可靠，似然打分是唯一可靠用法）

协议在适用处与官方仓库一致：逐通道 StandardScaler 只在 train 上 fit、WeightedRandomSampler、val macro-F1 早停、测试集每种子只评一次。指标：accuracy、binary F1（正类 = good）、macro F1、AUC；3 个种子（42/43/44）-> mean±std。

### 当前结果（test 集，截至 2026-08-06）

完整对比表见 `experiments/results/comparison_table.md`（可用 `python scripts/make_table.py` 重新生成）。要点：

- **经典 ML（RF / XGBoost + 48 维手工特征）目前最强**：test acc ≈ 0.70、AUC 0.79–0.80。
- **分布漂移是本基准的核心难度**：所有模型 val->test 显著跌落；test 集 good 占比（0.586）高于 train（0.470），多数类翻转。
- **ITFormer-QA（冻结 Qwen3 + 桥接）**：8B 两种子 test acc 0.6472±0.0204 / AUC 0.7166±0.0344；1.7B 单种子 test acc 0.6591 / AUC 0.7034。范式可行（仅训练 ≈34M 桥接参数，LLM 全冻结），但单周期输入下冻结 LLM 的语言先验未能补偿分布漂移，与 probe 消融、DLinear 等小模型大体同级。
- 文献锚点（Hahn et al., CIKM 2024）：VQ-VAE + Transformer ≈ 79.7% acc / 77% F1。
- ITFormer-QA 的 3 种子计划**未完成**（长时训练已按用户要求暂停），恢复方式见 [`reports/实验报告.md`](reports/实验报告.md) 第六节。

## SAW 时序基础模型实验

在 Zenodo [Submerged Arc Welding](https://doi.org/10.5281/zenodo.15083865) 数据集上尝试**时序基础模型**：将每道焊缝的 5 kHz 4 通道信号（电流/电压的 a/b 电极）切成 512 样本窗口，做**窗口级缺陷检测**（标签来自 `defects_xlocation.xlsx` 的缺陷采样区间），按试件划分（train PP3/4/5、val PP6、test PP7）。

对比：从零训练的 PatchTST 编码器（4.2M）vs 冻结的 **MOMENT-1-large** 时序基础模型（341M + 16K 线性探针）vs 经典 ML（RF）。**结果：从零小编码器最强（test AUC 0.635），冻结 MOMENT 跨试件泛化反而接近随机（AUC 0.489）**——预训练时序大模型的冻结特征未能迁移到焊缝 NDT。详见 [`reports/SAW时序基础模型实验报告.md`](reports/SAW时序基础模型实验报告.md)。

```bash
python scripts/saw_preprocess.py            # 窗口化 + 标签 + 划分
python scripts/saw_train.py --config configs/saw_encoder.yaml --seed 42   # 从零编码器
python scripts/saw_moment_probe.py          # MOMENT 冻结嵌入 + 线性探针
python scripts/saw_classic_ml.py            # RF / XGBoost
python scripts/saw_make_table.py            # 汇总表
```

## PAUT 相控阵超声缺陷检测实验（NDT 信号，本轮新进展）

延续项目方向，把输入从工艺信号推进到**真正的无损检测信号**。解析 SAW 数据集里的 **PAUT `.nde` 文件**（Evident/OmniScan X3 的 HDF5 容器，11 个，PP3–PP7 的 `2. ndt_data` 目录），提取每扫描位置的 49 波束 A-scan/B-scan（Group 0，71°，3500 采样 max-pool 降至 512），以 `defects_xlocation` 的局部缺陷（轴向 < 50 mm）为监督做**位置级缺陷检测**（贯穿型大裂纹 ≥ 50 mm 作背景，避免位置级退化）。划分与 SAW 一致（train PP3/4/5、val PP6、test PP7）以利跨模态对照。

模型：经典 ML（RF/XGB，包络手工特征）、从零 PatchTST 编码器（49 波束 B-scan，VarAttention 作注意力 MIL）、**谱-空-频 SSF 模型**（Alliance 启发，空间/时间谱/波束谱三分支）、冻结 MOMENT 探针。**结果（test AUC，主指标）**：SSF 最优且最稳 0.626±0.009，encoder 0.54±0.12（小数据下方差大），MOMENT 0.46（不迁移，与 SAW 互证），经典 ML ~0.49。跨模态对照：PAUT SSF(0.626) 与 SAW encoder(0.636) 相当；PAUT 经典特征(0.49)优于 SAW(0.22)；MOMENT 两模态均失败。详见 [`reports/PAUT相控阵缺陷检测实验报告.md`](reports/PAUT相控阵缺陷检测实验报告.md)。

> ⚠ 此为 **PP7 单点评估**。后续 LOOCV（见下节）显示跨试件泛化非PP4 AUC 仅 0.538，单点 0.626 偏乐观。

```bash
python scripts/paut_preprocess.py            # 解析 .nde -> A-scan/B-scan + 标签 + 划分
python scripts/paut_classic_ml.py            # RF / XGBoost
python scripts/paut_train.py --config configs/paut_encoder.yaml --seed 42   # 从零编码器
python scripts/paut_train.py --config configs/paut_ssf.yaml --seed 42       # 谱-空-频模型
python scripts/paut_moment_probe.py          # MOMENT 冻结探针
python scripts/paut_make_table.py            # 汇总表 + 跨模态对照
```

## PAUT 跨试件鲁棒性推进（P0–P3）

初始 PAUT 实验只在 PP7 单点评估，无法反映跨试件泛化。后续 4 个阶段改为 **5 折留一试件交叉验证（LOOCV）**，以 **非PP4 AUC**（剔除近零缺陷的 PP4 试件，4 折均值）为可信指标。

> **PP4 数据完整性已核实**：PP4 不是下载失败/解析 bug/标注遗漏。官方 AIMEN UT 报告证实 PP4 仅 1 个 2mm 可接受气孔、试件被接收，是 PENELOPE 零缺陷制造工作包下的近零缺陷试件。PP4 仅 3 个正样本，作 test 折时 AUC 纯噪声，故剔除。各试件缺陷标注数：PP3=68 / PP4=1 / PP5=50 / PP6=112 / PP7=12。
> 另：PP5 标注有 1 行 x_init>x_end 录入反转，旧版 `position_labels` 静默跳过（PP5 少计 18 个缺陷位置，占全量 0.6%，在 seed 噪声内）；`paut_preprocess.py` 已修复，P0–P3 结果沿用修复前标签，下次运行自动生效，结论不变。

| 阶段 | 方法 | 非PP4 AUC | 结论 | 报告 |
|---|---|---|---|---|
| P0 | LOOCV 上线 + 物理增强 / DANN 域对抗 / 多视角 / 温度缩放 | 0.538（裸 SSF） | **负面**：跨试件泛化差（远低于单点 0.626），增强/域对抗/多视角均未达 +0.03 门槛 | [`PAUT_P0`](reports/PAUT_P0_LOOCV实验报告.md) |
| P1 | SSL 掩码自编码器预训练 + McKnight Weibull 异常检测 | 0.572（+0.034） | **正面**：域内 SSL 预训练超基线，绕开有监督跨试件困难 | [`PAUT_P1`](reports/PAUT_P1_SSL预训练实验报告.md) |
| P2 | 多模态 LLM（Qwen3.6-27B）零样本 B-scan QA + LoRA | 0.593*（pooled） | **修正**：原"三阶段最优"是混口径伪影，统一口径后 SSL ≥ VLM（见 P4a 行） | [`PAUT_P2`](reports/PAUT_P2_多模态LLM实验报告.md) |
| P3 | 物理条件化 + 推理 CoT + LoRA 5 折 | 0.512 / 0.508 / 0.510 | **负面**：物理/CoT/微调均低于 bare(0.600)，瓶颈在感知而非推理 | [`PAUT_P3`](reports/PAUT_P3_物理条件化多模态LLM实验报告.md) |
| **P4a** | 信号原生表征变体（融合/微调/类型多任务/TTA）+ 天花板定位 | 0.579±0.007（SSL baseline 多seed） | **混合**：全杠杆多 seed 证伪；统一口径修正（SSL≥VLM）；天花板=表征级跨试件可判别性（+20% 标签无效） | [`PAUT_P4`](reports/PAUT_P4_信号原生表征LOOCV实验报告.md) |
| **P4b** | 深度掩码 SSL 预训练（depth 深度块 / both beam+depth） | 0.567±0.001（both）/ 0.513（depth） | **负面**：掩码目标不是杠杆，P1 beam-mask 已近表征上限；P4 证据图景完整，翻盘须新数据/合成数据（资源型） | [`PAUT_P4b`](reports/PAUT_P4b_深度掩码SSL预训练实验报告.md) |
| **P5** | 缺陷注入 SSL 预训练（注入 2D 高斯峰 + 多任务：MAE 重建 + 注入检测 + 注入定位） | 0.534±0.013（3 seed） | **负面 + 关键诊断**：注入任务本身可学到 ~100% 准确率 (inj_acc 0.998)，但学到的"高斯峰"特征不能迁移到真实缺陷；val-test gap ~0.34（比 P1 更严重）。H5 oracle 关键扩展：物理保真合成标签也无效 → 瓶颈是"真实缺陷形态空间"在 5 试件间的强变异 | [`PAUT_P5`](reports/PAUT_P5_缺陷注入SSL预训练实验报告.md) |
| **P5b** | **跨试件监督对比学习 (SupCon + 跨试件 batch 采样)** | **0.985±0.012（3 seed）** | 🎯 **重大突破**：nonPP4 0.985 远超 VLM 0.59+ 目标（+0.395）；val-test gap → 0；H5 oracle 结构性解药。机制：跨试件 batch 采样让 positives 天然来自不同试件 → 编码器被迫学"试件不变的判别特征" | [`PAUT_P5b`](reports/PAUT_P5b_跨试件监督对比学习实验报告.md) |

**主线结论**：PAUT 跨试件泛化困难（裸 SSF 非PP4 0.538）；域内 SSL 预训练（0.572）与通用多模态 LLM 视觉先验（0.593）两路有效，但注入物理/文本条件或微调反而退化--VLM 瓶颈是感知而非推理。**口径修正（P4a）**：P2 的 0.593（pooled）与 P1 的 0.572（逐折均值）不可比；统一口径下 SSL ≥ VLM。**天花板被打破（P5b）**：跨试件监督对比学习（SupCon + 跨试件 batch 采样）让编码器学"试件不变的判别特征"→ nonPP4 0.985±0.012（3 seed 稳定），远超 VLM 0.59+ 目标（+0.395），也远超 P1 baseline 0.579（+0.406）。**P5 关键诊断被 P5b 证伪**：P5a 物理保真合成注入失败不是"瓶颈不可破"，而是"SSL 任务与下游任务不对齐"——MAE 类任务学的是"重建被掩区域"（与判别缺陷正交），SupCon 直接用"是否有缺陷"作为对比信号（与下游任务完全对齐）。**P5b 价值**：H5 oracle 报告的"表征级天花板"的根本解药——不是新数据/合成数据，而是让对比信号自身携带跨试件不变量。汇总对比表见 [`experiments/results/paut_loocv_table.md`](experiments/results/paut_loocv_table.md)。

## 目录结构

```
configs/       每个模型族一份 YAML 配置
data/          原始 CSV + 预处理 memmap（不入 git）
src/wndt/      包：数据管线、模型、训练器、指标
scripts/       下载 / 预处理 / 训练 / 评测入口
tests/         单元测试（python tests/test_models.py [--with-llm]）
experiments/   runs/（checkpoint、日志）+ results/*.json + 汇总表
reports/       实验报告（中文，便于在 GitHub 上查看）
third_party/   官方 tmdt-buw 仓库 clone（仅一处必要补丁）
```

## 本地资源（不入 git，每台机器配置一次）

- **数据集**：由 `scripts/01_download_data.sh` 自动下载到 `data/raw/`（1.3 GB，MD5 校验），无需手动配置。
- **Qwen3 权重**：configs 中引用 `models/Qwen3-8B` 与 `models/Qwen3-1.7B-Base`（相对路径）。软链本地副本或从 HuggingFace 下载：
  ```bash
  ln -s /path/to/Qwen3-8B         models/Qwen3-8B
  ln -s /path/to/Qwen3-1.7B-Base  models/Qwen3-1.7B-Base
  # 或：huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
  ```
  需要时可用 `--model.llm_path /your/path` 单次覆盖；亦可设置 `QWEN_1P7B` 环境变量。
- **Conda 环境**：基础环境执行 `pip install -e .`（+ `xgboost`）；官方仓库环境由 `bash scripts/setup_official_env.sh` 构建。
- **官方仓库**：`scripts/run_official_repo.sh` 会自动 clone `tmdt-buw/VQ-VAE-Transformer-Arc-Welding` 到 `third_party/` 并应用 `scripts/official_repo.patch`（logging-tag 修复 + transformer checkpoint）。

## 复现

```bash
bash scripts/01_download_data.sh          # Zenodo CSV 下载 + MD5 校验
python scripts/02_preprocess.py           # 生成 memmap + 划分 + 归一化统计
python tests/test_models.py --with-llm    # 单元测试（GPU，Qwen3-1.7B）
bash scripts/smoke_test.sh                # 1-epoch 小子集管线冒烟
bash scripts/run_baselines.sh             # 我方 baseline，3 个种子
python scripts/run_classic_ml.py          # RF / XGBoost / SVM
bash scripts/setup_official_env.sh        # 官方仓库的 py3.11 环境
bash scripts/run_official_repo.sh         # 复现官方管线（自动 clone + 打补丁）
python scripts/eval_official_ckpt.py ...  # 在 canonical split 上重评官方 ckpt
bash scripts/run_itformer_qa.sh sweep     # 1.7B 学习率探针
bash scripts/run_itformer_qa.sh full 8b   # ITFormer-QA 正式运行
python scripts/make_table.py              # -> experiments/results/comparison_table.md
```

## 参考文献

- 数据集：Hahn et al., *Metal Arc Welding – Predictive Quality Arc Welding Dataset*, Zenodo, DOI [10.5281/zenodo.10017718](https://doi.org/10.5281/zenodo.10017718)（CC-BY-4.0，v2）。
- 基准论文：Hahn et al., *Quality Prediction in Arc Welding: Leveraging Transformer Models and Discrete Representations from Vector Quantised-VAE*, CIKM 2024, DOI [10.1145/3627673.3680031](https://doi.org/10.1145/3627673.3680031)；代码 [tmdt-buw/VQ-VAE-Transformer-Arc-Welding](https://github.com/tmdt-buw/VQ-VAE-Transformer-Arc-Welding)。
- 模型论文：Wang et al., *ITFormer: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset*, ICML 2025, [arXiv:2506.20093](https://arxiv.org/abs/2506.20093)；代码 [Pandalin98/ITFormer-ICML25](https://github.com/Pandalin98/ITFormer-ICML25)。
- LLM：Qwen3-8B / Qwen3-1.7B-Base（本地权重，bf16）。
