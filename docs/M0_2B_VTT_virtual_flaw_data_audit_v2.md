# M0-2B VTT 虚拟缺陷数据审计 v2（audit_v2）

> 日期：2026-08-20（audit_v2 修正版，取代 v1 报告的结论部分）
> 对象：ML-NDT（arXiv:1903.11399）与 NDT_ML_Flaw（VTT / koomas），M0-2B 外部
> 超声 MAE 预训练素材。
> **本报告明确区分三类内容**：
> - **A. 上游公开说明的事实**（论文 / 官方 README / 元数据中可直接核实的）；
> - **B. 当前实验能够支持的结论**（基于 det_v2 实验与审计脚本的实测）；
> - **C. 不能支持的推断**（当前证据不足以得出、本报告明确不宣称的）。
>
> 只用第⑤项（作者未公开说明的异常）有明确证据时才讨论不当行为；**本报告未发现
> 第⑤项，不使用"造假"作为正式结论。**

---

## A. 上游公开说明的事实（可直接核实）

**A1. ML-NDT（Virkkunen et al., 2019, arXiv:1903.11399）**
- 1 个物理试件（316L 奥氏体管道单对焊接头），**3 条真实热疲劳裂纹**
  （深度 1.6 / 4.0 / 8.6 mm）："Three thermal fatigue cracks with depths 1.6,
  4.0 and 8.6 mm"；"The raw data contained only three real cracks, that were
  then modified to give the total data set."
- **20,000 张变体由这 3 条裂纹经 eFlaw 虚拟缺陷流程生成**："Altogether 20000
  variations were generated"；"The extracted flaw signal can then be implanted
  into different locations of the scan data"；"The flaw signals extracted can be
  moved to different samples."
- **`.bins(100,256,256)` 是 100 张 B-scan 图的 minibatch 容器，不是三维体积
  采集**："The data was stored in minibatches of 100 UT-images per file."
- 反捷径设计（作者声明）："the virtual flaw process had been used to copy
  unflawed section to another location"（避免模型学会识别植入过程）。

**A2. NDT_ML_Flaw（VTT / koomas 官方 README）**
- 1 个物理试件（P41 异种金属焊缝）；每批 1000 条带，~50% 缺陷 / 50% 无缺陷；
- "**2xx batches are simulated flaws**"（CIVA 仿真）；"The data size is
  480x7168 and the flaw area 1100-3100"；augmentation factor 0.4–1.0。

**A3. 元数据（本仓库实测，与 A1/A2 一致）**
- ML-NDT：201 个 `.bins` = **20,010 张增强图**（200×100 + 1×10），其中
  **12,128 张 flaw-positive、7,882 张 clean/control**；`.jsons` 的
  `original_location` / `location` / `factor` 字段记录"模板重复植入 + 缩放"；
  3 个真实裂纹模板（size 1.6/4.0/8.6）驱动全部 flaw-positive 图，另有 2 个
  size=0 噪声模板（label=0）。
- NDT_ML_Flaw：17 批 × 1000 = 17,000 条带；**6 个真实缺陷**
  （P41_01..05 裂纹 + P41_06_notch EDM）+ **10 个 CIVA 仿真模板**；每真实批
  混合多个缺陷类型 + clean；`.lzma` 文件实际为 XZ 压缩（原始数据 ~117 GB，
  压缩 235.9 MB）。

## B. 当前实验能够支持的结论（det_v2 实测）

**B1. 有效独立单元远小于 nominal 数量**
- ML-NDT：20,010 张图 ≠ 20,010 个独立采集；有效独立单元 ≈ **3 条真实裂纹**
  （12,128 张缺陷图是其重复植入，7,882 张干净图是共享背景/噪声）。
- NDT_ML_Flaw：17,000 条带 ≠ 17,000 个独立缺陷；有效独立单元 ≈ **6 个真实
  缺陷 + 10 个 CIVA 模板**。

**B2. 三种子迁移结论（主判据，det_v2 实验）**
> 命名说明：这里的 **E0 / E1 / E2 / E3 是 M0-2B 的 PAUT 严格 LOOCV 条件**
> （E0=冻结随机 encoder 探针、E1=目标域 SSL、E2=外部 SSL、E3=外部→目标域），
> **与本审计的小 CNN 缺陷检测协议（random_image_level / leave_* /
> metadata_only 等，见 B3）是两套东西**，不混用。

- PAUT 严格 LOOCV 非PP4 逐折：E0 0.5484±0.0037 / E1 0.5273±0.0142 /
  E2 0.5410±0.0111；**平均 E2−E0 = −0.0075**（1/3 seed 为正）→
  **+0.01 且 2/3 seed 判据未过 → 公开超声外部迁移停止，不扩大公开超声模型/数据**。
- 初始版本"E2 外部直接迁移有正信号 / E3 目标域继续 MAE 有害"两结论均被
  deterministic v2 证伪（初始化 seed 顺序问题的伪影）。

**B3. 捷径学习审计（小 CNN，audit_v2 修正后）**
- 随机样本级缺陷检测 **AUC≈1.0**（ML-NDT 与 NDT_ML_Flaw 均接近完美）；
- **近重复（template 泄漏）是随机高分的重要机制**：ML-NDT test 样本 **99.7%**
  能在 train 中找到 cos>0.99 的同模板近重复（最近邻 100% 同模板）；
- **clean 复用泄漏修复后，ML-NDT leave-template-out 明显崩塌**：8.6mm 大裂纹
  AUC=1.0、**1.6mm AUC=0.41（低于机会）、4.0mm AUC=0.76** → 模型对**新尺寸
  模板（尤其小裂纹）不能泛化**；v1 报告被 clean 泄漏高估；
- NDT leave-one-real-defect-out 修复后 **AUC 仍=1.0**（同一试件 P41 内缺陷间
  泛化，非跨试件）；leave-container / leave-batch 后 AUC 仍≈1.0；
- **背景/上下文捷径（探索性）**：ML-NDT background-only **AUC≈0.99**（启发式
  bbox 定位缺陷区域，非真实 mask → 探索性结论）；metadata-only（批次/容器指纹）
  弱（~0.58 / 0.49）；
- **结论：随机样本级性能主要受模板复用 + 背景/上下文（植入残留）影响，不能
  代表对新真实缺陷的泛化**（按任务规定措辞）。

**B4. E2 预训练实际看到了什么（data_seed=42 采样计划实测）**
- ML-NDT 采样帧 60.4% 为虚拟缺陷（3 条裂纹副本）、39.6% 背景；
- NDT_ML_Flaw 随机 (480,256) 窗口仅 32.6% 与缺陷扫描区 [1100,3100] 相交，
  ~2/3 只是背景 → **E2 可能学到的是 VTT 采集背景/纹理 + 少量缺陷模板特征**，
  而非大量独立真实缺陷响应。

## C. 不能支持的推断（本报告明确不宣称）

1. **不能**把 20,010 / 17,000 当作"数万条独立真实缺陷"或"大规模独立真实缺陷
   数据"。
2. **不能**据随机样本级 AUC≈1.0 宣称"接近 100% 真实缺陷识别/泛化"——该性能含
   模板近重复与背景捷径，不代表对新真实缺陷的泛化。
3. **不能**把 E2/E3 的任何信号解释为"学到通用真实缺陷物理表征"或"跨新焊缝 /
   新试件 / 新真实缺陷泛化"。
4. **不能**把 det_v2 的外部超声迁移结果当作"PAUT 天花板被突破"的证据
   （E2−E0 平均为负，判据未过）。
5. **E3 仅一个确定性种子（seed42）**，**不能**据此下"目标域继续 MAE 有害"或
   "有效"的结论（audit_v2 措辞）。
6. **ML-NDT flaw-only / background-only / boundary-only 是探索性结果**：缺陷
   回波区域由启发式最亮像素边界框定位，**不是真实植入 mask**；在获得真实植入
   mask 前，不能据此下"缺陷检测由背景驱动"的正式结论。
7. **NDT_ML_Flaw 的 shortcut（flaw/background 裁剪）已删除**：原实现用缺陷
   metadata 位置裁剪（标签相关特征，存在泄漏），且无真实 mask；不报告其结果。

## 结论

**A（上游公开事实）** 确认：两数据集均为 VTT 虚拟缺陷（eFlaw）生成，作者公开
声明，非"造假"；ML-NDT 仅 1 试件 3 真实裂纹、NDT_ML_Flaw 仅 1 试件 6 真实
缺陷 + 10 CIVA 模板，`.bins` 为 minibatch 容器。

**B（实验可支持）**：det_v2 三种子 E2−E0 = −0.0075，公开超声外部迁移停止；
审计显示随机样本级性能含模板近重复与背景/上下文捷径，不能代表对新真实缺陷的
泛化；E2 预训练约 2/3 NDT 窗口只是背景。

**C（不支持）**：不得把 nominal 数量当独立缺陷、不得宣称随机级高分为真实缺陷
泛化、不得说 E3 有害/有效（单种子）、不得把启发式 mask 的 shortcut 当正式结论。

> 允许 / 禁止用途分级见 v1 报告 §9（本节继承）：允许做数据管线测试、virtual-flaw
> 方法研究、无标签背景/采集纹理预训练、PENELOPE 目标集迁移探索；禁止作为独立
> 真实缺陷、随机划分泛化证据、跨试件泛化证据、大模型真实训练数据依据。
