# Phase 1：公开数据集候选审计（多源信号型 NDT 全景）

> 阶段：General NDT Foundation / Physics-Aware SSL — Phase 1
> 分支：`research/general-ndt-foundation`
> 审计日期：2026-09-01
> 方法：优先一手来源（Zenodo API / arXiv API / OpenAlex / Crossref / PMC / GitHub API /
> Mendeley / Figshare / Kaggle / 机构仓库），辅以仓库内 `docs/M0_public_ndt_dataset_audit.md`
> （2026-08-18）与 M0-2C 审计交叉核对。无法在线确认的项目明确标注"待核实"，不臆造。
>
> **审计纪律**：严格区分 样本数 / 独立试件数 / 独立缺陷实例数；一个缺陷重复扫描数千次
> ≠ 数千个独立缺陷。**不把普通表面 RGB 缺陷检测数据混入本阶段。**
>
> 分级定义：
> - **A 核心 benchmark**：多试件/多独立单元 + 可靠标签 + 许可清晰，可做严格下游评测；
> - **B 无标签预训练**：原始信号规模大或模态独特，但标签弱/单试件，仅作预训练语料；
> - **C 外部迁移验证**：模态相关、可下载，但规模小或单结构，仅作跨域迁移对照；
> - **D 暂不使用**：模态错位（如纯振动）、下载失效、许可不明、仅为处理后图像等。

---

## 〇、总览表（12 候选 + 补充）

| # | 数据集 | 模态 | 独立单元 | 样本(记录) | 原始波形 | license | 下载 | 分级 |
|---|---|---|---|---|---|---|---|---|
| 1 | PENELOPE PAUT（补充） | 超声 PAUT | 5 coupon | 3,000 位置 | ✅ .nde | CC-BY-4.0 | ✅ 已本地 | **A**（小规模，勿作唯一基准） |
| 2 | EddyCus-HDF5 | 涡流 ECT | 148 **推断配置组**（非显式试件） | 738 扫描 | ✅ HDF5 | CC-BY-4.0 | ✅ 已本地 | **B/C pending admission**（无显式试件 ID） |
| 3 | ML-NDT | 超声 B-scan（minibatch 容器） | 1 试件 / **3 真实裂纹模板** | 20,010 B-scan | ✅ .bins | LGPL-3.0 | ✅ 已本地 | **D (quarantined)** |
| 4 | NDT_ML_Flaw | 超声 B-scan | 1 试件 / **6 真实缺陷 + 10 CIVA 模板** | 17,000 条带 | ✅ .xz | LGPL-3.0 | ✅ 已本地 | **D (quarantined)** |
| 5 | Long-term GW SHM（Sci. Data 2025） | 超声导波 | **1 铝板** / 13 损伤 | ~640 万测量 | ✅ 1 MHz pickle | CC BY-NC-ND | ⚠ Figshare 需人工验证 | **B/C**（单结构，非 A） |
| 6 | USimgAIST | 超声成像（图像） | 未知 | 7,000+ 图 | ❌ 图像 | 未知 | ⚠ 无法确认 | **D**（先确认下载） |
| 7 | Open Guided Waves | 超声导波 | 多板/长桁 | 多组 h5 波场 | ✅ HDF5 | #4 CC-BY-4.0 | ✅ 免登录 | **C** |
| 8 | UGW-3Mat-2SN（补充） | 超声导波 | 3 板×2 阵列 | 43.5 GB | ✅ .mat | CC-BY-4.0 | ✅ 免登录 | **C** |
| 9 | Pipeline UGW（DiB 2022） | 超声导波 | 1 钢管 | 236 信号 | ✅ 195 kHz | CC BY 4.0 | ✅ Mendeley | **C** |
| 10 | MDDECT | 涡流 ECT | 18 缺陷 | 48,000 扫描 | ✅（1D/2D 待核实） | 未知 | ⚠ Kaggle 登录 | **C**（分组待调查） |
| 11 | NASA SiC/SiC AE（补充） | 声发射 | 3 试件 | ~11 MB | ✅ 波形 | Unlicense | ✅ GitHub | **B/C** |
| 12 | ORION-AE（补充） | 声发射 | 螺栓结构 | 5 MHz | ✅ .mat | 待核实 | ✅ Harvard Dataverse | **C** |
| 13 | Composite AE（广义） | 声发射 | 视子集 | 特征/波形 | 混合 | 混合 | 混合 | **C/D** |
| 14 | Stanford AE waveforms | 声发射 | 无法确认 | — | — | — | ❌ | **D** |
| 15 | LANL SHM benchmark | 振动加速度 | 1 结构×2 | MB 级 | ✅ | 未核实 | ❌ 页面失效 | **D** |
| 16 | Wind-turbine-blade SHM | 振动加速度 | 缩比风机 | — | ✅ 102.4 kHz | CC BY 4.0 | ✅ Mendeley | **C/D** |
| 17 | Evident NDE examples（补充） | 超声 .nde | 示例 | 13 文件 | ✅ | 无声明 | ✅ 免登录 | **C**（工具） |

> **quarantined 红线**：ML-NDT / NDT_ML_Flaw 因 shortcut 高风险被隔离（见 §"Quarantined
> datasets and shortcut-learning evidence"），**不得作为核心 benchmark / 论文主结果 / 跨试件
> 泛化 claim 的证据**。
>
> 模态红线：**LANL SHM 与风机叶片数据为振动/加速度模态**，严格说属于结构动力学而非
> 超声/涡流/导波/声发射这类"波传播型 NDT 信号"。为避免混模态污染第一阶段预训练，均按 D 处理
> （LANL 同时因官方下载页失效）。若后续需要振动域外部迁移，可单独降级讨论为 C。

---

## 一、超声 / PAUT

### 1. PENELOPE / Submerged Arc Welding Open Repository → **A（核心 benchmark）**

- 来源：Zenodo 15083865，DOI 10.5281/zenodo.15083865；EU PENELOPE 项目（grant 958303）
- 内容：SAW 埋弧焊 X 型接头，7 coupon（Coupon1/2 仅工艺，PP3–PP7 完整 PAUT）
- 格式：PAUT `.nde`（HDF5，Evident/OmniScan X3）；G0=71°=49 波束/3500 采样 int16，
  G1=47°=22 波束/4500 采样；90/270 族；11 个 .nde
- 样本：3,000 位置级 B-scan（1 mm 分辨率）；标签 243 行（无深度/尺寸）
- license：CC-BY-4.0；下载免登录，已本地（12.7 GB zip）
- 泄漏：按 coupon 严格划分（Protocol V2）；PP4 近零缺陷已知
- 定位：**下游严格评测基准之一**；5 试件缺陷率耦合（0.5%–76%）是评估时的已知约束

### 2. ML-NDT → **D（quarantined / shortcut 高风险）**

- 来源：https://github.com/iikka-v/ML-NDT；arXiv:1903.11399（Virkkunen et al. 2019）
- 内容：316L 奥氏体管道单焊头；**仅 3 条真实热疲劳裂纹**（1.6/4.0/8.6 mm，Trueflaw）；
  其余 20,010 张图全部由 **eFlaw 流程生成**（将提取的裂纹信号**植入**到扫描数据的
  不同位置/背景），作者公开声明（"The flaw signals extracted can be moved to different
  samples"）
- 格式：`.bins` UInt16 256×256×100 = **minibatch 容器（100 张 B-scan 图，非三维体积采集）**
- 样本：20,010 B-scan（12,128 缺陷 / 7,882 干净）；有效独立单元 ≈ **3 条真实裂纹模板**
- 设备：Zetec Dynaray 64/64PR-Lite + Imasonic 1.5 MHz 矩阵探头；TRS 相控阵单 45°
- 标签：0/1 + equivalent_flaw_size；location（帧范围）
- license：LGPL-3.0（对数据授权语义模糊）
- **shortcut 证据（docs/M0_2B_VTT_virtual_flaw_data_audit_v2.md, 2026-08-20）**：
  - 随机样本级小 CNN 缺陷检测 **AUC≈1.0**（近饱和）；
  - test 样本 **99.3–99.7%** 可在 train 找到 **cos>0.99 同模板近重复**（最近邻 100% 同模板）
    → 泄漏单元是**缺陷模板**；
  - **leave-template-out**：8.6mm AUC=1.0 / **1.6mm AUC=0.41（低于机会）** / 4.0mm AUC=0.76
    → 对新尺寸模板（尤其小裂纹）**不能泛化**；
  - background-only（探索性）AUC≈0.9887 → 存在背景/植入残留捷径的探索性证据；
  - E2 外部预训练对 PAUT 负迁移（E2−E0 = −0.0075，判据未过）。
- 定位：**D（quarantined）**——仅限受控用途（smoke/合成缺陷消融/shortcut-leakage 负对照/
  模板依赖验证），**不得用于论文主结果、跨试件 claim、与真实工业数据直接比较**

### 3. NDT_ML_Flaw → **D（quarantined / shortcut 高风险）**

- 来源：https://github.com/koomas/NDT_ML_Flaw（VTT，Tuomas Koskinen——与 ML-NDT 同源）
- 内容：异种金属焊缝 P41；**6 个真实缺陷**（5 裂纹 + 1 EDM notch）+ **10 批 CIVA 仿真模板**；
  真实批含 0.4–1.0 幅度缩放增强
- 格式：`.xz`/`.lzma` uint16 条带（480 深度 × 7168 扫描）；17 批 × 1000 = 17,000 条带
- 标签：7 列（Flaw 0/1, 增强量, 深度, 位置, 尺寸, 索引, 类型）
- license：LGPL-3.0（授权语义模糊，建议使用前与作者确认）
- **shortcut 证据（同上审计）**：
  - 随机样本级小 CNN 缺陷检测 **AUC≈1.0**（近饱和）；
  - **leave-one-real-defect-out 修复后 AUC 仍=1.0**，但 acc 跌到 ~0.17–0.21（阈值不校准）——
    这是**同试件内（P41）缺陷间泛化，不是跨试件/新焊缝泛化**；
  - leave-container / leave-batch 后 AUC 仍≈1.0；
  - 有效独立单元 ≈ 6 真实缺陷 + 10 CIVA 模板，远小于 nominal 17,000。
- 定位：**D（quarantined）**——与 ML-NDT 同为 VTT 单试件 + 虚拟/仿真缺陷语料，均保留
  **high shortcut risk**；仅限受控用途（同上）
- ⚠ 注入机制表述：ML-NDT 为 **eFlaw 植入（提取真实裂纹信号重植入）**，NDT_ML_Flaw 为
  **CIVA 仿真模板 + 幅度缩放增强**——两者**生成机制不同**（未证实完全相同），但都满足
  "重复缺陷模板 / 虚拟缺陷 / 样本相关性" → 均判 high shortcut risk。

### 4. USimgAIST → **D（暂不使用；先人工核实下载）**

- 来源：Ye, J. & Toyama, N., IEEE Access 9:36986–36994 (2021)，DOI 10.1109/ACCESS.2021.3062860
  （AIST 産総研）
- 内容：>7,000 张已标注超声检测**图像**（处理后图像，非原始波形）
- 下载：**无法确认**任何公开托管（OpenAlex 全文检索仅命中该组论文；疑似按需索取）
- 定位：处理后图像 + 下载不可确认 + 无 license → **D**。若作者回邮件提供原始 A-scan，可重评。

### 5. Evident NDE Open File Format examples（补充，工具性）→ **C**

- 来源：https://ndeformat.com/4.0/examples/example-files/（S3 公开直链）
- 内容：13 个 `.nde` 示例（PAUT/TOFD/TFM/FMC，0.4–274 MB）；**无缺陷标签**
- license：格式库 MIT；示例数据无独立 license（商用前联系 Evident）
- 定位：**解析器兼容/无标签工程测试**；非训练数据

---

## 二、导波（Guided Waves）

### 5. Long-term Guided Wave SHM dataset（Sci. Data 2025）→ **B/C（单结构，非 A）**

- 来源：Yang, K. et al., *Dataset on guided waves from long-term SHM under uncontrolled and dynamic
  conditions*, **Scientific Data 12 (2025)**, DOI 10.1038/s41597-025-05300-5（OA，PMC12162875）
- 托管：**Figshare** DOI 10.6084/m9.figshare.28112504；配套代码
  https://github.com/SmartDATA-Lab/Long_Term_Guided_Waves
- 内容：**单块铝板**（犹他大学户外货架）2018-03 至 2022-10（~4.5 年）长期监测；
  **13 种递进损伤**（D1–D13：小凹痕→通孔）
- 采集：8 个 PZT（SM412，300 kHz 径向谐振）；1 ms 线性扫频 5–350 kHz；8 条路径；
  **采样率 1 MHz**；原始时域波形 `.pickle`（8 通道 × 2000 点/次）
- 数据量：~640 万条导波测量（约数十 GB，估算待核实）
- 标签：损伤引入时间线 + 环境信道（温度/湿度/光照/气压/天气）
- license：**CC BY-NC-ND 4.0**（非商业、禁改——合规需评估）
- 泄漏：**高（时间相关）**——单板跨 4.5 年连续采样，必须按时间分段划分
- 定位：**B/C**——"原始波形 + 多损伤 + 长期多环境 + 正式发表"的导波代表；
  **但只有 1 块铝板（单结构）→ 不满足 A 级核心基准准入（多独立 specimen）**，
  仅作无标签预训练语料（B）+ 单结构损伤严重度/环境迁移验证（C）；
  NC-ND 许可与单结构是硬约束

### 6. Open Guided Waves → **C（迁移验证）**

- 来源：https://openguidedwaves.de/downloads/
- 内容：4 数据集（Moll 2019 基础导波 / 变温 CFRP / omega 长桁 + 参考损伤 / Kudela 2022
  CFRP+长桁脱粘全波场）
- 格式：HDF5 `.h5` + MATLAB/Python 脚本；#4 CC-BY-4.0（Zenodo 10.5281/zenodo.5105861）；
  #1–3 license 待核实；各波场 zip ~6 GB
- 定位：**C**（复合材料板导波，SHM 域；低层表征/域适配对照）

### 7. UGW-3Mat-2SN（补充）→ **C**

- 来源：Zenodo 10.5281/zenodo.15688321；Gonzalez-Jimenez et al., *SHM* 2023，
  doi:10.1177/14759217231189972
- 内容：3 种复合板 × 2 PZT 阵列，Lamb 波；Before/After MPCA（域适配处理，跨材料/跨网络）
- 数据量：`Database.zip` **43.5 GB**（MATLAB `.mat`，单文件直链）
- license：CC-BY-4.0
- 定位：**C**（复合材料板导波对照；>10 GB 需用户批准下载）

### 8. Pipeline Ultrasonic Guided Waves dataset → **C（外部迁移验证）**

- 来源：El Mountassir, M. & Yaacoubi, S., *Data in Brief* 42 (2022) 108756，
  DOI 10.1016/j.dib.2022.108756（OA，PMC9747640）
- 托管：**Mendeley Data** DOI 10.17632/ttb63krg6d.1（已验证在线，**CC BY 4.0**）
- 内容：**单根钢管**（6.4 m，152.4 mm 外径）；Wavemaker G4 系统；激励 14/18/24/30/37 kHz；
  **采样率 195 kHz**；原始 A 扫（扭转+弯曲模态），信号矩阵 2057×4
- 样本：236 条 UGW 信号（健康 207 + 损伤 29）；缺陷=内壁腐蚀磨削（6 级递增）
- 泄漏：低-中（单管时间序列；同损伤状态多次采集需注意）
- 定位：**C**（真实原始管道导波、样本小、CC BY 4.0 易下载；外部迁移验证种子）

---

## 三、涡流（ECT）

### 9. EddyCus-HDF5 / Open-Source Multi-Sensor ECT Database → **B/C（pending admission，Phase 2A 修正）**

- 来源：Zenodo 19251759，DOI 10.5281/zenodo.19251759（TU Dresden + Fraunhofer IKTS，2026-03 v1.0）
- 内容：CFRP 碳纤维（3 种无屈曲织物）；**8 个 Fraunhofer IKTS 传感器**（3 绝对 + 5 差分半透射，
  6.1–24.3 MHz）
- 格式：HDF5 四层组（measurement_metadata / spatial_data / signal_data/fN/（real/imaginary/
  complex_impedance）/ analysis_results/fN/（magnitude/phase））；gzip L6+shuffle
- 样本：738 扫描（695 有信号）；**8 类缺陷**；**148 = 推断配置组**（material×fiber×layup×
  description×defect×thickness 的 SHA1 哈希），**非显式物理试件**
- **⚠ 准入修正（2026-09-02）**：数据集无显式 specimen ID；`specimen_id_available: false`、
  `inferred_group_available: true`、`inferred_group_contains_label: true`；
  `benchmark_tier: B/C pending admission`、`core_benchmark: false`、
  `headline_results_allowed: false`、`cross_specimen_claim_allowed: false`。
- license：数据 **CC BY 4.0**；转换软件 MIT
- 泄漏：无显式试件 → 无试件级分组；组内多传感器/频率相关（切片级中）
- 定位：**B/C pending admission**——可作**无标签预训练数据** + **cross-sensor/cross-material
  探索实验（结果标 exploratory）**；**不得声称 cross-specimen 泛化**。只有原始文件/官方论文/
  作者元数据证明配置组 == 独立物理试件，才可重新申请 A 级（见
  `phase2_eddycus_admission.md`）。

### 10. MDDECT → **C（先核实 license 与分组）**

- 来源：*Depth Evaluation for Metal Surface Defects by ECT using DRNN*, arXiv:2104.02472
- 内容：304 不锈钢薄板机加工表面缺陷；**18 深度档**（0.3–2.0 mm，步长 0.1 mm）；
  **48,000 次扫描 = 扫描次数，非独立缺陷数**；多人人工扫描；lift-off 变化
- 任务：缺陷深度分类（1-D；论文 ResNeXt-38 93.58%）
- license：**未知**；下载需 Kaggle 登录
- 泄漏：必须按 defect×operator 组合划分，**禁止随机扫描级划分**
- 定位：**C**（真实金属 ECT、语义接近焊缝 ECT；但平面试件 vs 焊缝几何差异大，license 不明）

---

## 四、声发射（Acoustic Emission）

### 11. NASA SiC/SiC Composite AE（补充）→ **B/C**

- 来源：Muir et al., npj Comput. Mater. 7:146 (2021)，DOI 10.1038/s41524-021-00620-7（OA）
- 托管：https://github.com/Muir-UCSB/AE-ML_Framework（Unlicense，~11 MB；含 `*_data.json` +
  `Waveforms_and_filters.zip`；**3 试件**）
- 内容：SiC/SiC 陶瓷基复合 AE 波形；任务=损伤机制识别（matrix/fiber/interface 无监督聚类）
- 定位：**B/C**——少数"NASA + AE 原始波形 + 多试件"的公开数据；材料非焊缝

### 12. ORION-AE（螺栓松动 AE，补充）→ **C**

- 来源：Harvard Dataverse DOI 10.7910/DVN/FBRDU0；代码
  https://github.com/emmanuelramasso/ORION_AE_acoustic_emission_multisensor_datasets_bolts_loosening
- 内容：3 AE 传感器（micro80/F50A/micro200HF）+ 激光测振仪；**5 MHz 采样**；`.mat`；
  7 级拧紧扭矩（5–60 cNm）
- 定位：**C**（多传感器 AE 原始波形，航空航天级螺栓连接；非 CFRP/焊缝）

### 13. Composite acoustic-emission datasets（广义）→ **C/D**

- "ASEA" 未能在任何来源确认存在 → 未知
- Kaggle "Acoustic Emission Dataset"（ziya07, CC0）：**特征级**（峰幅等）非原始波形 → **D**
- 定位：**C/D**（视具体子集；特征级数据集不参与预训练）

### 14. Stanford acoustic-emission waveforms → **D（无法确认）**

- 未能确认存在名为 "Stanford AE waveforms" 的独立公开数据集；NASA 侧实得为 11（SiC/SiC），
  PHM 挑战 2021/2023 均非 NDT-AE
- 定位：**D**；若用户有具体论文/链接可据以重新定位

---

## 五、振动 / SHM（模态红线区）

### 15. LANL SHM benchmark → **D（暂不使用）**

- 来源：Farrar et al. 2007（8-DOF 质量-弹簧，OSTI 10.2172/922532）；Figueiredo et al. 2009
  （三层建筑，LA-14393，OSTI 10.2172/961604）
- 模态：**振动加速度**（非超声/涡流等波传播型 NDT 信号）
- 下载：**官方页面失效**（institute.lanl.gov/ei/software-and-data 不可达，lanl.gov 404）；镜像未验证
- 定位：**D**——模态错位 + 下载失效。若未来需振动域外部迁移，先找有效镜像再降级 C

### 16. Wind-turbine-blade SHM datasets → **C/D**

- 真实信号候选：Ogaili et al., *Data in Brief* 48 (2023) 109414，DOI 10.1016/j.dib.2023.109414
  （OA）；Mendeley DOI 10.17632/5d7vbdp8f7（**CC BY 4.0**）：Edibon EEEC **实验室缩比风机**，
  PCB 352C65 单轴加速度计，**102.4 kHz**，故障=表面侵蚀/裂纹/质量不平衡/叶片扭转 + 健康
  → **C**（真实原始振动信号、多类故障，但缩比风机 + 振动模态）
- Kaggle 模拟 AE（ziya07）：模拟 + 特征级 → **D**
- Chalmers 45 kW 风机（870 MB）：**运行/SCADA 型数据，非损伤检测信号 → 排除（不混入）**
- 定位：**C/D**（按模态红线，非第一阶段核心）

---

## 六、Quarantined datasets and shortcut-learning evidence

> 本节依据仓库既有审计证据，**不重复训练**。核心来源：
> `docs/M0_2B_VTT_virtual_flaw_data_audit.md` 与 `docs/M0_2B_VTT_virtual_flaw_data_audit_v2.md`
> （2026-08-20，det_v2 实验 + 审计脚本 `scripts/m0_2b_vtt_data_audit.py` 实测）；
> 迁移结论交叉验证于 `docs/M0_2B_external_ultrasound_transfer_report.md`。

### 6.1 缺陷生成 / 注入方式（区分表述，不宣称两者机制完全相同）

| 项 | ML-NDT | NDT_ML_Flaw |
|---|---|---|
| 物理试件 | 1 个（316L 奥氏体管道单焊头） | 1 个（P41 异种金属焊缝） |
| 真实缺陷 | **3 条热疲劳裂纹**（1.6/4.0/8.6 mm） | **6 个**（5 裂纹 + 1 EDM notch） |
| 生成/注入方式 | **eFlaw 流程**：提取真实裂纹信号，**植入**到扫描数据不同位置/背景（作者公开声明） | **CIVA 仿真模板**（10 批）+ 真实批 **0.4–1.0 幅度缩放**增强 |
| nominal 样本 | 20,010 张 B-scan（minibatch 容器，非体积采集） | 17,000 条带 |
| 有效独立单元 | ≈ **3 条真实裂纹模板** | ≈ **6 真实缺陷 + 10 CIVA 模板** |
| 共享缺陷模板 | **是**（3 模板驱动全部 12,128 张缺陷图） | **是**（同缺陷沿扫描轴连续条带 + 模板复用） |

> ⚠ **表述纪律**：ML-NDT（eFlaw 植入）与 NDT_ML_Flaw（CIVA 仿真 + 缩放）的**生成机制并不
> 完全相同**，现有证据不支持"两数据集完全同一种注入机制"的结论。但两者都满足
> **重复缺陷模板 + 虚拟/仿真缺陷 + 样本高度相关** → 均保留 **high shortcut risk** 标记。

### 6.2 已有 shortcut 证据（小 CNN 缺陷检测，审计实测）

| 证据 | ML-NDT | NDT_ML_Flaw |
|---|---|---|
| 随机样本级检测 | **AUC≈1.0**（近饱和） | **AUC≈1.0**（近饱和） |
| 近重复（template 泄漏） | test 样本 **99.3–99.7%** 在 train 有 cos>0.99 同模板近重复；最近邻 100% 同模板 | 同缺陷沿扫描轴连续条带 |
| leave-template-out | 8.6mm AUC=1.0 / **1.6mm AUC=0.41（低于机会）** / 4.0mm AUC=0.76 → 新尺寸小裂纹不能泛化 | — |
| leave-real-defect-out | — | **AUC 仍=1.0** 但 acc≈0.17–0.21（同试件内泛化，非跨试件） |
| leave-container/batch | — | AUC 仍≈1.0 |
| background-only（探索性） | **AUC≈0.9887**（背景/植入残留捷径的探索性证据） | 因标签相关裁剪已删除 |
| metadata-only（批次指纹） | ~0.58（弱，非主导捷径） | ~0.49（弱） |

### 6.3 为什么随机切分结果不能代表跨试件泛化

- 泄漏单元是**缺陷模板**（近重复**跨容器**但**同模板** 100%）而非采集批次；
- 随机样本级 AUC≈1.0 主要是模型匹配 train 中近重复模板的缺陷形态，不是学习"新"缺陷泛化
  规律（leave-template-out 崩塌即为直接证据）；
- 同试件内缺陷间泛化（NDT_ML_Flaw leave-one-real-defect-out 仍=1.0）**不等于**跨试件/新焊缝
  泛化。

### 6.4 为什么不能与真实工业 NDT 数据直接比较

- 两数据集均**单试件** + 虚拟/仿真缺陷为主，不存在独立多试件结构；
- 背景/采集纹理（VTT 超声系统 + 单试件）是共享捷径，会污染与真实工业数据（不同试件/设备/
  环境）的比较；
- 预训练（E2）实际看到的是"单试件虚拟缺陷语料"（ML-NDT 60.4% 帧为 3 裂纹副本；NDT_ML_Flaw
  ~2/3 窗口只是背景），且对 PAUT 为**负迁移**（E2−E0 = −0.0075，判据未过）——不能作为
  "通用真实缺陷物理表征"的证据。

### 6.5 仍可用的受控用途（不删除数据/loader/历史结果）

1. dataset loader 与训练管线 **smoke test**（格式兼容 / adapter 验证）；
2. **合成缺陷预训练的受控消融**（virtual-flaw 方法学本身）；
3. **shortcut learning / leakage detection 负对照**；
4. 比较 **random split vs grouped split 的虚高差异**；
5. 验证模型是否**依赖注入模板**（template-dependence 探针）；
6. 研究**合成数据如何导致负迁移**。

**禁止用途**：论文主结果 / 通用 NDT 表征有效性的主要证据 / 跨试件泛化 claim /
与工业真实数据直接比较 / SOTA claim。

---

## 七、分级汇总（v2，quarantined 后重新评估）

> 准入原则：**没有足够证据放入 A 就放 B/C**，不为凑数量降低标准（详见
> phase1_experiment_protocol.md §"Core Benchmark Admission Criteria"）。

### A 核心严格 benchmark（唯一能支撑严格下游评测的集合）
1. **PENELOPE PAUT**（超声，5 coupon，CC-BY-4.0，已本地）——coupon LOOCV；
   ⚠ **仅 5 coupon，作为严格外部/探索验证基准，不宜作为唯一核心基准**。
2. ~~**EddyCus-HDF5**~~ → **已降级为 B/C pending admission（Phase 2A，2026-09-02）**：
   148 为推断配置组（非显式物理试件），`specimen_id_available: false`、
   `core_benchmark: false`、`headline_results_allowed: false`；仅无标签预训练 +
   cross-config/cross-sensor 探索（exploratory）。只有证明配置组 == 独立物理试件才能
   重新申请 A 级。

### B 仅用于无标签预训练（非 quarantined）
3. **Long-term GW SHM**（导波，1 板/13 损伤，CC BY-NC-ND）——**单结构 → 不作 A**，仅预训练 +
   单结构迁移验证（B/C）
4. **NASA SiC/SiC AE**（AE 波形，3 试件，Unlicense）
5. **合成超声**（synth_ut_50x2k，10 万，本地）
6. **external_weld_ut**（真实 FMC，4 试件，Strathclyde **CC BY 4.0**（Phase 2A 确认））——
   无下游标签 → 仅无标签预训练（候选 B，exploratory）

### C 仅用于外部迁移/泛化验证
7. **Open Guided Waves**（导波，按结构/传感器分组）
8. **UGW-3Mat-2SN**（导波，3 板×2 阵列，43.5 GB，需批准）
9. **Pipeline UGW**（管道导波，CC BY 4.0，易下载，单管小样本）
10. **MDDECT**（涡流，**先调查物理缺陷/试件/扫描批次/采集条件分组**，license 待核实）
11. **ORION-AE**（AE 螺栓，多传感器）
12. **Evident NDE examples**（工具/格式测试，无标签）

### D quarantined / shortcut-risk
13. **ML-NDT**（见 §6）
14. **NDT_ML_Flaw**（见 §6）

### D（hold）暂不使用（原因不变）
15. **USimgAIST**（图像 + 下载不可确认）｜16. **LANL SHM**（振动 + 下载失效）｜
17. **Wind-turbine SHM**（振动；用户确认后可作 C）｜18. **Stanford AE**（无法确认存在）｜
19. **EddyNet**（无 license、纯仿真、停更）｜20. **Chalmers 风机**（SCADA，非 NDT 信号）｜
21. **任意表面 RGB 缺陷检测数据**（本阶段明确排除）

---

## 八、待人工核实清单

1. **EddyCus-HDF5（B/C → A 级恢复所需）**：已确认 148 为**推断配置组**（无显式试件 ID，
   Phase 2A 审计）；仅当原始文件/官方论文/作者元数据证明配置组 == 独立物理试件，才可
   重新申请 A 级（当前保持 B/C pending admission，cross-config 结果标 exploratory）。
2. **MDDECT（C→可能的 A）**：Kaggle 目录结构 / license / **能否按物理缺陷、试件、扫描批次或
   采集条件分组** / operator 与 lift-off（需登录；若可分独立单元可升级评估）。
3. **Long-term GW SHM**：Figshare 实际文件列表/大小；NC-ND 许可使用合规性；时间分段划分元数据
   （单结构 → 维持 B/C，不作 A）。
4. **USimgAIST**：联系 Ye/Toyama（或查 IEEE Access 正文 Data Availability）确认下载与许可
   （若取得，须按独立试板划分，禁止随机按图像划分）。
5. **LANL SHM**：查找有效镜像（如确需振动域）。
6. **external_weld_ut**（本地）：来源已确认为 **Strathclyde Pure Portal / CC BY 4.0**
   （Phase 2A）；仍待补配套元数据（.ods/.xlsx）与逐位置标签（无标签 → 仅预训练）。
7. **ML-NDT / NDT_ML_Flaw（quarantined 用途前置）**：LGPL-3.0 对"数据"的授权边界
   （建议使用前与作者确认，即使只做受控消融）。

## 九、结论（v2）

1. **A 级核心严格基准**：**PENELOPE（超声, 5 coupon, 需联合多基准）**（Phase 2A 修正：
   EddyCus 已降为 B/C pending admission，因其无显式物理试件 ID）——**不用作唯一基准**；
   其余数据集在准入证据不足时一律放 B/C。
2. **ML-NDT / NDT_ML_Flaw 已 quarantine**：shortcut 证据确凿（随机 AUC≈1.0、近重复
   99.3–99.7%、leave-template 崩塌、E2 负迁移）→ 仅限受控用途，禁止主结果/跨试件/工业对比。
3. **B 级预训练语料**：Long-term GW SHM（单结构）+ NASA AE + 合成超声 +（待补许可的）
   external_weld_ut。
4. **C 级迁移验证**：Open GW / UGW-3Mat-2SN / Pipeline UGW / MDDECT（分组待调查）/ ORION-AE。
5. **模态红线**：振动/SCADA/表面 RGB 数据不混入第一阶段；LANL 与风机叶片按 D 处理。
6. **下载纪律**：MDDECT（Kaggle）与 Long-term GW SHM（Figshare）需用户批准/人工下载；
   其余大文件（UGW 43.5 GB）也需批准。**本阶段不立即下载，registry 先落 YAML。**
