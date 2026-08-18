# M0-1：公开 NDT 数据集审计报告

> 阶段：M0-1 公开数据集审计与统一数据架构设计
> 审计日期：2026-08-18
> 审计方法：Zenodo API / GitHub API / arXiv 全文 / 代表性文件实际下载解压交叉核实；
> 未能在线确认的项目明确标注"待下载核实"，不臆造。
>
> **审计纪律（贯穿全文）**：严格区分 **样本数（记录数）/ 独立缺陷数 / 独立试件数 /
> 独立操作者-设备数**。一个缺陷重复扫描数千次 ≠ 数千个独立缺陷。

---

## 〇、总览表

| 数据集 | 模态 | 材料/对象 | 独立试件 | 独立缺陷 | 样本(记录)数 | license | 压缩大小 | 下载 | 优先级 |
|---|---|---|---|---|---|---|---|---|---|
| PENELOPE / SAW Open Repository | 超声 PAUT | SAW 埋弧焊 | 7（5 含完整 PAUT） | 243 标注（PP3=68/PP4=1/PP5=50/PP6=112/PP7=12） | 3000 位置级 | CC-BY-4.0 | 12.7 GB | ✅ 免登录 | **P0** |
| NDT_ML_Flaw | 超声 B-scan | 异种金属焊缝 | 1（P41） | 6（5 裂纹+1 EDM notch） | ~17,000 条带 | LGPL-3.0 | 236 MB | ✅ 免登录 | **P1**（预训练素材） |
| ML-NDT | 超声 PAUT 体积 | 316L 奥氏体管道焊缝 | 1 | 3（真实热疲劳裂纹） | 201 体积 / 20,100 帧 | LGPL-3.0 | 174 MB | ✅ 免登录 | **P1**（预训练素材） |
| Evident NDE examples | 超声 .nde | 多类（PAUT/TFM/TOFD/FMC） | 多（示例） | 无可靠标签 | 13 个示例文件（0.4–274 MB） | 格式 MIT / 示例数据无声明 | ~1 GB | ✅ 免登录直链 | **P2**（格式兼容/无标签测试） |
| UGW-3Mat-2SN | 超声导波 | 3 种复合板（Lamb 波） | 待核实 | 待核实（pseudo-damage） | 43.5 GB 压缩 | CC-BY-4.0 | 43.5 GB | ✅ 免登录 | **P2**（低层表征/域适配对照） |
| Open Guided Waves | 超声导波 | CFRP 板/omega 长桁 | 多 | 人工缺陷/脱粘 | 多组 h5 波场 | #4 CC-BY-4.0 / #1–3 待核实 | ~6 GB/波场 | ✅ 免登录 | **P2**（对照） |
| MDDECT | 涡流 ECT | 304 不锈钢薄板 | 待核实 | 18（=18 深度档 0.3–2.0 mm） | 48,000 扫描 | **待核实** | 待核实 | Kaggle 需登录 | **P1**（真实金属 ECT） |
| EddyCus-HDF5 | 涡流 ECT | CFRP 碳纤维 | 待下载核实 | 8 类缺陷 | 738 扫描 | CC-BY-4.0 | 3.7 GB | ✅ 免登录 | **P1**（数据管线范式） |
| EddyNet | 涡流（逆问题） | 仿真图片 | — | — | — | 无 license | <5 MB | ✅ clone | **Reject** |
| UT+ECT 融合（WAAM, NDT&E 2026） | 超声+涡流 | WAAM 增材 | 待核实 | 人工缺陷+钨夹杂 | 待核实 | 论文 CC-BY-4.0 / **数据未见公开** | 待核实 | **仅论文公开，数据需联系作者** | **Reject（当前）/ 联系作者** |

---

## 一、超声数据集（必须核查）

### 1. PENELOPE / Submerged Arc Welding Open Repository（已接入，目标域预研）

| 项 | 事实 | 状态 |
|---|---|---|
| 官方名称 | Submerged Arc Welding Open Repository | 已确认 |
| 来源 | Zenodo record 15083865；DOI 10.5281/zenodo.15083865 | 已确认 |
| 描述 | PENELOPE 项目（EU grant 958303）SAW 埋弧焊试件：工艺数据 + NDT 数据 | 已确认 |
| license | **CC-BY-4.0** | 已确认 |
| 文件 | 单文件 `ZENODO_Penelope_vs2.zip`，**12,679,424,288 字节 ≈ 12.7 GB**（>10 GB），MD5 已校验一致 | 已确认 |
| 下载 | Zenodo 直接下载，**免登录** | 已确认 |
| 内部结构 | 7 个试件目录：Coupon1（仅工艺）、Coupon2（仅工艺+1 张 RX）、**PP3–PP7（完整 NDT）** | 已确认 |
| 格式 | `2. ndt_data/`：PAUT `.nde`（HDF5 容器，Evident/OmniScan X3）+ `defects_xlocation.xlsx` + RX/UT pdf | 已确认 |
| 信号维度 | .nde DataGroup：**G0=71°/49 波束/3500 采样 int16**；**G1=47°/22 波束/4500 采样**；90 族 + 270 族（探头两侧扫查），PP4 额外 90+；共 **11 个 .nde** | 已确认 |
| 采集设备 | Evident/Olympus OmniScan X3 相控阵（NDE-FileFormat-Schema-3.1.0）；UCoordinate 分辨率 1 mm/位置 | 已确认 |
| 材料/试件 | SAW 埋弧焊 X 型接头试件；AIMEN 制造 | 已确认 |
| 独立 specimen | **7**（Coupon1/2 + PP3–PP7）；PAUT 完整仅 **5**（PP3–PP7） | 已确认 |
| 独立 defect | 标注行 PP3=68 / PP4=1 / PP5=50 / PP6=112 / PP7=12（=243 行；PP4 已被官方 UT 报告证实近零缺陷） | 已确认 |
| nominal sample | 仓库当前用 3000 位置级样本（PP3=601/PP4=601/PP5=596/PP6=601/PP7=601） | 已确认 |
| 标签层级 | 位置级（仅轴向 x，无深度 z、无波束级标注）；类型码 1–6（气孔/未熔合/夹渣/金属夹杂/Projections/裂纹）；贯穿型大裂纹作背景 | 已确认 |
| 位置/深度/尺寸 | 有轴向位置与类型；**无深度、无尺寸** | 已确认 |
| 真实/人工/仿真 | **真实**制造缺陷（含 1 个近零缺陷试件） | 已确认 |
| 域变量 | 无 operator；有 90/270 族、G0/G1 视角、试件身份 | 已确认 |
| 推荐分组单位 | **coupon（试件）**；LOOCV 非PP4 逐折均值（沿用仓库主协议） | 已确认 |
| 可迁移价值 | **最高**：当前目标域 PAUT 预研数据；试件-缺陷率耦合已知（0.5%–76%） | 已确认 |
| 阻塞问题 | 5 试件缺陷率强耦合 + 标签无深度/尺寸；单试件评估方差大；zip >10 GB（已在本机） | 已确认 |

### 2. NDT_ML_Flaw（超声源域：异种金属焊缝）

| 项 | 事实 | 状态 |
|---|---|---|
| 来源 | https://github.com/koomas/NDT_ML_Flaw | 已确认 |
| 内容 | `datasets/`：17 批（7 真实 `.xz` + 10 仿真 `.lzma`）+ 17 个 `.txt` 元数据，无源码 | 已确认 |
| 格式 | **原始 B-scan 信号**（非图像）：每批 1000 条 × 480(深度)×7168(扫描轴) uint16 | 已确认（实测解压 batch_013） |
| 数据量 | 原始约 117 GB；压缩约 **236 MB**（GitHub API 211 MB） | 已确认 |
| 独立 specimen | **1**（试件 P41；README 未声明更多，待下载核实） | 部分确认 |
| 独立 defect | **6**：P41_01~05（裂纹，尺寸 2–26 mm）+ P41_06_notch（**EDM 人工缺陷**） | 已确认（实测 7 批 txt） |
| nominal sample | ~17,000 条带（真实 ~7000 + 仿真 ~10000）；每条带约 6.88 MB 原始 | 已确认 |
| 标签 | 7 列：`[Flaw 0/1, 增强量 0.4–1, 缺陷深度, 缺陷位置, 原始尺寸 mm, 索引, 缺陷类型]`；仿真批 6 列 | 已确认 |
| 位置/深度/尺寸 | **有**（深度与扫描轴数值坐标、原始尺寸） | 已确认 |
| 真实/人工/仿真 | 真实裂纹 + **EDM notch（人工）** + **CIVA 仿真**（201–210 批） | 已确认 |
| license | **LGPL-3.0**（对"数据"的授权语义模糊，建议使用前与作者确认） | 已确认 |
| 采集设备 | 仓库无设备信息；上传者 **Tuomas Koskinen（VTT）**——与 ML-NDT 论文共同作者同源；是否 PAUT 待核实 | 部分确认 |
| 可迁移价值 | **高（预训练/缺陷模板池）**：异种金属焊缝缺陷形态与 PAUT 目标匹配；原始 B-scan 可直接做 A/B-scan 级预训练（呼应 Synth-UT 方向）；同一缺陷大量重复 + 增强量标注适合缺陷模板学习 | 评估结论 |
| 阻塞问题 | **单试件 6 缺陷 → 与"5 试件天花板"同构**，不能作独立多试件基准；无设备信息；LGPL 语义模糊 | 已确认 |

### 3. ML-NDT（超声源域：PAUT 体积）

| 项 | 事实 | 状态 |
|---|---|---|
| 来源 | https://github.com/iikka-v/ML-NDT；论文 arXiv:1903.11399（Virkkunen et al., 2019） | 已确认 |
| 内容 | 每批 UUID：`.bins`(原始体积) + `.meta` + `.jsons` + `.labels` | 已确认（实测） |
| 格式 | **原始 3D 超声体积** `UInt16 256×256×100`（100 帧 × 256×256 B-scan）= 每体积 13.1 MB | 已确认（实测逐字节吻合） |
| 数据量 | 201 体积 ≈ 2.6 GB 原始；git 压缩约 **174 MB** | 已确认 |
| 独立 specimen | **1**（316L 奥氏体管道单对焊接头） | 已确认 |
| 独立 defect | **3 条真实热疲劳裂纹**（深度 1.6/4.0/8.6 mm，Trueflaw 制造）+ eFlaw 幅度缩放 virtual flaws | 已确认 |
| nominal sample | **201 体积 = 20,100 帧**（train 199 体积 + val 2 体积） | 已确认 |
| 标签 | `.labels` 每体积 100 行 `[flaw 0/1, equivalent flaw size]`；`.jsons` 含 `size`/`equivalent_flawsize`/`original_location`(帧范围)/`location`/`max_amplitude`/`factor` | 已确认（实测） |
| 位置/深度/尺寸 | **有**（帧范围坐标 + 源/等效尺寸）；无三维物理坐标 | 已确认 |
| 真实/人工/仿真 | 3 条**真实**热疲劳裂纹 + 基于真实裂纹重植入的 virtual flaws（仿真增强） | 已确认 |
| 采集设备 | **Zetec Dynaray 64/64PR-Lite** + **Imasonic 1.5 MHz 矩阵探头** + ADUX577A 楔块；**TRS 相控阵单 45°**，管道扫查架 0.21 mm 分辨率，水耦合 | 已确认（论文全文） |
| license | **LGPL-3.0**（同 NDT_ML_Flaw） | 已确认 |
| 可迁移价值 | **很高（模态完全匹配）**：可能是开源社区最接近"焊缝 PAUT 缺陷检测"的公开数据；原始体积适合体积级/帧级自监督预训练；逐帧标签+等效尺寸支持检测/回归；有高引用论文背书 | 评估结论 |
| 阻塞问题 | **单试件、3 缺陷**，跨试件泛化不可验证；200 体积大量重叠（独立信息量小，需防试件级泄露）；1.5 MHz 低频与目标频率可能 feature shift | 已确认 |

> **关键交叉发现**：NDT_ML_Flaw 与 ML-NDT **同源**（芬兰 VTT，作者 Tuomas Koskinen），
> 都是原始信号、都是单一试件、独立缺陷数都极少——都不能作独立"多试件"基准，
> 但都是**极佳的 PAUT 原始信号预训练素材**。NDT_ML_Flaw 缺陷形态最接近
> （异种金属焊缝），ML-NDT 模态最匹配（真·相控阵）。

### 4. Evident NDE Open File Format examples（格式兼容 / 无标签测试）

| 项 | 事实 | 状态 |
|---|---|---|
| 来源 | https://ndeformat.com/4.0/examples/example-files/（S3 公开直链，免登录） | 已确认 |
| 内容 | **13 个示例 `.nde` 文件**（全部可下载，HEAD 200）：Weld_Plate_UT(444KB)、Weld_Plate_PA-Sect(6.4MB)、PA-Comp 2gr(3.5MB)、4TFM(3.7MB)、PCI(1.9MB)、TOFD+PAUT(41.9MB)、DualTFM-Analysis(235.9MB)、CFRP PA/TFM/PCI-Raster(194–274MB)、Corr PAUT×2(113–166MB)、**fmc.nde(37.4MB 全矩阵)** | 已确认 |
| 格式 | .nde = **HDF5 容器 + JSON 元数据**；官方格式仓库 MIT license（2026-08-05 仍有更新）；示例数据文件本身无独立 license（商用前建议联系 Evident） | 已确认 |
| 缺陷标签 | **无**（解析器兼容/无标签工程测试用途） | 已确认 |
| ⚠ 附加发现 | 官方 README：.nde 当前**仅支持 UT 模态，ET（涡流）模态"即将支持"**——若未来要用 .nde 存 UT+ECT 融合数据需关注更新 | 已确认 |
| 用途 | **解析器兼容测试 + 无标签工程测试**；不默认具有可靠缺陷标签 | 定位明确 |
| 优先级 | **P2**（对照/工具测试，非训练数据） | — |

### 5. UGW-3Mat-2SN（超声导波，可选核查）

| 项 | 事实 | 状态 |
|---|---|---|
| 来源 | Zenodo DOI 10.5281/zenodo.15688321 | 已确认 |
| 内容 | **3 种复合材料板（K8、G16、K2G4S）**，每板 **2 个 PZT 压电传感器阵列**，Lamb 波导波；含 Before/After MPCA（域适配处理，专为迁移学习：跨材料/跨网络子目录 97%/99%/99.9% 方差保留） | 已确认 |
| 损伤 | 记录页给出 damage positions；物理损伤类型与 PZT 节点数**待下载核实**（关联论文用"伪损伤"增强 CNN 损伤诊断） | 部分确认 |
| license | **CC BY 4.0** | 已确认 |
| 数据量 | `Database.zip` **43.5 GB**（43,526,172,763 B），MATLAB `.mat`，单文件直链免登录可下 | 已确认 |
| 关联文献 | Gonzalez-Jimenez et al., *SHM* 2023, doi:10.1177/14759217231189972；Pinello et al., arXiv:2508.02726 | 已确认 |
| 定位 | **复合材料板结构导波（SHM 域）**，非焊缝 PAUT；只能作超声低层表征或域适配/迁移学习对照 | 定位明确 |
| 优先级 | **P2**（对照；43.5 GB >10 GB，若要下需用户批准） | — |

### 6. Open Guided Waves（超声导波，可选核查）

| 项 | 事实 | 状态 |
|---|---|---|
| 来源 | https://openguidedwaves.de/downloads/（公开浏览，无需注册） | 已确认 |
| 内容 | 4 个数据集：Moll 2019 基础导波（多频+人工缺陷+激光多普勒波场+数字射线照相）、Moll 2019 变温 CFRP 导波（figshare DOI 10.6084/m9.figshare.c.4488089）、Moll 2020 omega 长桁+参考损伤（含仿真脚本）、**Kudela 2022 CFRP+omega 长桁脱粘全波场（Zenodo 10.5281/zenodo.5105861，各波场 zip 约 6 GB）** | 已确认 |
| 格式/license | HDF5 `.h5` + MATLAB/Python 脚本；**#4 为 CC BY 4.0**；#1–3 license 待核实（Nextcloud/figshare） | 部分确认 |
| 下载 | 公开直链 + Nextcloud 目录浏览，**免登录** | 已确认 |
| 传感器/试件 | CFRP 板 + omega 长桁，人工缺陷/脱粘/冲击损伤/变温；PZT + 3D 激光多普勒测振仪 | 已确认 |
| 定位 | 复合材料板导波（SHM 域），非焊缝 PAUT；低层表征/域适配对照 | 定位明确 |
| 优先级 | **P2**（对照） | — |

---

## 二、涡流数据集（必须核查）

### 7. MDDECT（真实金属 ECT，深度评估）

| 项 | 事实 | 状态 |
|---|---|---|
| 论文 | *Depth Evaluation for Metal Surface Defects by ECT using DRNN*, arXiv:2104.02472 | 已确认 |
| 数据规模 | **48,000 次扫描 = 扫描次数**，不是独立缺陷数（论文原文 "48,000 scans from 18 defects"） | 已确认 |
| 独立缺陷 | **18**（= 18 深度档 0.3–2.0 mm，步长 0.1 mm，每深度一缺陷） | 已确认（论文） |
| operator | **多人人工扫描**（"constructed by human operators"）；operator 数量**待下载核实** | 部分确认 |
| lift-off | 存在变化；档位数**待下载核实** | 部分确认 |
| 试件 | 304 不锈钢薄板表面机加工缺陷（平面，非焊缝几何） | 已确认 |
| 任务 | **缺陷深度分类**（1-D 信号；论文 1-D ResNeXt-38 达 93.58%） | 已确认 |
| 采集设备 | 自研便携式 ECT（Zynq-7020 SoC FPGA 采集 + I/Q 解调）；探头型号待核实 | 部分确认 |
| 标签层级 | 深度档位分类；扫描形态 1D vs 2D **待登录核实** | 部分确认 |
| license | **未知（待核实）**——arXiv 未声明，Kaggle license 字段需登录 | 待核实 |
| 下载 | Kaggle 需登录（可免费注册，API 可下）；总量未知 | 待核实 |
| 可迁移价值 | **中偏上**（真实金属 ECT，与焊缝 ECT 语义接近）；但平面试件 vs 焊缝几何差异大 | 评估结论 |
| 阻塞问题 | license 不明、Kaggle 登录、operator/lift-off 分组结构待核实；**split 必须按 defect/operator 组合，禁止随机扫描级划分** | 待核实 |

### 8. EddyCus-HDF5 / Open-Source Multi-Sensor Eddy Current Database

| 项 | 事实 | 状态 |
|---|---|---|
| 来源 | Zenodo record 19251759；DOI 10.5281/zenodo.19251759；发布于 2026-03-27 v1.0 | 已确认 |
| 内容 | **CFRP 碳纤维**复合材料多传感器多频涡流数据（**非金属焊缝**） | 已确认 |
| 数据量 | **738 次多频扫描**；`eddy_current_data.zip` **3.7 GB**（<10 GB，可下） | 已确认 |
| 传感器 | **8 个 Fraunhofer IKTS 传感器**（3 绝对式 + 5 差分半透射；6.1–24.3 MHz） | 已确认 |
| 材料 | 3 种碳纤维无屈曲织物（HP-U300/122C、ST 50 g/m²、TUD 自产 12K 织物） | 已确认 |
| 缺陷 | 8 类：Gap 492 / 无缺陷参考 84 / 错铺层 80 / PTFE 膜 24 / 铜膜 24 / 镀铜丝束 24 / 波纹 6 / 毛球 4 | 已确认 |
| HDF5 schema | 四层组：`measurement_metadata/`（frequencies+sample_properties）、`spatial_data/`（x/y/z mm）、`signal_data/fN/`（real/imaginary/complex_impedance，复合 dtype）、`analysis_results/fN/`（magnitude/phase）；gzip L6+shuffle；单文件 50–500 KB | 已确认 |
| 独立试件数 | **待下载核实**（zip 内 README/sample_properties） | 待核实 |
| license | 数据 **CC BY 4.0**；转换软件 MIT | 已确认 |
| 下载 | **免登录**，Zenodo 直接下载 | 已确认 |
| 作者 | TU Dresden（Mersch/Schulze/Heuer/Cherif）+ Fraunhofer IKTS；**与 MDDECT 无关** | 已确认 |
| 可迁移价值 | 模态同为 ECT 但对象是 CFRP（非焊缝）；价值在**多传感器、多频、结构化 HDF5 + 明确缺陷标签**的完整数据管线范式 | 评估结论 |
| 阻塞问题 | 独立试件数待核实；CFRP 与金属焊缝物理差异大（只能做 ECT 数据处理范式参考 + cross-sensor/cross-material 协议开发） | 待核实 |

### 9. EddyNet（涡流逆问题 / 仿真审计）

| 项 | 事实 | 状态 |
|---|---|---|
| 来源 | https://github.com/askerlee/EddyNet | 已确认 |
| 性质 | ECT **逆问题**：神经网络重建裂纹轮廓（NeurIPS 2019 ML4PS）；纯代码 + 4 张图 + 生成脚本 | 已确认 |
| 数据 | 疑似程序生成/仿真；README 无声明 | 待核实（倾向纯仿真） |
| license | **无 license（null）** → 保留所有权利，不可自由使用 | 已确认 |
| 维护 | 2019-10 最后推送，6 stars / 0 forks，**已停更约 7 年** | 已确认 |
| 定位 | 印证"**纯仿真逆问题方法不能直接迁移到真实 ECT**"（sim2real 鸿沟：lift-off、探头阻抗、噪声、几何、各向异性）；可作为反面案例引用，**不提供可迁移数据** | 评估结论 |
| 优先级 | **Reject** | — |

---

## 三、超声—涡流融合（必须核查）

### 10. Automated robotic system for dual UT+ECT integration... WAAM（NDT&E 2026）

| 项 | 事实 | 状态 |
|---|---|---|
| 论文 | *Automated robotic system for dual ultrasonic and eddy current array integration and data fusion in WAAM material inspection*, NDT & E International **Vol.160, p.103665**, DOI 10.1016/j.ndteint.2026.103665；2026-05 正式出版，**全文 CC BY 4.0 开放获取** | 已确认（Crossref/OpenAlex/Elsevier/S2 三重核实） |
| 内容 | 双传感器机器人系统**同时**相控阵 UT + ECT，WAAM 沉积过程逐层检测；闭环力-力矩控制免换刀；Ti-6Al-4V 参考块（人工缺陷）+ 含钨夹杂钛 WAAM 样件；**深度加权 C-scan 数据融合**使对比度噪声比 +4.44/+9.02 dB，ROC 确认融合 AUC 优于单模态 | 已确认（摘要） |
| 是否同试件/同坐标/成对 | **是**（摘要明确 simultaneous UT+ECT、逐层同扫描网格、深度加权 C-scan 融合） | 已确认（摘要） |
| 代码公开 | **未见**（GitHub API / Crossref relation 均无仓库） | 已确认未见 |
| 数据公开 | **未见**（Zenodo/GitHub/Crossref/OpenAlex 均无数据链接）；正文 data availability statement 原文因反爬未能读取（唯一未直接核实项） | 未见公开 |
| 数据获取 | **只能联系作者**（Strathclyde 先进制造/NDT 团队，第一作者 Vedran Tunukovic，ORCID 0000-0002-3102-9098，on-request 最可能） | 已确认 |
| 对本项目 | 是"同试件同坐标成对 UT+ECT + 融合"的强参照，但**数据获取门槛高**；P7/sim2real 应视为独立实验设计，不指望复现该数据 | 评估结论 |

> **融合红线**：没有可验证的**同试件、同坐标、成对 UT+ECT 公共数据**时，
> 本阶段**不得进行真正的融合训练**。严禁把不同试件/材料/任务的 UT 与 ECT
> 强行拼接后称为"多模态融合"。未来配对融合数据须含：`shared_specimen_id`、
> `shared_coordinate_system`、`UT/ECT registration transform`、
> `modality availability mask`、`acquisition time/order`（见统一 schema）。

---

## 四、数据优先级

| 等级 | 数据集 | 理由 |
|---|---|---|
| **P0（必须接入）** | PENELOPE | 目标域 PAUT 预研；已在本机（12.7 GB，CC-BY-4.0）；统一 schema 应优先为其落地 manifest |
| **P1（建议接入）** | ML-NDT | 模态最匹配（真·相控阵管道焊缝）；原始体积 + 逐帧标签 + 等效尺寸；~174 MB 免登录；预训练素材 |
| **P1** | NDT_ML_Flaw | 异种金属焊缝缺陷形态最接近目标；原始 B-scan + 位置/深度/尺寸标签；~236 MB；与 ML-NDT 同源 VTT |
| **P1** | MDDECT | 真实金属 ECT 中与焊缝最相关；但 license/分组待核实（先登录 Kaggle 再定） |
| **P1** | EddyCus-HDF5 | 结构化完整、CC BY 4.0、3.7 GB 可下；作 ECT 数据管线范式和 cross-sensor/cross-material 协议开发；对象 CFRP |
| **P2（只做对照）** | Evident NDE examples | 解析器兼容 + 无标签工程测试，无可靠缺陷标签 |
| **P2** | UGW-3Mat-2SN / Open Guided Waves | 导波，仅超声低层表征/域适配对照，不与焊缝 PAUT 混任务 |
| **Reject** | EddyNet | 无 license、无真实数据、停更 7 年；仅作"仿真逆问题不迁移"反面案例 |
| **Reject（当前）** | 融合训练 | 无公开成对 UT+ECT 数据；推迟至取得配对数据后 |

> 说明：NDT_ML_Flaw / ML-NDT 因**单试件 + 极少独立缺陷**，只能作预训练素材与
> 单模态源域，**不能作独立多试件评估基准**（与本仓库 5 试件天花板问题同构）。

---

## 五、推荐 split（详见 docs/M0_unified_ndt_schema.md）

| 数据集 | 最小物理独立单元 | 推荐协议 |
|---|---|---|
| PENELOPE | coupon | `specimen` LOOCV（非PP4 逐折均值） |
| ML-NDT | 原始 flaw（3 裂纹）| `flaw` 分组；或 volume 分组 + 防重叠泄露 |
| NDT_ML_Flaw | 原始 flaw（6）/ 真实 vs CIVA | `flaw` + `source`（real/sim）分组 |
| MDDECT | defect / operator 组合 | `defect×operator` 划分，**禁随机扫描级** |
| EddyCus-HDF5 | specimen / material / sensor | `domain` cross-material、cross-sensor |
| 未来融合 | specimen | 配对模态必须同 split |

---

## 六、待下载核实清单（本轮未能在线确认）

1. **MDDECT**：Kaggle 目录结构、license、文件大小、operator 数量、lift-off 档位、
   扫描形态（1D vs 2D C-scan）、缺陷几何信息（需登录）。
2. **EddyCus-HDF5**：独立试件数（zip 内 README）。
3. **NDT_ML_Flaw**：试件总数（是否仅 P41）、采集设备/频率/是否 PAUT、
   "segmentation crack" 含义、CIVA 参数；LGPL 对数据授权边界。
4. **ML-NDT**：validation 目录是否即论文盲测集；三维物理坐标缺失问题。
5. **融合论文**：正文 data availability statement 原文（数据获取 on-request 的正式
   确认）；作者联系方式（Strathclyde PURE）。
6. **UGW-3Mat-2SN**：每阵列 PZT 节点数、物理损伤类型。
7. **Open Guided Waves #1–3**：Nextcloud/figshare 上的具体 license。

---

## 七、审计结论

1. **可获取（免登录）**：PENELOPE（已在本机 12.7GB）、NDT_ML_Flaw（236MB）、
   ML-NDT（174MB）、EddyCus-HDF5（3.7GB）、Evident NDE examples（13 个直链）、
   Open Guided Waves（#4 等）、UGW-3Mat-2SN（43.5GB，>10GB 需批准）。
2. **需登录/许可待定**：MDDECT（Kaggle）、Open Guided Waves #1–3（license）。
3. **Reject**：EddyNet（无 license、无真实数据、停更）。
4. **融合数据**：NDT&E 2026 WAAM 论文确认同试件同坐标成对 UT+ECT，但**数据未见
   公开、只能联系作者**（Strathclyde）；当前无公开成对数据 → **融合训练推迟**。
5. **超声源域优先**：ML-NDT（模态最匹配）与 NDT_ML_Flaw（缺陷形态最接近）是
   P1 首选下载；两者均与 PENELOPE 同为"原始信号 + 单试件"结构，统一作为
   **PAUT 原始信号预训练素材**进入 M0-2。
