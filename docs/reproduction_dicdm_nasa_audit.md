# Reproduction Audit — Pilot B: DI-CDM + NASA CFRP

> 阶段：reproducibility audit（训练开始前）。
> 论文：Yang X. et al., "Damage imaging in structural health monitoring with
> fine-tuned conditional diffusion model",
> Mechanical Systems and Signal Processing 236 (2025) 112996, DOI: 10.1016/j.ymssp.2025.112996
> 状态标记：R0 / R1 / R2 / BLOCKED；B1 Gate 判定在 §3。

## 0. 当前判定

- **论文全文：已获得**（TU Delft 机构库 green OA 出版版 PDF，26 页，
  本地 `docs/papers/Yang2025_DICDM_MSSP_112996.pdf`，gitignored）。
  Bristol 门户也有 CC-BY 版但被 Cloudflare 挡（curl 403）。
- **NASA CFRP 数据：已下载解压盘点**（§1）。
- **官方代码：不存在**。Data availability 原文承诺
  "the code could be shared via the GitHub repository after the potential
  publication of this manuscript"——检索未发现任何已发布 repo。
  仅给两个 HF 依赖：`stabilityai/stable-diffusion-2-base`、
  `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`。
- **等级判定：R2（关键 protocol 部分缺失，近似重实现）**，详见 §3 Gate。
  训练侧可高保真重建（§2 回答了绝大部分）；**不可恢复的核心是
  X-ray → 二值 GT mask 的生成流程**（配准/二值化/去标注）。
- 复现策略（遵循 B1 Gate）：**B2 先做 DAS 成像对照论文图（可行）；
  diffusion（B3）在 GT-mask 缺口定级前不开跑**。

## 1. 数据来源（已验证）

### 1.1 NASA PCoE 数据集门户迁移

- 旧门户 `datasets.phl.pdx.edu` **已死亡**（Google 公共 DNS NXDOMAIN，2026-10-03 验证）。
- 现役官方入口：NASA Intelligent Systems Division PCoE Data Set Repository 页面
  `https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/`
  （HTTP 200 实测可达）。
- 该页面把数据集文件托管在 PHM Society 的 S3：
  `https://phm-datasets.s3.amazonaws.com/NASA/<名称>.zip`。
- **Composites 数据集**：`https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip`
  - HTTP 200，**Content-Length 4,638,330,974 B ≈ 4.6 GB**，Last-Modified 2022-09-18。
  - 已开始下载至 `data/raw/nasa_cfrp/2.Composites.zip`。
- zip 内容（panel 数、layup、PZT 通道数、X-ray 文件）待解压后核实；
  论文使用的具体 panel/cycles 以全文为准。

### 1.2 数据集实际内容（README.pdf 实读，2026-10-04）

- 实验：Stanford SACL + NASA Ames PCoE，CFRP 试件 tension-tension 疲劳，
  MTS 试验机，ASTM D3039/D3479，加载频率 5 Hz，应力比 R≈0.14。
- 试件：Torayca T700G 预浸料，15.24 cm × 25.4 cm dogbone，中心缺口
  5.08 mm × 19.3 mm 诱导应力集中。
- **三个 layup**：Layup1 [02/904]s；Layup2 [0/902/45/−45/90]s；Layup3 [902/45/−45]2s。
- **PZT 配置：两组 6-PZT SMART Layer（Acellent）= 6 actuators + 6 sensors，共 12 个 PZT，
  36 条 actuator-sensor 轨迹 × 7 个 interrogation 频率（150–450 kHz）= 252 条路径。**
- 采集：疲劳每 ~50,000 cycles 停机采集全部 PZT 路径 + X-ray（dye-penetrant 增强）。
- 每个 coupon 目录：LogBook xlsx、PZT-data（MATLAB .mat，含 path.data(i,j) 的
  actuator/sensor/amplitude/sampling_rate/frequency/gain/signal_actuator/signal_sensor）、
  StrainData、**XRay（jpg 灰度图，按 cycle 命名，如 L1S11_100000.jpg）**。
- Coupon 清单：Layup1: L1S11,S12,S17,S18,S19；Layup2: L2S11,S12,S17,S18,S20；
  Layup3: L3S11,S13,S14,S20。
  注意事项（README 原文）：L1S18/L1S17 只有 X-ray 胶片照片；
  **L3S13、L3S14 无 X-ray 图像，仅有手描图**；
  部分 coupon 无应变片数据。
- 信号采集条件变量：condition ∈ {Loaded, Clamped, Traction Free}（影响信号，需论文确认用哪种）。

### 1.3 ⚠ 与任务描述的出入（必须由论文裁决）

- 任务描述称 DI-CDM 用 **16 PZT**；该 NASA PCoE Composites 数据集是 **12 PZT（6+6）**。
- 可能性：(a) 论文实际用 12 PZT（任务描述记忆偏差）；
  (b) 论文用的是另一份 NASA CFRP 数据（如 TU Delft 自采或其它 PCoE 数据集）；
  (c) 论文只用 36 条轨迹的子集构成 sparse array。
- **在论文全文核实前，不得假定 panel/layup/sensor 数。**

### 1.6 解压后全量盘点（2026-10-04，13 个 coupon 目录）

| coupon | mat | X-ray 图 | 备注 |
|---|---|---|---|
| L1_S11 | 225 | 15 | |
| L1_S12 | 166 | 21 | |
| L1_S18 | 180 | 12 | 胶片照片（.JPG） |
| L1_S19 | 85 | 14 | |
| L2_S11 | 197 | 26 | |
| L2_S17 | 108 | 16 | |
| L2_S18 | 110 | 15 | |
| L2_S20 | 107 | 11 | |
| L3_S11 | 139 | 27 | |
| L3_S13 | 194 | 5 | README：另有手描图 |
| L3_S14 | 44 | **0** | README：仅手描图 |
| L3_S18 | 134 | 13 | |
| L3_S20 | 123 | 46 | |

- 目录命名大小写不一致：`XRay/XRays`、`.jpg/.JPG`、`PZT-data/PZTdata` — 读取器需 case-insensitive。
- **除 L3S14 外全部 coupon 都有 X-ray** → 训练对（DAS + X-ray GT）在多个 coupon 上可构建；
  具体用哪个 coupon 仍由论文裁决。

### 1.5 实物抽查（2026-10-04）

- mat 命名：`<coupon>_<record>_<type>_<n>.mat`，LogBook xlsx（146 行）给出
  日期/cycles/load/边界条件（Baseline/clamped/loaded）/Data File Name 完整映射。
- path_data 实测：252 条 = 36 轨迹 × 7 频率；单路径 `signal_sensor (2000,)` float32
  @ 1.2 MHz（1.67 ms），幅值为 ADC 计数量级（±2400）。
- X-ray 实读（L1S11_100000.jpg，2560×3280）：**原始胶片照片**——
  含手写标注（"L1 S11 100 kcycles (21, 1')"）、胶片边框/边缘刻字、
  可见的导线与 SMART Layer 走线、6+6 个 PZT 圆盘、dogbone 轮廓与缺口。
  分层损伤区域（缺口向右下扩展的浅色区）肉眼可辨。
- **含义**：X-ray → 训练 target 需要胶片裁剪、几何配准（像素↔mm）、损伤二值化、
  手写区域剔除等多步 undocumented 处理 → 这是 B1 Gate 最可能触发 R2 的环节。

### 1.4 本地保存约定

- raw 只读：`data/raw/nasa_cfrp/`（zip 原样保留 + sha256）。
- manifest：`data/manifests/nasa_cfrp/`（来源 URL、时间、checksum、
  panel/cycle/sensor 清单、X-ray 文件清单）。

## 2. 论文 protocol 恢复情况（B0 的 24 个问题，全文精读后）

> 依据：`docs/papers/Yang2025_DICDM_MSSP_112996.pdf`（出版版 26 页）。

| # | 问题 | 恢复情况 | 论文依据 |
|---|------|---------|---------|
| 1 | 哪个 panel | **L1 = 测试；L2+L3 = 训练**（"images from specimen L1 were used for testing"） | §4.1 |
| 2 | layup | Table 1：L1 [0/90₄]s，L2 [0/90₂/45/−45/90]s，L3 [90₂/45/−45]₂s；ply 0.132 mm | Table 1 |
| 3 | 哪些 cycles | **1, 10k, 20k, 30k, 40k**（实验部分 Fig 13/Table 4） | Fig 13, Table 4 |
| 4 | 哪些 PZT paths | 2×6 SMART Layer：6 actuators（上）+ 6 receivers（下），36 轨迹 × 7 频率（150–450 kHz）；**DAS 用 250 kHz**；6 个 actuator 分别成像后**平均** | §4.1, §6.1 |
| 5 | waveform preprocessing | 带通 **230–270 kHz**（4 阶 Butterworth 图示）→ **Hilbert 变换**取 S0 首到达 ToF | §5.1.1, Fig 7 |
| 6 | baseline subtraction | 仿真：与 baseline(undamaged) 信号在**时频域**相减后再 DAS；**实验部分未明确说明是否做 baseline 相减**（论文只说直接用 DAS；Fig 13a cycle 1 已显示 notch 处伪影） | §5.1.2 vs §6 |
| 7 | filtering | 230–270 kHz 带通（同 #5） | Fig 7(c) |
| 8 | time window | **未写明**（信号原始 2000 点 @1.2 MHz = 1.67 ms；Fig 7(d) 显示 Hilbert 包络取 0–0.3 ms 窗口作示意） | UNKNOWN（图示推断） |
| 9 | DAS 参数 | 成像网格 **152×178 px**（≈mm 级 1:1）；速度：**13 方向（0–360°，30° 步长）S0 群速度逐试件拟合角度曲线**（Fig 8，L1/L2/L3 各自拟合）；权重 ω=1（"weighting factors are the same"）；延迟 τᵢ(φ) 由像素-传感器距离/方向速度计算；6 actuator 图像平均 | Eq(1), §5.1.1, Fig 5b/8/12 |
| 10 | X-ray 图像处理 | **未描述**（只说 dye-penetrant 增强 X-ray；Fig 13c 展示的是裁剪后的 X-ray） | UNKNOWN |
| 11 | X-ray → training target | **未描述**。Table 4 给出 P_ref 像素数（156/985/1614/1677/1943），但 mask 生成/配准/二值化流程零描述 | **UNKNOWN — 核心 blocker** |
| 12 | X-ray ↔ DAS 配准 | **未描述**（成像区 152×178 mm 与 X-ray 的像素↔mm 映射、坐标对齐均未写） | **UNKNOWN — 核心 blocker** |
| 13 | diffusion backbone | **Stable Diffusion v2 base**，UNet 868,443,332 params；VAE+CLIP 冻结 | §3, Data availability |
| 14 | pretrained checkpoints | `stabilityai/stable-diffusion-2-base`（VAE=`sd-vae-ft-mse-original`）；`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`（DAS 图像编码）+ CLIP text encoder（cycle 文本） | Data availability |
| 15 | LoRA 插入位置 | UNet 内 Transformer 架构各层（"introduced into the training of each layer of the Transformer architecture within the UNet"），经 **PEFT** 实现 | §5.2.1 |
| 16 | LoRA rank | 消融 16/32/64/128 → **选 rank=32**（5,042,176 trainable = 0.5799%）；alpha/dropout 未写 | Table 2, §5.2.1 |
| 17 | conditioning mechanism | **cross-attention**：Q=X-ray latent zₜ，K/V=DAS 图像 CLIP 嵌入 ⊕ cycle 文本嵌入；公式 Eq(12) | §3, Eq(12), Fig 3 |
| 18 | cycle condition | 文本提示经 CLIP text encoder；具体 prompt 模板**未写** | §3（模板 UNKNOWN） |
| 19 | training steps | DDPM scheduler **1000 diffusion steps**；收敛 ~700 steps；warmup 300 steps | §5.2.1, Fig 10 |
| 20 | learning rate | AdamW **1e-6** + warmup（0→1e-6 by step 300 后线性衰减）；gradient accumulation 16 | §5.2.1, Fig 10b |
| 21 | sampler | **DDIM** | §5.2.1 |
| 22 | inference steps | **50**（从纯噪声生成） | §5.2.1 |
| 23 | random seed | **未写** | UNKNOWN |
| 24 | metrics | **IoU**：输出在强度≥0.5 二值化 vs X-ray GT delamination 区；**SNR**：20·log₁₀[(mean(A_D)−mean(A_{I\D}))/std(A_{I\,D})] | Eq(13)(14), §4.2 |

**论文主结果（复现对照目标）：**

- 实验（L1，Table 4）：DAS IoU = 0.0164/0.0623/0.0881/0.0761/0.0753
  （cycle 1/10k/20k/30k/40k）；DI-CDM IoU = 0.0846/0.1322/0.1872/0.2234/**0.3340**。
  SNR：DAS 除 cycle1 (8.77) 外 <7 dB；DI-CDM 6.34→7.81 dB。
- 仿真（Table 3）：DAS IoU 0.0050–0.1239；DI-CDM 0.0825/0.1006/0.6211/0.3878/0.2697。
- 硬件：2×A100 80GB（KU Leuven VSC）。

## 3. B1 Gate 判定

> 规则：X-ray↔配准 / X-ray target 生成 / DAS 参数 / train-test cycles /
> conditioning 实现任一不可恢复 → 不直接开跑 diffusion。

| 关键项 | 可恢复？ | 影响 |
|---|---|---|
| DAS 参数 | ✅ 基本完整（时间窗与实验 baseline 相减除外，可用图对照校准） | B2 可执行 |
| train/test cycles | ✅ 完整（1–40k；L2+L3 train / L1 test） | — |
| conditioning 实现 | ✅ 机制完整（cross-attention + CLIP）；仅 cycle prompt 模板缺失（影响小，可自定但需记录） | 可控 |
| LoRA 配置 | ⚠ 部分（rank/target modules/优化器/lr/步数完整；alpha、dropout、resolution 缺） | 中等，标准默认值近似 |
| **X-ray → GT mask** | ❌ **完全未描述** | **致命：IoU/SNR 的 P_ref 无法重建**；训练 target 同样依赖它 |

**判定：R2。** 具体：
- **B2（DAS 成像）已执行，结果：未能与论文 Fig 13a 建立对应 → 按阶段规则 STOP。**
  详见 `experiments/results/dicdm/das_reproduction/README.md`。要点：
  - 实测发现**数据频率标签 ×10 现象**（file `frequency==250` = 25.2 kHz @1.2 MHz）；
  - 论文速度场来自 **ABAQUS 仿真**（§5.1.2），实验信号首到达多模态色散，
    无法拟出一致速度场；
  - 尝试管道（baseline 相减 + 20–30 kHz 带通 + 直达波门控 + v=5200 m/s）
    得到边缘主导图像、无损伤聚焦、无渐进趋势；v∈{4.4,5.2,5.6} km/s 敏感性一致；
  - 6 项未写明的自由参数（触发对齐/时间窗/基线规则/速度数值/传感器坐标/滤波细节）
    逐一记录于 README。
- **B3（DI-CDM diffusion）双门受阻**：(1) B2 STOP 规则；(2) GT mask 流程不可恢复。
  解锁途径：邮件通讯作者（xin.yang@kuleuven.be）索取 DAS 预处理 + GT 生成细节，
  或用户批准降级为 R2 "自有 GT 管道 trend 对照"。

## 4. 官方代码检索

- **不存在**（GitHub/PapersWithCode 0 hit；作者无公开 SHM-diffusion repo）。
- 论文 Data availability 承诺"code could be shared after potential publication"，
  至今未见。→ **R2 paper-guided reimplementation**，报告不得写 official reproduction。
