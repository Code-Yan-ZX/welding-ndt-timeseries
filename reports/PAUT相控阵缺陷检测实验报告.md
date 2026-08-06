# 实验报告:相控阵超声(PAUT)信号缺陷检测——从工艺参数迈向真正的 NDT 信号

- **仓库**:welding-ndt-timeseries
- **报告日期**:2026-08-06
- **硬件**:NVIDIA RTX 4090 D(48 GB),单卡运行各模型
- **数据集**:Submerged Arc Welding Open Repository,Zenodo [10.5281/zenodo.15083865](https://doi.org/10.5281/zenodo.15083865)(v2,CC-BY-4.0,AIMEN / PENELOPE 项目)

---

## 一、实验目标

延续项目方向——把模型输入从"工艺参数(电流/电压)"转向**真正的无损检测信号**。本轮首次解析 SAW 数据集里的**相控阵超声(PAUT)`.nde` 文件**,提取 A-scan / B-scan 信号,以 `defects_xlocation` 标注为监督做**缺陷自动检测**,并尝试**谱-空-频(spectral-spatial-frequency)模型**(参考 Alliance)与时序基础模型,最后与同批焊件的工艺信号(电流/电压)编码器、MOMENT 结果**跨模态对照**。

> 说明:本轮所有模型均为**数值时序/张量模型**,不涉及图像输入,契合当前"纯文本大模型、传图会 400"的接入约束。后续可用 ITFormer 式 patch 化把 A-scan 编码成 token 桥接到 LLM。

参考论文:
- **ITFormer**(ICML 2025,arXiv:2506.20093):时序-语言桥接,本项目 GMAW 基线所用范式。
- **MOMENT**(CMU,NeurIPS 2024):开源时序基础模型,本实验的"时序大模型"探针。
- **Alliance**: All-in-One Spectral-Spatial-Frequency Awareness Foundation Model(谱-空-频联合感知基础模型),本实验 SSF 模型的设计参考。

## 二、PAUT `.nde` 数据解析

`.nde` 是 Evident/Olympus OmniScan X3 写出的 **HDF5 容器**(NDE-FileFormat-Schema-3.1.0),用 `h5py` 即可读取,无需第三方专有库。每个试件的 `2. ndt_data` 目录下有 2 个 `.nde`(90 族与 270 族,即探头两侧扫查),共 11 个文件(PP3–PP7):

```
data/raw/saw/ZENODO_Penelope/<PPx>/2. ndt_data/
  ├── PAUT_90.nde / PAUT_90+.nde / 1163421_PP7_030325_90.nde   (90 族)
  ├── PAUT_270.nde / 1163421_PP7_030325_270.nde                (270 族)
  ├── defects_xlocation.xlsx   (缺陷轴向位置 + 类型标注)
  └── RX.pdf / UT.pdf          (射线 / 超声报告)
```

每个 `.nde` 含 3 个 DataGroup,其中 **Group 0** 在所有文件中一致:`(n_pos, 49, 3500)` int16,49 波束、3500 采样、折射角 71°、skew 90°。Group 1 为 47°/22 波束组,Group 2 为单波束参考通道。本实验统一用 **Group 0**(主检 71° 组)。

**几何对齐**:从 Setup JSON 读 `UCoordinate` 分辨率(1 mm/位置)与首波束 `uCoordinateOffset`(80 mm,PP5 为 85 mm),故扫描位置 `i` 对应焊缝轴向 `x = offset + i·1 mm`,与 `defects_xlocation` 的 `x_init/x_end [mm]` 直接对齐(`x [# sample] = x[mm] × 625`,同坐标)。

**信号特征**(PP3 90 G0 实测):
- int16 已**整流(包络)**,取值 ≥0,max≈16834;每条 A-scan 极稀疏——每位置仅 **0.5–0.9%** 的采样点有回波。
- 沿深度(时间)轴能量集中在 0–23 mm(焊缝填充/缺陷区)与 113–227 mm(底面多次回波),中间为低幅体波区。

## 三、任务设定:位置级局部缺陷检测

### 3.1 为何不能直接做"位置级二分类(全部缺陷)"

`defects_xlocation` 的缺陷只有**轴向 x 位置**,无深度 z 标注。更关键的是:每个试件都有**贯穿整条焊缝的大区域缺陷**(如 PP7 bead16 上一条 x=80–680 mm 的裂纹),在 PAUT(焊后整体检测)位置级会**铺满所有扫描位置**——若把全部缺陷当正例,PP3/PP5/PP6/PP7 的 4/5 试件将 100% 为正,任务退化为无负样本(而 SAW 工艺信号是**逐焊道 bead** 标注,贯穿裂纹只影响该 bead 的窗口,故 SAW 缺陷率仅 7% 不退化)。

### 3.2 任务定义

- **样本单元**:扫描位置(1 mm/位置)。输入 = 该位置的 PAUT 信号。
- **标签(局部缺陷检测)**:位置 x 若与任一**轴向长度 < 50 mm 的局部缺陷**(气孔/未熔合/夹渣/金属夹杂/小裂纹等)重叠 → 1(缺陷),否则 0(正常)。**轴向 ≥ 50 mm 的贯穿型大裂纹/大区域作背景(0)**,属任务范围限制(见第六节讨论)。
- **规模**:5 个试件共 3000 个位置;缺陷率 38.4%。
- **划分(按试件,与 SAW 实验一致以利对照)**:
  - train:PP3 / PP4 / PP5(1798 位置,缺陷率 33.9%)
  - val:PP6(601,76.4%)
  - test:PP7(601,13.8%)
- **输入表示**:Group 0 的 49 波束 A-scan,逐波束 **max-pool 降采样 3500 → 512**(整流包络用 max-pool 保峰),得每位置 `(49, 512)` B-scan;另存 max-over-beams 包络 `(512,)` 供经典 ML / MOMENT。
- **归一化**:逐时间步 mean/std(仅在 train 上统计)。
- **指标**:AUC(阈值无关,主指标)、macro-F1、acc。⚠ **val(PP6 76% 缺陷)与 test(PP7 14% 缺陷)缺陷率严重失配**,固定 0.5 阈值与 val 上调阈值的 f1/acc 都不可靠跨试件迁移,**以 AUC 为准**;f1/acc 用 val 上最优 macro-F1 阈值校准后给出,仅作参考。

## 四、方法

### 4.1 经典 ML(RF / XGBoost)

在 max-over-beams 包络 `(512,)` 上提手工特征:时域统计(14)+ FFT 谱特征(5 带能量/质心/熵/主频/滚降/峰高,9)+ PAUT 专属(峰高、峰位深度、近场/底面能量比、过阈比例,5),共 28 维。RF / XGB,train 上拟合,test 上评估。复用 `wndt.features.handcrafted`。

### 4.2 EncoderOnly(from-scratch PatchTST 编码器)

复用项目 `WeldTSEncoder`(PatchTST 风格),输入每位置完整 `(49, 512)` B-scan(49 通道)。编码器的**变量自注意力(VarAttention)在 49 个波束上做注意力**,等价于**注意力 MIL**:在位置级标签下学会聚焦含回波的波束(因缺陷标签是位置级,无波束级标注,逐波束独立训练会误标缺陷位置上 47 个无回波波束,故必须用整 B-scan + 注意力聚合)。d_model=128,3 层 8 头,patch_len=64(每波束 8 patch),dropout=0.3,~0.8M 参数;AdamW + cosine warmup,WeightedRandomSampler,**val AUC 早停**(抗 val/test 失配),3 seed。

### 4.3 SSF:谱-空-频模型(Alliance 启发,本实验新贡献)

`src/wndt/models/ssf.py`,输入每位置 `(49, 512)` B-scan,三分支并行后融合:

1. **空间(spatial)**:对原始 B-scan(波束×时间)做 2D 卷积栈,捕捉缺陷回波的横向/时间几何。
2. **谱(spectral)**:对**时间轴**做 rFFT 取 log 幅值 → `(49, 257)`,再 2D 卷积,捕捉每条 A-scan 的深度频率内容。
3. **频率(frequency)**:对**波束轴**做 rFFT 取 log 幅值 → `(25, 512)`,再 2D 卷积,捕捉横向空间频率(结构化/周期性指示)。

三分支各经 Conv→BN→GELU→MaxPool + 全局均值池化 → 128 维,拼接后 MLP 分类。~0.7M 参数。这是纯数值 2D 张量模型,不渲染为图像,符合纯文本 LLM 约束。

### 4.4 MOMENT 时序基础模型(冻结 + 线性探针)

加载预训练 **MOMENT-1-large**(~341M,24 层 T5,d_model=1024),主干全冻结。对每位置 max-over-beams **包络**(1 通道,512)过共享编码器 → 1024 维特征;探针头 LayerNorm+Dropout+Linear。一次性预计算全部位置嵌入并缓存(8 s),val AUC 选最优探针,3 seed。

## 五、结果

### 5.1 PAUT 缺陷检测结果(test = PP7,位置级)

| 模型 | seeds | test AUC | test F1(macro) | test acc | 可训练参数 |
|---|---|---|---|---|---|
| classic_rf | 1 | 0.4901 | 0.4549 | 0.7288 | 0.0M |
| classic_xgb | 1 | 0.5024 | 0.4522 | 0.6423 | 0.0M |
| encoder_only | 3 | 0.5365±0.1241 | 0.4096±0.1329 | 0.5724±0.2626 | 0.8M |
| **ssf(谱-空-频)** | 3 | **0.6262±0.0092** | 0.3209±0.0587 | 0.3322±0.0673 | 0.7M |
| moment | 3 | 0.4588±0.0212 | 0.1580±0.0194 | 0.1658±0.0149 | 0.0M |

多数类基线(test):acc 0.8619 | F1(macro) 0.4629。

**要点**:
- **SSF 最优且最稳**:AUC 0.626±0.009,三 seed 一致优于经典 ML 与 MOMENT,方差远小于 encoder。说明谱-空-频多分支结构契合 PAUT B-scan 的多域结构。
- **encoder 高方差**:AUC 0.54±0.12,seed43 仅 0.36(未学到)。PatchTST 在 1798 个位置的小数据上不稳定;相比之下 SSF 的卷积归纳偏置更省样本。
- **MOMENT 失败**:AUC 0.459(低于随机),冻结时序基础模型在 PAUT 包络上**不迁移**,与 SAW 上的结论一致。
- **经典 ML 近随机**:包络手工特征 AUC 0.49–0.50,跨试件不泛化。

### 5.2 跨模态对照:PAUT(NDT 信号)vs SAW(工艺信号)

同一批焊件 PP3–PP7、同一 test=PP7、同一缺陷真值,两套输入:

| 模型族 | 模态 | test AUC | test F1(macro) |
|---|---|---|---|
| classic RF | PAUT(NDT) | 0.4901 | 0.4549 |
| classic RF | SAW(工艺) | 0.2159 | 0.4879 |
| from-scratch encoder | PAUT(NDT) | 0.5365 | 0.4096 |
| from-scratch encoder | SAW(工艺) | 0.6354 | 0.5231 |
| MOMENT(冻结探针) | PAUT(NDT) | 0.4588 | 0.1580 |
| MOMENT(冻结探针) | SAW(工艺) | 0.4885 | 0.4850 |
| **SSF(谱-空-频)** | **PAUT(NDT)** | **0.6262** | 0.3209 |

> SAW 行为既有结果(`experiments/results/saw_*.json`),输入为 4 通道电流/电压 512 窗口。PAUT 输入为 49 波束 B-scan(或包络),512。

**对照结论**:
1. **经典 ML**:PAUT 包络特征(AUC 0.49)明显优于 SAW 手工特征(0.22)——NDT 信号本身比工艺信号更接近缺陷物理表现,手工特征即可体现;但两者都近随机,跨试件不泛化。
2. **从零 encoder**:SAW(0.636)> PAUT(0.537)。SAW 有 10 万+ 窗口可训;PAUT 仅 1798 位置,transformer 数据不足、方差大。
3. **MOMENT**:两模态都失败(0.46 / 0.49),**冻结时序基础模型不迁移到焊缝 NDT**,与上一轮 SAW 结论互相印证。
4. **SSF(0.626)是 PAUT 上的最佳模型**,与 SAW encoder(0.636)相当,且更稳。说明在 NDT 小数据场景,**带领域归纳偏置的谱-空-频 CNN 优于通用 transformer 与冻结基础模型**。
5. **整体**:两种模态在跨试件泛化上都不强(AUC 0.5–0.64),核心瓶颈是 train(PP3/4/5)与 test(PP7)的焊接参数域与缺陷分布漂移,而非信号本身。

## 六、讨论与限制

1. **贯穿型大裂纹的处理**:≥50 mm 的大区域缺陷(如 PP7 bead16 全长裂纹)在 PAUT 位置级会铺满整条焊缝,使二分类退化。本实验将其作背景,聚焦**局部缺陷检测**(气孔/夹杂/小裂纹/夹渣),这是 PAUT 常规检测任务;但意味着模型不评估"整条焊缝是否裂纹贯穿"。如需纳入,可改为焊件级二分类或缺陷类型分类。
2. **val/test 缺陷率失配**:PP6(76%)vs PP7(14%),任何在 val 上调的阈值都难迁移到 test,f1/acc 噪声大。**AUC 是可靠主指标**。后续可考虑按试件缺陷率分层划分或加入校准(如温度标定)。
3. **标签只有轴向 x,无深度 z**:无法做 (位置×深度) 体素分割;本实验在位置级检测。波束级无标注,故不能逐波束训练(会误标),只能用整 B-scan + 注意力/卷积聚合(SSF、encoder 的 VarAttention)。
4. **encoder 不稳**:小数据下 transformer 方差大(seed43 崩)。SSF 的卷积归纳偏置更省样本、更稳。增大数据(更多试件/数据增强)或可改善 encoder。
5. **未用 270 族与 47° Group 1**:本实验仅用 90 族 Group 0(71°)。270 族与 47° 组提供不同视角,后续可作多通道输入提升。
6. **方向契合度**:本轮把输入成功从工艺信号推进到真正的相控阵超声 NDT 信号,验证了 SSF 谱-空-频思路在 PAUT 上的有效性,并印证冻结时序基础模型不迁移——为后续用 ITFormer 式 patch 化桥接文本 LLM(纯 token,不传图)打下了信号预处理与基线对照基础。

## 七、代码与复现

```
scripts/paut_preprocess.py      # 解析 .nde -> data/processed/paut/
scripts/paut_classic_ml.py      # RF / XGB 基线
scripts/paut_train.py           # encoder / SSF 训练(复用 ClassificationTrainer)
scripts/paut_moment_probe.py    # MOMENT 冻结探针
scripts/paut_make_table.py      # 结果汇总 + 跨模态对照
configs/paut_encoder.yaml / paut_ssf.yaml / paut_moment.yaml
src/wndt/data/paut_dataset.py   # PAUTSeriesDataset(env/bscan/expand 模式)
src/wndt/models/ssf.py          # 谱-空-频模型
```

```bash
python scripts/paut_preprocess.py
python scripts/paut_classic_ml.py
python scripts/paut_train.py --config configs/paut_encoder.yaml --seed 42
python scripts/paut_train.py --config configs/paut_ssf.yaml --seed 42
python scripts/paut_moment_probe.py --seeds 42 43 44
python scripts/paut_make_table.py
```
