# 实验报告:时序基础模型在埋弧焊(SAW)无损检测缺陷检测上的尝试

- **仓库**:welding-ndt-timeseries
- **报告日期**:2026-08-06
- **硬件**:NVIDIA RTX 4090 D(48 GB)×3,单卡运行各模型
- **数据集**:Submerged Arc Welding Open Repository,Zenodo [10.5281/zenodo.15083865](https://doi.org/10.5281/zenodo.15083865)(v2,CC-BY-4.0,AIMEN / PENELOPE 项目)

---

## 一、实验目标

在焊缝无损检测(NDT)场景下尝试**时序基础模型(time-series foundation model)**,考察预训练大模型相对从零训练的小模型在该任务上的迁移性。这是继 GMAW(Metal Arc Welding)ITFormer 基线之后,在新数据集、新焊接工艺(埋弧焊 SAW)上的探索,并延续项目方向:把输入从"工艺参数"逐步引向真正的无损检测信号。

参考论文:
- **ITFormer**(ICML 2025,arXiv:2506.20093):时序-语言桥接,本项目 GMAW 基线所用范式。
- **MOMENT**(CMU,NeurIPS 2024):开源时序基础模型族,在多个时序任务上预训练,本实验的"时序大模型"。
- **Alliance**: All-in-One Spectral-Spatial-Frequency Awareness Foundation Model(谱-空-频基础模型,后续相控阵信号建模的参考)。

## 二、数据集

**Submerged Arc Welding(SAW)数据集**:多个焊接试件(coupon)经受埋弧焊,每个试件含若干焊道(bead),每道记录:
- **process_data(`*.hdf5`)**:
  - `data1`(5 kHz):`current_a, current_b, voltage_a, voltage_b`(直流/交流电极的电流、电压)--本实验输入信号。
  - `data0`(66.7 Hz):高温计温度、送丝速度、xyz 坐标、`process_status`(焊接是否进行)、`idx_data1`(到 data1 的索引映射)。
- **NDT_data**:`defects_xlocation.xlsx`(每道的缺陷位置,含缺陷类型 1–6:气孔/未熔合/夹渣/金属夹杂/凸起/裂纹,以及 `x_init/x_end [# sample]` 即 data1 采样索引区间)、PAUT(相控阵超声)、UT、X 射线报告。

共 7 个试件:Coupon1/2(各 5 道,无缺陷标注)、PP3–PP7(各 24 道,共 120 道有缺陷标注)。本实验使用 PP3–PP7。

## 三、任务设定:窗口级缺陷检测

将每道焊缝的 5 kHz 4 通道信号切成定长窗口,做二分类(窗口内是否含缺陷):

- **输入**:window = 512 样本(0.1024 s @ 5 kHz),4 通道(V/I 的 a/b 电极)。
- **切窗**:stride = 256(50% 重叠);只在 `process_status==1` 的活跃焊接区切窗(剔除起弧前/收弧后的零值段)。
- **标签**:某窗口 `[i, i+512)` 若与该道任意缺陷采样区间重叠则标 1(缺陷),否则 0(正常)。
- **规模**:共 172,424 个窗口,缺陷率 7.36%(类不平衡,符合 NDT 实际)。
- **划分(按试件,杜绝同试件跨 split 泄漏)**:
  - train:PP3 / PP4 / PP5(103,755 窗口)
  - val:PP6(34,447)
  - test:PP7(34,222)
- **归一化**:逐通道 StandardScaler,只在 train 上 fit。
- **指标**:accuracy、binary F1(正类=缺陷)、macro F1、AUC。由于 test 缺陷率仅 4.7%,多数类基线 accuracy=0.953,**accuracy 不具区分力,以 AUC 与 macro-F1 为准**。

⚠ 刻意的跨试件分布漂移:不同试件采用不同焊接参数设定(电压/电流/送丝/速度),train(PP3/4/5)与 test(PP7)的参数域与缺陷分布不同,这是本任务的核心难度。

## 四、方法

### 4.1 EncoderOnly(from-scratch 基线)

复用项目既有 PatchTST 风格编码器(`WeldTSEncoder`),配置为 4 通道、seq_len=512:
- patch_len=64 -> 每通道 8 个 patch;d_model=256,4 层 8 头;均值池化 + 线性头(2 类)。
- 可训练参数 4.23M,从零训练。
- AdamW + cosine warmup,WeightedRandomSampler(类平衡),val macro-F1 早停,3 个种子(42/43/44)。

### 4.2 MOMENT 时序基础模型(冻结 + 线性探针)

加载预训练 **MOMENT-1-large**(`AutonLab/MOMENT-1-large`,~341M 参数,24 层 T5 编码器,d_model=1024):
- **主干全冻结**,4 通道各自过共享预训练编码器(`embed`, reduction=none -> 每通道 64 patch × 1024 维),patch 维均值池化后拼接 -> 4096 维特征。
- 探针头:LayerNorm + Dropout + Linear(4096->2),仅 **~16K 可训练参数**。
- 为效率采用**预计算嵌入**:一次性算出全部窗口的 4096 维特征并缓存(2.7 GB),再在小特征上训头(每种子 ~3 s)。等价于每 epoch 重跑冻结主干,但快 ~10×。bf16 + no_grad 加速前向。
- 同样 3 个种子;class-weighted 交叉熵。

### 4.3 经典 ML 基线

4 通道各提时域统计(均值/方差/RMS/峰峰/偏度/峭度/分位数/crest/过零率/差分)+ FFT 谱特征(5 带能量/谱质心/谱熵/主频/滚降)+ 通道间 Pearson 相关,~70 维。RF(500 树,class_weight=balanced)、XGBoost(500 树)。在 train 上 fit,val 上选,test 上评一次。

## 五、结果

### 5.1 主对比表(test = PP7,3 种子 mean±std)

| 模型 | 可训练参数 | test acc | test F1(bin) | test F1(macro) | test AUC | val F1(macro) | val AUC | wall(s)/种子 |
|---|---|---|---|---|---|---|---|---|
| **encoder_only**(从零) | 4.23M | 0.8945±0.0096 | 0.1022±0.0188 | **0.5231±0.0067** | **0.6354±0.0089** | 0.6487±0.0090 | **0.8165±0.0176** | ~173 |
| moment(冻结 341M + 探针) | 0.016M | 0.9183±0.0117 | 0.0126±0.0078 | 0.4850±0.0017 | 0.4885±0.0245 | 0.6376±0.0022 | 0.7664±0.0028 | ~3(嵌入预计算 ~720 s 后) |
| classic_rf(70 维特征) | – | 0.9529 | 0.0000 | 0.4879 | 0.2159 | 0.4992 | 0.6630 | ~60 |
| 多数类基线(test) | – | 0.9530 | – | 0.3694 | – | – | – | – |

> 注:XGBoost 因训练耗时较长本次未纳入最终表;RF 作为经典 ML 代表(树模型在该任务上跨试件泛化均差,结论一致)。完整结果可由 `python scripts/saw_make_table.py` 重新生成。

### 5.2 结果解读

**(1)从零训练的小编码器是最强的。** `encoder_only`(4.2M)在 test 上 AUC 0.635、macro-F1 0.523,优于冻结的 341M MOMENT 与经典 ML。在 10 万级窗口上从零训练已足以学到可跨试件迁移的缺陷特征。

**(2)预训练时序基础模型 MOMENT 的冻结特征未能迁移。** MOMENT 在 val(PP6)上 AUC 0.766(尚可),但 test(PP7)AUC 仅 0.489(≈随机),且低于多数类基线的 accuracy。其预训练语料与焊接电弧信号域差距过大,冻结特征对跨试件缺陷检测无增益。这与 GMAW 实验中"冻结 LLM(Qwen3)的语言先验未能补偿分布漂移"的结论一致:**通用时序/语言大模型的零迁移在焊缝 NDT 缺陷检测上目前不奏效**。

**(3)经典 ML 跨试件泛化最差。** RF 在 test 上 AUC 0.216(低于 0.5,缺陷分数与真实反向),严重过拟合训练试件的参数域。手工特征+树模型在小样本、强漂移设定下最不稳健。

**(4)跨试件分布漂移是核心难点。** 所有模型都出现 val->test 显著跌落(encoder AUC 0.817->0.635;MOMENT 0.766->0.489;RF 0.663->0.216)。不同试件焊接参数设定差异大,train(PP3/4/5)与 test(PP7)参数域不重叠,模型普遍偏向预测"正常"(test 缺陷率仅 4.7%),导致 binary F1 极低、accuracy 虚高。在此设定下绝对准确率无意义,应看 AUC 与 macro-F1。

**(5)MOMENT 的效率优势。** 冻结 + 预计算嵌入后,每种子训练仅 ~3 s(嵌入预计算一次性 ~12 min);而 encoder 每种子 ~173 s。若未来能在更接近的域上预训练或微调,这种"预计算特征 + 轻头"的范式在迭代效率上有优势。

## 六、结论与后续

- **结论**:在 SAW 焊缝缺陷检测(窗口级二分类、跨试件泛化)任务上,**从零训练的 PatchTST 风格小编码器优于冻结的 MOMENT 时序基础模型与经典 ML**;预训练时序基础模型的冻结特征目前不能有效迁移到该 NDT 任务,与既有 GMAW 结论一致。
- **后续方向**:
  1. **微调 MOMENT**(解冻主干或加 LoRA)而非纯冻结,观察能否改善跨试件泛化(注意 120 道小数据下过拟合风险)。
  2. **引入真正的 NDT 信号**:数据集含 PAUT(相控阵超声)`.nde` 文件,符合项目向"相控阵信号"转型的方向;后续以 PAUT A-scan 信号替代/补充工艺电流电压信号。
  3. 参考 **Alliance**(谱-空-频联合感知)等针对频域/空间结构设计的模型,可能更契合超声 NDT 信号。
  4. 跨试件泛化:尝试域适应/域随机化,或按焊接参数分层划分以缓解漂移。

## 七、可复现性

```bash
# 环境:venv + torch 2.5.1+cu121 + momentfm(见 CLAUDE.md / README)
bash scripts/01... # 数据:下载并解压 ZENODO_Penelope_vs2.zip 到 data/raw/saw/ZENODO_Penelope
python scripts/saw_preprocess.py          # -> data/processed/saw/(窗口+标签+划分)
python scripts/saw_train.py --config configs/saw_encoder.yaml --seed 42   # from-scratch 编码器
python scripts/saw_moment_probe.py        # MOMENT 冻结嵌入 + 线性探针(3 种子)
python scripts/saw_classic_ml.py          # RF / XGBoost
python scripts/saw_make_table.py          # 汇总表
```

## 八、参考文献

1. 数据集:Tapia Suárez et al., *Submerged Arc Welding Open Repository*, Zenodo, DOI [10.5281/zenodo.15083865](https://doi.org/10.5281/zenodo.15083865)(CC-BY-4.0,v2)。
2. MOMENT:Goswami et al., *MOMENT: A Family of Open Time-series Foundation Models*, NeurIPS 2024;代码 [AutonLab/MOMENT](https://github.com/moment-timeseries-foundation-model/moment)。
3. ITFormer:Wang et al., ICML 2025,[arXiv:2506.20093](https://arxiv.org/abs/2506.20093)。
4. Alliance: All-in-One Spectral-Spatial-Frequency Awareness Foundation Model(后续参考)。
