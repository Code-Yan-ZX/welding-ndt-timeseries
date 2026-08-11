# PAUT 方向调研:多模态 LLM 在焊缝 NDT 中的文献定位与创新空间

> 项目:大模型 + 焊缝无损检测(智能 NDT)｜ 方向:相控阵超声(PAUT)缺陷检测
> 调研日期:2026-08-11
> 触发:P2 阶段通用多模态 LLM(Qwen3.6-27B)零样本非PP4 AUC=0.593,为 P0–P2 三阶段最优。
> 本报告回答两个问题:(1)「多模态大模型做 NDT,文献里是不是已经做烂了?」
> (2)「后续可以在哪里创新?」并给出 P3 落地方案与可直接派发的 goal 提示词。

---

## 一、文献实况:有人开始做,但做的几乎都是「图像」,不是「信号」

### 1.1 检索方法与结果

在 arXiv 全文检索(2026-08-11)以下查询,命中情况:

| 查询 | 命中数 | 相关性 |
|---|---|---|
| `large language model nondestructive testing` | 0 | — |
| `GPT-4V defect detection industrial` | 0 | — |
| `foundation model ultrasonic testing` | 2 | 0 篇真正相关(关键词误匹配) |
| `deep learning phased array ultrasonic defect` | 0 | — |
| `ultrasonic NDT neural network B-scan` | 0 | — |
| `vision language model weld defect` | 1 | 高度相关(下文 Rao) |
| `large language model welding quality` | 1 | 同上 Rao |
| `multimodal LLM industrial inspection manufacturing` | 1 | 弱相关(时序异常,非视觉) |

**核心结论**:arXiv 上「LLM/VLM + 超声/PAUT 信号」几乎是空白;「LLM/VLM + 焊缝」仅有 1 篇,且做的是**射线照片**(本质是图像),不是超声信号。

### 1.2 最接近的对照工作

**Rao, "Answer-Conditioned Chain-of-Thought Distillation for Few-Shot Industrial Vision with Small VLMs"**(arXiv:[2607.10666](https://arxiv.org/abs/2607.10666),2026-07)

- 问题:制造业视觉检测需求频变、新缺陷类型涌现、标注稀缺。
- 方法:用 frontier VLM 对每张训练图(已知标签)生成「带理由的视觉解释」,再用 LoRA 在 3B 小模型上微调这些推理增强样本(answer-conditioning 保证推理始终指向正确结论)。
- 数据/模态:**4 个工业分类任务,跨 3 种图像模态,每任务仅 18–30 张标注**。其中含**焊缝射线照片(weld radiograph)分类**。
- 结果:在 16 种 seed×任务组合上全部超过直接微调,均值 +1.7~+4.4pp;焊缝射线照片上微调后的 3B 模型**仅用 24 张图就比 GPT-4.1 高 10.0pp**。
- 与本项目的关键差异:**它用的是射线照片(X-ray radiograph,本质是图像)**;**我们用的是 PAUT B-scan(波动信号的时-空表示,不是自然图像)**。CoT 蒸馏的方法学可借鉴,但模态完全不同。

### 1.3 相邻但不同的工作

- **时序基础模型做故障诊断**:Li et al., arXiv:[2511.23177](https://arxiv.org/abs/2511.23177),用 MOMENT/Mantis 做电机状态监测与故障诊断,1% 数据即达 2× 常规 DL。模态是**电机振动/电流**,非超声 NDT。→ 印证本项目 P1 结论:通用时序大模型(MOMENT)不迁移到焊缝 NDT(P1 MOMENT 冻结非PP4 AUC=0.470)。
- **物理先验 VLM 是当前热点,但全在机器人操控**:SpatioLM(arXiv:[2608.01899](https://arxiv.org/abs/2608.01899),物理空间智能)、Lift3D-VLA(arXiv:[2607.06564](https://arxiv.org/abs/2607.06564),3D 几何与动力学)、CausalPhys(arXiv:[2606.05966](https://arxiv.org/abs/2606.05966),因果物理推理)。**没有任何一篇把物理先验 VLM 用到 NDT 波动物理上**--这是本项目最干净的白地。
- **关键警示**:STEMGym(arXiv:[2606.29592](https://arxiv.org/abs/2606.29592))发现,在电子显微镜晶体缺陷分析上,**通用 VLM 比专用 CNN 差约 13 倍**,且瓶颈在感知管线而非导航策略。说明 VLM 在专门模态上并不自动占优,模态 gap 大就会崩。这对解读 P2 结果至关重要。
- **VLM 混淆「异常」与「危险」**:arXiv:[2607.18325](https://arxiv.org/abs/2607.18325) 指出 VLM 常把「异常度」误判为「危险度」。这恰是 NDT 误报的根源--超声检测中「异常回波」≠「缺陷回波」(可能是底面、角反射、模式转换)。

### 1.4 诚实的保留

NDT 是工程领域,大量工作发在期刊(NDT & E International、IEEE T-IM、Measurement、JNDE)和 QNDE 会议,不上 arXiv。所以「图像类 NDT + 深度学习」在期刊里其实不少(CNN 对 B-scan/radiograph 分类已较成熟)。但 **LLM/VLM 这个新角度,若真做了通常会发 arXiv**--因此「LLM + 超声 NDT」的空白是真实的,不是检索盲区造成的。可在大论文相关工作章节用期刊库(Scopus/Compendex)再补一次系统检索以坐实。

---

## 二、为什么 P2 的 0.593「最优」其实是机会而非天花板

### 2.1 三阶段轨迹

| 阶段 | 技术 | 非PP4 AUC | 结论 |
|---|---|---|---|
| P0 | SSF(谱-空-频)从零 | 0.542 | 跨试件过拟合;增强/DANN/多视角全失败 |
| P1 | 域内 SSL 掩码自编码器预训练 | 0.572 | 首个有效技术(+0.030) |
| P2 | 通用多模态 LLM(Qwen3.6-27B)零样本 | **0.593** | 三阶段最优;LoRA 视觉端微调 PP7 0.504→0.587 |

### 2.2 这是一场弱比赛的险胜

P0–P2 全部 AUC 都低(最优 0.593),说明任务本身极难(5 试件留一交叉验证,缺陷率 0.5%–76% 强分布偏移)。VLM 仅**小幅**领先 SSL,原因可定位:

**VLM 把 B-scan 当普通图片看**,靠自然图像的视觉先验硬吃,没有理解 B-scan 的物理含义--

- B-scan **纵轴是声程深度(时间)**,不是空间像素;
- **回波幅度 = 反射率**,需增益/TCG 归一化才有物理意义;
- 缺陷特征是**波场中的模式**(缺陷回波 vs 几何回波 vs 模式转换噪声),不是视觉纹理;
- **探头角度/focal law/波速** 决定成像几何,但这些信息完全没进入 P2 的 prompt。

这与 STEMGym 的教训一致:VLM 在非自然图像模态上会「看不懂」。**这个 gap 就是创新空间**--把 VLM 的「视觉先验」升级为「波动先验(wave-physics prior)」。

P2 的可解释性输出也佐证:VLM 能识别「底面回波」「亮指示」等结构,但仅停留在视觉描述,未做物理因果推断。把这一步补上,是直接的提升路径。

---

## 三、创新方向(按 novelty × 可行性排序,对接 P0/P1/P2)

### 方向 1:物理条件化 VLM(最干净的白地)★强推

不只喂 B-scan 图,同时把 **focal law、探头角度(71° 剪波)、材料声速(钢 ~3230 m/s 剪波)、空间分辨率(1 mm/波束)、偏移量、增益/归一化方式、试件几何**作为文本/结构化条件拼进 prompt,让模型像检验员一样推理。

- 文献依据:物理先验 VLM(SpatioLM / Lift3D-VLA / CausalPhys)在机器人领域已验证可行,**NDT 波动物理无人做**。
- 与 P2 连贯:在现有 Qwen3.6-27B + LoRA 框架上,把 prompt 从「Is there a defect? yes/no」扩展为带物理上下文的差分推理,起步成本极低。
- 本仓库已有的物理元数据(可直接用):探头角度 71°(G0)、49 波束、1 mm 分辨率、offset 80–85 mm、缺陷类型 6 类(porosity/LOF/slag/metallic/projections/cracks)、试件 ID。

### 方向 2:推理式缺陷鉴别(CoT 差分诊断)★强推

NDT 最难的不是「有没有异常」,而是**区分缺陷回波 vs 几何回波(底面/角/模式转换)**。arXiv:[2607.18325](https://arxiv.org/abs/2607.18325) 证明 VLM 会把「异常」误判为「危险」,这正是 NDT 误报根源。

- 做法:CoT 链显式做差分诊断,每步给出物理依据(声程→深度→是否落在熔合线/底面→回波形态→结论)。比 Rao [2607.10666](https://arxiv.org/abs/2607.10666) 的 CoT 蒸馏更进一步:他用 CoT 在**射线图**上分类,我们用**波动物理 CoT** 在**信号**上鉴别。
- 可用 frontier VLM 生成带物理理由的标注,再 answer-conditioning 蒸馏到小模型(直接复用 Rao 的思路)。

### 方向 3:信号原生 tokenization(接 ITFormer 底子)

现在 B-scan 被栅格化成 512×512 像素喂 VLM,丢掉了相位/频谱/包络信息。本仓库实际有更原始的资产:`ascans.npy`(3000×49×512)、`env.npy`(3000×512 包络)。

- 改成**物理意义 patch**(声程 bin × 频带 × 幅度),像 ITFormer/PatchTST 那样 patch 化喂 LLM,而不是裸像素。
- 直接复用本项目原本的 ITFormer 专长,把「时序-语言桥接」从 GMAW 电信号迁移到 A-scan,故事线天然连贯。

### 方向 4:跨试件泛化(直击 P0 痛点)

P0 LOOCV 负面说明像素统计跨试件不迁移。但 **VLM 的语义推理先验是域不变的**--「熔合线上的不规则回波」跨试件成立,而像素分布不成立。

- 做法:VLM 做**域不变推理先验** + 少样本物理适配,专门攻钢种/几何变化的场景。这正好是 P0 失败、P2 部分缓解的地方,做成即直接贡献。
- 评测必须用 5 折 LOOCV 非 PP4 AUC(P2 只做了 PP7 单折,需补全)。

### 方向 5:多模态 NDT 融合(LLM 当融合枢纽)

真实检验常是 UT + RT + ET + 表面视觉多模态,各模态物理完全不同。LLM 的语言/推理天然适合做异质模态融合的「推理枢纽」--目前没人用 LLM 这么做,是全新问题。可作为 P4/P5 长线方向。

### 方向 6:波动方程合成数据

用 CIVA 式波动方程仿真生成物理有效的合成缺陷,让 VLM 描述/标注,构建大规模预训练语料。**直接解决把所有人卡在 ~0.59 的数据稀缺问题**。可作为所有方向的共同支撑。

### 方向汇总

| 方向 | novelty | 可行性 | 与现有连贯性 | 建议阶段 |
|---|---|---|---|---|
| 1 物理条件化 VLM | 高 | 高 | 极连贯(P2 直接扩展) | **P3** |
| 2 推理式 CoT 鉴别 | 高 | 中高 | 连贯(P2 可解释性延伸) | **P3** |
| 3 信号原生 tokenization | 高 | 中 | 连贯(ITFormer 专长) | P3/P4 |
| 4 跨试件泛化 | 中高 | 中 | 直击 P0 痛点 | P3 评测/ P4 |
| 5 多模态融合 | 极高 | 低 | 需新数据 | P5 |
| 6 波动方程合成数据 | 中 | 中 | 支撑所有方向 | P3+/并行 |

---

## 四、P3 落地建议

**P3 = 方向 1 + 方向 2(+ 方向 4 的完整 LOOCV 评测)**

在现有 Qwen3.6-27B + LoRA 框架上:

1. **物理条件化**:把 focal law / 探头角度 / 声速 / 分辨率 / 缺陷类型候选拼进 prompt(方向 1)。
2. **推理式鉴别**:把「yes/no」单 token 打分,扩展为「差分诊断 → 结论」的 CoT 打分(方向 2);可选 answer-conditioning 蒸馏到小模型。
3. **完整 5 折 LOOCV 非 PP4 AUC 评测**(方向 4),与 P0/P1/P2 同口径对比。
4. 消融:有/无物理条件、有/无 CoT、灰度 vs 信号 patch。

**为什么是 1+2 组合**:与 P2 同模型栈、增量改动;文献空白最干净(物理条件化 VLM × 超声信号 = 0 篇);能讲清「0.593 不是天花板,因为之前没注入波动先验」;退可写成单篇「物理增强的多模态超声 NDT」,进可铺到 P4(信号原生 tokenization)、P5(跨模态融合)成一条线。

**成功判据**:非 PP4 AUC 显著超过 P2 的 0.593(目标 ≥0.62),且完整 5 折 LOOCV 稳健。

---

## 五、本仓库可复用资产

- **模型**:`models/Qwen3.6-27B`(Qwen3.5-VL,27.4B,bfloat16)。部署用 `.venv_p2`(transformers 5.14 + torch cu126 + flash-linear-attention,~0.3s/图)。**硬约束:CUDA 12.2(driver 535),不能用 vLLM 0.26(需 CUDA13),必须 transformers 直推**。
- **数据**:`data/processed/paut/`--`ascans.npy`(3000×49×512)、`ascans_mv.npy`(多视角 4 通道)、`env.npy`(3000×512 包络)、`meta_coupon.npy`(PP3–PP7)、`meta_pos.npy`、`meta_label.npy`、`meta_defect_type.npy`、`meta_summary.json`(含物理参数与缺陷类型码)、`images/`(6000 张已渲染图:bscan + spec)。
- **P2 代码(可直接扩展)**:`scripts/paut_render_images.py`(B-scan 渲染)、`scripts/paut_vlm_zeroshot.py`(零样本 QA 似然打分)、`scripts/paut_vlm_lora.py`(视觉端 LoRA)、`scripts/paut_vlm_describe.py`(可解释性)。
- **P2 结果**:`experiments/results/paut_vlm_zeroshot.json`(3000×2 分数)、`paut_vlm_zeroshot_summary.json`、`paut_vlm_lora.json`、`paut_vlm_describe.json`。
- **评测口径**:5 折 LOOCV(PP3–PP7 轮流 test),**非 PP4 AUC** 为可信指标(PP4 仅 3 正样本,AUC ±0.03 纯噪声)。主脚本 `scripts/paut_loocv.py`,汇总 `experiments/results/paut_loocv_table.md`。
- **物理上下文(用于条件化)**:G0 探头 71° 剪波;49 波束,1 mm/波束;offset 80–85 mm;钢剪波速 ~3230 m/s;缺陷类型 {1 孔/气孔,2 未熔合,3 夹渣,4 金属夹杂,5 凸起,6 裂纹}。

---

## 六、相关文献清单

- [arXiv:2607.10666](https://arxiv.org/abs/2607.10666) - Rao, Answer-Conditioned CoT Distillation for Few-Shot Industrial Vision with Small VLMs(焊缝射线照片,小 VLM CoT 蒸馏)
- [arXiv:2511.23177](https://arxiv.org/abs/2511.23177) - Li et al., Data-Efficient Motor Condition Monitoring with TSFMs(MOMENT/Mantis,电机振动)
- [arXiv:2608.01899](https://arxiv.org/abs/2608.01899) - SpatioLM,物理空间智能 VLM
- [arXiv:2607.06564](https://arxiv.org/abs/2607.06564) - Lift3D-VLA,3D 几何与动力学感知 VLA
- [arXiv:2606.05966](https://arxiv.org/abs/2606.05966) - CausalPhys,因果物理推理 VLM benchmark
- [arXiv:2606.29592](https://arxiv.org/abs/2606.29592) - STEMGym,VLM 在晶体缺陷上比 CNN 差 13×(模态 gap 警示)
- [arXiv:2607.18325](https://arxiv.org/abs/2607.18325) - Hazard or Anomaly,VLM 混淆异常与危险(NDT 误报警示)
- [arXiv:2506.20093](https://arxiv.org/abs/2506.20093) - ITFormer, ICML 2025(本项目基线范式:PatchTST 编码器 + 桥接 + 冻结 LLM)
