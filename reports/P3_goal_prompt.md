# P3 Goal Prompt(可派发给 agent)

> 把下面整段(从 `===` 开始)粘贴给一个新的 Claude Code 会话或子 agent 即可执行。

===

你是一名超声无损检测(NDT)+ 大模型方向的研究工程师。当前项目是「大模型 + 焊缝无损检测」,聚焦相控阵超声(PAUT)焊缝缺陷检测。已完成 P0–P2 三阶段,你要执行 **P3:物理条件化 + 推理式鉴别的多模态超声 NDT**。

## 背景(必读)

- 工作目录:`/home/yzx/doct/welding-ndt-timeseries`(git 仓库,主分支 main)。
- 三阶段现状(评估口径统一为 **5 折留一试件交叉验证 LOOCV,PP3–PP7 轮流做 test;非 PP4 AUC 为可信指标**,因 PP4 仅 3 正样本,AUC ±0.03 纯噪声):
  - P0:SSF 谱-空-频小模型从零,非PP4 AUC=0.542;物理增强/DANN/多视角全失败。
  - P1:域内 SSL 掩码自编码器预训练,非PP4 AUC=0.572(首个有效技术)。
  - P2:通用多模态 LLM Qwen3.6-27B 零样本 QA 似然打分,灰度 B-scan 非PP4 AUC=**0.593**(三阶段最优);LoRA 视觉端微调 PP7 单折 0.504->0.587。
- P2 的结论:通用 VLM 视觉先验可迁移到 PAUT,但只是弱比赛的险胜。根因是 VLM 把 B-scan 当普通图片看,没理解其物理含义(纵轴=声程深度,幅度=反射率,缺陷特征是波场模式而非视觉纹理,且 focal law/探头角度/声速完全没进 prompt)。
- 完整方向调研见 `reports/PAUT_方向调研_多模态LLM创新定位.md`,P2 细节见 `reports/PAUT_P2_多模态LLM实验报告.md`,三阶段汇总见 `reports/PAUT长期推进实验报告.md`。开始前先读这三份。

## P3 目标

在 Qwen3.6-27B + LoRA 框架上,把 P2 的「裸 B-scan 图 + yes/no 单 token 打分」升级为「**物理条件化 + 推理式差分诊断**」,验证非PP4 AUC 能否显著超过 P2 的 0.593(目标 ≥0.62),并用完整 5 折 LOOCV 证明稳健。

## 方法(三个组件,按顺序实现)

### 组件 A:物理条件化 prompt(必做)

把 P2 的 prompt 从单句「Is there a defect? yes/no」扩展为带物理上下文的检验员式 prompt,至少包含:
- 探头与几何:G0 探头 71° 剪波,49 波束,1 mm/波束,offset 80–85 mm,钢剪波速 ~3230 m/s。
- B-scan 语义:纵轴=声程(可换算深度),横轴=扫查位置,幅度=反射率(已百分位 2–98 归一化)。
- 试件与缺陷类型候选:试件 ID(PP3–PP7),缺陷类型 {1 气孔,2 未熔合,3 夹渣,4 金属夹杂,5 凸起,6 裂纹}。
- 任务:差分诊断,区分缺陷回波与几何回波(底面/角/模式转换)。

实现:新建 `scripts/paut_vlm_physics_zeroshot.py`,基于 `scripts/paut_vlm_zeroshot.py` 改 prompt 与条件拼装,复用其 transformers 直推 + yes/no 首 token logprob 差打分逻辑。物理参数可从 `data/processed/paut/meta_summary.json` 与 `meta_coupon.npy`/`meta_pos.npy` 动态读取,不要硬编码到无法泛化。

### 组件 B:推理式 CoT 差分诊断打分(必做)

把单 token 打分扩展为 CoT:让模型先输出差分诊断步骤(声程->深度->是否落在熔合线/底面->回波形态->结论),再给 yes/no。打分方式二选一(都做,消融对比):
- (B1)CoT 后末尾 yes/no 的 logprob 差;
- (B2)让模型输出结构化诊断 JSON(含 `reasoning` 与 `defect: yes/no`),按 `defect` 字段或末尾 token 打分。
可选:用 Qwen3.6-27B 自身对训练位置(已知标签)生成 answer-conditioned 的带理由描述,再用 LoRA 蒸馏(参考 Rao arXiv:2607.10666 思路)。时间不够则先做零样本 CoT,蒸馏列为 stretch。

### 组件 C:完整 5 折 LOOCV 评测(必做)

P2 只做了 PP7 单折微调,你必须补完整 5 折:
- 零样本:PP3–PP7 每折做 test,其余 4 折训练(零样本则全量推理),报告每折 AUC 与**非PP4 池化 AUC**。
- LoRA 微调:每折用训练折位置(可采样至 ~400–1000 位置控成本)微调视觉端(r=16,LLM 冻结,参考 `paut_vlm_lora.py`),test 折评测。
- 与 P0/P1/P2 同口径对比,写进 `experiments/results/paut_loocv_table.md`。

## 可用资产(直接复用,不要重造)

- 模型:`models/Qwen3.6-27B`(Qwen3.5-VL,27.4B)。**必须用 `.venv_p2`(transformers 5.14 + torch cu126 + flash-linear-attention)直推,~0.3s/图。硬约束:本机 CUDA 12.2(driver 535),禁止用 vLLM 0.26(需 CUDA13 会失败)**。
- 数据:`data/processed/paut/`--`ascans.npy`(3000×49×512)、`env.npy`(3000×512 包络)、`meta_coupon.npy`(PP3–PP7)、`meta_pos.npy`、`meta_label.npy`、`meta_defect_type.npy`、`meta_summary.json`、`images/`(6000 张:`{idx:05d}_bscan.png`/`_spec.png`,百分位归一化灰度 B-scan 已渲染好)。
- P2 代码(扩展起点):`scripts/paut_render_images.py`、`scripts/paut_vlm_zeroshot.py`、`scripts/paut_vlm_lora.py`、`scripts/paut_vlm_describe.py`。
- P2 结果(对比基线):`experiments/results/paut_vlm_zeroshot.json`、`paut_vlm_zeroshot_summary.json`、`paut_vlm_lora.json`、`paut_vlm_describe.json`。
- LOOCV 主脚本:`scripts/paut_loocv.py`(参考其折划分与指标实现,保持同口径)。

## 消融与对比

至少跑通以下对比(零样本为主,LoRA 视时间):
1. P2 baseline 复现(裸 prompt + 单 token):应 ~0.593,用于校准。
2. +物理条件(组件 A):看条件化是否提升。
3. +物理条件 + CoT(组件 A+B):完整方法。
4. (stretch)+ answer-conditioning 蒸馏:小模型 LoRA。
报告每配置的 5 折每折 AUC + 非PP4 池化 AUC + 与 P0/P1/P2 差值。

## 硬约束与规范

- **提交身份**:所有 commit 必须以 `严正兴 <85539067+Code-Yan-ZX@users.noreply.github.com>` 名义,**禁止任何 AI / Co-Authored-By 署名**。commit message 中英均可。
- 不要修改/破坏 P0–P2 已有代码与结果文件;新代码用新文件名(如 `paut_vlm_physics_*.py`),新结果存 `experiments/results/paut_vlm_physics_*.json`。
- README 用中文;实验报告放 `reports/`,命名 `PAUT_P3_物理条件化多模态LLM实验报告.md`。
- 凭证已在 `~/.git-credentials`,`git push` 直接用。不要把任何凭证写进会被提交的文件(CLAUDE.md 已 gitignore)。
- 控制显存与时间:Qwen3.6-27B 用 tensor_parallel 或 transformers 直推;若 5 折全量 3000 位置推理太久,可先在子集上验证 prompt 有效再全量。
- 诚实报告:若某配置不提升甚至退化,如实记录并分析原因,不要为凑结果而选择性报告。负面结果同样有价值(P0 就是负面结果)。

## 交付物

1. 代码:`scripts/paut_vlm_physics_zeroshot.py`(+ CoT/LoRA 变体)。
2. 结果:`experiments/results/paut_vlm_physics_*.json`(5 折分数 + 汇总)。
3. 汇总表更新:`experiments/results/paut_loocv_table.md` 加 P3 行。
4. 报告:`reports/PAUT_P3_物理条件化多模态LLM实验报告.md`,与 P0–P2 报告同结构(部署/方法/结果/消融/结论/产物)。
5. git 提交并 push。

## 成功判据

- 非PP4 AUC 显著超过 0.593(目标 ≥0.62),且完整 5 折 LOOCV 稳健(非PP4 每折不崩)。
- 若达成:这是「物理条件化多模态 VLM × 超声 NDT」的新组合(arXiv 上 0 篇),可作为论文核心贡献。
- 若未达成:给出清晰的失败原因分析(是 prompt 设计、CoT 漂移、还是 VLM 根本无法利用这些物理文本),并指出下一步(如转向方向 3 信号原生 tokenization)。无论正负,都要写成可发表的方法学结论。

开始前先读三份报告,然后给出你的执行计划(数据管线、prompt 模板草稿、评测脚本改动点、时间预算),再动手。

===
