# Phase 3: SWRD / 焊缝 RT（射线）数据集准入

> 分支：`research/general-ndt-foundation`，日期：2026-09-11
> 目的：为 F 系列（F2–F5）引入焊缝射线（RT/X-ray）无标签预训练语料，支撑
> 「同缺陷语义、不同物理模态」迁移检验。**判定目标 tier B（仅无标签预训练，禁止评测）。**
> 原则（沿用 phase2 准入矩阵）：无法确认就保持，不得猜测升级。

## 1. 候选数据集

| 项 | 首选：Zenodo 10618962 | 备选：SWRD 官方 |
|---|---|---|
| 名称 | X-ray weld seam image | SWRD（Standard Weld Radiographic Defects） |
| 来源 | https://zenodo.org/records/10618962 | Google Drive（bit628/RapidX-Annotator README 链接） |
| 规模 | my_dataset.zip 475MB（内容待审计） | 3,675 张焊缝底片，6 类缺陷，bbox 标注 |
| License | **apache-2.0**（Zenodo 元数据） | 待核实（论文 DOI 10.1007/s10921-025-01186-w） |
| 获取 | curl 直链（本机已验证可达） | gdown / 人工浏览器 |

## 2. 审计项（下载后逐项填写）

- [ ] 内容结构（目录/文件格式/图像数量/分辨率分布）
- [ ] 标注：bbox 或类别标签的存在性与格式（SWRD: 6 类缺陷）
- [ ] **影片数 ≠ 独立试件数**：radiograph 与物理焊缝的对应关系不明 → 即使只做
      无标签预训练也必须写明「样本数 ≠ 独立试件数」，禁止未来据其做评测或
      foundation-scale 声明
- [ ] 灰度方向（底片负片：缺陷偏亮/偏暗）与逐图自适应反相禁令
- [ ] license 落实（Zenodo apache-2.0 已确认；若用 SWRD 需核实其 license）

## 3. 判定

（待下载审计后填写：tier B / 候选 B 阻塞 / 拒绝）

## 4. 与 F 系列的关系

- F2–F5 依赖本准入结论为 tier B；否则 F2–F5 全部暂停，F1 不受影响。
- RT 为 2D 图像模态（新增 `radiographic`），走 Stem2D 通路；bbox 标注仅用于
  F4 缺陷感知掩码（region-biased mask），不参与任何监督训练。
- 下载方式：`bash scripts/download_swrd.sh`（Zenodo 优先 → gdown 兜底 → 退出码 2 人工下载）。
