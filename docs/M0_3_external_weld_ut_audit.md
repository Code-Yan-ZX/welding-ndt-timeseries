# M0-3 外部真实焊缝超声数据审计（Strathclyde FMC/PAUT）

> 审计对象：`data/raw/external_weld_ut/{A,B,C,D}/`（Strathclyde Pure Portal，CC BY 4.0）
> 审计脚本：`scripts/m0_3_external_weld_ut_audit.py`
> 下载清单 / 校验和 / 许可证：`data/manifests/external_weld_ut/download_manifest.json`
> 状态：**数据未下载（Cloudflare challenge 阻止自动下载），以下为网页级审计 +
> 审计基础设施 + 数据落地后的执行说明**。更新日期 2026-08-25。

## 1. 下载状态与失败记录

Pure Portal 的 `/files/` 端点由 Cloudflare **managed challenge**
（`cf-mitigated: challenge`）保护，无 JS 客户端一律 HTTP 403。自动下载全部
尝试均失败（详见 `download_manifest.json` 的 `attempts`）：

| 方法 | 结果 |
|---|---|
| curl（完整浏览器 header + cookie jar + referer） | 403 challenge HTML（~5-6KB） |
| WebFetch | 403 Forbidden |
| Firefox snap headless（用户批准） | 环境内启动即挂死（snap 沙箱），无法使用 |
| Playwright Chromium headless（本机代理 127.0.0.1:7890） | 挑战页不自动通过（无 `cf_clearance`），浏览器间歇 EPIPE 崩溃，下载事件永不触发 |
| Xvfb headed 模式 | 环境无 Xvfb |

按项目纪律不做 stealth/exploit 绕过。**需浏览器人工下载**：打开各 `source.page`
点 Download（公开数据、无需登录），或 `bash scripts/m0_3_download_external_weld_ut.sh --urls`
打印全部直链，按 `files[].path` 放入 `data/raw/external_weld_ut/` 后重跑审计。

## 2. 数据源网页级信息（下载前已知）

| id | 数据集 | DOI | 文件（大小） | 材料 / 焊缝 | 缺陷 | 探头发数 |
|---|---|---|---|---|---|---|
| A | Lack of fusion on welded 316L | 10.15129/086404bd-eb69-429b-978c-2c35cdbfcf87 | `Lack_of_fusion_FMC_DORT_2016.mat` (260MB) + `.ods` (53KB) | 316L 奥氏体不锈钢板焊缝 | lack-of-fusion 裂纹（与 x 轴 50°） | 页面上未给出，见 `.ods` 元数据 |
| B | Centreline crack in Inconel 82/182 | 10.15129/179e1b38-e701-443d-b995-a4449851330c | `FMC_2012_04_26_at_16_16.mat` (33.1MB) + `.xlsx` (8.86KB) | Inconel 82/182 焊缝（右 316L / 左碳钢+182 覆层） | 中心线竖向粗糙裂纹 12mm、内嵌距换能器面 37mm | 见 `.xlsx` 元数据 |
| C | 3mm SDH 304SS MMA weld | 10.15129/60b6a5b8-e78e-4742-8414-aaba9399a9c8 | `FMC_RR3_2_25MHz_3mmsdh.mat` (44.9MB) + `.xlsx` (10.1KB) | 304SS 含 MMA 焊缝 | 3mm 直径侧钻孔（SDH） | 文件名提示 2.25MHz；iNEED 项目 EP/P005268/1 |
| D | PAUT probe localisation | 10.15129/bfb5a77d-dabe-4be4-82c9-b10e8c237dea | `PAUT.zip` (98.9MB) | 焊缝（材料未声明） | 无（探头定位 / 焊缝材料识别） | zip 内格式待下载后审计；Gilmour et al. IEEE OJ I&C 2023 |

- 许可证：全部 CC BY 4.0；发布者 University of Strathclyde。
- 相关论文：Cunningham et al., Proc. R. Soc. A 2016 (10.1098/rspa.2015.0500)；
  Tant et al., Acta Acustica 2017 (10.3813/AAA.919125)。

## 3. 审计清单与执行方式（数据落地后）

`python scripts/m0_3_external_weld_ut_audit.py --full`（可选 `--probe <file>` 单文件）
输出结构探测 + 全量聚合到 `experiments/results/m0_3_external_weld_ut_audit_full.json`，
覆盖：

1. **文件格式 / MATLAB-HDF5 schema / dtype / shape / NaN / Inf**：
   `--probe` 递归展开 v5（scipy）/ v7.3（h5py）变量树，每数组给 min/max/mean/
   std/NaN/Inf；聚合按 dataset path 汇总 shape/dtype 集合。
2. **Tx / Rx / time / scan position / TFM-PAUT image 维度含义**：审计后按真实
   变量名与 shape 逐轴标注（`detect_fmc_arrays` 已内置 (Tx,Rx,T) 探测）。
3. **探头参数、采样率、材料、焊缝类型、缺陷类型、缺陷位置与尺寸**：A 的
   `.ods` / B、C 的 `.xlsx` / D 的 zip 内元数据解析到 dataset card。
4. **真正独立的物理 specimen 数与 acquisition 数**：只按物理试件/采集配置
   计数；网页/文件内的多个信号、通道、Tx、scan 位置**不计为独立试件**。
5. **group_id**：同一试件的 Tx×Rx、scan position、重复扫查**必须共享
   group_id**（adapter 默认一个 .mat = 一个 group，审计确认后再细分）。
6. **哈希、近重复与元数据重复检查**：sha256 逐文件 + 同 source 内重复分组。
7. **每源标记**：`ssl_pretrain_usable` / `downstream_label_usable` /
   `metadata_only` / `incompatible`。
8. **明确"多个信号/通道 ≠ 多个独立试件"**：写入 audit 输出与 dataset card
   `data_policy.forbid`。
9. **<10 独立试件标注 exploratory**：adapter 自动设
   `data_policy.exploratory = n_specimens < 10`。

## 4. 当前可用信息（合成数据 smoke 验证）

由于真实数据未落地，用**合成 FMC .mat**（每源 1 试件，Tx×Rx×T）验证了完整
链路：audit 脚本、`ExternalWeldUTAdapter`（24 views / 3 groups，exploratory=True）、
`build_manifest`（`data/manifests/external_weld_ut/dataset_card.json`）、
P-long/W→P 预训练、Protocol V2 LOOCV、聚合判据——**全部通过 smoke**（20 steps）。

⚠ 合成数据**不是**真实审计结果；真实数据落地后必须重跑 audit 并回填本节。
`dataset_card.json` 的 `n_specimens / n_records / 材料 / 焊缝 / 缺陷` 目前是
合成占位，真实审计后重建。

## 5. 预期结论口径（无论审计结果如何）

- 这些 Strathclyde 数据集大概率每个 `.mat` 只有 **1 个物理试件**（A/B/C 合计
  约 3 个独立试件，D 未知）——**远少于 10**，因此：
  - 正式报告必须标注 **exploratory external pretraining source**，
    不称为 foundation-scale dataset；
  - 仅可做低成本外部预训练 pilot（若下载成功），结论必须保持 exploratory；
  - **禁止用 Tx×Rx、scan position、切片数包装成"大规模真实试件数据"**。
- 若下载持续被 Cloudflare 阻止，M0-3 的真实数据实验无法进行，需人工下载后
  再继续（adapter / 预训练 / 评估 / 聚合 / 测试均已就绪）。
