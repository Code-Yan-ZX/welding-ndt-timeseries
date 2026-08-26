# M0-3 外部真实焊缝超声数据审计（Strathclyde FMC/PAUT）

> 审计对象：`data/raw/external_weld_ut/{A,B,C,D}/`（Strathclyde Pure Portal，CC BY 4.0）
> 审计脚本：`scripts/m0_3_external_weld_ut_audit.py --full`
> 下载清单 / 校验和 / 许可证：`data/manifests/external_weld_ut/download_manifest.json`
> 状态：**真实数据已落地（浏览器人工下载，2026-08-26），结构审计完成**。
> 全量 JSON：`experiments/results/m0_3_external_weld_ut_audit_full.json`

## 1. 下载记录

Pure Portal `/files/` 端点由 Cloudflare managed challenge 保护，自动下载全部
失败（curl / WebFetch / Firefox snap / Playwright Chromium headless），已按纪律
记录（见 `download_manifest.json`）。**最终由浏览器人工下载完成**，文件落地并
归位如下（md5/sha256 记录在 `data/raw/external_weld_ut/checksums.txt`）：

| 源 | 文件 | 大小 | md5 |
|---|---|---|---|
| A | `A/Lack_of_fusion_FMC_DORT_2016.mat` | 272,341,239 | `146afc07bcf4c9a8074f34f62dd75be1` |
| B | `B/FMC_2012_04_26_at_16_16.mat` | 34,699,543 | `e11add4c23befb334ec14f4db9875501` |
| C | `C/FMC_RR3_2_25MHz_3mmsdh.mat` | 47,060,975 | `e790588f491fdd64d33e8b91fc61ee72` |
| D | `D/PAUT.zip` | 103,673,425 | `582815382de6cd6926954ee04ef67cad` |

⚠ 配套元数据（A 的 `.ods`、B/C 的 `.xlsx`）未随下载落地；探头参数（阵元数/频率/
采样率）部分可从文件推断（见 §3），其余需查论文补全——**不影响 SSL 预训练**
（SSL 只用 FMC/B-scan 结构，不需要绝对时间/探头参数）。

## 2. 真实文件结构（逐文件，`--probe` 确认）

| 源 | 变量 | 原始 shape | dtype | 规范化布局 | 内容 |
|---|---|---|---|---|---|
| A | `FMC_new` | (10000, 128, 128) | int32 | **(Tx=128, Rx=128, T=10000)** | 316L lack-of-fusion 全矩阵（时间在 axis0，平滑度 43 vs 206/192） |
| B | `FMC_new` | (10000, 45, 45) | int32 | **(Tx=45, Rx=45, T=10000)** | Inconel 82/182 中心线裂纹全矩阵（平滑度 131 vs 1175/1172） |
| C | `fmc` | (128, 128, 976) | float64 | **(Tx=128, Rx=128, T=976)** | 304SS MMA 3mm SDH 全矩阵（时间在 axis-1，平滑度 431 vs 1287/1315） |
| D | `PAUT.zip` → 389×`.txt` | 每个 (762, 401) | float64(文本) | **(beam=401, time=762)** ×389 | 相控阵 B-scan 图像（time×beam，能量随深度衰减） |

- **A/B 时间轴判定**：axis0 相邻采样 mean|diff| 远小于阵元轴（43 vs 206/191、
  131 vs 1175/1172）→ 时间在 axis0；对角线（tx==rx）能量比 2.9×/5.3× 确认
  pulse-echo 全矩阵结构。
- **C**：时间在 axis-1（431 vs 1287/1315）；`n_nan=0, n_inf=0`。
- **D**：762 = 时间/深度（能量 485→26 随行衰减），401 = beam；A01–M30 命名 =
  13 字母 × 30（A 组 29 个）；相邻字母首图相关性 0.57–0.80 → **同一焊缝的
  多位置扫查**（非 13 个独立试件）。

## 3. 探头 / 采样参数（文件内无显式字段，推断）

| 源 | 阵元数 | 时间采样 | 推断采样窗口 | 备注 |
|---|---|---|---|---|
| A | 128 | 10000 | ~100µs @ 100MHz（推断） | 页/论文未给确切值，需 `.ods` |
| B | 45 | 10000 | 同上 | 需 `.xlsx` |
| C | 128 | 976 | 短窗口（2.25MHz 探头） | 文件名 2_25MHz |
| D | —（成像 beam 401） | 762 | 未给出 | 需论文（Gilmour 2023） |

## 4. 独立试件 / 采集 / group_id（核心纪律）

| 源 | 独立物理试件 | 独立采集数 | view 数 | group_id |
|---|---|---|---|---|
| A | **1**（316L 焊缝板，known lack-of-fusion） | 1 | 128（每 Tx 1 view） | `external_weld_ut:A:spec1` |
| B | **1**（Inconel 焊缝，known 中心线裂纹） | 1 | 45 | `external_weld_ut:B:spec1` |
| C | **1**（304SS MMA 块，3mm SDH） | 1 | 128 | `external_weld_ut:C:spec1` |
| D | **1**（焊缝，多位置 PAUT 扫查） | 389（每 txt 1 次采集） | 389 | `external_weld_ut:D:spec1` |
| **合计** | **4** | 392 | **690** | 4 |

- **Tx×Rx / 文件编号 / scan position 全部共享 group_id，禁止当独立试件**
  （adapter 与 dataset card `data_policy.forbid` 显式声明）。
- 独立试件 **4 < 10** → 按 M0-3 §三.9 标注 **exploratory external pretraining
  source**，不称为 foundation-scale dataset。
- 哈希/近重复检查：4 个文件 sha256 互不相同，`sha256_dups` 为空。

## 5. 每源标记（ssl_pretrain_usable / downstream_label_usable）

| 源 | ssl_pretrain_usable | downstream_label_usable | 说明 |
|---|---|---|---|
| A | ✅ | ❌ | 真实 FMC；仅整试件已知缺陷（无 per-position 标签） |
| B | ✅ | ❌ | 同上 |
| C | ✅ | ❌ | 同上（SDH 参考反射体，非缺陷标签） |
| D | ✅ | ❌ | 真实 PAUT B-scan；探头定位数据，无缺陷标签 |
| E(ndeformat) | 未接入 | — | 需确认下载地址/许可证/原始信号，本轮不阻塞 |

结论：**全部 4 源可作 SSL 预训练素材；均无下游逐位置缺陷标签**（与目标域 PAUT
PP3–PP7 有逐位置标签不同）——因此外部阶段只做无监督 SSL，不做监督适配。

## 6. 输入表示与预处理（M0-3 §四，已按真实结构实现）

- FMC：规范化 `(Tx, Rx, T)`，**每 transmit event = 1 个 view (Rx, T)**（保留
  接收阵元×时间二维物理结构，不 flatten）；view 继承 specimen/group_id。
- PAUT(D)：每 txt = 1 个 view `(beam, time)`（保持 beam×time 二维表示）。
- **变长输入**：按 `(Rx, T)` bucket 同批；`MAX_FMC_H=128, MAX_FMC_W=2048`
  等比例下采样（预先声明）：A→(26,2000)、B→(9,2000)、C→(128,976)、D→(101,191)。
- 每 view 独立 median/MAD robust 归一化；block=16×16 mask；recon loss 只算
  masked∩valid。
- 共享 P1 `MAEEncoder`；数据源专用 `FlexDecoder`（阶段 2 新建 PAUT decoder，
  不迁移外部 decoder）。

## 7. 审计执行记录

```bash
python scripts/m0_3_external_weld_ut_audit.py --probe <file>   # 单文件
python scripts/m0_3_external_weld_ut_audit.py --full            # 全量
python -c "from wndt.data.adapters.external_weld_ut import build_manifest; build_manifest()"
```
