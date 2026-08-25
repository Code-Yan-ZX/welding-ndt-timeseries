#!/usr/bin/env python3
"""M0-3 Strathclyde 真实焊缝超声数据下载（Firefox 无头浏览器，用户已批准）。

Pure Portal 的 /files/ 端点由 Cloudflare managed challenge 保护，curl/wget 等
无 JS 客户端一律 403。本脚本用真实 Firefox（headless）访问页面，让浏览器执行
challenge JS 后自动下载公开的 CC BY 4.0 数据——等价于真人浏览器下载。

- 逐文件下载，Firefox 内置断点语义由 download dir + 完成轮询保证（失败重试）；
- 下载完成后按 magic bytes 校验（防 challenge HTML 冒充），再算 md5/sha256；
- 校验和追加到 data/raw/external_weld_ut/checksums.txt。

用法:
  python scripts/m0_3_download_firefox.py            # 全部 A/B/C/D
  python scripts/m0_3_download_firefox.py A C        # 指定数据源
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEST = REPO / "data" / "raw" / "external_weld_ut"
MANIFEST = REPO / "data/manifests/external_weld_ut/download_manifest.json"
CHECKSUM_FILE = DEST / "checksums.txt"

# 数据源 -> 文件（id -> [(filename, url), ...]）
FILES = {
    "A": [
        ("Lack_of_fusion_FMC_DORT_2016.mat",
         "https://pureportal.strath.ac.uk/files/46166162/Lack_of_fusion_FMC_DORT_2016.mat"),
        ("Lack_of_fusion_FMC_DORT_2016.ods",
         "https://pureportal.strath.ac.uk/files/46166161/Lack_of_fusion_FMC_DORT_2016.ods"),
    ],
    "B": [
        ("FMC_2012_04_26_at_16_16.mat",
         "https://pureportal.strath.ac.uk/files/46012157/FMC_2012_04_26_at_16_16.mat"),
        ("12mm_verticle_crack_in_austenitic_weld_experimental_FMC_metadata_FMC_2012_04_26_AT_16_18.xlsx",
         "https://pureportal.strath.ac.uk/files/46012288/12mm_verticle_crack_in_austenitic_weld_experimental_FMC_metadata_FMC_2012_04_26_AT_16_18.xlsx"),
    ],
    "C": [
        ("FMC_RR3_2_25MHz_3mmsdh.mat",
         "https://pureportal.strath.ac.uk/files/67484223/FMC_RR3_2_25MHz_3mmsdh.mat"),
        ("FMC_RR3_2_25MHz_3mmsdh_metadata.xlsx",
         "https://pureportal.strath.ac.uk/files/67484224/FMC_RR3_2_25MHz_3mmsdh_metadata.xlsx"),
    ],
    "D": [
        ("PAUT.zip",
         "https://pureportal.strath.ac.uk/files/287644161/PAUT.zip"),
    ],
}

LANDING = {
    "A": "https://pureportal.strath.ac.uk/en/datasets/fmc-dataset-lack-of-fusion-crack-on-welded-316l-stainless-steel-p/",
    "B": "https://pureportal.strath.ac.uk/en/datasets/fmc-dataset-centreline-crack-in-inconel-82182-weld/",
    "C": "https://pureportal.strath.ac.uk/en/datasets/fmc-dataset-3mm-side-drilled-hole-304ss-mma-weld/",
    "D": "https://pureportal.strath.ac.uk/en/datasets/data-for-using-phased-array-ultrasound-to-localize-probes-during-/",
}


def sig_ok(path: Path) -> bool:
    """magic bytes 校验：.mat = MATLAB5/HDF5；.zip/.xlsx/.ods = PK。"""
    name = path.name
    if name.endswith(".mat"):
        head = path.read_bytes()[:8]
        return head.startswith(b"MatLAB") or b"HDF" in head
    if name.endswith((".zip", ".xlsx", ".ods")):
        return path.read_bytes()[:2] == b"PK"
    return False


def wait_download(dl_dir: Path, target: str, timeout: int = 600) -> Path | None:
    """等待 dl_dir 中出现 target（无 .part）且 5s 内大小不再增长。"""
    t0 = time.time()
    last_size = -1
    stable = 0
    while time.time() - t0 < timeout:
        parts = list(dl_dir.glob(f"{target}.*")) + list(dl_dir.glob("*.part"))
        for p in parts:                 # 清理残留 .part
            p.unlink(missing_ok=True)
        final = dl_dir / target
        if final.exists():
            sz = final.stat().st_size
            if sz > 1000 and sig_ok(final):
                if sz == last_size:
                    stable += 1
                else:
                    stable = 0
                last_size = sz
                if stable >= 3:         # ~5s 稳定
                    return final
            else:
                final.unlink(missing_ok=True)   # challenge HTML / 坏文件，删除重试
        time.sleep(1.5)
    return None


def main() -> int:
    ids = sys.argv[1:] or list(FILES)
    DEST.mkdir(parents=True, exist_ok=True)

    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options

    dl_dir = DEST / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    opts = Options()
    opts.add_argument("-headless")
    opts.set_preference("browser.download.dir", str(dl_dir))
    opts.set_preference("browser.download.folderList", 2)
    opts.set_preference("browser.download.useDownloadDir", True)
    opts.set_preference("browser.download.manager.showWhenStarting", False)
    opts.set_preference("browser.helperApps.neverAsk.saveToDisk",
                        "application/octet-stream,application/zip,"
                        "application/x-zip-compressed,application/x-msdownload,"
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet,application/vnd.oasis.opendocument."
                        "spreadsheet,application/vnd.ms-excel,text/csv")
    opts.set_preference("browser.helperApps.alwaysAsk.force", False)
    opts.set_preference("pdfjs.disabled", True)
    opts.set_preference("security.fileuri.strict_origin_policy", False)
    opts.set_preference("dom.disable_beforeunload", True)

    print(f"[firefox] headless download -> {DEST}")
    results = {}
    with webdriver.Firefox(options=opts) as drv:
        drv.set_page_load_timeout(180)
        for sid in ids:
            files = FILES[sid]
            out_dir = DEST / sid
            out_dir.mkdir(parents=True, exist_ok=True)
            results[sid] = {}
            print(f"== [{sid}] {len(files)} file(s) ==")
            try:
                drv.get(LANDING[sid])
                time.sleep(3)           # 先拿页面 cookie / 通过基础 challenge
            except Exception as e:
                print(f"  [warn] landing {sid}: {e}")
            for fn, url in files:
                t0 = time.time()
                got = None
                for attempt in range(3):
                    try:
                        drv.get(url)                    # 触发 challenge+下载
                    except Exception:
                        pass
                    got = wait_download(dl_dir, fn, timeout=600)
                    if got:
                        break
                    print(f"  retry {fn} (attempt {attempt + 1})")
                if not got:
                    results[sid][fn] = {"ok": False,
                                        "reason": "firefox 未在超时内下载成功（challenge 未过或网络）"}
                    print(f"  [FAIL] {fn}")
                    continue
                # 移动 + 校验和
                final = out_dir / fn
                got.replace(final)
                md5 = hashlib.md5(final.read_bytes()).hexdigest()
                sha = hashlib.sha256(final.read_bytes()).hexdigest()
                sz = final.stat().st_size
                with CHECKSUM_FILE.open("a") as fh:
                    fh.write(f"{sid}|{fn}|{sz}|{md5}|{sha}\n")
                results[sid][fn] = {"ok": True, "size": sz, "md5": md5,
                                    "sha256": sha, "wall_s": round(time.time() - t0, 1)}
                print(f"  ok {fn} size={sz} md5={md5[:12]}… ({results[sid][fn]['wall_s']}s)")

    print("\n=== 结果 ===")
    n_ok = sum(1 for r in results.values() for v in r.values() if v.get("ok"))
    n_tot = sum(len(v) for v in results.values())
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"ok {n_ok}/{n_tot}")
    return 0 if n_ok == n_tot else 1


if __name__ == "__main__":
    sys.exit(main())
