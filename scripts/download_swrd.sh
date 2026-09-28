#!/usr/bin/env bash
# S2: SWRD / 焊缝 RT (射线) 数据获取脚本 — F2/F3/F4/F5 前置。
#
# 首选: Zenodo 10618962 "X-ray weld seam image" (my_dataset.zip 475MB, apache-2.0,
#       免登录直链, 本机已验证可达)。
# 兜底: SWRD 官方 Google Drive (bit628/RapidX-Annotator README; 3,675 张焊缝底片 +
#       6 类缺陷 bbox 标注, 论文 DOI 10.1007/s10921-025-01186-w) — 需 gdown/代理。
# 两者都失败 → 退出码 2, 提示人工浏览器下载 (external_weld_ut Cloudflare 先例)。
#
# 用法: bash scripts/download_swrd.sh
# 输出: data/raw/swrd/<zip> + checksums.txt (sha256)
set -u
OUT_DIR="data/raw/swrd"
ZENODO_URL="https://zenodo.org/records/10618962/files/my_dataset.zip?download=1"
SWRD_GDRIVE_ID="1LNUt101wufTBJpRAgrZAU1h-Tfx629wO"   # SWRD folder (folder → 需 gdown --folder)

mkdir -p "$OUT_DIR"

download_zenodo() {
    local dest="$OUT_DIR/my_dataset.zip"
    echo "[zenodo] ${ZENODO_URL}"
    curl -sL --max-time 3600 -o "$dest" "$ZENODO_URL" || return 1
    python3 -m zipfile -t "$dest" 2>/dev/null || return 1
    return 0
}

download_gdrive() {
    local dest="$OUT_DIR/swrd_gdrive.zip"
    echo "[gdrive] 尝试 gdown (SWRD 官方)"
    python -m pip show gdown >/dev/null 2>&1 || python -m pip install -q gdown || return 1
    python -m gdown --folder "$SWRD_GDRIVE_ID" -O "$dest" || return 1
    return 0
}

if download_zenodo; then
    SRC="my_dataset.zip (Zenodo 10618962, apache-2.0)"
elif download_gdrive; then
    SRC="swrd_gdrive (SWRD 官方 Google Drive)"
else
    echo "自动下载失败。请人工浏览器下载:"
    echo "  1) https://zenodo.org/records/10618962 (my_dataset.zip)"
    echo "  2) SWRD: https://drive.google.com/drive/folders/${SWRD_GDRIVE_ID}"
    echo "放入 ${OUT_DIR}/ 后重跑预处理。"
    exit 2
fi

echo "[ok] 下载源: ${SRC}"
cd "$OUT_DIR" && sha256sum *.zip > checksums.txt && cat checksums.txt
