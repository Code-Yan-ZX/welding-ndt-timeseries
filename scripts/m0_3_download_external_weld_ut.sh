#!/usr/bin/env bash
# M0-3 真实焊缝多源超声数据下载脚本（Strathclyde Pure Portal，CC BY 4.0）。
#
# 下载 4 个数据源（A 316L lack-of-fusion FMC / B Inconel 82/182 FMC /
# C 304SS MMA 3mm SDH FMC / D PAUT 探头定位），全部支持断点续传（curl -C -）。
#
# ⚠ 已知限制（2026-08-25 实测）：
#   Pure Portal 的 /files/ 端点由 Cloudflare managed challenge 保护，curl/
#   wget/WebFetch 等无 JS 客户端一律 403（cf-mitigated: challenge）。本脚本在
#   普通服务器上会失败——失败时**不要绕过权限**，改用浏览器手工下载，或在与
#   浏览器同网段/有浏览器会话的机器上运行本脚本。
#
# 下载校验：完成后计算 md5 + sha256，写入 data/raw/external_weld_ut/checksums.txt
# （覆盖写，供审计脚本读取）。
#
# 用法：
#   bash scripts/m0_3_download_external_weld_ut.sh          # 下载全部
#   bash scripts/m0_3_download_external_weld_ut.sh A B C D  # 指定数据源
#   bash scripts/m0_3_download_external_welt_ut.sh --urls   # 只打印人工下载地址
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/data/raw/external_weld_ut"
mkdir -p "$DEST"

UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# 数据源：id | 子目录 | 页面 | 文件（URL 与大小）
declare -A SRC_URL SRC_DIR SRC_PAGE SRC_SIZE
add_src() { # id dir page size url...
  local id="$1" dir="$2" page="$3" size="$4"; shift 4
  SRC_DIR[$id]="$dir"; SRC_PAGE[$id]="$page"; SRC_SIZE[$id]="$size"
  SRC_URL[$id]="$*"
}
add_src A A "https://pureportal.strath.ac.uk/en/datasets/fmc-dataset-lack-of-fusion-crack-on-welded-316l-stainless-steel-p/" "260MB+53KB" \
  "https://pureportal.strath.ac.uk/files/46166162/Lack_of_fusion_FMC_DORT_2016.mat" \
  "https://pureportal.strath.ac.uk/files/46166161/Lack_of_fusion_FMC_DORT_2016.ods"
add_src B B "https://pureportal.strath.ac.uk/en/datasets/fmc-dataset-centreline-crack-in-inconel-82182-weld/" "33.1MB+8.9KB" \
  "https://pureportal.strath.ac.uk/files/46012157/FMC_2012_04_26_at_16_16.mat" \
  "https://pureportal.strath.ac.uk/files/46012288/12mm_verticle_crack_in_austenitic_weld_experimental_FMC_metadata_FMC_2012_04_26_AT_16_18.xlsx"
add_src C C "https://pureportal.strath.ac.uk/en/datasets/fmc-dataset-3mm-side-drilled-hole-304ss-mma-weld/" "44.9MB+10.1KB" \
  "https://pureportal.strath.ac.uk/files/67484223/FMC_RR3_2_25MHz_3mmsdh.mat" \
  "https://pureportal.strath.ac.uk/files/67484224/FMC_RR3_2_25MHz_3mmsdh_metadata.xlsx"
add_src D D "https://pureportal.strath.ac.uk/en/datasets/data-for-using-phased-array-ultrasound-to-localize-probes-during-/" "98.9MB" \
  "https://pureportal.strath.ac.uk/files/287644161/PAUT.zip"

if [[ "$*" == *--urls* ]]; then
  echo "人工下载地址（浏览器访问即可下载；放到 data/raw/external_weld_ut/<id>/ 下）："
  for id in A B C D; do
    echo "== [$id] ${SRC_PAGE[$id]} =="
    for u in ${SRC_URL[$id]}; do echo "  $u"; done
  done
  exit 0
fi

IDS=("$@"); [ ${#IDS[@]} -eq 0 ] && IDS=(A B C D)
CHECKSUM_FILE="$DEST/checksums.txt"
: > "$CHECKSUM_FILE"

# 按扩展名校验 magic bytes：.mat = MATLAB5/HDF5；.zip/.xlsx/.ods = PK
sig_ok() { # <file> <name>
  local f="$1" n="$2"
  case "$n" in
    *.mat)
      local magic; magic="$(head -c 8 "$f" | od -An -tx1 | tr -d ' \n')"
      # MATLAB 5.0 文本头 (4d 61 74 4c 41 42 = "MatLAB") 或 HDF5 (\x89HDF)
      [[ "$magic" == 4d* ]] || [[ "$magic" == *484446* ]];;
    *.zip|*.xlsx|*.ods)
      [[ "$(head -c 2 "$f")" == "PK" ]];;
    *)
      # 未知扩展名：拒绝纯 HTML（<）与过小文件
      [[ "$(head -c 1 "$f")" != "<" ]] && [[ "$(stat -c%s "$f")" -gt 1000 ]];;
  esac
}

for id in "${IDS[@]}"; do
  dir="$DEST/${SRC_DIR[$id]}"; mkdir -p "$dir"
  echo "== [$id] $(basename "${SRC_PAGE[$id]}") (${SRC_SIZE[$id]}) =="
  for u in ${SRC_URL[$id]}; do
    fn="$(basename "$u")"
    echo "  -> $fn"
    # 断点续传；--retry 处理瞬时失败
    if ! curl -sS -L -C - --retry 3 --retry-delay 2 -A "$UA" \
         -H "Accept: application/octet-stream,*/*;q=0.5" \
         -H "Referer: ${SRC_PAGE[$id]}" \
         -o "$dir/$fn" "$u"; then
      echo "    [FAIL] $fn 传输失败。请浏览器手工下载。"
      rm -f "$dir/$fn"
      continue
    fi
    # 校验文件签名（防 Cloudflare challenge HTML 冒充真实文件）
    if ! sig_ok "$dir/$fn" "$fn"; then
      echo "    [FAIL] $fn 内容非预期文件（Cloudflare challenge/HTML 403 页面），已删除。"
      rm -f "$dir/$fn"
      continue
    fi
    md5="$(md5sum "$dir/$fn" | awk '{print $1}')"
    sha="$(sha256sum "$dir/$fn" | awk '{print $1}')"
    sz="$(stat -c%s "$dir/$fn")"
    echo "$id|$fn|$sz|$md5|$sha" >> "$CHECKSUM_FILE"
    echo "    ok $fn  size=$sz  md5=$md5  sha256=$sha"
  done
done
echo "checksums -> $CHECKSUM_FILE"
