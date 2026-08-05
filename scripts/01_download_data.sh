#!/usr/bin/env bash
# Download the Zenodo "Metal Arc Welding" dataset (processed version only).
# Record: https://zenodo.org/records/10017718 (CC-BY-4.0, Hahn et al., Univ. Wuppertal)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"
mkdir -p "${RAW_DIR}"

BASE="https://zenodo.org/records/10017718/files"

echo "[1/3] Downloading processed_asimow_dataset.csv (~1.3 GB) ..."
wget -c -q --show-progress -O "${RAW_DIR}/processed_asimow_dataset.csv" \
    "${BASE}/processed_asimow_dataset.csv?download=1"

echo "[2/3] Downloading dataset README ..."
wget -c -q -O "${RAW_DIR}/dataset_README.md" "${BASE}/README.md?download=1"

echo "[3/3] Verifying MD5 checksums ..."
cd "${RAW_DIR}"
echo "8245bfaa33d432341a2a9e7d4b4082b5  processed_asimow_dataset.csv" | md5sum -c -
echo "f62cc2a44de41b99db75813ff8e5de08  dataset_README.md" | md5sum -c -

echo "Download + checksum OK."
