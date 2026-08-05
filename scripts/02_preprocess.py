#!/usr/bin/env python
"""Stage 2 entrypoint: CSV -> memmap -> canonical splits -> norm stats.

Usage: python scripts/02_preprocess.py [--repo-root .] [--no-verify-counts]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wndt.data.preprocess import csv_to_memmap, spot_check, compute_norm_stats  # noqa: E402
from wndt.data.splits import build_splits  # noqa: E402
from wndt.utils.logging import get_logger  # noqa: E402

log = get_logger("preprocess")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--no-verify-counts", action="store_true")
    parser.add_argument("--skip-csv", action="store_true",
                        help="skip CSV->memmap conversion (waves.npy already built)")
    args = parser.parse_args()

    csv_path = args.repo_root / "data/raw/processed_asimow_dataset.csv"
    out_dir = args.repo_root / "data/processed"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} missing -- run scripts/01_download_data.sh")

    if not args.skip_csv:
        n = csv_to_memmap(csv_path, out_dir, chunksize=args.chunksize)
        log.info("wrote %d rows to %s", n, out_dir)
        spot_check(csv_path, out_dir)
    else:
        log.info("skipping CSV conversion")

    idx = build_splits(out_dir, verify_counts=not args.no_verify_counts)
    compute_norm_stats(out_dir, idx["train"])
    log.info("preprocessing done.")


if __name__ == "__main__":
    main()
