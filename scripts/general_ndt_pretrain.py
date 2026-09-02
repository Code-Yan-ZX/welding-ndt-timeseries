#!/usr/bin/env python
"""General NDT Foundation — M0 vanilla MAE 预训练入口 (Phase 2A smoke)。

用法:
  python scripts/general_ndt_pretrain.py [--config configs/general_ndt_mae_smoke.yaml]
                                        [--smoke] [--seed N]

- 只实现 M0: vanilla MAE + random mask; 多源/时频双视图/物理掩码 暂不启用。
- checkpoint 含数据集指纹, probe 时校验。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from general_ndt.datasets.registry import build_dataset          # noqa: E402
from general_ndt.models.mae import MaskedAutoencoder              # noqa: E402
from general_ndt.trainers.ssl_trainer import SSLTrainer, dataset_fingerprint  # noqa: E402

DEFAULT_CONFIG = REPO / "configs" / "general_ndt_mae_smoke.yaml"


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--smoke", action="store_true", help="覆盖为最小步数")
    ap.add_argument("--seed", type=int, default=None, help="覆盖 model/data seed")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.smoke:
        cfg["train"]["n_steps"] = min(cfg["train"]["n_steps"], 30)
        cfg["dataset_config"]["sample_limit"] = min(
            cfg["dataset_config"].get("sample_limit", 10**9), 40)
    if args.seed is not None:
        cfg["train"]["model_seed"] = args.seed
        cfg["train"]["data_seed"] = args.seed

    print(f"[pretrain] dataset={cfg['dataset']} steps={cfg['train']['n_steps']} "
          f"batch={cfg['train']['batch_size']}")
    samples = build_dataset(cfg["dataset"], cfg.get("dataset_config", {}))
    if not samples:
        print("[pretrain] 空数据集", file=sys.stderr)
        return 2
    print(f"[pretrain] loaded {len(samples)} samples; "
          f"specimens={len({s.specimen_id for s in samples})}")

    m = cfg["model"]
    model = MaskedAutoencoder(
        d_model=int(m["d_model"]), patch_len=int(m["patch_len"]), patch2d=int(m["patch2d"]),
        n_layers_enc=int(m["n_layers_enc"]), n_heads=int(m["n_heads"]),
        d_decoder=int(m["d_decoder"]), n_layers_dec=int(m["n_layers_dec"]),
        mask_ratio=float(m["mask_ratio"]), n_modalities=int(m.get("n_modalities", 8)),
        n_sensors=int(m.get("n_sensors", 32)), dropout=float(m.get("dropout", 0.0)),
    )
    tr = cfg["train"]
    trainer = SSLTrainer(
        model, dict(tr),
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    fp = dataset_fingerprint(samples)
    ckpt = trainer.train(
        samples, n_steps=int(tr["n_steps"]), batch_size=int(tr["batch_size"]),
        log_every=int(tr.get("log_every", 10)), ckpt_every=int(tr.get("ckpt_every", 500)),
        output_dir=cfg["output_dir"],
    )
    print(f"[pretrain] final checkpoint: {ckpt} (fingerprint={fp})")
    # 写元信息
    meta = {"dataset": cfg["dataset"], "fingerprint": fp, "config": cfg, "ckpt": str(ckpt)}
    (Path(cfg["output_dir"]) / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
