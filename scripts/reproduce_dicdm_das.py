#!/usr/bin/env python
"""B2: Reproduce DAS damage imaging of Yang et al., MSSP 236 (2025) 112996, §5.1/§6.1.

Pipeline (paper):
  raw PZT pitch-catch (36 paths x 7 freqs, 2000 samples @ 1.2 MHz)
  -> bandpass around the 250 kHz excitation (paper Fig 7c: 230-270 kHz)
  -> Hilbert envelope (paper §5.1.1: S0 first-arrival processing)
  -> DAS on a 152x178 px grid (Eq 1, omega = 1), delays from actuator/pixel/
     receiver distances and the S0 group velocity
  -> average the 6 per-actuator images (paper §6.1, Fig 12b)

Key audit finding baked in here (see APPROX): the paper's velocity field comes
from ABAQUS SIMULATION of each layup (Fig 8), NOT from the experimental signals;
and the dataset's frequency labels are 10x the stored-signal scale, so the
paper's 230-270 kHz band is applied as 20-30 kHz at the data's 1.2 MHz scale.

Every choice the paper leaves open is recorded in APPROX and in the output JSON.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert, sosfilt

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/raw/nasa_cfrp/2. Composites/L1_S11_F"
OUT = REPO / "experiments/results/dicdm/das_reproduction"
OUT.mkdir(parents=True, exist_ok=True)

CFG = {
    "coupon": "L1S11",
    "excitation_khz": 250.0,          # paper: DAS at center frequency 250 kHz
    "bandpass_khz": [20.0, 30.0],     # paper Fig 7c 230-270 kHz -> data scale /10,
                                      # widened to the 5-cycle burst bandwidth
    "grid_wh": [152, 178],            # paper Fig 5b, 1 px = 1 mm
    "cycles": [1, 10000, 20000, 30000, 40000],   # paper Fig 13 / Table 4
    "condition_filter": "loaded",     # paper does not state; Loaded = in-fatigue state
    "velocity_ms": 5200.0,            # paper §5.1.2/Fig 8a (simulation-based); Fig 8a
                                      # (L1 quasi-isotropic) reads ~4.5-6 km/s
    "direct_gate_us": 8.0,            # gate margin after the direct arrival
    "seed": 0,                        # DAS is deterministic; kept for traceability
}

# sensor geometry, mm, in the 152x178 imaging frame (origin = top-left).
# UNCERTAINTY: paper gives no numeric coordinates. Assumption: 6 PZT evenly spaced
# (pitch 25.4 mm; 6 x 25.4 = 152.4 mm plate width), actuators on the top edge,
# receivers on the bottom edge; actuator 1 at right (paper Fig 5b: "6 5 4 3 2 1"
# left->right on top; receivers "7..12" left->right on bottom).
_X = np.array([12.7 + 25.4 * k for k in range(6)])  # left->right
CFG["actuator_xy"] = {str(i + 1): [float(_X[5 - i]), 0.0] for i in range(6)}
CFG["receiver_xy"] = {str(i + 7): [float(_X[i]), 178.0] for i in range(6)}

APPROX = {
    "frequency_scale": "VERIFIED: paper kHz = 10 x stored-signal scale. Actuator "
                       "reference for file frequency==250 peaks at 25.2 kHz "
                       "(fs=1.2 MHz); a data-scale 23-27 kHz band is the literal "
                       "mapping of the paper's 230-270 kHz.",
    "velocity_source": "paper §5.1.2: Fig 8 velocity curves are ABAQUS simulations of "
                       "each layup; experimental signals are NOT used to estimate "
                       "velocity. Experimental first arrivals here are multi-modal/"
                       "dispersive (no consistent single-mode ToF), so the paper's "
                       "own choice (simulation velocity) is reproduced: uniform "
                       "5.2 km/s from Fig 8a reading; sensitivity 4.4/5.6 km/s.",
    "sensor_geometry": "even 25.4 mm pitch, rows at y=0 / y=178 (paper gives no numbers)",
    "filter": "4th-order Butterworth, zero-phase filtfilt for imaging (paper: no "
              "order stated); record starts with a quiet pre-arrival segment so "
              "t=0 = excitation start is consistent across paths",
    "das_signal": "|Hilbert envelope| sampled at the delayed time, summed over the 6 "
                  "receivers per actuator, then averaged over actuators (paper Eq 1 "
                  "with omega=1, Fig 12b). A per-path time gate removes the direct "
                  "arrival (t < ToF_direct + margin) because pixels on the straight "
                  "actuator-receiver line share the direct-path delay and would "
                  "otherwise light up as vertical stripes (verified without gate). "
                  "The paper does not state a gate; this is a standard pitch-catch "
                  "DAS practice required to obtain damage-focused images.",
    "baseline_subtraction": "time-domain subtraction of the matching cycle-0 baseline "
                            "acquisition (same condition 'Loaded', closest actuation "
                            "load) before DAS. Paper §5.1.2 describes baseline "
                            "subtraction for the simulation branch; without it the "
                            "experimental DAS image is dominated by direct-path/boundary "
                            "energy (verified), which contradicts the paper's "
                            "damage-focused Fig 13a, so the experimental branch must "
                            "subtract a baseline too (exact matching rule unstated).",
    "time_window": "full 2000-sample record (1.667 ms); paper does not state a window",
    "condition": "first 'Loaded' mat per cycle used (clamped/traction-free variants exist)",
}


def bandpass(x, fs, lo_khz, hi_khz, order=4):
    lo, hi = lo_khz * 1e3, hi_khz * 1e3
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    x = x - x.mean()
    # long padlen suppresses forward-backward pre-ring in the quiet pre-arrival segment
    return filtfilt(b, a, x, padlen=int(fs * 200e-6))


def _load_mat(fname):
    m = loadmat(DATA / "PZT-data" / fname, struct_as_record=False, squeeze_me=True)
    return m["coupon"]


def _paths_from_coupon(c):
    paths = {}
    for p in np.atleast_1d(c.path_data):
        if float(p.frequency) != CFG["excitation_khz"]:
            continue
        paths[(int(p.actuator), int(p.sensor))] = np.asarray(
            p.signal_sensor, dtype=np.float64)
    return paths


def load_paths(cycle: int):
    """Return paths for `cycle` baseline-subtracted by the closest cycle-0
    'Loaded' acquisition, plus provenance of both mats."""
    folder = DATA / "PZT-data"
    cands = sorted(p.name for p in folder.glob(f"{CFG['coupon']}_{cycle}_*.mat")
                   if not p.name.startswith("._"))
    if not cands:
        raise FileNotFoundError(f"no mat for cycle {cycle}")

    def cond_load(names):
        out = []
        for n in names:
            c = _load_mat(n)
            out.append((str(c.condition).lower(), float(c.load), c, n))
        return out

    infos = cond_load(cands)
    picked = next((x for x in infos
                   if x[0].startswith(CFG["condition_filter"][:4])), infos[0])
    cond, load, cmat, fname = picked[0], picked[1], picked[2], picked[3]

    # baseline: cycle-0 mats, 'Loaded', closest actuation load
    b_cands = sorted(p.name for p in folder.glob(f"{CFG['coupon']}_0_*.mat")
                     if not p.name.startswith("._"))
    b_infos = [x for x in cond_load(b_cands) if x[0].startswith("load")]
    if b_infos:
        binfo = min(b_infos, key=lambda x: abs(x[1] - load))
        bmat, bfname = binfo[2], binfo[3]
    else:  # fall back to the Baseline-condition mat
        binfo = next(x for x in cond_load(b_cands) if x[0].startswith("baseline"))
        bmat, bfname = binfo[2], binfo[3]

    cp = _paths_from_coupon(cmat)
    bp = _paths_from_coupon(bmat)
    paths = []
    for key, sig in cp.items():
        base = bp.get(key)
        if base is None or base.size != sig.size:
            base = np.zeros_like(sig)
        paths.append({
            "actuator": key[0], "sensor": key[1],
            "signal": sig - base, "fs": 1.2e6,
        })
    return paths, fname, cond, load, bfname


def das_image(paths, velocity):
    """Six per-actuator DAS images averaged -> (178,152) normalized image."""
    W, H = CFG["grid_wh"]
    ys, xs = np.mgrid[0:H, 0:W]
    acc = np.zeros((H, W), dtype=np.float64)
    for a in range(1, 7):
        ax, ay = CFG["actuator_xy"][str(a)]
        img = np.zeros((H, W), dtype=np.float64)
        for p in paths:
            if p["actuator"] != a:
                continue
            rx, ry = CFG["receiver_xy"][str(p["sensor"])]
            d_direct = np.hypot(ax - rx, ay - ry)
            t_direct = d_direct / velocity * 1e3          # us
            d1 = np.hypot(xs - ax, ys - ay)
            d2 = np.hypot(rx - xs, ry - ys)
            # mm/(m/s) * 1e3 -> microseconds
            tau = (d1 + d2) / velocity * 1e3
            sig = bandpass(p["signal"], p["fs"], *CFG["bandpass_khz"])
            env = np.abs(hilbert(sig))
            t_us = np.arange(sig.size) / p["fs"] * 1e6
            # gate out the direct arrival (and any pre-arrival transients)
            env[t_us < t_direct + CFG["direct_gate_us"]] = 0.0
            img += np.interp(tau, t_us, env, left=0.0, right=0.0)
        acc += img
    acc /= max(len(paths) / 6.0, 1.0)
    acc -= acc.min()
    if acc.max() > 0:
        acc /= acc.max()
    return acc


def main():
    np.random.seed(CFG["seed"])
    velocities = [CFG["velocity_ms"], 4400.0, 5600.0]   # primary + sensitivity
    summary = {"config": CFG, "approximations": APPROX, "runs": []}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cyc in CFG["cycles"]:
        paths, fname, cond, load, bfname = load_paths(cyc)
        for vel in velocities:
            img = das_image(paths, vel)
            tag = "primary" if vel == CFG["velocity_ms"] else f"sens_{int(vel)}"
            np.save(OUT / f"das_{CFG['coupon']}_{cyc}_{tag}.npy", img)
            fig, axx = plt.subplots(figsize=(3.2, 3.8), dpi=150)
            im = axx.imshow(img, cmap="jet", origin="upper", vmin=0, vmax=1, aspect="auto")
            axx.set_title(f"{CFG['coupon']} cyc {cyc} DAS v={vel:.0f}m/s ({tag})")
            axx.set_xticks([]); axx.set_yticks([])
            fig.colorbar(im, ax=axx, fraction=0.046)
            fig.tight_layout()
            fig.savefig(OUT / f"das_{CFG['coupon']}_{cyc}_{tag}.png")
            plt.close(fig)
            # damage-side focus metric: left half vs right half energy
            summary["runs"].append({
                "cycle": cyc, "velocity_ms": vel, "tag": tag, "mat": fname,
                "baseline_mat": bfname, "condition": cond, "load_kips": load,
                "n_paths": len(paths),
                "left_half_energy": float(img[:, :76].sum()),
                "right_half_energy": float(img[:, 76:].sum()),
            })
        print(f"cycle {cyc:>6}: {len(paths)} paths, images saved (v sweep done)")

    (OUT / "das_repro_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
