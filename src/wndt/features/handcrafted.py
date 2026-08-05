"""Hand-crafted features for classic-ML baselines (RF / XGBoost / SVM).

Per channel (V, I): time-domain statistics + FFT spectral features.
Cross-channel: Pearson correlation and RMS ratio.
~50 features total, computed from raw (unnormalized) cycles.
"""
from __future__ import annotations

import numpy as np

N_BANDS = 5


def _time_features(x: np.ndarray) -> np.ndarray:
    """x: (n, 200) one channel -> 14 features."""
    mean = x.mean(axis=1)
    std = x.std(axis=1) + 1e-12
    rms = np.sqrt((x ** 2).mean(axis=1)) + 1e-12
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    p2p = mx - mn
    centered = x - mean[:, None]
    skew = (centered ** 3).mean(axis=1) / (std ** 3)
    kurt = (centered ** 4).mean(axis=1) / (std ** 4)
    med = np.median(x, axis=1)
    p10 = np.percentile(x, 10, axis=1)
    p90 = np.percentile(x, 90, axis=1)
    crest = mx / rms
    zcr = (np.abs(np.diff(np.signbit(centered), axis=1)).sum(axis=1)) / x.shape[1]
    deriv = np.diff(x, axis=1)
    mad = np.abs(deriv).mean(axis=1)
    sderiv = deriv.std(axis=1)
    return np.stack([mean, std, rms, mn, mx, p2p, skew, kurt, med, p10, p90,
                     crest, zcr, mad + sderiv], axis=1)


def _spectral_features(x: np.ndarray, fs: float = 100_000.0) -> np.ndarray:
    """x: (n, 200) -> 9 features from the one-sided FFT spectrum."""
    xc = x - x.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(xc, axis=1))                  # (n, 101)
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / fs)         # (101,)
    total = spec.sum(axis=1, keepdims=True) + 1e-12
    psd = spec / total                                      # normalized
    # band energies (5 equal-width bands over the 101 bins)
    bands = np.array_split(spec, N_BANDS, axis=1)
    band_e = np.stack([(b ** 2).sum(axis=1) for b in bands], axis=1)
    band_e = band_e / (band_e.sum(axis=1, keepdims=True) + 1e-12)
    centroid = (psd * freqs[None, :]).sum(axis=1)
    entropy = -(psd * np.log(psd + 1e-12)).sum(axis=1)
    dom_freq = freqs[spec.argmax(axis=1)]
    cum = np.cumsum(spec, axis=1)
    rolloff_idx = (cum >= 0.85 * cum[:, -1:]).argmax(axis=1)
    rolloff = freqs[rolloff_idx]
    peak_mag = spec.max(axis=1)
    return np.stack([*band_e.T, centroid, entropy, dom_freq, rolloff], axis=1)


def extract_features(waves: np.ndarray, fs: float = 100_000.0) -> np.ndarray:
    """waves: (n, 2, 200) channel order (V, I) -> (n, ~50) float32 matrix."""
    feats = []
    for c in range(waves.shape[1]):
        x = waves[:, c, :]
        feats.append(_time_features(x))
        feats.append(_spectral_features(x, fs=fs))
    v, i = waves[:, 0, :], waves[:, 1, :]
    # cross-channel: Pearson correlation + RMS ratio
    vc = v - v.mean(axis=1, keepdims=True)
    ic = i - i.mean(axis=1, keepdims=True)
    corr = (vc * ic).sum(axis=1) / (
        np.sqrt((vc ** 2).sum(axis=1)) * np.sqrt((ic ** 2).sum(axis=1)) + 1e-12)
    rms_ratio = np.sqrt((v ** 2).mean(axis=1)) / (np.sqrt((i ** 2).mean(axis=1)) + 1e-12)
    feats.append(np.stack([corr, rms_ratio], axis=1))
    out = np.concatenate(feats, axis=1).astype(np.float32)
    assert np.isfinite(out).all(), "non-finite features produced"
    return out
