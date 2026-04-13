"""Feature extraction for PPG windows."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew


FEATURE_NAMES = [
    "mean",
    "std",
    "pulse_amplitude",
    "variance",
    "skewness",
    "kurtosis",
    "d1_mean",
    "d1_std",
    "d1_energy",
    "d2_mean",
    "d2_std",
    "d2_energy",
    "dominant_frequency_hz",
    "spectral_energy",
    "spectral_entropy",
]


def extract_features_from_segment(
    ppg_segment: np.ndarray,
    sampling_rate_hz: float,
) -> Dict[str, float]:
    """Extract required time, statistical, derivative, and frequency features."""
    x = np.asarray(ppg_segment, dtype=np.float64)

    d1 = np.gradient(x)
    d2 = np.gradient(d1)

    centered = x - np.mean(x)
    fft_complex = rfft(centered)
    freqs = rfftfreq(x.size, d=1.0 / sampling_rate_hz)
    power = np.abs(fft_complex) ** 2

    band_mask = (freqs >= 0.5) & (freqs <= 8.0)
    if np.any(band_mask):
        band_freqs = freqs[band_mask]
        band_power = power[band_mask]

        dominant_frequency_hz = float(band_freqs[int(np.argmax(band_power))])
        spectral_energy = float(np.sum(band_power))

        power_sum = float(np.sum(band_power))
        if power_sum > 0:
            p = band_power / power_sum
            spectral_entropy = float(-np.sum(p * np.log2(p + 1e-12)))
        else:
            spectral_entropy = 0.0
    else:
        dominant_frequency_hz = 0.0
        spectral_energy = 0.0
        spectral_entropy = 0.0

    features = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "pulse_amplitude": float(np.max(x) - np.min(x)),
        "variance": float(np.var(x)),
        "skewness": float(skew(x, bias=False)) if x.size > 2 else 0.0,
        "kurtosis": float(kurtosis(x, fisher=True, bias=False)) if x.size > 3 else 0.0,
        "d1_mean": float(np.mean(d1)),
        "d1_std": float(np.std(d1)),
        "d1_energy": float(np.sum(d1**2)),
        "d2_mean": float(np.mean(d2)),
        "d2_std": float(np.std(d2)),
        "d2_energy": float(np.sum(d2**2)),
        "dominant_frequency_hz": dominant_frequency_hz,
        "spectral_energy": spectral_energy,
        "spectral_entropy": spectral_entropy,
    }

    return features


def build_feature_dataset(
    segments: List[Dict[str, Any]],
    sampling_rate_hz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame, pd.DataFrame]:
    """Convert segmented windows into feature matrix, labels, and metadata tables."""
    feature_rows: List[Dict[str, float]] = []
    sbp_targets: List[float] = []
    dbp_targets: List[float] = []
    metadata_rows: List[Dict[str, Any]] = []

    for segment in segments:
        features = extract_features_from_segment(
            ppg_segment=segment["ppg_segment"],
            sampling_rate_hz=sampling_rate_hz,
        )
        feature_rows.append(features)
        sbp_targets.append(float(segment["sbp"]))
        dbp_targets.append(float(segment["dbp"]))
        metadata_rows.append(
            {
                "group_id": segment["group_id"],
                "source_file": segment["source_file"],
                "record_index": segment["record_index"],
                "start_index": segment["start_index"],
                "end_index": segment["end_index"],
            }
        )

    feature_df = pd.DataFrame(feature_rows)
    for required_name in FEATURE_NAMES:
        if required_name not in feature_df.columns:
            feature_df[required_name] = 0.0

    feature_df = feature_df[FEATURE_NAMES]
    metadata_df = pd.DataFrame(metadata_rows)

    X = feature_df.to_numpy(dtype=np.float64)
    y_sbp = np.asarray(sbp_targets, dtype=np.float64)
    y_dbp = np.asarray(dbp_targets, dtype=np.float64)

    return X, y_sbp, y_dbp, FEATURE_NAMES, metadata_df, feature_df
