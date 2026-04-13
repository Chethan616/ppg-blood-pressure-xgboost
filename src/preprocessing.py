"""Signal preprocessing for PPG and ABP waveforms."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt


def _longest_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive run of True values."""
    longest = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def clean_signal_with_interpolation(
    signal: np.ndarray,
    max_nan_ratio: float,
    max_nan_gap_samples: int,
) -> Optional[np.ndarray]:
    """Replace short missing spans using linear interpolation and reject poor-quality signals."""
    cleaned = np.asarray(signal, dtype=np.float64).copy()
    cleaned[~np.isfinite(cleaned)] = np.nan

    nan_mask = np.isnan(cleaned)
    if nan_mask.all():
        return None

    nan_ratio = float(np.mean(nan_mask))
    if nan_ratio > max_nan_ratio:
        return None

    if _longest_true_run(nan_mask) > max_nan_gap_samples:
        return None

    if np.any(nan_mask):
        valid_idx = np.flatnonzero(~nan_mask)
        missing_idx = np.flatnonzero(nan_mask)
        cleaned[missing_idx] = np.interp(missing_idx, valid_idx, cleaned[valid_idx])

    return cleaned


def bandpass_filter_ppg(
    signal: np.ndarray,
    sampling_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> Optional[np.ndarray]:
    """Apply zero-phase Butterworth bandpass filtering to PPG signal."""
    nyquist = 0.5 * sampling_rate_hz
    if nyquist <= 0:
        return None

    low = max(low_hz / nyquist, 1e-5)
    high = min(high_hz / nyquist, 0.999)
    if high <= low:
        return None

    b, a = butter(order, [low, high], btype="bandpass")
    min_required_length = 3 * max(len(a), len(b))
    if signal.size <= min_required_length:
        return None

    try:
        filtered = filtfilt(b, a, signal)
    except ValueError:
        return None

    return filtered.astype(np.float64)


def normalize_signal(signal: np.ndarray, method: str) -> Optional[np.ndarray]:
    """Normalize one signal segment using selected method."""
    x = np.asarray(signal, dtype=np.float64)

    if method == "none":
        return x.copy()

    if method == "zscore":
        std = float(np.std(x))
        if std < 1e-12:
            return None
        return (x - float(np.mean(x))) / std

    if method == "minmax":
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        denom = x_max - x_min
        if denom < 1e-12:
            return None
        return (x - x_min) / denom

    raise ValueError(f"Unsupported normalization method: {method}")


def preprocess_ppg_signal(
    signal: np.ndarray,
    sampling_rate_hz: float,
    low_hz: float,
    high_hz: float,
    filter_order: int,
    max_nan_ratio: float,
    max_nan_gap_seconds: float,
    min_signal_std: float,
) -> Optional[np.ndarray]:
    """Run full preprocessing chain for PPG: clean NaNs then bandpass filter."""
    max_nan_gap_samples = int(round(max_nan_gap_seconds * sampling_rate_hz))

    cleaned = clean_signal_with_interpolation(
        signal, max_nan_ratio=max_nan_ratio, max_nan_gap_samples=max_nan_gap_samples
    )
    if cleaned is None or float(np.std(cleaned)) < min_signal_std:
        return None

    filtered = bandpass_filter_ppg(
        cleaned,
        sampling_rate_hz=sampling_rate_hz,
        low_hz=low_hz,
        high_hz=high_hz,
        order=filter_order,
    )
    if filtered is None or float(np.std(filtered)) < min_signal_std:
        return None

    return filtered


def preprocess_abp_signal(
    signal: np.ndarray,
    sampling_rate_hz: float,
    max_nan_ratio: float,
    max_nan_gap_seconds: float,
    min_signal_std: float,
) -> Optional[np.ndarray]:
    """Clean ABP signal while preserving pressure amplitudes (no filtering)."""
    max_nan_gap_samples = int(round(max_nan_gap_seconds * sampling_rate_hz))
    cleaned = clean_signal_with_interpolation(
        signal, max_nan_ratio=max_nan_ratio, max_nan_gap_samples=max_nan_gap_samples
    )
    if cleaned is None or float(np.std(cleaned)) < min_signal_std:
        return None

    return cleaned
