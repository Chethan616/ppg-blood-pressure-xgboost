"""Window segmentation for synchronized PPG and ABP signals."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .label_extraction import extract_sbp_dbp, is_valid_bp_label
from .preprocessing import normalize_signal

Segment = Dict[str, Any]


def compute_window_parameters(
    sampling_rate_hz: float,
    window_seconds: float,
    overlap_ratio: float,
) -> Tuple[int, int]:
    """Compute window and hop lengths in samples."""
    window_size = int(round(window_seconds * sampling_rate_hz))
    if window_size <= 0:
        raise ValueError("window_seconds is too small for the given sampling rate.")

    step_size = int(round(window_size * (1.0 - overlap_ratio)))
    step_size = max(1, step_size)
    return window_size, step_size


def segment_records(
    records: List[Dict[str, Any]],
    sampling_rate_hz: float,
    window_seconds: float,
    overlap_ratio: float,
    min_segment_std: float,
    segment_normalization: str,
    label_config: Dict[str, float],
) -> Tuple[List[Segment], Dict[str, int]]:
    """Create valid training segments and aligned SBP/DBP labels."""
    window_size, step_size = compute_window_parameters(
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
        overlap_ratio=overlap_ratio,
    )

    segments: List[Segment] = []
    stats = {
        "records_considered": 0,
        "records_used": 0,
        "records_too_short": 0,
        "dropped_nan": 0,
        "dropped_low_variance": 0,
        "dropped_invalid_label": 0,
        "segments_total": 0,
    }

    for record in records:
        stats["records_considered"] += 1

        ppg = np.asarray(record["ppg"], dtype=np.float64)
        abp = np.asarray(record["abp"], dtype=np.float64)

        min_len = min(ppg.size, abp.size)
        if min_len < window_size:
            stats["records_too_short"] += 1
            continue

        ppg = ppg[:min_len]
        abp = abp[:min_len]

        valid_segment_count = 0
        for start in range(0, min_len - window_size + 1, step_size):
            end = start + window_size
            ppg_segment = ppg[start:end]
            abp_segment = abp[start:end]

            if not np.isfinite(ppg_segment).all() or not np.isfinite(abp_segment).all():
                stats["dropped_nan"] += 1
                continue

            if float(np.std(ppg_segment)) < min_segment_std:
                stats["dropped_low_variance"] += 1
                continue

            sbp, dbp = extract_sbp_dbp(abp_segment)
            valid_label = is_valid_bp_label(
                sbp=sbp,
                dbp=dbp,
                sbp_min=float(label_config["sbp_min"]),
                sbp_max=float(label_config["sbp_max"]),
                dbp_min=float(label_config["dbp_min"]),
                dbp_max=float(label_config["dbp_max"]),
                min_pulse_pressure=float(label_config["min_pulse_pressure"]),
            )
            if not valid_label:
                stats["dropped_invalid_label"] += 1
                continue

            normalized_segment = normalize_signal(ppg_segment, method=segment_normalization)
            if normalized_segment is None:
                stats["dropped_low_variance"] += 1
                continue

            segments.append(
                {
                    "ppg_segment": normalized_segment,
                    "abp_segment": abp_segment,
                    "sbp": sbp,
                    "dbp": dbp,
                    "group_id": record["group_id"],
                    "source_file": record["source_file"],
                    "record_index": int(record["record_index"]),
                    "start_index": int(start),
                    "end_index": int(end),
                }
            )
            valid_segment_count += 1

        if valid_segment_count > 0:
            stats["records_used"] += 1

    stats["segments_total"] = len(segments)
    return segments, stats
