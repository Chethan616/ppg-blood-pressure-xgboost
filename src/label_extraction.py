"""Label extraction from ABP segments."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def extract_sbp_dbp(abp_segment: np.ndarray) -> Tuple[float, float]:
    """Compute systolic and diastolic blood pressure from one ABP segment."""
    abp = np.asarray(abp_segment, dtype=np.float64)
    return float(np.max(abp)), float(np.min(abp))


def is_valid_bp_label(
    sbp: float,
    dbp: float,
    sbp_min: float,
    sbp_max: float,
    dbp_min: float,
    dbp_max: float,
    min_pulse_pressure: float,
) -> bool:
    """Validate extracted BP targets against physiological bounds."""
    if not np.isfinite(sbp) or not np.isfinite(dbp):
        return False
    if sbp < sbp_min or sbp > sbp_max:
        return False
    if dbp < dbp_min or dbp > dbp_max:
        return False
    if sbp <= dbp:
        return False
    if (sbp - dbp) < min_pulse_pressure:
        return False
    return True
