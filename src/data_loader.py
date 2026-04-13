"""Data acquisition utilities for the Kaggle BloodPressureDataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat

Record = Dict[str, Any]


def discover_mat_files(raw_data_dir: str | Path, mat_glob: str = "part_*.mat") -> List[Path]:
    """Return sorted .mat files that match the configured pattern."""
    root = Path(raw_data_dir)
    return sorted(root.glob(mat_glob))


def _find_signal_container(
    mat_content: Dict[str, Any], preferred_key: Optional[str] = None
) -> Tuple[str, np.ndarray]:
    """Find the MATLAB variable that stores records."""
    if preferred_key is not None:
        value = mat_content.get(preferred_key)
        if isinstance(value, np.ndarray):
            return preferred_key, value
        raise KeyError(f"Preferred key '{preferred_key}' was not found as a numpy array.")

    object_candidates: List[Tuple[str, np.ndarray]] = []
    numeric_candidates: List[Tuple[str, np.ndarray]] = []

    for key, value in mat_content.items():
        if key.startswith("__"):
            continue
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype == object:
            object_candidates.append((key, value))
        elif np.issubdtype(value.dtype, np.number):
            numeric_candidates.append((key, value))

    if object_candidates:
        object_candidates.sort(key=lambda item: item[1].size, reverse=True)
        return object_candidates[0]

    if numeric_candidates:
        numeric_candidates.sort(key=lambda item: item[1].size, reverse=True)
        return numeric_candidates[0]

    raise ValueError("No array-like signal container found inside MAT file.")


def _unwrap_cell_value(value: Any) -> Any:
    """Unwrap deeply nested single-object MATLAB cell wrappers."""
    current = value
    while isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
        current = current.item()
    return current


def _iter_record_candidates(container: np.ndarray) -> Iterable[Any]:
    """Yield each record candidate from either an object cell array or a numeric matrix."""
    if isinstance(container, np.ndarray) and container.dtype == object:
        for cell in np.ravel(container, order="C"):
            yield _unwrap_cell_value(cell)
        return

    yield container


def _extract_ppg_abp_from_matrix(matrix: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract aligned PPG and ABP vectors from a record matrix."""
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        return None

    if arr.shape[0] in (2, 3):
        channels = arr
    elif arr.shape[1] in (2, 3):
        channels = arr.T
    elif min(arr.shape) <= 6:
        channels = arr if arr.shape[0] < arr.shape[1] else arr.T
    else:
        return None

    if channels.shape[0] < 2:
        return None

    try:
        ppg = np.asarray(channels[0], dtype=np.float64).ravel()
        abp = np.asarray(channels[1], dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None

    min_len = min(ppg.size, abp.size)
    if min_len < 2:
        return None

    return ppg[:min_len], abp[:min_len]


def load_records(
    raw_data_dir: str | Path,
    mat_glob: str = "part_*.mat",
    preferred_mat_key: Optional[str] = None,
    group_block_size: int = 10,
) -> List[Record]:
    """Load all records from MAT files and return a list of PPG/ABP pairs with metadata."""
    mat_files = discover_mat_files(raw_data_dir, mat_glob=mat_glob)
    if not mat_files:
        raise FileNotFoundError(
            f"No MAT files were found in '{raw_data_dir}' with pattern '{mat_glob}'."
        )

    records: List[Record] = []
    skipped_records = 0
    global_index = 0
    safe_group_block = max(1, int(group_block_size))

    for mat_path in mat_files:
        mat_content = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
        container_key, container = _find_signal_container(mat_content, preferred_key=preferred_mat_key)

        loaded_in_part = 0
        for record_index, candidate in enumerate(_iter_record_candidates(container)):
            extracted = _extract_ppg_abp_from_matrix(candidate)
            if extracted is None:
                skipped_records += 1
                continue

            ppg, abp = extracted
            group_id = f"{mat_path.stem}_g{record_index // safe_group_block}"

            records.append(
                {
                    "global_index": global_index,
                    "record_index": record_index,
                    "source_file": mat_path.name,
                    "group_id": group_id,
                    "ppg": ppg,
                    "abp": abp,
                }
            )
            global_index += 1
            loaded_in_part += 1

        print(
            f"[DataLoader] {mat_path.name}: loaded {loaded_in_part} records "
            f"(source key='{container_key}')"
        )

    print(
        f"[DataLoader] Total valid records: {len(records)} | "
        f"Skipped/invalid records: {skipped_records}"
    )

    if not records:
        raise RuntimeError("Signal extraction failed for all records. Please inspect MAT structure.")

    return records
