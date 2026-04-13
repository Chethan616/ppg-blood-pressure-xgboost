"""Shared utilities for reproducible BP estimation experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(payload: Dict[str, Any], output_path: str | Path) -> None:
    """Persist a dictionary as formatted JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration into a Python dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} did not parse into a dictionary.")

    return config
