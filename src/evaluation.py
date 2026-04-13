"""Evaluation and visualization helpers for BP regression."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return MAE, RMSE, and R2 for one regression target."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """Save scatter plot comparing actual and predicted values."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.45, edgecolors="k", linewidths=0.2)
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal")
    plt.xlabel("Actual BP (mmHg)")
    plt.ylabel("Predicted BP (mmHg)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_feature_importance(
    model,
    feature_names: List[str],
    title: str,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """Save horizontal bar chart of the top feature importances."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    importances = np.asarray(importances, dtype=np.float64)
    order = np.argsort(importances)[::-1][:top_n]

    selected_importances = importances[order][::-1]
    selected_names = [feature_names[idx] for idx in order[::-1]]

    plt.figure(figsize=(9, 6))
    plt.barh(selected_names, selected_importances)
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def build_sample_predictions(
    y_sbp_true: np.ndarray,
    y_sbp_pred: np.ndarray,
    y_dbp_true: np.ndarray,
    y_dbp_pred: np.ndarray,
    rows: int,
) -> pd.DataFrame:
    """Build a compact table of predictions and absolute errors."""
    sample_rows = min(rows, y_sbp_true.size)

    table = pd.DataFrame(
        {
            "SBP_Actual": y_sbp_true[:sample_rows],
            "SBP_Predicted": y_sbp_pred[:sample_rows],
            "DBP_Actual": y_dbp_true[:sample_rows],
            "DBP_Predicted": y_dbp_pred[:sample_rows],
        }
    )
    table["SBP_AbsError"] = np.abs(table["SBP_Actual"] - table["SBP_Predicted"])
    table["DBP_AbsError"] = np.abs(table["DBP_Actual"] - table["DBP_Predicted"])
    return table
