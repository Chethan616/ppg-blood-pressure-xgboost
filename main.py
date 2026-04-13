"""End-to-end pipeline for cuff-less BP estimation from PPG using XGBoost."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_records
from src.evaluation import (
    build_sample_predictions,
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_feature_importance,
)
from src.feature_extraction import build_feature_dataset
from src.model_training import train_dual_xgboost_models
from src.preprocessing import preprocess_abp_signal, preprocess_ppg_signal
from src.segmentation import segment_records
from src.utils import ensure_directory, load_config, save_json, set_global_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PPG to blood pressure estimation pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def preprocess_records(
    records: List[Dict[str, Any]],
    sampling_rate_hz: float,
    preprocessing_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Preprocess raw PPG and ABP for each record and remove invalid records."""
    valid_records: List[Dict[str, Any]] = []
    skipped_records = 0

    for record in records:
        ppg_processed = preprocess_ppg_signal(
            signal=record["ppg"],
            sampling_rate_hz=sampling_rate_hz,
            low_hz=float(preprocessing_cfg["ppg_bandpass_low_hz"]),
            high_hz=float(preprocessing_cfg["ppg_bandpass_high_hz"]),
            filter_order=int(preprocessing_cfg["filter_order"]),
            max_nan_ratio=float(preprocessing_cfg["max_nan_ratio"]),
            max_nan_gap_seconds=float(preprocessing_cfg["max_nan_gap_seconds"]),
            min_signal_std=float(preprocessing_cfg["min_signal_std"]),
        )

        abp_processed = preprocess_abp_signal(
            signal=record["abp"],
            sampling_rate_hz=sampling_rate_hz,
            max_nan_ratio=float(preprocessing_cfg["max_nan_ratio"]),
            max_nan_gap_seconds=float(preprocessing_cfg["max_nan_gap_seconds"]),
            min_signal_std=float(preprocessing_cfg["min_signal_std"]),
        )

        if ppg_processed is None or abp_processed is None:
            skipped_records += 1
            continue

        min_len = min(ppg_processed.size, abp_processed.size)
        if min_len < 2:
            skipped_records += 1
            continue

        clean_record = dict(record)
        clean_record["ppg"] = ppg_processed[:min_len]
        clean_record["abp"] = abp_processed[:min_len]
        valid_records.append(clean_record)

    print(
        f"[Preprocessing] Valid records: {len(valid_records)} | "
        f"Rejected records: {skipped_records}"
    )
    return valid_records


def create_split_indices(
    X: np.ndarray,
    y_sbp: np.ndarray,
    metadata_groups: np.ndarray,
    test_size: float,
    random_seed: int,
    use_group_split: bool,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Create train/test indices with optional group-wise split."""
    all_indices = np.arange(X.shape[0])

    if use_group_split and metadata_groups.size == X.shape[0]:
        unique_groups = np.unique(metadata_groups)
        if unique_groups.size > 1:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_seed,
            )
            train_idx, test_idx = next(splitter.split(X, y_sbp, groups=metadata_groups))
            return np.asarray(train_idx), np.asarray(test_idx), "group"

    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )
    return np.asarray(train_idx), np.asarray(test_idx), "random"


def main() -> None:
    """Run the complete BP estimation workflow from data to plots."""
    args = parse_args()
    config = load_config(args.config)

    project_cfg = config["project"]
    paths_cfg = config["paths"]

    random_seed = int(project_cfg["random_seed"])
    sampling_rate_hz = float(project_cfg["sampling_rate_hz"])
    n_jobs = int(project_cfg["n_jobs"])
    verbose = int(project_cfg["verbose"])

    set_global_seed(random_seed)
    print(f"[Setup] Random seed set to {random_seed}")

    raw_data_dir = Path(paths_cfg["raw_data_dir"])
    processed_dir = ensure_directory(paths_cfg["processed_dir"])
    split_dir = ensure_directory(paths_cfg["split_dir"])
    model_dir = ensure_directory(paths_cfg["model_dir"])
    plot_dir = ensure_directory(paths_cfg["plot_dir"])
    results_dir = ensure_directory(model_dir.parent)

    print(f"[Setup] Loading data from: {raw_data_dir}")
    records = load_records(
        raw_data_dir=raw_data_dir,
        mat_glob=config["data_loader"]["mat_glob"],
        preferred_mat_key=config["data_loader"]["mat_variable_name"],
        group_block_size=int(config["data_loader"]["group_block_size"]),
    )

    preprocessed_records = preprocess_records(
        records=records,
        sampling_rate_hz=sampling_rate_hz,
        preprocessing_cfg=config["preprocessing"],
    )
    if not preprocessed_records:
        raise RuntimeError("No valid records left after preprocessing.")

    segments, segment_stats = segment_records(
        records=preprocessed_records,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=float(config["segmentation"]["window_seconds"]),
        overlap_ratio=float(config["segmentation"]["overlap_ratio"]),
        min_segment_std=float(config["segmentation"]["min_segment_std"]),
        segment_normalization=config["preprocessing"]["segment_normalization"],
        label_config=config["labeling"],
    )

    print(f"[Segmentation] Segment stats: {segment_stats}")
    if not segments:
        raise RuntimeError("No valid segments were generated. Adjust preprocessing or label limits.")

    X, y_sbp, y_dbp, feature_names, metadata_df, feature_df = build_feature_dataset(
        segments=segments,
        sampling_rate_hz=sampling_rate_hz,
    )

    print(f"[Features] Feature matrix shape: {X.shape}")
    print("[Features] First 3 feature rows:")
    print(feature_df.head(3).to_string(index=False))

    train_idx, test_idx, split_type = create_split_indices(
        X=X,
        y_sbp=y_sbp,
        metadata_groups=metadata_df["group_id"].astype(str).to_numpy(),
        test_size=float(config["split"]["test_size"]),
        random_seed=random_seed,
        use_group_split=bool(config["split"]["use_group_split"]),
    )

    print(
        f"[Split] Split type: {split_type} | "
        f"Train samples: {train_idx.size} | Test samples: {test_idx.size}"
    )

    split_payload = {
        "split_type": split_type,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
    }
    save_json(split_payload, split_dir / "split_indices.json")

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_sbp_train = y_sbp[train_idx]
    y_sbp_test = y_sbp[test_idx]
    y_dbp_train = y_dbp[train_idx]
    y_dbp_test = y_dbp[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_df.to_csv(processed_dir / "features.csv", index=False)
    metadata_df.to_csv(processed_dir / "segment_metadata.csv", index=False)

    if bool(config["outputs"]["save_processed_arrays"]):
        np.savez(
            processed_dir / "dataset_arrays.npz",
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_sbp_train=y_sbp_train,
            y_sbp_test=y_sbp_test,
            y_dbp_train=y_dbp_train,
            y_dbp_test=y_dbp_test,
        )

    training_results = train_dual_xgboost_models(
        X_train=X_train_scaled,
        y_sbp_train=y_sbp_train,
        y_dbp_train=y_dbp_train,
        fixed_params=config["model"]["fixed_params"],
        param_grid=config["model"]["param_grid"],
        cv_folds=int(config["model"]["cv_folds"]),
        scoring=config["model"]["scoring"],
        random_seed=random_seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    sbp_model = training_results["sbp"]["model"]
    dbp_model = training_results["dbp"]["model"]

    y_sbp_pred_train = sbp_model.predict(X_train_scaled)
    y_sbp_pred_test = sbp_model.predict(X_test_scaled)
    y_dbp_pred_train = dbp_model.predict(X_train_scaled)
    y_dbp_pred_test = dbp_model.predict(X_test_scaled)

    metrics = {
        "sbp": {
            "train": compute_regression_metrics(y_sbp_train, y_sbp_pred_train),
            "test": compute_regression_metrics(y_sbp_test, y_sbp_pred_test),
            "best_params": training_results["sbp"]["best_params"],
            "best_cv_mae": training_results["sbp"]["best_cv_mae"],
        },
        "dbp": {
            "train": compute_regression_metrics(y_dbp_train, y_dbp_pred_train),
            "test": compute_regression_metrics(y_dbp_test, y_dbp_pred_test),
            "best_params": training_results["dbp"]["best_params"],
            "best_cv_mae": training_results["dbp"]["best_cv_mae"],
        },
    }

    save_json(metrics, results_dir / "metrics.json")

    joblib.dump(sbp_model, model_dir / "xgb_sbp_model.joblib")
    joblib.dump(dbp_model, model_dir / "xgb_dbp_model.joblib")
    joblib.dump(scaler, model_dir / "feature_scaler.joblib")
    save_json({"feature_names": feature_names}, model_dir / "feature_names.json")

    sample_predictions = build_sample_predictions(
        y_sbp_true=y_sbp_test,
        y_sbp_pred=y_sbp_pred_test,
        y_dbp_true=y_dbp_test,
        y_dbp_pred=y_dbp_pred_test,
        rows=int(config["outputs"]["sample_prediction_rows"]),
    )
    sample_predictions.to_csv(processed_dir / "sample_predictions.csv", index=False)

    print("[Evaluation] Sample predictions:")
    print(sample_predictions.to_string(index=False))

    plot_actual_vs_predicted(
        y_true=y_sbp_test,
        y_pred=y_sbp_pred_test,
        title="SBP: Actual vs Predicted",
        output_path=plot_dir / "actual_vs_predicted_sbp.png",
    )
    plot_actual_vs_predicted(
        y_true=y_dbp_test,
        y_pred=y_dbp_pred_test,
        title="DBP: Actual vs Predicted",
        output_path=plot_dir / "actual_vs_predicted_dbp.png",
    )
    plot_feature_importance(
        model=sbp_model,
        feature_names=feature_names,
        title="SBP Feature Importance",
        output_path=plot_dir / "feature_importance_sbp.png",
    )
    plot_feature_importance(
        model=dbp_model,
        feature_names=feature_names,
        title="DBP Feature Importance",
        output_path=plot_dir / "feature_importance_dbp.png",
    )

    print("[Final] SBP test metrics:", metrics["sbp"]["test"])
    print("[Final] DBP test metrics:", metrics["dbp"]["test"])
    print(f"[Final] Saved metrics to {results_dir / 'metrics.json'}")
    print(f"[Final] Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
