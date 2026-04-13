"""Model training and hyperparameter tuning with XGBoost regressors."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor


TrainingResult = Dict[str, Any]


def tune_xgb_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fixed_params: Dict[str, Any],
    param_grid: Dict[str, Any],
    cv_folds: int,
    scoring: str,
    random_seed: int,
    n_jobs: int,
    verbose: int,
) -> TrainingResult:
    """Tune one XGBoost regressor using grid search over basic hyperparameters."""
    estimator = XGBRegressor(
        random_state=random_seed,
        n_jobs=n_jobs,
        **fixed_params,
    )

    cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_strategy,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )

    search.fit(X_train, y_train)

    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_mae": float(-search.best_score_),
    }


def train_dual_xgboost_models(
    X_train: np.ndarray,
    y_sbp_train: np.ndarray,
    y_dbp_train: np.ndarray,
    fixed_params: Dict[str, Any],
    param_grid: Dict[str, Any],
    cv_folds: int,
    scoring: str,
    random_seed: int,
    n_jobs: int,
    verbose: int,
) -> Dict[str, TrainingResult]:
    """Train separate tuned XGBoost models for SBP and DBP estimation."""
    print("[Training] Tuning SBP regressor...")
    sbp_result = tune_xgb_regressor(
        X_train=X_train,
        y_train=y_sbp_train,
        fixed_params=fixed_params,
        param_grid=param_grid,
        cv_folds=cv_folds,
        scoring=scoring,
        random_seed=random_seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    print("[Training] Tuning DBP regressor...")
    dbp_result = tune_xgb_regressor(
        X_train=X_train,
        y_train=y_dbp_train,
        fixed_params=fixed_params,
        param_grid=param_grid,
        cv_folds=cv_folds,
        scoring=scoring,
        random_seed=random_seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return {"sbp": sbp_result, "dbp": dbp_result}
