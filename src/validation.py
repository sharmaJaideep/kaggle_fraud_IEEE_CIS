"""Out-of-fold and time-aware validation utilities for the fraud ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.features.preprocessing import TransactionFeatureEngineer
from src.models.baseline import build_model


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_pred))


def kfold_oof_validation(
    frame: pd.DataFrame,
    target: pd.Series | np.ndarray,
    model_configs: dict[str, dict[str, Any]],
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit each model on K folds and collect OOF predictions for every row."""
    labels = target.to_numpy() if isinstance(target, pd.Series) else np.asarray(target)
    if len(labels) != len(frame):
        raise ValueError("Target length must match frame length")

    oof_predictions: dict[str, np.ndarray] = {name: np.zeros(len(frame), dtype="float64") for name in model_configs}
    fold_scores: dict[str, list[float]] = {name: [] for name in model_configs}
    skf = StratifiedKFold(n_splits=min(n_splits, len(np.unique(labels))), shuffle=True, random_state=random_state)

    for fold_number, (train_idx, valid_idx) in enumerate(skf.split(frame, labels), start=1):
        train_frame = frame.iloc[train_idx].copy()
        valid_frame = frame.iloc[valid_idx].copy()
        y_train = labels[train_idx]
        y_valid = labels[valid_idx]

        transformer = TransactionFeatureEngineer(id_column="TransactionID").fit(train_frame)
        X_train = transformer.transform(train_frame)
        X_valid = transformer.transform(valid_frame)

        for name, parameters in model_configs.items():
            scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
            model = build_model(
                name,
                {"random_state": random_state + fold_number, "n_jobs": -1, **parameters},
                scale_pos_weight,
            )
            model.fit(X_train, y_train)
            valid_predictions = model.predict_proba(X_valid)[:, 1]
            oof_predictions[name][valid_idx] = valid_predictions
            fold_scores[name].append(_safe_auc(y_valid, valid_predictions))

    summary: dict[str, dict[str, Any]] = {}
    for name, scores in fold_scores.items():
        summary[name] = {
            "fold_scores": scores,
            "mean_auc": float(np.mean(scores)),
            "std_auc": float(np.std(scores, ddof=0)),
            "oof_auc": _safe_auc(labels, oof_predictions[name]),
        }

    return {
        "oof_predictions": oof_predictions,
        "summary": summary,
        "fold_scores": fold_scores,
        "n_splits": min(n_splits, len(np.unique(labels))),
    }


def time_based_validation(
    frame: pd.DataFrame,
    target: pd.Series | np.ndarray,
    model_configs: dict[str, dict[str, Any]],
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate models on a chronological split based on TransactionDT."""
    if "TransactionDT" not in frame.columns:
        raise ValueError("The time-based validation requires a TransactionDT column.")

    labels = target.to_numpy() if isinstance(target, pd.Series) else np.asarray(target)
    cut_point = frame["TransactionDT"].quantile(0.8)
    train_mask = frame["TransactionDT"] < cut_point
    valid_mask = ~train_mask

    train_frame = frame.loc[train_mask].copy()
    valid_frame = frame.loc[valid_mask].copy()
    y_train = labels[train_mask.to_numpy()]
    y_valid = labels[valid_mask.to_numpy()]

    transformer = TransactionFeatureEngineer(id_column="TransactionID").fit(train_frame)
    X_train = transformer.transform(train_frame)
    X_valid = transformer.transform(valid_frame)

    scores: dict[str, float] = {}
    prediction_map: dict[str, np.ndarray] = {}
    for name, parameters in model_configs.items():
        scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        model = build_model(name, {"random_state": random_state, "n_jobs": -1, **parameters}, scale_pos_weight)
        model.fit(X_train, y_train)
        valid_predictions = model.predict_proba(X_valid)[:, 1]
        prediction_map[name] = valid_predictions
        scores[name] = _safe_auc(y_valid, valid_predictions)

    return {
        "train_rows": int(train_frame.shape[0]),
        "valid_rows": int(valid_frame.shape[0]),
        "split_threshold": float(cut_point),
        "summary": scores,
        "predictions": prediction_map,
    }
