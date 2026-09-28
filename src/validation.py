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
    id_column: str = "TransactionID",
    feature_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit each model on K folds and collect OOF predictions for every row."""
    labels = target.to_numpy() if isinstance(target, pd.Series) else np.asarray(target)
    if len(labels) != len(frame):
        raise ValueError("Target length must match frame length")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not model_configs:
        raise ValueError("At least one model configuration is required")
    class_counts = np.unique(labels, return_counts=True)[1]
    actual_splits = min(n_splits, int(class_counts.min()))
    if actual_splits < 2:
        raise ValueError("Each target class needs at least two rows for stratified OOF validation")

    oof_predictions: dict[str, np.ndarray] = {name: np.zeros(len(frame), dtype="float64") for name in model_configs}
    oof_seen = np.zeros(len(frame), dtype="int8")
    oof_folds = np.zeros(len(frame), dtype="int16")
    fold_scores: dict[str, list[float]] = {name: [] for name in model_configs}
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)

    for fold_number, (train_idx, valid_idx) in enumerate(skf.split(frame, labels), start=1):
        train_frame = frame.iloc[train_idx].copy()
        valid_frame = frame.iloc[valid_idx].copy()
        y_train = labels[train_idx]
        y_valid = labels[valid_idx]
        oof_seen[valid_idx] += 1
        oof_folds[valid_idx] = fold_number

        transformer = TransactionFeatureEngineer(id_column=id_column, **(feature_flags or {})).fit(
            train_frame, y_train
        )
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

    if not np.all(oof_seen == 1):
        raise RuntimeError("OOF validation must predict every row exactly once")

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
        "n_splits": actual_splits,
        "oof_coverage": int(oof_seen.sum()),
        "oof_folds": oof_folds,
    }


def time_based_validation(
    frame: pd.DataFrame,
    target: pd.Series | np.ndarray,
    model_configs: dict[str, dict[str, Any]],
    random_state: int = 42,
    blend_weights: dict[str, float] | None = None,
    id_column: str = "TransactionID",
    feature_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate models on a chronological split based on TransactionDT."""
    if "TransactionDT" not in frame.columns:
        raise ValueError("The time-based validation requires a TransactionDT column.")

    labels = target.to_numpy() if isinstance(target, pd.Series) else np.asarray(target)
    if len(labels) != len(frame):
        raise ValueError("Target length must match frame length")
    ordered_indices = np.argsort(frame["TransactionDT"].to_numpy(), kind="stable")
    split_index = int(len(ordered_indices) * 0.8)
    if split_index == 0 or split_index == len(ordered_indices):
        raise ValueError("Not enough rows for an 80/20 chronological validation split")
    time_values = frame["TransactionDT"].to_numpy()
    boundary_time = time_values[ordered_indices[split_index]]
    train_indices = ordered_indices[time_values[ordered_indices] < boundary_time]
    valid_indices = ordered_indices[time_values[ordered_indices] >= boundary_time]
    if len(train_indices) == 0 or len(valid_indices) == 0:
        raise ValueError("Chronological boundary produced an empty train or validation partition")

    train_frame = frame.iloc[train_indices].copy()
    valid_frame = frame.iloc[valid_indices].copy()
    y_train = labels[train_indices]
    y_valid = labels[valid_indices]

    transformer = TransactionFeatureEngineer(id_column=id_column, **(feature_flags or {})).fit(train_frame, y_train)
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

    if blend_weights is None:
        blend_weights = {name: 1.0 / len(prediction_map) for name in prediction_map}
    if blend_weights:
        if not set(blend_weights).issubset(prediction_map):
            raise ValueError("Blend weights reference models without time-validation predictions")
        blended = sum(blend_weights[name] * prediction_map[name] for name in blend_weights)
        scores["blend"] = _safe_auc(y_valid, blended)

    return {
        "train_rows": int(train_frame.shape[0]),
        "valid_rows": int(valid_frame.shape[0]),
        "split_threshold": float(boundary_time),
        "train_max_time": float(time_values[train_indices].max()),
        "valid_min_time": float(time_values[valid_indices].min()),
        "blend_weights": blend_weights,
        "summary": scores,
        "predictions": prediction_map,
        "valid_indices": valid_indices,
    }
