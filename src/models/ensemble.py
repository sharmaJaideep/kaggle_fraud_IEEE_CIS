"""Validation-weighted gradient boosting ensemble and artifact contract."""

from __future__ import annotations

import pickle
from typing import Iterator
from pathlib import Path
from typing import Any

import cloudpickle
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.baseline import build_model


DEFAULT_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "lightgbm": {"n_estimators": 500, "learning_rate": 0.04, "num_leaves": 63, "max_depth": -1},
    "xgboost": {"n_estimators": 450, "learning_rate": 0.04, "max_depth": 8, "min_child_weight": 5},
    "catboost": {"n_estimators": 450, "learning_rate": 0.04, "max_depth": 8, "l2_leaf_reg": 5},
}

DEFAULT_TUNING_CANDIDATES: dict[str, list[dict[str, Any]]] = {
    "lightgbm": [
        {"n_estimators": 350, "learning_rate": 0.05, "num_leaves": 31},
        DEFAULT_MODEL_CONFIGS["lightgbm"],
    ],
    "xgboost": [
        {"n_estimators": 350, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5},
        DEFAULT_MODEL_CONFIGS["xgboost"],
    ],
    "catboost": [
        {"n_estimators": 350, "learning_rate": 0.05, "max_depth": 6, "l2_leaf_reg": 5},
        DEFAULT_MODEL_CONFIGS["catboost"],
    ],
}


def tune_model_configs(
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    candidates: dict[str, list[dict[str, Any]]] | None = None,
    random_state: int = 42,
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Select one explicit parameter set per booster using validation ROC-AUC."""
    selected: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    scale_pos_weight = float((np.asarray(y_train) == 0).sum() / max((np.asarray(y_train) == 1).sum(), 1))
    for name, parameter_candidates in (candidates or DEFAULT_TUNING_CANDIDATES).items():
        best_score = -np.inf
        best_parameters = parameter_candidates[0]
        for parameters in parameter_candidates:
            model = build_model(name, {"random_state": random_state, "n_jobs": -1, **parameters}, scale_pos_weight)
            model.fit(X_train, y_train)
            score = float(roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1]))
            if score > best_score:
                best_score, best_parameters = score, parameters
        selected[name] = best_parameters
        scores[name] = best_score
    return selected, scores


def fit_validation_weighted_ensemble(
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    model_configs: dict[str, dict[str, Any]] | None = None,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    """Fit configured models and weight probabilities by validation ROC-AUC.

    Each model is tuned through its explicit configuration, allowing a small,
    reproducible parameter sweep to be run by the caller without hiding choices
    in a search object that cannot be serialized with the final artifact.
    """
    configs = model_configs or DEFAULT_MODEL_CONFIGS
    scale_pos_weight = float((np.asarray(y_train) == 0).sum() / max((np.asarray(y_train) == 1).sum(), 1))
    models: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    for name, parameters in configs.items():
        parameters = {"random_state": random_state, "n_jobs": -1, **parameters}
        model = build_model(name, parameters, scale_pos_weight)
        model.fit(X_train, y_train)
        model_probabilities = model.predict_proba(X_valid)[:, 1]
        models[name] = model
        probabilities[name] = model_probabilities
        scores[name] = float(roc_auc_score(y_valid, model_probabilities))

    raw_weights = {name: max(score - 0.5, 1e-6) for name, score in scores.items()}
    normalizer = sum(raw_weights.values())
    weights = {name: value / normalizer for name, value in raw_weights.items()}
    blended = sum(weights[name] * probabilities[name] for name in models)
    scores["blend"] = float(roc_auc_score(y_valid, blended))
    return models, weights, scores


def optimize_oof_blend_weights(oof_predictions: dict[str, np.ndarray], y_true: np.ndarray) -> tuple[dict[str, float], float]:
    """Search a non-negative weight simplex using OOF ROC-AUC."""
    if not oof_predictions:
        raise ValueError("Need at least one model prediction array")

    names = list(oof_predictions)
    y_true = np.asarray(y_true).reshape(-1)
    if len(np.unique(y_true)) < 2:
        raise ValueError("OOF blend optimization requires both target classes")
    predictions = {name: np.asarray(values, dtype="float64").reshape(-1) for name, values in oof_predictions.items()}
    if any(len(values) != len(y_true) for values in predictions.values()):
        raise ValueError("Every OOF prediction array must match the target length")
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("OOF predictions must contain only finite values")

    resolution = 20

    def compositions(total: int, parts: int, prefix: tuple[int, ...] = ()) -> Iterator[tuple[int, ...]]:
        if parts == 1:
            yield prefix + (total,)
            return
        for value in range(total + 1):
            yield from compositions(total - value, parts - 1, prefix + (value,))

    best_score = -np.inf
    best_weights = {name: 1.0 / len(names) for name in names}
    for integer_weights in compositions(resolution, len(names)):
        candidate = {name: value / resolution for name, value in zip(names, integer_weights)}
        blended = np.zeros_like(y_true, dtype="float64")
        for name in names:
            blended += candidate[name] * predictions[name]
        blended = np.clip(blended, 0.0, 1.0)
        score = float(roc_auc_score(y_true, blended))
        if score > best_score:
            best_score = score
            best_weights = {name: float(value) for name, value in candidate.items()}

    return best_weights, float(best_score)


def save_artifact(
    path: str | Path,
    transformer: Any,
    models: dict[str, Any],
    weights: dict[str, float],
    metadata: dict[str, Any],
) -> None:
    """Persist preprocessing, models, weights, and schema as one deployable file."""
    artifact = {
        "version": 1,
        "transformer": transformer,
        "models": models,
        "weights": weights,
        "feature_names": list(transformer.get_feature_names_out()),
        "metadata": metadata,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        cloudpickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load a saved training artifact and validate its required contract."""
    with Path(path).open("rb") as handle:
        try:
            artifact = cloudpickle.load(handle)
        except Exception:
            handle.seek(0)
            artifact = joblib.load(handle)
    required = {"transformer", "models", "weights", "feature_names", "metadata"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Artifact is missing required keys: {sorted(missing)}")
    return artifact


def predict_artifact(artifact: dict[str, Any], records: Any) -> np.ndarray:
    """Generate blended fraud probabilities using the persisted feature logic."""
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)
    transformed = artifact["transformer"].transform(records)
    predictions = np.zeros(len(transformed), dtype="float64")
    for name, model in artifact["models"].items():
        model_predictions = model.predict_proba(transformed)
        predictions += artifact["weights"][name] * model_predictions[:, 1]
    return np.clip(predictions, 0.0, 1.0)