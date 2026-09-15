"""Baseline model factory."""

from typing import Any


def build_baseline_model(parameters: dict[str, Any], scale_pos_weight: float):
    """Build a LightGBM classifier with imbalance-aware weighting."""
    from lightgbm import LGBMClassifier

    model_parameters = {
        key: value
        for key, value in parameters.items()
        if key in {"n_estimators", "learning_rate", "num_leaves", "max_depth", "subsample", "colsample_bytree", "random_state", "n_jobs"}
    }
    return LGBMClassifier(**model_parameters, objective="binary", scale_pos_weight=scale_pos_weight, verbosity=-1)
