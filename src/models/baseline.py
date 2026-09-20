"""Model factories used by the training and serving pipeline."""

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


def build_model(name: str, parameters: dict[str, Any], scale_pos_weight: float):
    """Build one of the supported, imbalance-aware gradient boosting models."""
    common = {
        "random_state": parameters.get("random_state", 42),
        "n_jobs": parameters.get("n_jobs", -1),
    }
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary", scale_pos_weight=scale_pos_weight, verbosity=-1,
            n_estimators=parameters.get("n_estimators", 400),
            learning_rate=parameters.get("learning_rate", 0.05),
            num_leaves=parameters.get("num_leaves", 63),
            max_depth=parameters.get("max_depth", -1),
            subsample=parameters.get("subsample", 0.8),
            colsample_bytree=parameters.get("colsample_bytree", 0.8),
            **common,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic", eval_metric="auc", scale_pos_weight=scale_pos_weight,
            n_estimators=parameters.get("n_estimators", 400), learning_rate=parameters.get("learning_rate", 0.05),
            max_depth=parameters.get("max_depth", 8), min_child_weight=parameters.get("min_child_weight", 5),
            subsample=parameters.get("subsample", 0.8), colsample_bytree=parameters.get("colsample_bytree", 0.8),
            tree_method=parameters.get("tree_method", "hist"), **common,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
            iterations=parameters.get("n_estimators", 400), learning_rate=parameters.get("learning_rate", 0.05),
            depth=parameters.get("max_depth", 8), l2_leaf_reg=parameters.get("l2_leaf_reg", 5),
            verbose=False, thread_count=parameters.get("n_jobs", -1), random_seed=parameters.get("random_state", 42),
        )
    raise ValueError(f"Unsupported model: {name}")
