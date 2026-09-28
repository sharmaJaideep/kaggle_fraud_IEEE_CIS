"""Train the feature-engineered ensemble and save one serving artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.calibration import brier_score, compute_calibration_curve, select_decision_threshold
from src.data.loading import load_training_data
from src.features.preprocessing import TransactionFeatureEngineer
from src.models.baseline import build_model
from src.models.ensemble import (
    DEFAULT_MODEL_CONFIGS,
    DEFAULT_TUNING_CANDIDATES,
    optimize_oof_blend_weights,
    save_artifact,
    tune_model_configs,
)
from src.reporting import write_validation_report
from src.validation import kfold_oof_validation, time_based_validation


def train(config_path: str | Path = "config/config.yaml", artifact_path: str | Path = "models/fraud_ensemble.joblib") -> dict[str, float]:
    started_at = perf_counter()
    with Path(config_path).open() as config_file:
        config = yaml.safe_load(config_file)
    data_config = config["data"]
    seed = int(config["project"]["random_seed"])
    frame = load_training_data(
        data_config["raw_dir"], data_config["train_transaction"], data_config["train_identity"]
    )
    target = frame.pop(data_config["target_column"]).astype("int8")
    validation_config = config.get("validation", {})
    holdout_size = float(validation_config.get("test_size", 0.2))
    fold_count = int(validation_config.get("n_splits", 5))
    selection_indices, holdout_indices = train_test_split(
        np.arange(len(frame)), test_size=holdout_size, stratify=target, random_state=seed
    )
    selection_frame = frame.iloc[selection_indices].copy()
    selection_target = target.iloc[selection_indices].to_numpy()
    holdout_frame = frame.iloc[holdout_indices].copy()
    holdout_target = target.iloc[holdout_indices].to_numpy()

    model_settings = config.get("model", {})
    model_names = model_settings.get("models", list(DEFAULT_MODEL_CONFIGS))
    id_column = data_config["id_column"]
    feature_flags: dict[str, bool] = config.get("features", {})

    tune_hyperparameters = bool(model_settings.get("tune_hyperparameters", True))
    tuning_scores: dict[str, float] | None = None
    tuning_rows = 0
    if tune_hyperparameters:
        tuning_size = float(validation_config.get("tuning_size", 0.2))
        tune_train_idx, tune_valid_idx = train_test_split(
            np.arange(len(selection_frame)), test_size=tuning_size, stratify=selection_target, random_state=seed
        )
        tune_train_frame = selection_frame.iloc[tune_train_idx].copy()
        tune_valid_frame = selection_frame.iloc[tune_valid_idx].copy()
        tuning_transformer = TransactionFeatureEngineer(id_column, **feature_flags).fit(
            tune_train_frame, selection_target[tune_train_idx]
        )
        tune_train_features = tuning_transformer.transform(tune_train_frame)
        tune_valid_features = tuning_transformer.transform(tune_valid_frame)
        candidate_pool = model_settings.get("tuning_candidates") or DEFAULT_TUNING_CANDIDATES
        tuning_candidates = {name: candidate_pool[name] for name in model_names if name in candidate_pool}
        model_configs, tuning_scores = tune_model_configs(
            tune_train_features,
            selection_target[tune_train_idx],
            tune_valid_features,
            selection_target[tune_valid_idx],
            candidates=tuning_candidates,
            random_state=seed,
        )
        tuning_rows = int(len(tune_valid_idx))
    else:
        model_configs = {name: dict(DEFAULT_MODEL_CONFIGS[name]) for name in model_names}
        for name, parameters in model_configs.items():
            parameters["n_estimators"] = int(model_settings.get("n_estimators", parameters["n_estimators"]))
            parameters["learning_rate"] = float(model_settings.get("learning_rate", parameters["learning_rate"]))
            if name == "lightgbm":
                parameters["num_leaves"] = int(model_settings.get("num_leaves", parameters["num_leaves"]))

    oof_result = kfold_oof_validation(
        selection_frame,
        selection_target,
        model_configs,
        n_splits=fold_count,
        random_state=seed,
        id_column=id_column,
        feature_flags=feature_flags,
    )
    oof_predictions = oof_result["oof_predictions"]
    blend_weights, selected_oof_auc = optimize_oof_blend_weights(oof_predictions, selection_target)

    temporal_result = time_based_validation(
        selection_frame,
        selection_target,
        model_configs,
        random_state=seed,
        id_column=id_column,
        feature_flags=feature_flags,
    )

    selection_transformer = TransactionFeatureEngineer(id_column, **feature_flags).fit(
        selection_frame, selection_target
    )
    selection_features = selection_transformer.transform(selection_frame)
    holdout_features = selection_transformer.transform(holdout_frame)
    selection_class_weight = float((selection_target == 0).sum() / max((selection_target == 1).sum(), 1))
    holdout_model_scores: dict[str, float] = {}
    holdout_predictions: dict[str, np.ndarray] = {}
    for model_number, (name, parameters) in enumerate(model_configs.items()):
        model = build_model(
            name,
            {"random_state": seed + model_number, "n_jobs": -1, **parameters},
            selection_class_weight,
        )
        model.fit(selection_features, selection_target)
        probabilities = model.predict_proba(holdout_features)[:, 1]
        holdout_predictions[name] = probabilities
        holdout_model_scores[name] = float(roc_auc_score(holdout_target, probabilities))
    holdout_blend = sum(blend_weights[name] * holdout_predictions[name] for name in blend_weights)
    holdout_model_scores["blend"] = float(roc_auc_score(holdout_target, holdout_blend))

    evaluation_settings = config.get("evaluation", {})
    calibration_bins = int(evaluation_settings.get("calibration_bins", 10))
    cost_false_negative = float(evaluation_settings.get("cost_false_negative", 5.0))
    cost_false_positive = float(evaluation_settings.get("cost_false_positive", 1.0))
    threshold_resolution = float(evaluation_settings.get("threshold_resolution", 0.01))

    calibration_curve, calibration_error = compute_calibration_curve(
        holdout_target, holdout_blend, n_bins=calibration_bins
    )
    holdout_brier_score = brier_score(holdout_target, holdout_blend)
    selected_threshold, expected_cost, threshold_curve = select_decision_threshold(
        holdout_target,
        holdout_blend,
        cost_false_negative=cost_false_negative,
        cost_false_positive=cost_false_positive,
        resolution=threshold_resolution,
    )
    metrics_at_selected_threshold = next(
        row for row in threshold_curve if abs(row["threshold"] - selected_threshold) < 1e-9
    )

    final_transformer = TransactionFeatureEngineer(id_column, **feature_flags).fit(frame, target)
    final_features = final_transformer.transform(frame)
    full_class_weight = float((target.to_numpy() == 0).sum() / max((target.to_numpy() == 1).sum(), 1))
    final_models = {}
    for model_number, (name, parameters) in enumerate(model_configs.items()):
        model = build_model(
            name,
            {"random_state": seed + model_number, "n_jobs": -1, **parameters},
            full_class_weight,
        )
        model.fit(final_features, target.to_numpy())
        final_models[name] = model

    artifact_metadata = {
        "random_seed": seed,
        "model_configs": model_configs,
        "fraud_rate": float(target.mean()),
        "training_rows": int(len(frame)),
        "feature_count": int(final_features.shape[1]),
        "kfold_oof": oof_result["summary"],
        "time_validation": temporal_result["summary"],
        "time_validation_blend_weights": temporal_result["blend_weights"],
        "holdout_validation": holdout_model_scores,
        "blend_weights": blend_weights,
        "blend_oof_auc_selection": selected_oof_auc,
        "hyperparameter_tuning": {
            "enabled": tune_hyperparameters,
            "tuning_rows": tuning_rows,
            "scores": tuning_scores,
        },
        "decision_threshold": selected_threshold,
        "calibration": {
            "brier_score": holdout_brier_score,
            "expected_calibration_error": calibration_error,
            "bins": calibration_curve,
        },
        "threshold_selection": {
            "cost_false_negative": cost_false_negative,
            "cost_false_positive": cost_false_positive,
            "selected_threshold": selected_threshold,
            "expected_cost_at_selected": expected_cost,
            "metrics_at_selected": metrics_at_selected_threshold,
            "curve": threshold_curve,
        },
        "feature_version": "features-v1",
        "feature_flags": feature_flags,
        "training_fingerprint": hashlib.sha256(
            pd.util.hash_pandas_object(pd.DataFrame({id_column: frame[id_column], data_config["target_column"]: target}), index=False).values.tobytes()
        ).hexdigest(),
    }
    save_artifact(artifact_path, final_transformer, final_models, blend_weights, artifact_metadata)

    oof_output = pd.DataFrame({
        id_column: selection_frame[id_column].to_numpy() if id_column in selection_frame else selection_frame.index.to_numpy(),
        data_config["target_column"]: selection_target,
        "fold": oof_result["oof_folds"],
    })
    for name, predictions in oof_predictions.items():
        oof_output[f"oof_{name}"] = predictions
    oof_path = Path(config.get("reports", {}).get("oof_predictions", "reports/oof_predictions.csv"))
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_output.to_csv(oof_path, index=False)

    report_path = Path(config.get("reports", {}).get("validation_report", "reports/validation_report.json"))
    elapsed = perf_counter() - started_at
    write_validation_report(report_path, {
        "dataset": {"rows": int(len(frame)), "columns": int(frame.shape[1]), "fraud_rate": float(target.mean())},
        "validation_protocol": {"selection_rows": int(len(selection_frame)), "untouched_holdout_rows": int(len(holdout_frame)), "folds": oof_result["n_splits"]},
        "stratified_oof": oof_result["summary"],
        "time_validation": temporal_result["summary"],
        "time_validation_blend_weights": temporal_result["blend_weights"],
        "time_validation_split": {
            "train_rows": temporal_result["train_rows"],
            "valid_rows": temporal_result["valid_rows"],
            "train_max_time": temporal_result["train_max_time"],
            "valid_min_time": temporal_result["valid_min_time"],
            "split_threshold": temporal_result["split_threshold"],
        },
        "untouched_holdout": holdout_model_scores,
        "blend_weights": blend_weights,
        "blend_oof_auc_selection_only": selected_oof_auc,
        "hyperparameter_tuning": {
            "enabled": tune_hyperparameters,
            "tuning_rows": tuning_rows,
            "scores": tuning_scores,
        },
        "decision_threshold": selected_threshold,
        "calibration": {
            "brier_score": holdout_brier_score,
            "expected_calibration_error": calibration_error,
            "bins": calibration_curve,
        },
        "threshold_selection": {
            "cost_false_negative": cost_false_negative,
            "cost_false_positive": cost_false_positive,
            "selected_threshold": selected_threshold,
            "expected_cost_at_selected": expected_cost,
            "metrics_at_selected": metrics_at_selected_threshold,
            "curve": threshold_curve,
        },
        "feature_count": int(final_features.shape[1]),
        "feature_flags": feature_flags,
        "model_configs": model_configs,
        "training_seconds": elapsed,
        "training_fingerprint": artifact_metadata["training_fingerprint"],
        "kaggle_public_leaderboard_auc": None,
        "kaggle_private_leaderboard_auc": None,
    })

    print(f"Saved full-train artifact: {artifact_path}")
    print(f"Saved OOF predictions: {oof_path}")
    print(f"Saved validation report: {report_path}")
    if tune_hyperparameters:
        print(f"Tuned hyperparameters (validation scores): {tuning_scores}")
    print(f"Selected model configs: {model_configs}")
    print(f"OOF-selected blend weights: {blend_weights}")
    print(f"Untouched holdout ROC-AUC: {holdout_model_scores}")
    print(f"Time-based ROC-AUC: {temporal_result['summary']}")
    print(f"Calibration on holdout: Brier={holdout_brier_score:.4f}, ECE={calibration_error:.4f}")
    print(
        f"Selected decision threshold: {selected_threshold:.2f} "
        f"(expected cost {expected_cost:.1f} at cost_fn={cost_false_negative}, cost_fp={cost_false_positive})"
    )
    return holdout_model_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--artifact", default="models/fraud_ensemble.joblib")
    arguments = parser.parse_args()
    train(arguments.config, arguments.artifact)