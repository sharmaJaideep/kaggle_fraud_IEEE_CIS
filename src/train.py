"""Train the feature-engineered ensemble and save one serving artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.data.loading import load_training_data
from src.features.preprocessing import TransactionFeatureEngineer
from src.models.ensemble import fit_validation_weighted_ensemble, save_artifact, tune_model_configs
from src.reporting import optimize_oof_blend_weights, write_validation_report
from src.validation import kfold_oof_validation, time_based_validation


def train(config_path: str | Path = "config/config.yaml", artifact_path: str | Path = "models/fraud_ensemble.joblib") -> dict[str, float]:
    with Path(config_path).open() as config_file:
        config = yaml.safe_load(config_file)
    data_config = config["data"]
    frame = load_training_data(
        data_config["raw_dir"], data_config["train_transaction"], data_config["train_identity"]
    )
    target = frame.pop(data_config["target_column"]).astype("int8")

    model_configs = {
        "lightgbm": {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1},
        "xgboost": {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5},
        "catboost": {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 6, "l2_leaf_reg": 5},
    }

    kfold_result = kfold_oof_validation(frame, target, model_configs, n_splits=5, random_state=config["project"]["random_seed"])
    oof_predictions = kfold_result["oof_predictions"]
    oof_weights, oof_score = optimize_oof_blend_weights(oof_predictions, target.to_numpy())

    time_result = time_based_validation(frame, target, model_configs, random_state=config["project"]["random_seed"])
    train_frame = frame.copy()
    valid_frame = frame.copy()
    y_train = target.copy()
    y_valid = target.copy()
    transformer = TransactionFeatureEngineer(data_config["id_column"]).fit(train_frame)
    X_train = transformer.transform(train_frame)
    X_valid = transformer.transform(valid_frame)
    tuned_configs, tuning_scores = tune_model_configs(
        X_train, y_train, X_valid, y_valid, random_state=config["project"]["random_seed"]
    )
    models, weights, scores = fit_validation_weighted_ensemble(
        X_train, y_train, X_valid, y_valid, model_configs=tuned_configs, random_state=config["project"]["random_seed"]
    )

    artifact_metadata = {
        "random_seed": config["project"]["random_seed"],
        "validation_scores": scores,
        "tuning_scores": tuning_scores,
        "fraud_rate": float(np.mean(target)),
        "kfold_oof": kfold_result["summary"],
        "time_validation": time_result["summary"],
        "blend_weights": oof_weights,
        "blend_oof_auc": oof_score,
        "feature_version": "features-v1",
    }
    save_artifact(
        artifact_path,
        transformer,
        models,
        weights,
        artifact_metadata,
    )

    report_path = Path("reports/validation_report.json")
    write_validation_report(report_path, {
        "kfold_oof": kfold_result["summary"],
        "time_validation": time_result["summary"],
        "blend_weights": oof_weights,
        "blend_oof_auc": oof_score,
        "validation_scores": scores,
    })

    print(f"Saved artifact: {artifact_path}")
    print(f"Saved report: {report_path}")
    for name, score in scores.items():
        print(f"{name} validation ROC-AUC: {score:.6f}")
    print(f"Tuning scores: {tuning_scores}")
    print(f"Blend weights: {weights}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--artifact", default="models/fraud_ensemble.joblib")
    arguments = parser.parse_args()
    train(arguments.config, arguments.artifact)