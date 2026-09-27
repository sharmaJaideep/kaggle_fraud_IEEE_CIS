"""Adversarial validation: how easily can a model tell train rows from test rows?

A high OOF ROC-AUC here means the train and test sets are genuinely different
distributions, not just the same population sampled twice - which is exactly
the kind of drift a random validation split can't detect, but that hurts real
leaderboard performance. The most important features for that classifier are
the specific columns driving the shift, which is what to prioritize fixing
before spending time on other feature engineering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data.loading import load_training_data
from src.models.baseline import build_model


def _encode_for_adversarial_check(combined: pd.DataFrame) -> pd.DataFrame:
    """Factorize categorical columns jointly so unseen test-only categories
    don't crash, leaving numeric columns as-is except for a missing sentinel."""
    encoded = combined.copy()
    for column in encoded.select_dtypes(include=["object", "string", "category"]).columns:
        codes, _ = pd.factorize(encoded[column].astype("string"))
        encoded[column] = codes.astype("int32")
    return encoded.replace([np.inf, -np.inf], np.nan).fillna(-999).astype("float32")


def run_adversarial_validation(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    id_column: str = "TransactionID",
    target_column: str = "isFraud",
    n_splits: int = 5,
    random_state: int = 42,
    top_n: int = 25,
) -> dict[str, Any]:
    """Train a classifier to distinguish train rows from test rows.

    id_column is dropped because it is a row-assignment artifact (test IDs
    are a disjoint, higher range than train IDs), not a real feature - it
    would trivially and uninformatively dominate the ranking otherwise.
    """
    train_features = train_frame.drop(columns=[id_column, target_column], errors="ignore").copy()
    test_features = test_frame.drop(columns=[id_column, target_column], errors="ignore").copy()

    shared_columns = [column for column in train_features.columns if column in test_features.columns]
    train_features = train_features[shared_columns]
    test_features = test_features[shared_columns]

    labels = np.concatenate([np.zeros(len(train_features)), np.ones(len(test_features))])
    combined_raw = pd.concat([train_features, test_features], ignore_index=True)
    combined_encoded = _encode_for_adversarial_check(combined_raw)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.zeros(len(combined_encoded), dtype="float64")
    fold_aucs: list[float] = []
    importances = np.zeros(combined_encoded.shape[1], dtype="float64")

    for fold_number, (train_idx, valid_idx) in enumerate(skf.split(combined_encoded, labels), start=1):
        model = build_model(
            "lightgbm",
            {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "random_state": random_state + fold_number,
                "n_jobs": -1,
            },
            scale_pos_weight=1.0,
        )
        model.fit(combined_encoded.iloc[train_idx], labels[train_idx])
        fold_predictions = model.predict_proba(combined_encoded.iloc[valid_idx])[:, 1]
        oof_predictions[valid_idx] = fold_predictions
        fold_aucs.append(float(roc_auc_score(labels[valid_idx], fold_predictions)))
        importances += model.feature_importances_.astype("float64")

    importances /= n_splits
    overall_auc = float(roc_auc_score(labels, oof_predictions))

    categorical_columns = set(train_features.select_dtypes(include=["object", "string", "category"]).columns)
    ranking = sorted(zip(shared_columns, importances), key=lambda pair: pair[1], reverse=True)
    top_features: list[dict[str, Any]] = []
    for column, importance in ranking[:top_n]:
        entry: dict[str, Any] = {"feature": column, "importance": float(importance)}
        if column in categorical_columns:
            train_top = train_features[column].value_counts(normalize=True, dropna=False).head(3)
            test_top = test_features[column].value_counts(normalize=True, dropna=False).head(3)
            entry["kind"] = "categorical"
            entry["train_top_values"] = {str(key): float(value) for key, value in train_top.items()}
            entry["test_top_values"] = {str(key): float(value) for key, value in test_top.items()}
        else:
            entry["kind"] = "numeric"
            entry["train_mean"] = float(pd.to_numeric(train_features[column], errors="coerce").mean())
            entry["test_mean"] = float(pd.to_numeric(test_features[column], errors="coerce").mean())
            entry["train_missing_rate"] = float(train_features[column].isna().mean())
            entry["test_missing_rate"] = float(test_features[column].isna().mean())
        top_features.append(entry)

    return {
        "n_train": int(len(train_features)),
        "n_test": int(len(test_features)),
        "n_shared_columns": len(shared_columns),
        "fold_aucs": fold_aucs,
        "oof_auc": overall_auc,
        "top_features": top_features,
    }


def run_from_config(
    config_path: str | Path = "config/config.yaml",
    output_path: str | Path = "reports/adversarial_validation.json",
    top_n: int = 25,
) -> dict[str, Any]:
    with Path(config_path).open() as config_file:
        config = yaml.safe_load(config_file)
    data_config = config["data"]
    seed = int(config["project"]["random_seed"])
    raw_dir = data_config["raw_dir"]

    train_frame = load_training_data(raw_dir, data_config["train_transaction"], data_config["train_identity"])
    test_frame = load_training_data(
        raw_dir,
        data_config.get("test_transaction", "test_transaction.csv"),
        data_config.get("test_identity", "test_identity.csv"),
    )

    result = run_adversarial_validation(
        train_frame,
        test_frame,
        id_column=data_config["id_column"],
        target_column=data_config["target_column"],
        random_state=seed,
        top_n=top_n,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print(f"Adversarial validation OOF AUC: {result['oof_auc']:.4f} (0.5 = indistinguishable, 1.0 = fully separable)")
    print("Top drifting features:")
    for entry in result["top_features"][:10]:
        print(f"  {entry['feature']:25s} importance={entry['importance']:.1f}  kind={entry['kind']}")
    print(f"Saved report to {destination}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="reports/adversarial_validation.json")
    parser.add_argument("--top-n", type=int, default=25)
    arguments = parser.parse_args()
    run_from_config(arguments.config, arguments.output, arguments.top_n)
