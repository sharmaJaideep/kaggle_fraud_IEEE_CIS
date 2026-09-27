"""Generate a Kaggle-style submission file from the trained artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.models.ensemble import load_artifact, predict_artifact
from src.reporting import generate_submission_file


def generate_submission(
    config_path: str | Path = "config/config.yaml",
    artifact_path: str | Path = "models/fraud_ensemble.joblib",
    output_path: str | Path = "reports/submission.csv",
) -> pd.DataFrame:
    with Path(config_path).open() as config_file:
        config = yaml.safe_load(config_file)
    data_config = config["data"]
    raw_dir = data_config["raw_dir"]
    transactions = pd.read_csv(Path(raw_dir) / data_config.get("test_transaction", "test_transaction.csv"))
    identity = pd.read_csv(Path(raw_dir) / data_config.get("test_identity", "test_identity.csv"))
    id_column = data_config["id_column"]
    test_frame = transactions.merge(identity, on=id_column, how="left", validate="one_to_one")
    sample_submission = pd.read_csv(Path(raw_dir) / data_config.get("sample_submission", "sample_submission.csv"))
    artifact = load_artifact(artifact_path)
    predictions = predict_artifact(artifact, test_frame)
    return generate_submission_file(
        predictions=pd.Series(predictions),
        transaction_ids=test_frame[id_column],
        destination=output_path,
        sample_submission=sample_submission,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--artifact", default="models/fraud_ensemble.joblib")
    parser.add_argument("--output", default="reports/submission.csv")
    args = parser.parse_args()
    generate_submission(args.config, args.artifact, args.output)
    print(f"Saved submission to {args.output}")
