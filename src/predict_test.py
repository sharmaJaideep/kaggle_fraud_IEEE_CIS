"""Generate a Kaggle-style submission file from the trained artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.loading import load_training_data
from src.models.ensemble import load_artifact, predict_artifact
from src.reporting import generate_submission_file


def generate_submission(
    config_path: str | Path = "config/config.yaml",
    artifact_path: str | Path = "models/fraud_ensemble.joblib",
    output_path: str | Path = "reports/submission.csv",
) -> pd.DataFrame:
    config = __import__("yaml").safe_load(Path(config_path).read_text())
    data_config = config["data"]
    raw_dir = data_config["raw_dir"]
    transactions = pd.read_csv(Path(raw_dir) / "test_transaction.csv")
    identity = pd.read_csv(Path(raw_dir) / "test_identity.csv")
    test_frame = transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    sample_submission = pd.read_csv(Path(raw_dir) / "sample_submission.csv")
    artifact = load_artifact(artifact_path)
    predictions = predict_artifact(artifact, test_frame)
    submission = generate_submission_file(predictions=pd.Series(predictions), transaction_ids=test_frame["TransactionID"], destination=output_path)

    if len(submission) != len(sample_submission):
        raise ValueError(f"Submission row count mismatch: expected {len(sample_submission)}, got {len(submission)}")
    if not submission["TransactionID"].equals(sample_submission["TransactionID"]):
        raise ValueError("Submission TransactionID values do not match sample_submission.csv")
    if submissions_invalid := submission["isFraud"].isna().any() or ((submission["isFraud"] < 0) | (submission["isFraud"] > 1)).any():
        raise ValueError("Submission probabilities are invalid")
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--artifact", default="models/fraud_ensemble.joblib")
    parser.add_argument("--output", default="reports/submission.csv")
    args = parser.parse_args()
    generate_submission(args.config, args.artifact, args.output)
    print(f"Saved submission to {args.output}")
