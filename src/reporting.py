"""Reporting utilities for validation summaries and Kaggle submission outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.ensemble import optimize_oof_blend_weights


def generate_submission_file(
    predictions: pd.Series,
    transaction_ids: pd.Series,
    destination: str | Path,
    sample_submission: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write and validate a Kaggle submission, preserving sample ID order."""
    values = np.asarray(predictions, dtype="float64").reshape(-1)
    ids = pd.Series(transaction_ids).reset_index(drop=True)
    if len(values) != len(ids):
        raise ValueError("Prediction and transaction ID counts must match")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Transaction IDs must be present and unique")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")

    output = pd.DataFrame({"TransactionID": ids, "isFraud": values})
    if sample_submission is not None:
        required_columns = {"TransactionID", "isFraud"}
        if not required_columns.issubset(sample_submission.columns):
            raise ValueError("sample_submission must contain TransactionID and isFraud columns")
        sample_ids = sample_submission["TransactionID"]
        if sample_ids.isna().any() or sample_ids.duplicated().any():
            raise ValueError("Sample submission IDs must be present and unique")
        if len(sample_ids) != len(output) or set(sample_ids) != set(output["TransactionID"]):
            raise ValueError("Test TransactionIDs do not exactly match sample_submission.csv")
        output = output.set_index("TransactionID").loc[sample_ids.to_list()].reset_index()
    output = output[["TransactionID", "isFraud"]]

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    written = pd.read_csv(destination)
    if list(written.columns) != ["TransactionID", "isFraud"]:
        raise ValueError("Written submission has an invalid column schema")
    if len(written) != len(output) or written.isna().any().any():
        raise ValueError("Written submission row count or values are invalid")
    if not np.isfinite(written["isFraud"].to_numpy()).all() or not written["isFraud"].between(0, 1).all():
        raise ValueError("Written submission contains invalid probabilities")
    if sample_submission is not None and not written["TransactionID"].equals(sample_submission["TransactionID"].reset_index(drop=True)):
        raise ValueError("Written submission IDs do not match sample_submission order")
    return output


def write_validation_report(report_path: str | Path, payload: dict[str, Any]) -> str:
    """Persist a compact validation summary to disk as JSON."""
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return str(destination)
