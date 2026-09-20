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
) -> pd.DataFrame:
    """Write a Kaggle-style test submission with a TransactionID and isFraud column."""
    output = pd.DataFrame({"TransactionID": transaction_ids.to_numpy(), "isFraud": np.clip(predictions.to_numpy(), 0.0, 1.0)})
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return output


def write_validation_report(report_path: str | Path, payload: dict[str, Any]) -> str:
    """Persist a compact validation summary to disk as JSON."""
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return str(destination)
