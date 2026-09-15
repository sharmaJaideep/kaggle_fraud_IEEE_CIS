"""Data loading utilities for the IEEE-CIS dataset."""

from pathlib import Path

import pandas as pd


def reduce_memory_usage(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns while preserving values and missingness."""
    for column in dataframe.columns:
        dtype = dataframe[column].dtype
        if pd.api.types.is_float_dtype(dtype):
            dataframe[column] = pd.to_numeric(dataframe[column], downcast="float")
        elif pd.api.types.is_integer_dtype(dtype):
            dataframe[column] = pd.to_numeric(dataframe[column], downcast="integer")
    return dataframe


def load_training_data(raw_dir: str | Path, transaction_file: str, identity_file: str) -> pd.DataFrame:
    """Load and left-join transaction and identity training data."""
    raw_path = Path(raw_dir)
    transactions = pd.read_csv(raw_path / transaction_file)
    identity = pd.read_csv(raw_path / identity_file)
    merged = transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    return reduce_memory_usage(merged)
