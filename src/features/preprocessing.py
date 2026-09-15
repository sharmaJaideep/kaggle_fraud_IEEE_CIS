"""Feature preparation helpers for tree-based baseline models."""

import pandas as pd


def encode_categoricals(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode object/category columns consistently across train and validation."""
    train_encoded = train.copy()
    valid_encoded = valid.copy()
    for column in train_encoded.columns:
        if train_encoded[column].dtype == "object" or pd.api.types.is_categorical_dtype(train_encoded[column]):
            combined = pd.concat([train_encoded[column], valid_encoded[column]], axis=0).astype("category")
            train_encoded[column] = combined.iloc[: len(train_encoded)].cat.codes.astype("int32")
            valid_encoded[column] = combined.iloc[len(train_encoded) :].cat.codes.astype("int32")
    return train_encoded.fillna(-999), valid_encoded.fillna(-999)
