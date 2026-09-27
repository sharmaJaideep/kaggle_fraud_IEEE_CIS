"""Leakage-conscious, reusable feature engineering for transaction records."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create numeric features and retain the exact state needed at inference.

    Aggregation and frequency lookup tables are fitted on the training frame only.
    Unknown categories and groups receive stable fallback values, which makes the
    transformer suitable for both batch validation and one-row API requests.
    """

    def __init__(self, id_column: str = "TransactionID") -> None:
        self.id_column = id_column
        self.group_columns = ("card1", "addr1", "P_emaildomain", "ProductCD", "uid")
        self.interaction_columns = (
            ("card1", "addr1"),
            ("card1", "P_emaildomain"),
            ("ProductCD", "card4"),
        )

    def _base_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = frame.copy()
        features = features.drop(columns=[self.id_column, "isFraud"], errors="ignore")

        numeric = features.select_dtypes(include=[np.number]).columns
        features["missing_count"] = features.isna().sum(axis=1).astype("int16")
        features["has_missing"] = (features["missing_count"] > 0).astype("int8")
        if "TransactionAmt" in features:
            features["TransactionAmt_log1p"] = np.log1p(features["TransactionAmt"].clip(lower=0))
        if "TransactionDT" in features:
            seconds = features["TransactionDT"]
            features["transaction_hour"] = ((seconds // 3600) % 24).astype("float32")
            features["transaction_day"] = (seconds // 86400).astype("float32")
            features["transaction_week"] = (seconds // (86400 * 7)).astype("float32")
        for left, right in self.interaction_columns:
            if left in features and right in features:
                features[f"{left}__{right}"] = (
                    features[left].astype("string").fillna("__MISSING__")
                    + "__"
                    + features[right].astype("string").fillna("__MISSING__")
                )

        uid_components = [
            column for column in ("card1", "card2", "card3", "card5", "addr1", "addr2") if column in features
        ]
        if uid_components:
            uid = features[uid_components[0]].astype("string").fillna("__MISSING__")
            for column in uid_components[1:]:
                uid = uid + "_" + features[column].astype("string").fillna("__MISSING__")
            if "D1" in features and "TransactionDT" in features:
                # D1 tracks days since account creation, so D1 minus elapsed days is a
                # near-constant residual per client even as TransactionDT grows, which
                # reconstructs a pseudo customer identity far more reliably than raw
                # card/address fields alone (cards get reused across real customers).
                account_day = np.floor(
                    pd.to_numeric(features["D1"], errors="coerce") - features["TransactionDT"] / 86400.0
                )
                uid = uid + "_" + account_day.astype("string").fillna("__MISSING__")
            features["uid"] = uid

            if "TransactionDT" in features:
                # Relative to whatever rows are in this transform() call, not full
                # history: a single-row API payload has no prior transaction to compare
                # against, and falls back to the fitted median via numeric_medians_.
                by_time = features.sort_values("TransactionDT", kind="stable")
                features["uid_seconds_since_last_transaction"] = (
                    by_time.groupby("uid")["TransactionDT"].diff().reindex(features.index)
                )
                features["uid_transactions_seen_before"] = (
                    by_time.groupby("uid").cumcount().reindex(features.index).astype("float32")
                )

        return features

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TransactionFeatureEngineer":
        frame = self._base_features(X)
        self.input_columns_ = list(frame.columns)
        self.categorical_maps_: dict[str, dict[str, int]] = {}
        self.frequency_maps_: dict[str, dict[str, int]] = {}
        for column in frame.select_dtypes(include=["object", "string", "category"]).columns:
            values = frame[column].astype("string").fillna("__MISSING__")
            categories = pd.Index(values.unique())
            self.categorical_maps_[column] = {str(value): index for index, value in enumerate(categories)}
            self.frequency_maps_[column] = values.value_counts(dropna=False).astype("int64").to_dict()

        self.numeric_medians_ = frame.select_dtypes(include=[np.number]).median().to_dict()
        self.group_stats_: dict[str, pd.DataFrame] = {}
        for column in self.group_columns:
            if column not in frame or "TransactionAmt" not in frame:
                continue
            group_key = frame[column].astype("string").fillna("__MISSING__")
            stats = frame.assign(_group_key=group_key).groupby("_group_key")["TransactionAmt"].agg(
                group_count="size", group_amount_mean="mean", group_amount_std="std", group_amount_median="median"
            )
            self.group_stats_[column] = stats.fillna(0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._base_features(X)
        # API payloads may contain only a subset of the columns seen during training.
        frame = frame.reindex(columns=self.input_columns_, fill_value=np.nan)
        for column, frequencies in self.frequency_maps_.items():
            values = frame[column].astype("string").fillna("__MISSING__")
            frame[f"{column}__frequency"] = values.map(frequencies).fillna(0).astype("float32")
        for column, stats in self.group_stats_.items():
            values = frame[column].astype("string").fillna("__MISSING__")
            for statistic in stats.columns:
                frame[f"{column}__{statistic}"] = values.map(stats[statistic]).fillna(0).astype("float32")
            if "TransactionAmt" in frame:
                mean = frame[f"{column}__group_amount_mean"]
                frame[f"{column}__amount_delta"] = (frame["TransactionAmt"] - mean).astype("float32")

        for column, mapping in self.categorical_maps_.items():
            values = frame[column].astype("string").fillna("__MISSING__")
            frame[column] = values.map(mapping).fillna(-1).astype("int32")
        frame = frame.drop(columns=[column for column in frame.columns if frame[column].dtype.name in {"object", "string", "category"}])
        frame = frame.reindex(columns=self.get_feature_names_out(), fill_value=0)
        for column, median in self.numeric_medians_.items():
            if column in frame:
                frame[column] = frame[column].fillna(median)
        return frame.replace([np.inf, -np.inf], np.nan).fillna(-999).astype("float32")

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "input_columns_"):
            raise RuntimeError("TransactionFeatureEngineer must be fitted before feature names are requested.")
        names = list(self.input_columns_)
        names.extend(f"{column}__frequency" for column in self.frequency_maps_)
        for column, stats in self.group_stats_.items():
            names.extend(f"{column}__{statistic}" for statistic in stats.columns)
            names.append(f"{column}__amount_delta")
        return np.asarray([name for name in names if name not in self.categorical_maps_ or name in self.input_columns_])


def encode_categoricals(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper using the fitted feature engineer."""
    transformer = TransactionFeatureEngineer().fit(train)
    return transformer.transform(train), transformer.transform(valid)
