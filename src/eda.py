"""Correlation analysis for card/client-identity related columns.

There is no real customer ID in this dataset, so `uid` (see
src/features/preprocessing.py) reconstructs one from card1/card2/card3/card5/
addr1/addr2 plus a D1-derived day offset. This module answers the question
that choice of fields depends on: which of these fields actually carry
distinct client-identity information, and which are redundant with each
other or with the fraud target?
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display needed for scripts, tests, or CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.data.loading import load_training_data

DEFAULT_CARD_COLUMNS: tuple[str, ...] = (
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "D1",
    "D2",
    "D3",
    "D4",
    "D10",
    "D11",
    "D15",
    "TransactionAmt",
)


def compute_card_correlation(
    frame: pd.DataFrame,
    target: pd.Series | np.ndarray | None = None,
    columns: tuple[str, ...] = DEFAULT_CARD_COLUMNS,
) -> pd.DataFrame:
    """Pairwise Pearson correlation among numeric card/client-identity columns.

    High correlation between two fields suggests they carry overlapping
    identity information - redundant as a pair inside a composite key like
    `uid`. Low correlation among a candidate set suggests each contributes
    distinct signal. Including the target (as an extra "isFraud" row/column)
    shows which of these fields are individually predictive versus just along
    for the ride.
    """
    available = [column for column in columns if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])]
    if len(available) < 2:
        raise ValueError("Need at least two numeric columns to compute a correlation matrix")
    subset = frame[available].copy()
    if target is not None:
        subset["isFraud"] = pd.Series(np.asarray(target), index=frame.index)
    return subset.corr()


def plot_card_correlation_heatmap(
    correlation: pd.DataFrame,
    output_path: str | Path = "reports/card_correlation_heatmap.png",
    title: str = "Card / client-identity field correlation",
) -> str:
    """Render and save a correlation heatmap; returns the written path."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    side = len(correlation.columns)
    fig, ax = plt.subplots(figsize=(max(8, side * 0.7), max(6, side * 0.6)))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": "Pearson correlation"},
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return str(destination)


def summarize_high_correlation_pairs(correlation: pd.DataFrame, threshold: float = 0.5) -> list[dict[str, Any]]:
    """List column pairs (excluding the diagonal) with |correlation| >= threshold, sorted strongest-first."""
    pairs: list[dict[str, Any]] = []
    columns = list(correlation.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({"left": left, "right": right, "correlation": float(value)})
    return sorted(pairs, key=lambda entry: abs(entry["correlation"]), reverse=True)


def run_from_config(
    config_path: str | Path = "config/config.yaml",
    output_path: str | Path = "reports/card_correlation_heatmap.png",
    columns: tuple[str, ...] = DEFAULT_CARD_COLUMNS,
) -> pd.DataFrame:
    with Path(config_path).open() as config_file:
        config = yaml.safe_load(config_file)
    data_config = config["data"]

    frame = load_training_data(data_config["raw_dir"], data_config["train_transaction"], data_config["train_identity"])
    target = frame.pop(data_config["target_column"])

    correlation = compute_card_correlation(frame, target, columns)
    path = plot_card_correlation_heatmap(correlation, output_path)

    print(f"Saved heatmap to {path}")
    print(correlation.round(3).to_string())
    print()
    print("Strongest pairs (|corr| >= 0.5):")
    for entry in summarize_high_correlation_pairs(correlation, threshold=0.5):
        print(f"  {entry['left']:15s} <-> {entry['right']:15s}  corr={entry['correlation']:+.3f}")
    return correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="reports/card_correlation_heatmap.png")
    arguments = parser.parse_args()
    run_from_config(arguments.config, arguments.output)
