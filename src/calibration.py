"""Calibration diagnostics and cost-aware decision-threshold selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss


def compute_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[list[dict[str, Any]], float]:
    """Bin predictions and compare mean predicted probability to observed fraud rate.

    Returns the per-bin curve plus the Expected Calibration Error (ECE): the
    count-weighted average gap between predicted and observed rates. ROC-AUC
    only measures ranking quality, not whether a predicted 0.2 corresponds to
    a real ~20% fraud rate, so this is a separate check.
    """
    y_true = np.asarray(y_true, dtype="float64").reshape(-1)
    y_prob = np.asarray(y_prob, dtype="float64").reshape(-1)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    total = len(y_true)
    curve: list[dict[str, Any]] = []
    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        count = int(mask.sum())
        entry: dict[str, Any] = {
            "bin_lower": float(edges[bin_index]),
            "bin_upper": float(edges[bin_index + 1]),
            "count": count,
            "mean_predicted": None,
            "observed_fraud_rate": None,
        }
        if count > 0:
            mean_predicted = float(y_prob[mask].mean())
            observed_rate = float(y_true[mask].mean())
            entry["mean_predicted"] = mean_predicted
            entry["observed_fraud_rate"] = observed_rate
            ece += (count / total) * abs(mean_predicted - observed_rate)
        curve.append(entry)
    return curve, float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes."""
    return float(brier_score_loss(np.asarray(y_true).reshape(-1), np.asarray(y_prob).reshape(-1)))


def select_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_false_negative: float = 5.0,
    cost_false_positive: float = 1.0,
    resolution: float = 0.01,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Sweep thresholds and pick the one minimizing expected misclassification cost.

    A missed fraud (false negative) and a wrongly blocked transaction (false
    positive) rarely cost the same, so this reports precision/recall/F1 and an
    expected-cost estimate at every threshold rather than defaulting to 0.5.
    """
    y_true = np.asarray(y_true, dtype="int64").reshape(-1)
    y_prob = np.asarray(y_prob, dtype="float64").reshape(-1)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Threshold selection requires both target classes")
    if not (0.0 < resolution < 1.0):
        raise ValueError("resolution must be between 0 and 1")

    thresholds = np.arange(resolution, 1.0, resolution)
    positives = y_true == 1
    negatives = ~positives

    curve: list[dict[str, Any]] = []
    best_threshold = float(thresholds[0])
    best_cost = np.inf
    for threshold in thresholds:
        predicted_positive = y_prob >= threshold
        true_positives = int((predicted_positive & positives).sum())
        false_positives = int((predicted_positive & negatives).sum())
        false_negatives = int((~predicted_positive & positives).sum())
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        expected_cost = false_positives * cost_false_positive + false_negatives * cost_false_negative
        curve.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "expected_cost": float(expected_cost),
            }
        )
        if expected_cost < best_cost:
            best_cost = expected_cost
            best_threshold = float(threshold)

    return best_threshold, float(best_cost), curve
