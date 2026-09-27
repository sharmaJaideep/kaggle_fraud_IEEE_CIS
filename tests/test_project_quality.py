import json

import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient

from app import app
from src.adversarial_validation import run_adversarial_validation
from src.calibration import compute_calibration_curve, select_decision_threshold
from src.features.preprocessing import TransactionFeatureEngineer
from src.models.ensemble import load_artifact, optimize_oof_blend_weights, predict_artifact, save_artifact
from src.predict_test import generate_submission
from src.train import train
from src.validation import kfold_oof_validation, time_based_validation


def _make_frame(rows=40, fraud_rate=0.2):
    rng = np.random.default_rng(7)
    rows_df = []
    for _ in range(rows):
        rows_df.append(
            {
                "TransactionID": _ + 1,
                "TransactionAmt": float(rng.integers(5, 200)),
                "card1": int(rng.integers(1000, 2000)),
                "card2": int(rng.integers(100, 200)),
                "addr1": int(rng.integers(10, 100)),
                "ProductCD": "W" if rng.random() > 0.3 else "C",
                "card4": "visa" if rng.random() > 0.5 else "mastercard",
                "P_emaildomain": "gmail.com" if rng.random() > 0.4 else "yahoo.com",
                "isFraud": 1 if rng.random() < fraud_rate else 0,
            }
        )
    return pd.DataFrame(rows_df)


def test_feature_engineer_produces_numeric_matrix_without_object_columns():
    train = _make_frame(60, fraud_rate=0.3)
    valid = _make_frame(20, fraud_rate=0.2)
    transformer = TransactionFeatureEngineer().fit(train)
    transformed_train = transformer.transform(train)
    transformed_valid = transformer.transform(valid)

    assert transformed_train.columns.tolist() == transformed_valid.columns.tolist()
    assert transformed_train.shape[0] == len(train)
    assert transformed_valid.shape[0] == len(valid)
    assert not any(pd.api.types.is_object_dtype(dtype) for dtype in transformed_train.dtypes)
    assert not np.isinf(transformed_train.to_numpy()).any()


def test_uid_reconstruction_groups_shared_cards_and_tracks_time_since_last_transaction():
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [1000, 5000, 2000, 100000],
            "TransactionAmt": [10.0, 20.0, 15.0, 50.0],
            "card1": [111, 111, 111, 222],
            "card2": [1, 1, 1, 2],
            "addr1": [50, 50, 50, 99],
            "isFraud": [0, 0, 0, 1],
        }
    )
    transformer = TransactionFeatureEngineer().fit(frame)
    transformed = transformer.transform(frame)

    assert "uid_seconds_since_last_transaction" in transformed.columns
    assert "uid_transactions_seen_before" in transformed.columns
    assert any(name.startswith("uid__") for name in transformed.columns)

    counts = transformed["uid_transactions_seen_before"]
    gaps = transformed["uid_seconds_since_last_transaction"]

    # Rows 0-2 share the same card1/card2/addr1 combo (chronological order: 0, 2, 1);
    # row 3 has a distinct combo and is its own client with no prior transaction.
    assert counts.iloc[0] == 0
    assert counts.iloc[2] == 1
    assert counts.iloc[1] == 2
    assert counts.iloc[3] == 0
    assert gaps.iloc[2] == 1000
    assert gaps.iloc[1] == 3000

    # A single-row transform (no history in the batch) should not crash and should
    # fall back to a fitted default rather than leaving a raw NaN in the matrix.
    single_row = transformer.transform(frame.iloc[[1]])
    assert np.isfinite(single_row["uid_seconds_since_last_transaction"].to_numpy()).all()


def test_ablation_flags_each_independently_disable_their_behavior():
    rng = np.random.default_rng(9)
    n = 200
    base = rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.arange(n) * 3600,
            "isFraud": rng.integers(0, 2, n),
            "TransactionAmt": rng.uniform(5, 200, n),
            "card1": np.tile(np.arange(1, 11), n // 10),
            "card2": np.tile(np.arange(1, 11), n // 10),
            "addr1": np.tile(np.arange(1, 11), n // 10),
            "V1": base,
            "V2": base + rng.normal(0, 1e-4, n),
        }
    )

    default_transformer = TransactionFeatureEngineer().fit(frame)
    default_columns = set(default_transformer.transform(frame).columns)
    assert "uid_seconds_since_last_transaction" in default_columns
    assert "V2" not in default_columns
    assert "card1" not in default_columns

    no_uid_time = TransactionFeatureEngineer(enable_uid_time_features=False).fit(frame)
    no_uid_time_columns = set(no_uid_time.transform(frame).columns)
    assert "uid_seconds_since_last_transaction" not in no_uid_time_columns
    assert "uid_transactions_seen_before" not in no_uid_time_columns
    assert any(name.startswith("uid__") for name in no_uid_time_columns)  # uid grouping still active

    no_v_pruning = TransactionFeatureEngineer(enable_v_column_pruning=False).fit(frame)
    assert no_v_pruning.dropped_v_columns_ == []
    no_v_pruning_columns = set(no_v_pruning.transform(frame).columns)
    assert "V2" in no_v_pruning_columns

    keep_raw_ids = TransactionFeatureEngineer(drop_raw_id_columns=False).fit(frame)
    keep_raw_ids_columns = set(keep_raw_ids.transform(frame).columns)
    assert "card1" in keep_raw_ids_columns
    assert "card2" in keep_raw_ids_columns
    assert "addr1" in keep_raw_ids_columns


def test_artifact_round_trip_and_predictions_are_valid():
    train = _make_frame(80, fraud_rate=0.25)
    transformer = TransactionFeatureEngineer().fit(train)
    X_train = transformer.transform(train)

    class DummyModel:
        def predict_proba(self, features):
            feature_values = np.asarray(features)
            scores = np.clip(feature_values[:, 0], 0, 1)
            return np.column_stack([1.0 - scores, scores])

    model = DummyModel()
    save_artifact("/tmp/test_artifact.joblib", transformer, {"dummy": model}, {"dummy": 1.0}, {"version": "test"})
    artifact = load_artifact("/tmp/test_artifact.joblib")
    preds = predict_artifact(artifact, train.iloc[:5].to_dict(orient="records"))

    assert preds.shape[0] == 5
    assert ((0.0 <= preds) & (preds <= 1.0)).all()


def test_kfold_oof_validation_and_blend_optimization():
    frame = _make_frame(120, fraud_rate=0.35)
    target = frame["isFraud"].astype(int)
    model_configs = {
        "lightgbm": {"n_estimators": 30, "learning_rate": 0.05, "num_leaves": 15},
        "xgboost": {"n_estimators": 30, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 1},
    }

    result = kfold_oof_validation(frame, target, model_configs, n_splits=3, random_state=42)

    assert set(result["oof_predictions"]) == {"lightgbm", "xgboost"}
    assert all(len(preds) == len(frame) for preds in result["oof_predictions"].values())
    assert set(result["summary"]).issuperset({"lightgbm", "xgboost"})

    weights, score = optimize_oof_blend_weights(result["oof_predictions"], target.to_numpy())
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert 0.0 <= score <= 1.0

    labels = np.tile([0, 1], 30)
    perfect = np.where(labels == 1, 0.9, 0.1)
    inverted = 1.0 - perfect
    weights, score = optimize_oof_blend_weights(
        {"best": perfect, "weak": inverted, "also_weak": inverted}, labels
    )
    assert weights["best"] >= 0.55
    assert score == 1.0


def test_time_validation_keeps_chronological_boundary():
    frame = _make_frame(100, fraud_rate=0.4)
    frame["TransactionDT"] = np.repeat(np.arange(20), 5)
    target = frame.pop("isFraud")
    shuffled = np.random.default_rng(11).permutation(len(frame))
    frame = frame.iloc[shuffled].reset_index(drop=True)
    target = target.iloc[shuffled].reset_index(drop=True)

    result = time_based_validation(
        frame,
        target,
        {"lightgbm": {"n_estimators": 5, "num_leaves": 7}},
        random_state=7,
    )

    assert result["train_max_time"] < result["valid_min_time"]
    assert result["valid_min_time"] == result["split_threshold"]
    assert result["train_rows"] + result["valid_rows"] == len(frame)
    assert "blend" in result["summary"]
    assert result["blend_weights"] == {"lightgbm": 1.0}


def test_training_to_kaggle_submission_pipeline(tmp_path):
    raw_dir = tmp_path / "data"
    raw_dir.mkdir()
    train_rows = 100
    train_data = pd.DataFrame(
        {
            "TransactionID": np.arange(1, train_rows + 1),
            "TransactionDT": np.arange(train_rows) * 3600,
            "TransactionAmt": np.linspace(5.0, 150.0, train_rows),
            "card1": np.tile(np.arange(10, 20), train_rows // 10),
            "isFraud": np.tile([0, 0, 0, 1, 0], train_rows // 5),
        }
    )
    train_data.to_csv(raw_dir / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": train_data["TransactionID"], "DeviceType": "mobile"}).to_csv(
        raw_dir / "train_identity.csv", index=False
    )

    test_ids = np.arange(1001, 1013)
    pd.DataFrame(
        {
            "TransactionID": test_ids,
            "TransactionDT": np.arange(train_rows, train_rows + len(test_ids)) * 3600,
            "TransactionAmt": np.linspace(20.0, 80.0, len(test_ids)),
            "card1": np.tile(np.arange(10, 16), 2),
        }
    ).to_csv(raw_dir / "test_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": test_ids, "DeviceType": "mobile"}).to_csv(
        raw_dir / "test_identity.csv", index=False
    )
    pd.DataFrame({"TransactionID": test_ids[::-1], "isFraud": 0.0}).to_csv(
        raw_dir / "sample_submission.csv", index=False
    )

    report_path = tmp_path / "validation.json"
    oof_path = tmp_path / "oof.csv"
    config = {
        "project": {"random_seed": 7},
        "data": {
            "raw_dir": str(raw_dir),
            "train_transaction": "train_transaction.csv",
            "train_identity": "train_identity.csv",
            "target_column": "isFraud",
            "id_column": "TransactionID",
        },
        "validation": {"test_size": 0.2, "n_splits": 3},
        "model": {"models": ["lightgbm"], "tune_hyperparameters": False, "n_estimators": 8, "learning_rate": 0.1},
        "reports": {"validation_report": str(report_path), "oof_predictions": str(oof_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    artifact_path = tmp_path / "final.joblib"

    train(config_path, artifact_path)
    artifact = load_artifact(artifact_path)
    oof = pd.read_csv(oof_path)
    assert artifact["metadata"]["training_rows"] == train_rows
    assert len(oof) == int(train_rows * 0.8)
    assert set(oof["fold"]) == {1, 2, 3}
    assert report_path.exists()

    metadata = artifact["metadata"]
    assert 0.0 < metadata["decision_threshold"] < 1.0
    assert metadata["calibration"]["brier_score"] >= 0.0
    assert len(metadata["calibration"]["bins"]) == 10
    assert metadata["threshold_selection"]["selected_threshold"] == metadata["decision_threshold"]
    assert metadata["threshold_selection"]["metrics_at_selected"]["threshold"] == metadata["decision_threshold"]

    report = json.loads(report_path.read_text())
    assert report["decision_threshold"] == metadata["decision_threshold"]
    assert "calibration" in report and "threshold_selection" in report

    submission_path = tmp_path / "submission.csv"
    submission = generate_submission(config_path, artifact_path, submission_path)
    assert submission["TransactionID"].tolist() == test_ids[::-1].tolist()
    assert len(submission) == len(test_ids)
    assert submission["isFraud"].between(0, 1).all()


def test_calibration_curve_reports_observed_vs_predicted_rates():
    y_prob = np.array([0.05] * 5 + [0.95] * 5)
    y_true = np.array([0, 0, 0, 0, 1] + [1, 1, 1, 1, 0])

    curve, ece = compute_calibration_curve(y_true, y_prob, n_bins=10)

    assert len(curve) == 10
    low_bin = curve[0]
    high_bin = curve[9]
    assert low_bin["count"] == 5
    assert abs(low_bin["mean_predicted"] - 0.05) < 1e-9
    assert abs(low_bin["observed_fraud_rate"] - 0.2) < 1e-9
    assert high_bin["count"] == 5
    assert abs(high_bin["mean_predicted"] - 0.95) < 1e-9
    assert abs(high_bin["observed_fraud_rate"] - 0.8) < 1e-9
    assert abs(ece - 0.15) < 1e-9

    empty_bins = [entry for entry in curve if entry["count"] == 0]
    assert all(entry["mean_predicted"] is None for entry in empty_bins)


def test_select_decision_threshold_finds_zero_cost_cutoff_for_separable_data():
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.array([0.1] * 50 + [0.9] * 50)

    threshold, expected_cost, curve = select_decision_threshold(
        y_true, y_prob, cost_false_negative=5.0, cost_false_positive=1.0, resolution=0.1
    )

    assert 0.1 < threshold <= 0.9
    assert expected_cost == 0.0
    selected_row = next(row for row in curve if abs(row["threshold"] - threshold) < 1e-9)
    assert selected_row["precision"] == 1.0
    assert selected_row["recall"] == 1.0


def test_v_column_correlation_pruning_drops_a_near_duplicate():
    rng = np.random.default_rng(9)
    n = 200
    base = rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "isFraud": rng.integers(0, 2, n),
            "TransactionAmt": rng.uniform(5, 200, n),
            "V1": base,
            "V2": base + rng.normal(0, 1e-4, n),  # near-perfect duplicate of V1
            "V3": rng.normal(0, 1, n),  # independent
        }
    )
    transformer = TransactionFeatureEngineer().fit(frame)

    assert transformer.dropped_v_columns_ == ["V2"]
    transformed = transformer.transform(frame)
    assert "V1" in transformed.columns
    assert "V3" in transformed.columns
    assert "V2" not in transformed.columns


def test_raw_high_cardinality_id_columns_are_dropped_but_their_stats_remain():
    frame = _make_frame(80, fraud_rate=0.25)
    frame["card2"] = np.tile(np.arange(1, 11), 8)
    transformer = TransactionFeatureEngineer().fit(frame)
    transformed = transformer.transform(frame)

    for raw_column in ("card1", "card2", "addr1"):
        assert raw_column not in transformed.columns
        assert f"{raw_column}__group_count" in transformed.columns
        assert f"{raw_column}__amount_delta" in transformed.columns


def test_d_columns_get_a_time_detrended_companion_feature():
    frame = _make_frame(40, fraud_rate=0.3)
    frame["TransactionDT"] = np.arange(40) * 3600 * 24  # one day apart
    frame["D1"] = 10.0  # constant "days since account creation" regardless of time
    transformer = TransactionFeatureEngineer().fit(frame)
    transformed = transformer.transform(frame)

    assert "D1_detrended" in transformed.columns
    # D1 stays flat while elapsed time grows, so the detrended value should trend
    # downward across rows (each row is one more day further from the reference).
    detrended = transformed["D1_detrended"].to_numpy()
    assert detrended[0] > detrended[-1]


def test_adversarial_validation_flags_a_genuinely_drifting_column():
    rng = np.random.default_rng(3)
    n = 300
    train = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "isFraud": rng.integers(0, 2, n),
            "stable_feature": rng.normal(0, 1, n),
            "drifting_feature": rng.normal(0.0, 1, n),
            "ProductCD": rng.choice(["W", "C"], n),
        }
    )
    test = pd.DataFrame(
        {
            "TransactionID": np.arange(n, 2 * n),
            "stable_feature": rng.normal(0, 1, n),
            "drifting_feature": rng.normal(6.0, 1, n),
            "ProductCD": rng.choice(["W", "C"], n),
        }
    )

    result = run_adversarial_validation(train, test, n_splits=3, random_state=1, top_n=5)

    assert result["n_train"] == n
    assert result["n_test"] == n
    assert result["oof_auc"] > 0.9

    top_names = [entry["feature"] for entry in result["top_features"]]
    assert "drifting_feature" in top_names[:2]
    drifting_entry = next(entry for entry in result["top_features"] if entry["feature"] == "drifting_feature")
    assert drifting_entry["kind"] == "numeric"
    assert drifting_entry["test_mean"] - drifting_entry["train_mean"] > 4.0


def test_train_selects_hyperparameters_via_tuning_when_enabled(tmp_path):
    raw_dir = tmp_path / "data"
    raw_dir.mkdir()
    rows = 200
    train_data = pd.DataFrame(
        {
            "TransactionID": np.arange(1, rows + 1),
            "TransactionDT": np.arange(rows) * 3600,
            "TransactionAmt": np.linspace(5.0, 150.0, rows),
            "card1": np.tile(np.arange(10, 20), rows // 10),
            "isFraud": np.tile([0, 0, 0, 1, 0], rows // 5),
        }
    )
    train_data.to_csv(raw_dir / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": train_data["TransactionID"], "DeviceType": "mobile"}).to_csv(
        raw_dir / "train_identity.csv", index=False
    )

    report_path = tmp_path / "validation.json"
    oof_path = tmp_path / "oof.csv"
    candidates = {
        "lightgbm": [
            {"n_estimators": 3, "learning_rate": 0.3, "num_leaves": 7},
            {"n_estimators": 6, "learning_rate": 0.1, "num_leaves": 15},
        ]
    }
    config = {
        "project": {"random_seed": 5},
        "data": {
            "raw_dir": str(raw_dir),
            "train_transaction": "train_transaction.csv",
            "train_identity": "train_identity.csv",
            "target_column": "isFraud",
            "id_column": "TransactionID",
        },
        "validation": {"test_size": 0.2, "n_splits": 3, "tuning_size": 0.25},
        "model": {"models": ["lightgbm"], "tune_hyperparameters": True, "tuning_candidates": candidates},
        "reports": {"validation_report": str(report_path), "oof_predictions": str(oof_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    artifact_path = tmp_path / "final.joblib"

    train(config_path, artifact_path)
    artifact = load_artifact(artifact_path)

    selected_config = artifact["metadata"]["model_configs"]["lightgbm"]
    assert selected_config in candidates["lightgbm"]
    tuning_info = artifact["metadata"]["hyperparameter_tuning"]
    assert tuning_info["enabled"] is True
    assert tuning_info["tuning_rows"] > 0
    assert "lightgbm" in tuning_info["scores"]

    report = json.loads(report_path.read_text())
    assert report["hyperparameter_tuning"]["enabled"] is True
    assert report["model_configs"]["lightgbm"] == selected_config


def test_api_supports_health_and_prediction_contract():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    model_info = client.get("/model-info")
    assert model_info.status_code in {200, 503}

    payload = {
        "TransactionID": 1,
        "TransactionAmt": 25.0,
        "card1": 1000,
        "card2": 150,
        "addr1": 10,
        "ProductCD": "W",
        "card4": "visa",
        "P_emaildomain": "gmail.com",
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        body = response.json()
        assert 0.0 <= float(body["fraud_probability"]) <= 1.0
    else:
        assert response.status_code == 503

    invalid = client.post("/predict", json={})
    assert invalid.status_code == 422
