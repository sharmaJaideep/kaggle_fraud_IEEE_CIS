import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app import app
from src.features.preprocessing import TransactionFeatureEngineer
from src.models.ensemble import load_artifact, optimize_oof_blend_weights, predict_artifact, save_artifact
from src.validation import kfold_oof_validation


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
