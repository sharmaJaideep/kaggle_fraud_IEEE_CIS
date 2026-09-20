"""FastAPI service for the persisted IEEE-CIS fraud ensemble."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.models.ensemble import load_artifact, predict_artifact


ARTIFACT_PATH = Path(os.getenv("FRAUD_ARTIFACT_PATH", "models/fraud_ensemble.joblib"))
DEFAULT_THRESHOLD = float(os.getenv("FRAUD_DECISION_THRESHOLD", "0.5"))


class TransactionInput(BaseModel):
    TransactionID: Optional[int] = None
    TransactionAmt: Optional[float] = None
    card1: Optional[Union[int, float]] = None
    card2: Optional[Union[int, float]] = None
    addr1: Optional[Union[int, float]] = None
    ProductCD: Optional[str] = None
    card4: Optional[str] = None
    P_emaildomain: Optional[str] = None
    DeviceInfo: Optional[str] = None
    browser: Optional[str] = None
    OS: Optional[str] = None


class BatchTransactionInput(BaseModel):
    records: List[TransactionInput] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    fraud_probability: float
    predicted_label: int
    threshold_used: float
    model_version: Union[str, int]
    feature_version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    artifact_loaded: bool
    artifact_path: str


class ModelInfoResponse(BaseModel):
    artifact_loaded: bool
    artifact_path: str
    model_version: Optional[Union[str, int]] = None
    metadata: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global artifact
    if ARTIFACT_PATH.exists():
        artifact = load_artifact(ARTIFACT_PATH)
    else:
        artifact = None
    yield


app = FastAPI(title="IEEE-CIS Fraud Risk API", version="1.0.0", lifespan=lifespan)
artifact: dict[str, Any] | None = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        artifact_loaded=artifact is not None,
        artifact_path=str(ARTIFACT_PATH),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model artifact not found at {ARTIFACT_PATH}")
    metadata = artifact.get("metadata", {})
    return ModelInfoResponse(
        artifact_loaded=True,
        artifact_path=str(ARTIFACT_PATH),
        model_version=artifact.get("version", "unknown"),
        metadata=metadata,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionInput) -> PredictionResponse:
    payload_dict = payload.model_dump(exclude_none=True)
    if not payload_dict:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Transaction JSON must contain at least one field")
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model artifact not found at {ARTIFACT_PATH}")
    try:
        probability = float(predict_artifact(artifact, [payload_dict])[0])
    except (TypeError, ValueError, KeyError) as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid transaction payload: {error}") from error
    threshold = float(artifact.get("metadata", {}).get("decision_threshold", DEFAULT_THRESHOLD))
    return PredictionResponse(
        fraud_probability=probability,
        predicted_label=int(probability >= threshold),
        threshold_used=threshold,
        model_version=artifact.get("version", "unknown"),
        feature_version=artifact.get("metadata", {}).get("feature_version"),
    )


@app.post("/predict_batch")
def predict_batch(payload: BatchTransactionInput) -> list[PredictionResponse]:
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model artifact not found at {ARTIFACT_PATH}")
    threshold = float(artifact.get("metadata", {}).get("decision_threshold", DEFAULT_THRESHOLD))
    requests = [record.model_dump(exclude_none=True) for record in payload.records]
    probabilities = predict_artifact(artifact, requests)
    return [
        PredictionResponse(
            fraud_probability=float(probability),
            predicted_label=int(float(probability) >= threshold),
            threshold_used=threshold,
            model_version=artifact.get("version", "unknown"),
            feature_version=artifact.get("metadata", {}).get("feature_version"),
        )
        for probability in probabilities
    ]