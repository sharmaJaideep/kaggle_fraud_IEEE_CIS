# IEEE-CIS Fraud Detection

## Executive Summary & Project Intention

This repository provides a reproducible machine learning foundation for the Kaggle IEEE-CIS Fraud Detection competition. The objective is to identify fraudulent online transactions from joined transaction and identity signals while keeping data preparation, validation, and model artifacts auditable.

The baseline has grown from a single fast gradient-boosted model into a validation-weighted **three-model ensemble** (LightGBM, XGBoost, CatBoost) with out-of-fold blend-weight optimization, a chronological validation check, and a served FastAPI endpoint. It remains a foundation for disciplined experimentation rather than a claim of leaderboard optimality — Kaggle leaderboard scores are not yet recorded.

## Core Machine Learning Goals

- Handle severe class imbalance without allowing the majority class to dominate training.
- Optimize and report ROC-AUC, the competition's primary evaluation metric.
- Preserve useful missingness and categorical information from both transaction and identity tables.
- Prevent train/validation preprocessing drift by fitting consistent encodings across both partitions.
- Build a repeatable path from experiment to saved model artifact, through to a servable API.

## Current Status (as of 2026-09-27)

**Implemented and tested:**

- **Data loading** (`src/data/loading.py`) — left-joins the transaction and identity tables on `TransactionID` and downcasts numeric dtypes to reduce memory pressure.
- **Feature engineering** (`src/features/preprocessing.py`) — a fitted, sklearn-compatible `TransactionFeatureEngineer` that adds missingness flags, log-scaled transaction amount, time-of-day/day/week features, categorical interaction columns, frequency encodings, and per-group (card/address/email/product) amount aggregates. All learned state is fit on training data only and tolerates partial-column inference payloads (e.g. a single-transaction API request).
- **Models** (`src/models/baseline.py`) — LightGBM, XGBoost, and CatBoost, each configured with imbalance-aware class weighting (`scale_pos_weight` / `auto_class_weights`).
- **Ensembling** (`src/models/ensemble.py`) — blend weights are chosen by grid-searching the weight simplex against out-of-fold predictions (`optimize_oof_blend_weights`), and the fitted transformer, models, weights, and metadata are serialized into a single deployable artifact (`save_artifact` / `load_artifact` / `predict_artifact`).
- **Validation** (`src/validation.py`) — stratified K-fold OOF validation (every row scored exactly once, out-of-fold) and a chronological (`TransactionDT`-ordered) 80/20 split to catch temporal drift that a random split would hide.
- **Training entry point** (`src/train.py`) — holds out an untouched stratified partition, tunes blend weights via OOF on the remainder, evaluates on the untouched holdout and a time-based split, then refits final models on all labeled rows for the deployed artifact. Emits `reports/oof_predictions.csv` and `reports/validation_report.json` (per-fold metrics, blend weights, training fingerprint, runtime).
- **Hyperparameter tuning** (`src/train.py`, `tune_model_configs` in `src/models/ensemble.py`) — before OOF validation, a further stratified split is carved from the selection partition, and each booster's hyperparameters are chosen from a candidate pool (`DEFAULT_TUNING_CANDIDATES`, or a `model.tuning_candidates` override in config) by validation ROC-AUC. Controlled by `model.tune_hyperparameters` in `config.yaml` (default `true`); set to `false` to use the fixed `n_estimators`/`learning_rate`/`num_leaves` values instead. The selected configs and their tuning scores are persisted in both the artifact metadata and `validation_report.json` under `hyperparameter_tuning`.
- **Submission generation** (`src/predict_test.py`, `src/reporting.py`) — scores the Kaggle test set with the saved artifact and writes a schema- and order-validated `submission.csv`.
- **Serving** (`app.py`) — a FastAPI app exposing `/health`, `/model-info`, `/predict`, and `/predict_batch`.
- **Tests** (`tests/test_project_quality.py`) — 7 passing tests covering the feature engineer, OOF/blend-weight logic, time-based validation, hyperparameter-tuning wiring, a full synthetic train → artifact → submission pipeline, and the API contract.
- **Continuous integration** (`.github/workflows/tests.yml`) — runs `pytest tests/ -v` on every push and pull request targeting `main` (Python 3.11, dependencies from `requirements.txt`). The suite is self-contained (synthetic data via `tmp_path`), so it needs no Kaggle CSVs or pre-trained artifact.
- A trained artifact already exists locally at `models/fraud_ensemble.joblib` (~12 MB) from a prior run. Model artifacts and validation reports are gitignored as generated outputs, so they are reproduced locally via the commands below rather than committed.

**Not yet done:**

- No EDA summary or figures are committed under `reports/`, despite being part of the intended workflow below.
- No calibration analysis or business-facing decision-threshold study yet, beyond the fixed default threshold used by the API.
- Kaggle public/private leaderboard scores are unset (`null` placeholders in `validation_report.json`) — no submission has been uploaded to the competition.

## Next Steps

1. Run and commit an EDA summary (target prevalence, missingness, cardinality, train/test drift) under `reports/`.
2. Upload a submission to Kaggle and record the public/private leaderboard AUC in `validation_report.json`.
3. Add calibration diagnostics and a threshold-selection analysis to support a real deployment decision, not just the default 0.5 cutoff.

## Project Structure

```text
.
├── config/
│   └── config.yaml               # Data paths, validation, and model parameters
├── input_data/                   # Existing raw Kaggle files; never modified by setup
│   └── ieee-fraud-detection/
├── models/                       # Persisted model artifacts (gitignored)
├── notebooks/
│   └── 01_baseline_model.ipynb   # End-to-end baseline experiment
├── reports/                      # Evaluation outputs, OOF predictions, submissions (gitignored)
├── logs/                         # Run logs
├── app.py                        # FastAPI fraud-risk service
├── src/
│   ├── data/loading.py           # Join and memory reduction utilities
│   ├── features/preprocessing.py # Fitted aggregation/frequency feature logic
│   ├── models/
│   │   ├── baseline.py           # Per-booster model factories (LightGBM/XGBoost/CatBoost)
│   │   └── ensemble.py           # Blend-weight optimization, artifact save/load/predict
│   ├── validation.py             # Stratified OOF and chronological validation
│   ├── reporting.py              # Submission writer and validation report writer
│   ├── train.py                  # Reproducible training entry point
│   └── predict_test.py           # Kaggle test-set submission generator
├── tests/
│   └── test_project_quality.py   # Feature engineer, validation, pipeline, and API tests
├── requirements.txt
├── setup_project.py               # Idempotent layout creator
└── README.md
```

## Workflow Pipeline

### 1. Exploratory Data Analysis

Profile target prevalence, missingness, cardinality, transaction time behavior, and train/test distribution differences. EDA outputs should be saved under `reports/` so findings are reproducible.

### 2. Feature Engineering

Join transaction and identity records on `TransactionID`, reduce numeric memory safely, encode categoricals consistently, and develop time, frequency, aggregation, and missingness features. Every transformation must be fit using training data only when it learns state.

### 3. Modelling

Use stratified validation for quick iteration, then move to time-aware or adversarial validation if distribution shift is material. Compare weighted and unweighted tree models, tune only after establishing a trustworthy baseline, and persist the exact configuration with each artifact.

### 4. Evaluation

Report ROC-AUC on an untouched validation set, alongside the fraud rate, confusion-oriented diagnostics, calibration checks, and inference cost. For competition submissions, generate probabilities in the required `sample_submission.csv` format.

## Getting Started

1. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Confirm the Kaggle CSV files remain in `input_data/ieee-fraud-detection/`.

4. Recreate any missing project directories safely.

   ```bash
   python setup_project.py
   ```

5. Launch Jupyter and run `notebooks/01_baseline_model.ipynb` from top to bottom.

   ```bash
   jupyter lab
   ```

6. Record the resulting ROC-AUC, configuration, and environment before comparing experiments. The notebook uses a stratified split for a quick baseline; production decisions should include stronger validation and monitoring.

## Ensemble training and deployment

The reusable training path selects blend weights from stratified out-of-fold predictions on a model-selection partition, reports performance on a separate untouched stratified holdout and a chronological validation split, then refits the selected models on all labeled rows. Feature mappings are fitted inside each fold for OOF evaluation and on the complete labeled dataset for the final artifact:

```bash
python -m src.train --config config/config.yaml --artifact models/fraud_ensemble.joblib
python -m src.predict_test --config config/config.yaml --artifact models/fraud_ensemble.joblib --output reports/submission.csv
uvicorn app:app --host 0.0.0.0 --port 8000
```

Training writes `reports/oof_predictions.csv` with one OOF row and fold assignment per selection row, plus `reports/validation_report.json` with OOF fold metrics, time-validation scores, untouched-holdout scores, blend weights, runtime, and a training fingerprint. Time-validation blend scores use fixed equal weights; OOF-selected weights are assessed on the untouched stratified holdout. The OOF blend score is a selection metric, not an unbiased performance estimate.

The submission command joins the test identity table, scores the Kaggle test rows with the saved full-train artifact, validates probabilities and IDs against `sample_submission.csv`, and writes exactly `TransactionID,isFraud` in sample order. It does not use test labels. Kaggle public and private leaderboard scores remain unset until a submission is uploaded.

The service loads `models/fraud_ensemble.joblib` by default. Set `FRAUD_ARTIFACT_PATH` to use another artifact. Send one transaction JSON object to `POST /predict`; the response contains `fraud_probability`. `GET /health` reports whether the artifact loaded successfully.

## Notes on Scale and Reproducibility

The merged training table is large. Run the notebook on a machine with sufficient RAM, close unused kernels, and avoid retaining duplicate full-size DataFrames. The memory reduction helper downcasts numeric columns, but it cannot eliminate the cost of the initial CSV parse. Pin dependencies and retain random seeds when publishing results.
