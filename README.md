# IEEE-CIS Fraud Detection

[![Tests](https://github.com/sharmaJaideep/kaggle_fraud_IEEE_CIS/actions/workflows/tests.yml/badge.svg)](https://github.com/sharmaJaideep/kaggle_fraud_IEEE_CIS/actions/workflows/tests.yml)

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
- **Feature engineering** (`src/features/preprocessing.py`) — a fitted, sklearn-compatible `TransactionFeatureEngineer` that adds missingness flags, log-scaled transaction amount, time-of-day/day/week features, categorical interaction columns, frequency encodings, and per-group (card/address/email/product/**uid**) amount aggregates. All learned state is fit on training data only and tolerates partial-column inference payloads (e.g. a single-transaction API request).
- **Client identity reconstruction** (`src/features/preprocessing.py`) — the dataset has no real customer ID, so `uid` reconstructs a pseudo customer identity from `card1/card2/card3/card5/addr1/addr2` plus a `D1`-derived account-age residual (`D1 - TransactionDT/86400`, which stays near-constant per client even as `TransactionDT` grows). This feeds the existing per-group aggregation machinery (`uid__group_count`, `uid__amount_delta`, etc.) and two new derived columns: `uid_seconds_since_last_transaction` and `uid_transactions_seen_before`, computed by sorting each `transform()` call's rows by `TransactionDT` within each reconstructed client. These are scoped to whatever rows are in that single `transform()` call (not persisted history across API calls), and any row with no prior transaction to compare against — including every single-row API prediction — falls back to the fitted training-set median gap via `numeric_medians_`, not a raw `NaN`.
- **Models** (`src/models/baseline.py`) — LightGBM, XGBoost, and CatBoost, each configured with imbalance-aware class weighting (`scale_pos_weight` / `auto_class_weights`).
- **Ensembling** (`src/models/ensemble.py`) — blend weights are chosen by grid-searching the weight simplex against out-of-fold predictions (`optimize_oof_blend_weights`), and the fitted transformer, models, weights, and metadata are serialized into a single deployable artifact (`save_artifact` / `load_artifact` / `predict_artifact`).
- **Validation** (`src/validation.py`) — stratified K-fold OOF validation (every row scored exactly once, out-of-fold) and a chronological (`TransactionDT`-ordered) 80/20 split to catch temporal drift that a random split would hide.
- **Training entry point** (`src/train.py`) — holds out an untouched stratified partition, tunes blend weights via OOF on the remainder, evaluates on the untouched holdout and a time-based split, then refits final models on all labeled rows for the deployed artifact. Emits `reports/oof_predictions.csv` and `reports/validation_report.json` (per-fold metrics, blend weights, training fingerprint, runtime).
- **Hyperparameter tuning** (`src/train.py`, `tune_model_configs` in `src/models/ensemble.py`) — before OOF validation, a further stratified split is carved from the selection partition, and each booster's hyperparameters are chosen from a candidate pool (`DEFAULT_TUNING_CANDIDATES`, or a `model.tuning_candidates` override in config) by validation ROC-AUC. Controlled by `model.tune_hyperparameters` in `config.yaml` (default `true`); set to `false` to use the fixed `n_estimators`/`learning_rate`/`num_leaves` values instead. The selected configs and their tuning scores are persisted in both the artifact metadata and `validation_report.json` under `hyperparameter_tuning`.
- **Submission generation** (`src/predict_test.py`, `src/reporting.py`) — scores the Kaggle test set with the saved artifact and writes a schema- and order-validated `submission.csv`.
- **Calibration and decision-threshold selection** (`src/calibration.py`) — on the untouched holdout set, `compute_calibration_curve` bins predictions and reports the Brier score and Expected Calibration Error (whether a predicted 0.2 really means a ~20% fraud rate — ROC-AUC alone can't tell you that), and `select_decision_threshold` sweeps candidate cutoffs to find the one minimizing `cost_false_negative * FN + cost_false_positive * FP` (configurable under `evaluation` in `config.yaml`; default assumes a missed fraud costs 5x a wrongly blocked transaction). The result is saved as `decision_threshold` in the artifact metadata, which `app.py` was already reading but which nothing previously set — so the API now serves a cost-aware threshold instead of a hardcoded 0.5. Full curves and per-threshold metrics are also persisted in `validation_report.json`.
- **Serving** (`app.py`) — a FastAPI app exposing `/health`, `/model-info`, `/predict`, and `/predict_batch`.
- **Tests** (`tests/test_project_quality.py`) — 11 passing tests covering the feature engineer (including uid reconstruction and time-since-last-transaction), OOF/blend-weight logic, time-based validation, hyperparameter-tuning wiring, calibration and threshold-selection logic, a full synthetic train → artifact → submission pipeline, and the API contract.
- **Continuous integration** (`.github/workflows/tests.yml`) — runs `python -m pytest tests/ -v` on every push and pull request targeting `main` (Python 3.11, dependencies from `requirements.txt`). The suite is self-contained (synthetic data via `tmp_path`), so it needs no Kaggle CSVs or pre-trained artifact. See [Continuous Integration](#continuous-integration) below for what it checks and why it's set up the way it is.
- **A full training run against the real Kaggle CSVs has been completed and submitted.** On the untouched holdout: blend ROC-AUC 0.964; on a chronological split: blend ROC-AUC 0.922. A real Kaggle submission (`reports/submission.csv`, generated from this artifact) scored **0.937 public / 0.897 private** — both numbers are visible immediately since this competition's deadline has passed. The public→private drop (and the fact that even the time-based internal estimate was more optimistic than the private score) confirms real temporal drift in this data: the further into the future a test row is, the harder it is to score. This is recorded in `reports/validation_report.json`. For reference, the competition's top teams historically scored in roughly the 0.945 AUC range on the private leaderboard, so there's a meaningful, structural gap left to close — see below.
- **Adversarial validation** (`src/adversarial_validation.py`) — trains a classifier to distinguish real train rows from real test rows; a high OOF AUC means the two sets are genuinely different distributions, and its feature importances point at exactly which columns are driving that. Run against the real data, this came back **OOF AUC 1.0000** (fold AUCs all ≥0.99999) with two distinct causes: (1) the `D`-columns (`D3/D4/D5/D10/D11/D15`, not just `D1`) show large, systematic mean shifts and missingness-rate shifts from train to test — genuine time drift; (2) `card1`/`card2`/`addr1`/`TransactionAmt` are top-ranked despite having nearly identical train/test means — meaning a tree model is exploiting specific individual ID values rather than a real distribution shift, closer to memorization than signal. Both findings are saved to `reports/adversarial_validation.json` (gitignored, like other generated reports).

**Not yet done:**

- No EDA summary or figures are committed under `reports/`, despite being part of the intended workflow below.
- **Feature selection** on the ~340 anonymized `V` columns — many are redundant/noisy (868 pairs found with \|correlation\| > 0.95 in a quick check); no correlation-based pruning has been done yet.
- **Acting on the adversarial validation findings above**: detrending `D3/D4/D5/D10/D11/D15` the same way `D1` already is inside `uid` construction, and dropping the raw `card1`/`card2`/`addr1` numeric columns from the final feature matrix (keeping only their derived frequency/group-stat encodings) so the model can't lean on ID memorization.
- Only a 2-candidate hyperparameter search per booster — a broader random/Bayesian search would likely help, though probably not enough alone to close the ~4.8-point gap to the historical winning scores.
- **The uid/time-since-last-transaction features (and the adversarial validation findings above) haven't been measured against a real retrain yet** — `train.py` hasn't been re-run against the real CSVs since adding them, so their actual impact on the holdout/time-based/leaderboard scores is still unknown.

## Next Steps

1. Detrend `D3/D4/D5/D10/D11/D15` by elapsed time, and drop the raw `card1`/`card2`/`addr1` numeric columns from the final feature matrix — both directly evidenced by the adversarial validation results above.
2. Re-run `src/train.py` against the real Kaggle CSVs with those changes plus the existing uid/time-since-last-transaction features, and compare the resulting holdout, time-based, and (after a fresh submission) leaderboard scores against the 0.964 / 0.922 / 0.937 / 0.897 baseline recorded above.
3. Run and commit an EDA summary (target prevalence, missingness, cardinality, train/test drift) under `reports/`.
4. Prune the noisy/redundant `V` columns via correlation clustering or permutation importance.
5. Broaden the hyperparameter search beyond the current 2 candidates per booster.
6. Set `evaluation.cost_false_negative`/`cost_false_positive` in `config.yaml` to reflect actual business costs rather than the illustrative 5:1 default used so far.

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
│   ├── adversarial_validation.py # Train-vs-test separability check and drift diagnosis
│   ├── calibration.py            # Calibration curve/Brier score and cost-aware threshold selection
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

The same untouched holdout is also used for calibration and threshold selection (`src/calibration.py`): the report's `calibration` block gives the Brier score and Expected Calibration Error, and `threshold_selection` gives the full precision/recall/F1/expected-cost curve across candidate thresholds plus the one that was selected. That selected threshold is saved as `decision_threshold` in the artifact metadata and is what `app.py` uses at serving time — override the cost assumptions behind it via `evaluation.cost_false_negative` / `evaluation.cost_false_positive` in `config.yaml` before retraining if your real fraud/false-positive costs differ from the illustrative 5:1 default.

The submission command joins the test identity table, scores the Kaggle test rows with the saved full-train artifact, validates probabilities and IDs against `sample_submission.csv`, and writes exactly `TransactionID,isFraud` in sample order. It does not use test labels. Kaggle public and private leaderboard scores remain unset until a submission is uploaded.

The service loads `models/fraud_ensemble.joblib` by default. Set `FRAUD_ARTIFACT_PATH` to use another artifact. Send one transaction JSON object to `POST /predict`; the response contains `fraud_probability`, `predicted_label` (thresholded at the artifact's `decision_threshold`, falling back to `FRAUD_DECISION_THRESHOLD`/0.5 if the artifact predates this feature), and `threshold_used`. `GET /health` reports whether the artifact loaded successfully.

## Continuous Integration

Every push and pull request to `main` triggers `.github/workflows/tests.yml`, which spins up a clean Ubuntu runner on Python 3.11, installs `requirements.txt`, and runs the full test suite. This is the project's safety net: nobody has to remember to run `pytest` by hand before merging, and a regression in the feature engineer, blend-weight logic, validation, artifact contract, or API shows up as a failed check within a couple of minutes rather than being discovered later.

To reproduce the CI run locally:

```bash
python -m pytest tests/ -v
```

Two things matter here that are easy to get wrong:

- **Use `python -m pytest`, not a bare `pytest` invocation.** Since `tests/` has no `__init__.py`, plain `pytest tests/` does not add the repository root to `sys.path`, so `from app import app` fails with `ModuleNotFoundError` even though the exact same tests pass under `python -m pytest`. This bit the CI workflow itself before it was corrected.
- **`requirements.txt` must include everything the tests need, including `pytest` and `httpx`.** A dependency that's only installed by hand in a local virtual environment (and never added to `requirements.txt`) will work for you and silently fail on any fresh machine, including CI runners and new contributors' setups.

The suite generates its own synthetic CSVs under `tmp_path` for the end-to-end training/submission test, so CI needs neither the Kaggle dataset nor a pre-trained `models/fraud_ensemble.joblib` artifact to pass.

## Notes on Scale and Reproducibility

The merged training table is large. Run the notebook on a machine with sufficient RAM, close unused kernels, and avoid retaining duplicate full-size DataFrames. The memory reduction helper downcasts numeric columns, but it cannot eliminate the cost of the initial CSV parse. Pin dependencies and retain random seeds when publishing results.
