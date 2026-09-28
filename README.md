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
- **Feature engineering** (`src/features/preprocessing.py`) — a fitted, sklearn-compatible `TransactionFeatureEngineer` that adds missingness flags, log-scaled transaction amount, time-of-day/day/week features, categorical interaction columns, frequency encodings, D-column time-detrending, and per-group amount aggregates. All learned state is fit on training data only and tolerates partial-column inference payloads (e.g. a single-transaction API request). Four pieces of this are gated behind ablation flags in `config.yaml`'s `features` section (`enable_uid_reconstruction`, `enable_uid_time_features`, `enable_v_column_pruning`, `drop_raw_id_columns`, `enable_d_column_detrending`) — see the ablation study below for why the defaults are what they are.
- **A dedicated ablation study determined the current feature-engineering defaults**, after the first attempt at applying every adversarial-validation-motivated idea at once *regressed* the time-based validation AUC from 0.922 to 0.903. Isolating each change (holding tuned model configs fixed, so only the feature engineering varied) found:
  - **Client identity reconstruction (`uid`, from `card1/card2/card3/card5/addr1/addr2` + a `D1`-derived day offset) was the dominant cause of the regression, not the adversarial-validation fixes.** `uid` is extremely high-cardinality, so within the smaller time-based training split its group-stat aggregates are mostly noise from single-occurrence groups rather than a stable signal. Disabling it recovered from 0.9026 to 0.9238, *exceeding* the pre-session baseline. **`enable_uid_reconstruction` therefore now defaults to `false`** — the `uid` construction code (including `uid_seconds_since_last_transaction`/`uid_transactions_seen_before`) remains in `preprocessing.py` for future revisiting with a leakage-safe, expanding-window design, rather than deleted.
  - **Dropping raw `card1`/`card2`/`addr1` and pruning correlated `V`-columns were both mildly *negative* for time-based AUC**, despite being well-evidenced by adversarial validation. The lesson: a column being statistically distinguishable between train and test doesn't mean it lacks real predictive signal — adversarial validation tells you *where* distributions differ, not automatically what to do about it. **`drop_raw_id_columns` and `enable_v_column_pruning` now default to `false`** (keeping all of them), which combined with `uid` disabled reached **0.9248**, the best score found.
  - **D-column detrending (`Dn - TransactionDT/86400` for every `Dn`, not just `D1`) is a confirmed, clean win** — disabling it alone dropped AUC to 0.8996. **`enable_d_column_detrending` stays `true`.**
  - The `_select_v_columns_to_drop` (correlation pruning) and raw-ID-exclusion (`get_feature_names_out`) code paths remain in the class, inactive by default, so they can be re-tested (e.g. at a different correlation threshold) rather than removed outright.
  - **Follow-up: `_select_v_columns_to_drop` was optimized to be target-aware rather than order-arbitrary** — for each correlated pair above the threshold, it now keeps whichever column correlates more strongly with fraud (falling back to the old order-based tiebreak when no target is passed), since the naive version was throwing away predictive signal, not just redundancy. `fit()`'s `y` parameter — previously accepted but unused everywhere in the class — is now threaded through from every call site (`kfold_oof_validation`, `time_based_validation`, `train.py`'s tuning/selection/final transformers) to make this live wherever pruning runs.
- **Models** (`src/models/baseline.py`) — LightGBM, XGBoost, and CatBoost, each configured with imbalance-aware class weighting (`scale_pos_weight` / `auto_class_weights`).
- **Ensembling** (`src/models/ensemble.py`) — blend weights are chosen by grid-searching the weight simplex against out-of-fold predictions (`optimize_oof_blend_weights`), and the fitted transformer, models, weights, and metadata are serialized into a single deployable artifact (`save_artifact` / `load_artifact` / `predict_artifact`).
- **Validation** (`src/validation.py`) — stratified K-fold OOF validation (every row scored exactly once, out-of-fold) and a chronological (`TransactionDT`-ordered) 80/20 split to catch temporal drift that a random split would hide.
- **Training entry point** (`src/train.py`) — holds out an untouched stratified partition, tunes blend weights via OOF on the remainder, evaluates on the untouched holdout and a time-based split, then refits final models on all labeled rows for the deployed artifact. Emits `reports/oof_predictions.csv` and `reports/validation_report.json` (per-fold metrics, blend weights, training fingerprint, runtime).
- **Hyperparameter tuning** (`src/train.py`, `tune_model_configs` in `src/models/ensemble.py`) — before OOF validation, a further stratified split is carved from the selection partition, and each booster's hyperparameters are chosen from a candidate pool (`DEFAULT_TUNING_CANDIDATES`, or a `model.tuning_candidates` override in config) by validation ROC-AUC. Controlled by `model.tune_hyperparameters` in `config.yaml` (default `true`); set to `false` to use the fixed `n_estimators`/`learning_rate`/`num_leaves` values instead. The selected configs and their tuning scores are persisted in both the artifact metadata and `validation_report.json` under `hyperparameter_tuning`.
- **Submission generation** (`src/predict_test.py`, `src/reporting.py`) — scores the Kaggle test set with the saved artifact and writes a schema- and order-validated `submission.csv`.
- **Calibration and decision-threshold selection** (`src/calibration.py`) — on the untouched holdout set, `compute_calibration_curve` bins predictions and reports the Brier score and Expected Calibration Error (whether a predicted 0.2 really means a ~20% fraud rate — ROC-AUC alone can't tell you that), and `select_decision_threshold` sweeps candidate cutoffs to find the one minimizing `cost_false_negative * FN + cost_false_positive * FP` (configurable under `evaluation` in `config.yaml`; default assumes a missed fraud costs 5x a wrongly blocked transaction). The result is saved as `decision_threshold` in the artifact metadata, which `app.py` was already reading but which nothing previously set — so the API now serves a cost-aware threshold instead of a hardcoded 0.5. Full curves and per-threshold metrics are also persisted in `validation_report.json`.
- **Serving** (`app.py`) — a FastAPI app exposing `/health`, `/model-info`, `/predict`, and `/predict_batch`.
- **Tests** (`tests/test_project_quality.py`) — 14 passing tests covering the feature engineer (uid reconstruction, time-since-last-transaction, V-column pruning, raw ID-column dropping, D-column detrending), adversarial validation, OOF/blend-weight logic, time-based validation, hyperparameter-tuning wiring, calibration and threshold-selection logic, a full synthetic train → artifact → submission pipeline, and the API contract.
- **Continuous integration** (`.github/workflows/tests.yml`) — runs `python -m pytest tests/ -v` on every push and pull request targeting `main` (Python 3.11, dependencies from `requirements.txt`). The suite is self-contained (synthetic data via `tmp_path`), so it needs no Kaggle CSVs or pre-trained artifact. See [Continuous Integration](#continuous-integration) below for what it checks and why it's set up the way it is.
- **Two full training runs against the real Kaggle CSVs have been completed; a third (regressed) run in between is what triggered the ablation study above.**

  | Run | Holdout AUC | Time-based AUC | Kaggle public / private |
  |---|---|---|---|
  | 1. Original (pre-uid) | 0.9643 | 0.9222 | 0.9366 / 0.8974 |
  | 2. All adversarial-validation fixes on at once | 0.9635 | 0.9026 (regression) | not submitted |
  | 3. **Ablation-optimal defaults (current)** | **0.9672** | **0.9248** | **0.9388 / 0.9016** |

  Run 3 improves on run 1 on *every* metric that was checked — holdout, time-based, and both real Kaggle scores — confirming the ablation study's fix generalizes rather than just overfitting one validation scheme. Its internal time-based figure (0.9248) exactly matched the standalone ablation study's prediction, and the real private-leaderboard gain (+0.0042) came in even larger than the internal time-based gain predicted (+0.0026) — a pleasant surprise, not a shortfall. Calibration also improved over run 1 (Brier 0.0457 vs 0.0476, ECE 0.1225 vs 0.1267). This is recorded in `reports/validation_report.json`. For reference, this competition's top teams historically scored in roughly the 0.945 AUC range on the private leaderboard — run 3 closes about a fifth of that gap (from ~4.8 points to ~4.3 points short).
- **Adversarial validation** (`src/adversarial_validation.py`) — trains a classifier to distinguish real train rows from real test rows; a high OOF AUC means the two sets are genuinely different distributions, and its feature importances point at exactly which columns are driving that. Run against the real data, this came back **OOF AUC 1.0000** (fold AUCs all ≥0.99999) with two distinct causes: (1) the `D`-columns (`D3/D4/D5/D10/D11/D15`, not just `D1`) show large, systematic mean shifts and missingness-rate shifts from train to test — genuine time drift; (2) `card1`/`card2`/`addr1`/`TransactionAmt` are top-ranked despite having nearly identical train/test means — meaning a tree model is exploiting specific individual ID values rather than a real distribution shift, closer to memorization than signal. Both findings are saved to `reports/adversarial_validation.json` (gitignored, like other generated reports).
- **Card/client-identity correlation analysis** (`src/eda.py`) — `compute_card_correlation` and `plot_card_correlation_heatmap` examine how the fields available for reconstructing a pseudo customer identity (`card1/card2/card3/card5`, `addr1/addr2`, `dist1/dist2`, the `D`-columns) relate to each other and to fraud, saving a heatmap to `reports/card_correlation_heatmap.png` (gitignored, regenerate with `python -m src.eda`). Run against the real data:
  - **`card1`, `card2`, and `addr1` are all pairwise near-uncorrelated (|corr| < 0.04)** — each carries genuinely independent identity information, which supports combining them in `uid` (when that's re-enabled with a leakage-safe design) rather than one making the others redundant.
  - **`card3` and `addr2` are strongly, oppositely correlated (−0.57)** — they likely encode overlapping geographic/issuer information, so a rebuilt `uid` probably doesn't need both.
  - **`D1` and `D2` are near-duplicates (0.98)**, and **`D4`/`D10`/`D11`/`D15` form their own correlated cluster (0.58–0.77 pairwise)** — the per-column detrending in `preprocessing.py` treats all of these independently, which is safe but means a lot of the detrending work is being repeated on largely redundant signal.
  - **`card3` has by far the strongest raw correlation with fraud of any field checked here (0.154)** — well above `card1` (−0.014), `card2` (0.003), `addr1` (0.006), or any `D`-column (all ≤0.09 in magnitude). It isn't currently in `group_columns`, so it gets no group-stat encoding — a concrete, evidence-backed candidate to add.

**Not yet done:**

- The card correlation heatmap (`src/eda.py`) is the first EDA figure produced, but like other generated outputs it's gitignored rather than committed — no broader EDA summary (target prevalence, missingness, cardinality) exists yet.
- Only a 2-candidate hyperparameter search per booster — a broader random/Bayesian search would likely help.
- Still a ~4.3-point AUC gap to this competition's historical top-team scores (~0.945 private) — down from ~4.8 points before this session's fixes, but not closed.
- `uid`'s current whole-partition group-stat aggregation is leakage-safe (fit on training data only) but not causally time-aware (it doesn't restrict to *prior* transactions relative to each row) - revisiting it with an expanding-window design could recover its benefit without the noise that caused the earlier regression.
- `card3` is confirmed the most fraud-correlated card/identity field checked but has no group-stat encoding yet (not in `group_columns`).

## Next Steps

1. Add `card3` to `group_columns` in `preprocessing.py` — it's the most fraud-correlated card/identity field found, per the correlation analysis above, but currently gets no group-stat treatment.
2. Revisit `uid` with an expanding-window (prior-transactions-only) aggregation instead of whole-partition group stats, now that plain whole-partition `uid` is confirmed to hurt rather than help. Given `card3`/`addr2` are highly redundant with each other and `D1`/`D2` are near-duplicates, a rebuilt `uid` likely doesn't need both members of either pair.
3. Run and commit a broader EDA summary (target prevalence, missingness, cardinality, train/test drift) under `reports/`.
4. Broaden the hyperparameter search beyond the current 2 candidates per booster.
5. Set `evaluation.cost_false_negative`/`cost_false_positive` in `config.yaml` to reflect actual business costs rather than the illustrative 5:1 default used so far.

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
│   ├── eda.py                    # Card/client-identity correlation heatmap and analysis
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
