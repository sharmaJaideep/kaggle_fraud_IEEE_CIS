# IEEE-CIS Fraud Detection

## Executive Summary & Project Intention

This repository provides a reproducible machine learning foundation for the Kaggle IEEE-CIS Fraud Detection competition. The objective is to identify fraudulent online transactions from joined transaction and identity signals while keeping data preparation, validation, and model artifacts auditable.

The initial baseline is deliberately practical: it prioritizes a reliable data contract, memory-aware loading, leakage-conscious validation, and a fast gradient-boosted model. It is a starting point for disciplined experimentation rather than a claim of leaderboard optimality.

## Core Machine Learning Goals

- Handle severe class imbalance without allowing the majority class to dominate training.
- Optimize and report ROC-AUC, the competition's primary evaluation metric.
- Preserve useful missingness and categorical information from both transaction and identity tables.
- Prevent train/validation preprocessing drift by fitting consistent encodings across both partitions.
- Build a repeatable path from experiment to saved model artifact.

## Project Structure

```text
.
├── config/
│   └── config.yaml              # Data paths, validation, and model parameters
├── input_data/                  # Existing raw Kaggle files; never modified by setup
│   └── ieee-fraud-detection/
├── models/                      # Persisted model artifacts
├── notebooks/
│   └── 01_baseline_model.ipynb  # End-to-end baseline experiment
├── reports/                     # Evaluation outputs and figures
├── logs/                        # Run logs
├── src/
│   ├── data/loading.py          # Join and memory reduction utilities
│   ├── features/preprocessing.py# Consistent categorical encoding
│   └── models/baseline.py       # Imbalance-aware LightGBM factory
├── requirements.txt
├── setup_project.py              # Idempotent layout creator
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

## Notes on Scale and Reproducibility

The merged training table is large. Run the notebook on a machine with sufficient RAM, close unused kernels, and avoid retaining duplicate full-size DataFrames. The memory reduction helper downcasts numeric columns, but it cannot eliminate the cost of the initial CSV parse. Pin dependencies and retain random seeds when publishing results.
