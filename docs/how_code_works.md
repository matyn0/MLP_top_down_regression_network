# How the Current Code Works

## What This Project Does

This project predicts `Irradiance` from meteorological inputs using `MLPRegressor`. Model and feature choices are made on validation only (with multi-seed checks), and test is reserved for the final checkpoint.

## Shared Pipeline Used by All Experiment Scripts

All experiment scripts follow the same core steps:

1. Load datasets with `data_loader.load_datasets()`.
2. Add cyclic time features with `preprocessing.add_time_cyclis_features()`.
3. Train/evaluate with `train.run_experiment(...)`.
4. Collect RMSE plus training diagnostics (`n_iter_`, `n_params`).
5. Print results and mirror output to timestamped logs in `docs_majk_djuk/`.

## Core Engine: `train.py`

### `run_experiment(...)`

One complete train/validation run (and optional test run).

Main behavior:

1. Build feature subset as `all_features - features_to_remove`.
2. Validate that removed feature names exist (safety check).
3. Split to `X_train/y_train`, `X_val/y_val` (and test if enabled).
4. Fill missing values using train medians only.
5. Fit `StandardScaler` on train only, then transform val/test.
6. Build MLP params from defaults + optional `mlp_params` overrides.
7. Train `MLPRegressor` and compute validation metrics.
8. Return metrics + metadata (feature count, epochs, params, seed).

### `run_leave_one_feature_out_ablation(...)`

Validation-only ablation utility used in exploratory phase.

- Removes one feature at a time from a baseline subset.
- Reports `delta_rmse = rmse_without_feature - baseline_rmse`.

### `run_seed_stability(...)`

Runs multiple experiment configs across multiple seeds.

- Returns per-seed table for paired comparisons and stability checks.

## Script Flows

### `main.py` (Exploratory Workflow)

- Historical exploration script.
- Runs baseline setup, leave-one-feature-out ablation, and seed stability for one selected candidate.
- Useful to show the feature-testing path used during development.

### `multi_seed_model_grid.py` (Model Selection)

- Compares multiple model structures/alphas across seeds.
- Produces model ranking by validation RMSE summary.
- Used to select top model candidates.

### `multi_seed_second_feature_compare.py` (Second-Feature Screening)

- Uses `baseline_without_TiltAngle` and tests second-feature removals.
- Current seed setting: `0..29`.
- Compares candidates (`PressureTemp`, `hour_cos`, `HumidityTemp`) on top selected models.

### `final_30seed_feature_compare.py` (Final Focused Validation Compare)

- Uses one selected model: `mlp_256_128_64_32_a1e4`.
- Compares exactly 3 fixed subsets across 30 seeds:
  - `baseline_all_features`
  - `baseline_without_TiltAngle`
  - `without_TiltAngle_and_hour_cos`
- Prints raw table, summary table, paired deltas, wins/win-rate, and final ranking.

## How To Read the Outputs

- Lower RMSE is better.
- In paired blocks, delta is always `candidate - baseline`.
- Negative delta means candidate is better for that seed.
- `wins` = number of seeds where delta is negative.
- `n_iter_` = epochs used; `n_params` = network complexity.

## Logs and Reproducibility

Each script saves timestamped logs in `docs_majk_djuk/`:

- `run_YYYYMMDD_HHMMSS.txt` (from `main.py`)
- `model_grid_YYYYMMDD_HHMMSS.txt`
- `second_feature_compare_YYYYMMDD_HHMMSS.txt`
- `final_30seed_compare_YYYYMMDD_HHMMSS.txt`

## Current Project Stage

- Model/hyperparameter selection done for current workflow.
- `TiltAngle` removal is a stable improvement over all-features baseline.
- `hour_cos` is a close second-removal candidate, less stable than first removal.
- Next step is to freeze final subset and run final test checkpoint once.
