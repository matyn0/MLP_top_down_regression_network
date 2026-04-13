## MLP Top-Down Regression Network

Python ML thesis project where `MLPRegressor` predicts `Irradiance` from meteorological data.

## Objective

Perform top-down feature selection for a regression task while optimizing MLP structure and hyperparameters.

## Methodology (Locked)

- Use `train` for fitting, `validation` for model/feature selection, and `test` only as final checkpoint.
- Run multi-seed comparisons for stability.
- Use top-down feature ablation for feature selection.
- Prevent leakage: fill missing values from train medians only, fit scaling on train only.
- Track and report `RMSE`, `n_iter_` (epochs), and `n_params` (complexity).
- Treat unstable improvements as candidates, not final conclusions.

## Project Scripts

- `main.py`: exploratory pipeline (ablation + seed stability; useful history of feature-testing workflow).
- `multi_seed_model_grid.py`: model structure/alpha comparison across seeds.
- `multi_seed_second_feature_compare.py`: second-feature comparison across seeds for selected top models.
- `final_30seed_feature_compare.py`: final focused 30-seed comparison of:
  - `baseline_all_features`
  - `baseline_without_TiltAngle`
  - `without_TiltAngle_and_hour_cos`

Core modules:

- `data_loader.py`: dataset loading.
- `preprocessing.py`: feature engineering (time cyclic features).
- `train.py`: experiment execution, ablation, seed stability helpers.
- `evaluate.py`: metrics computation.
- `utils.py`: helper utilities (`Tee`, parameter counting).

## Data Files

Expected in project root:

- `training_meteo_data_cleaned.csv`
- `validation_meteo_data_cleaned.csv`
- `testing_meteo_data_cleaned.csv`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn
```

## Run

```bash
python3 main.py
python3 multi_seed_model_grid.py
python3 multi_seed_second_feature_compare.py
python3 final_30seed_feature_compare.py
```

## Logging

All run scripts mirror output to terminal and save timestamped logs in `docs_majk_djuk/`:

- `run_YYYYMMDD_HHMMSS.txt`
- `model_grid_YYYYMMDD_HHMMSS.txt`
- `second_feature_compare_YYYYMMDD_HHMMSS.txt`
- `final_30seed_compare_YYYYMMDD_HHMMSS.txt`

## Current Validation Status

- Best model setup used in final focused runs: `hidden_layer_sizes=(256, 128, 64, 32)`, `alpha=1e-4`.
- Confirmed improvement: remove `TiltAngle`.
- `hour_cos` is a close candidate second removal.
- `PressureTemp` did not show robust second-removal gain.

## Next Step

Freeze final selected subset and run the final test checkpoint once.

## Docs

- `docs/methodology.md`
- `docs/experiments.md`
- `docs/how_code_works.md`
- `docs/results_baseline_vs_no_tilt.md`
- `docs/results_final_feature_selection.md`
