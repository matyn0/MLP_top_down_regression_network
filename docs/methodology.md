# Methodology

## Objective

Predict `Irradiance` from meteorological features using feedforward neural networks (`MLPRegressor`) with top-down feature selection and validation-driven model/hyperparameter optimization.

## Dataset Split Policy

- `train`: fitting model parameters.
- `validation`: model selection and feature-selection decisions.
- `test`: final checkpoint only after selection is frozen.

## Locked Evaluation Rules

- Validation is used for:
  - network structure and hyperparameter comparison,
  - feature subset comparison.
- Test set is **never** used during model/feature selection.
- Multi-seed evaluation is required for stability checks.
- Top-down feature selection is required.
- Unstable gains are treated as **candidates**, not final conclusions.

## Leakage Prevention

- Missing values are filled using **train-set medians only**.
- Feature scaling (`StandardScaler`) is fit on **train only** and applied to validation/test.
- Validation and test metrics are computed only after these train-derived transforms.

## Optimization Workflow

### 1) Model Structure + Hyperparameter Optimization

- Script: `multi_seed_model_grid.py`
- Compares candidate MLP structures and `alpha` values across seeds.
- Ranking is based on validation RMSE summary (`mean`, `std`, `median`, `min`, `max`).
- Top model candidates are selected for next stage.

### 2) Top-Down Feature Selection

- Exploratory ablation in `main.py` and focused candidate checks in `multi_seed_second_feature_compare.py`.
- Baseline subset after first-step removal: `without_TiltAngle`.
- Second-feature candidates are compared using paired per-seed deltas.

### 3) Final Focused Validation Comparison

- Script: `final_30seed_feature_compare.py`
- Uses one selected model (`mlp_256_128_64_32_a1e4`) and compares:
  - `baseline_all_features`
  - `baseline_without_TiltAngle`
  - `without_TiltAngle_and_hour_cos`
- Uses 30-seed stability (`0..29`) with paired-delta reporting.

## Decision Criteria

- Primary metric: validation RMSE (lower is better).
- Stability indicators:
  - paired `mean_delta`,
  - paired `median_delta`,
  - `wins / total` and `win_rate`.
- Complexity and training diagnostics are always reported:
  - `n_iter_` (epochs used),
  - `n_params` (trainable parameters).

## Current Status

- Selected model setup for focused runs: hidden layers `(256, 128, 64, 32)`, `alpha=1e-4`.
- `TiltAngle` removal shows stable improvement vs all-features baseline.
- `hour_cos` is a close second-removal candidate with weaker stability.

## Final Checkpoint Protocol

- Freeze final model + final feature subset.
- Run final test checkpoint once.
- Report test metrics without further selection changes.

## Reproducibility Notes

- Multi-seed runs are standard evaluation.
- Terminal output is mirrored to timestamped logs in `docs_majk_djuk/`.
