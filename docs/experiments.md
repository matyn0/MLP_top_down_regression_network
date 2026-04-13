# Experiments Overview

This document describes the current experiment scripts, what each script tests, and how to interpret outputs.

## Common Rules

- Validation is used for model and feature selection.
- Test set is reserved for final checkpoint only.
- Multi-seed comparisons are used for stability.
- Lower RMSE is better.

## 1) Model Grid Selection

Run:

```bash
python3 multi_seed_model_grid.py
```

Purpose:

- Compare model structure and `alpha` across multiple seeds.
- Identify top model candidates for feature-comparison stage.

Current config:

- Seeds: `0..29`.
- Feature set: all features.
- Models compared:
  - `mlp_256_128_64_32_a1e4`
  - `mlp_256_128_64_32_a1e3`
  - `mlp_128_64_32_a1e4`
  - `mlp_128_64_32_a1e3`

Key outputs:

- Raw per-seed table.
- Summary by model (`mean/std/median/min/max` RMSE).
- Top-2 model names and parameters.

Log file pattern:

- `docs_majk_djuk/model_grid_YYYYMMDD_HHMMSS.txt`

## 2) Multi-Seed Second-Feature Compare

Run:

```bash
python3 multi_seed_second_feature_compare.py
```

Purpose:

- For each selected top model, compare second-feature removal candidates on top of `no_tilt` baseline.

Current config:

- Seeds: `0..29`.
- Baseline remove list: `['TiltAngle']`.
- Candidates: `PressureTemp`, `hour_cos`, `HumidityTemp`.
- Top models tested:
  - `mlp_256_128_64_32_a1e4`
  - `mlp_128_64_32_a1e4`

Key outputs:

- Raw per-seed candidate comparison table.
- Summary by `(model_name, candidate_feature)`.
- Overall candidate ranking across both models.

Log file pattern:

- `docs_majk_djuk/second_feature_compare_YYYYMMDD_HHMMSS.txt`

## 3) Final Focused 30-Seed Comparison

Run:

```bash
python3 final_30seed_feature_compare.py
```

Purpose:

- Final validation-stage comparison of three fixed subsets under one selected model.

Current config:

- Seeds: `0..29`.
- Model: `mlp_256_128_64_32_a1e4`.
- Experiments:
  - `baseline_all_features`
  - `baseline_without_TiltAngle`
  - `without_TiltAngle_and_hour_cos`

Key outputs:

- Raw per-seed stability table (90 rows).
- Per-experiment summary (`mean/std/median/min/max` RMSE, mean epochs, mean params).
- Paired deltas and decision metrics for:
  - `no_tilt - all_features`
  - `no_tilt_hour_cos - no_tilt`
  - `no_tilt_hour_cos - all_features`
- Final ranking (best to worst).

Log file pattern:

- `docs_majk_djuk/final_30seed_compare_YYYYMMDD_HHMMSS.txt`

## 4) Legacy Exploratory Script

Run:

```bash
python3 main.py
```

Purpose:

- Historical exploratory workflow: baseline setup, leave-one-feature-out ablation, and seed-stability checks.
- Keeps transparent record of earlier feature-testing path.

Log file pattern:

- `docs_majk_djuk/run_YYYYMMDD_HHMMSS.txt`

## How to Read Paired Deltas

- Delta is always `candidate - baseline` in the given comparison block.
- Negative delta means candidate is better on that seed.
- Positive delta means baseline is better on that seed.
- Use `mean_delta`, `median_delta`, and `win_rate` together for decisions.

## Current Decision Status

- Selected model setup for final focused runs: `(256, 128, 64, 32)` with `alpha=1e-4`.
- Removing `TiltAngle` is a stable improvement over all-features baseline.
- Removing `hour_cos` on top of `no_tilt` is a close candidate but less stable.
- `PressureTemp` second-removal signal is not robust.

## Next Step

- Freeze final selected subset.
- Run final checkpoint on test set once.
