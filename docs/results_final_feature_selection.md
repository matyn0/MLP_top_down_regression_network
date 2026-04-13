# Final Feature Selection Results

## Purpose

This document tracks final validation-stage feature-selection results for the selected MLP setup. It currently includes the 30-seed final run and will be updated with the 100-seed final run.

## Fixed Evaluation Setup

- Model: `mlp_256_128_64_32_a1e4`
- Hidden layers: `(256, 128, 64, 32)`
- Alpha: `1e-4`
- Selection metric: validation RMSE (lower is better)
- Compared subsets:
  - `baseline_all_features`
  - `baseline_without_TiltAngle`
  - `without_TiltAngle_and_hour_cos`
- Policy: validation is used for selection, test is reserved for final checkpoint only.

## 30-Seed Final Run

### Run Metadata

- Script: `final_30seed_feature_compare.py`
- Seed range: `0..29`
- Log file: `docs_majk_djuk/final_30seed_compare_20260413_123252.txt`

### Validation Summary by Experiment

| Experiment | mean_rmse | std_rmse | median_rmse | min_rmse | max_rmse | mean_epochs | mean_params |
|---|---:|---:|---:|---:|---:|---:|---:|
| without_TiltAngle_and_hour_cos | 91.175308 | 2.719863 | 90.613387 | 86.953706 | 96.850728 | 183.066667 | 48129.0 |
| baseline_without_TiltAngle | 91.298999 | 2.264732 | 91.403809 | 87.718712 | 97.069165 | 182.666667 | 48385.0 |
| baseline_all_features | 92.804078 | 2.049424 | 92.903064 | 88.935910 | 97.107585 | 194.600000 | 48641.0 |

### Paired Delta Decision Metrics

Delta definition in each row: `candidate - baseline`

| Comparison | mean_delta | std_delta | median_delta | wins | win_rate |
|---|---:|---:|---:|---:|---:|
| no_tilt - all_features | -1.5051 | 3.3456 | -2.0005 | 21/30 | 0.7000 |
| no_tilt_hour_cos - no_tilt | -0.1237 | 3.2608 | -0.4094 | 15/30 | 0.5000 |
| no_tilt_hour_cos - all_features | -1.6288 | 3.2501 | -1.4509 | 21/30 | 0.7000 |

### Ranking (Best to Worst)

1. `without_TiltAngle_and_hour_cos`
2. `baseline_without_TiltAngle`
3. `baseline_all_features`

### Interpretation

- Removing `TiltAngle` is a clear improvement over all-features baseline.
- Removing `hour_cos` on top of `no_tilt` gives only a small average gain and weak stability in this run (`15/30` wins).
- Practical conclusion from 30-seed run: `no_tilt` is a robust choice; `no_tilt + hour_cos` remains a close candidate.

## 100-Seed Final Run (Planned Update)

### Planned Metadata

- Script: `final_30seed_feature_compare.py` (seed list switched to `0..99` for this run)
- Seed range: `0..99`
- Log file: `docs_majk_djuk/<to_be_added>.txt`

### Results Placeholder

Fill this section after the 100-seed run using the same structure as the 30-seed section:

1. Validation summary by experiment (`mean/std/median/min/max`, epochs, params)
2. Paired delta metrics (`mean/std/median`, wins, win_rate)
3. Ranking and interpretation

## Current Decision Status

- Model/hyperparameter optimization complete for current workflow.
- `TiltAngle` removal is confirmed as stable improvement.
- `hour_cos` second-removal candidate is promising but not strongly stable in the 30-seed direct comparison.

## Next Action

- Run 100-seed final comparison and update this file.
- Freeze final subset and run final test checkpoint once.
