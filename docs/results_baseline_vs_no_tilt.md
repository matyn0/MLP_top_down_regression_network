Results: Baseline vs No TiltAngle
 Compared Experiments
- `baseline_all_features`
- `baseline_not_TiltAngle_included`
 Multi-Seed Validation Summary
- `baseline_all_features`: mean RMSE = `93.3775`, std = `2.1781`
- `baseline_not_TiltAngle_included`: mean RMSE = `91.9683`, std = `1.9747`
 Paired Per-Seed Delta
Delta definition:
- `delta = no_tilt - baseline`
Interpretation:
- negative delta => no_tilt better for that seed
- positive delta => baseline better for that seed
Observed decision metrics:
- `mean_delta = -1.4092`
- `std_delta = 2.7532`
- `wins_no_tilt = 7/10`
 Current Interpretation
- Removing `TiltAngle` is a **candidate improvement** on validation.
- Evidence is favorable but not perfectly stable across all seeds.
- This is not final test-based proof; selection remains validation-driven at this stage.
 Methodology Compliance
- Validation used for model/feature selection
- Test reserved for final checkpoint
- Multi-seed evaluation applied
