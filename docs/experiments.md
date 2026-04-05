Experiments
 Run Command
python3 main.py
What main.py does
1. Loads train/validation/test datasets
2. Adds time cyclic features
3. Runs baseline experiment with all features (validation-only selection phase)
4. Runs leave-one-feature-out ablation (validation-only)
5. Runs multi-seed stability comparison for:
   - baseline_all_features
   - baseline_not_TiltAngle_included
6. Prints validation summaries and paired per-seed deltas
7. Mirrors terminal output to docs_majk_djuk/run_YYYYMMDD_HHMMSS.txt
Key Outputs
- Baseline validation metrics
- Ablation table sorted by delta_rmse
- Stability table across seeds
- Validation-only summary stats per experiment
- Paired deltas: no_tilt - baseline
- Decision helper:
  - mean_delta
  - std_delta
  - wins_no_tilt
Interpretation Rules
- Lower RMSE is better
- For paired delta:
  - negative = no_tilt better for that seed
  - positive = baseline better for that seed
- Feature-removal decisions are based on validation behavior and seed stability
- Test set remains final checkpoint only
