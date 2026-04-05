Methodology
 Objective
Predict **Irradiance** from meteorological features using `MLPRegressor`.
 Dataset Split Policy
- `train`: model fitting
- `validation`: model selection and feature selection
- `test`: final checkpoint only
 Locked Evaluation Rules
- Validation set is used for:
  - feature subset decisions
  - model comparison
- Test set is **not** used during feature/model selection.
- Multi-seed evaluation is required for stability checks.
- Feature ablation is required (leave-one-feature-out / top-down workflow).
- Unstable removals are treated as **candidates**, not final conclusions.
 Leakage Prevention
- Missing values are filled using **train-set medians only**.
- Feature scaling is fit on **train only**, then applied to validation/test.
- Validation and test metrics are computed only after these train-derived transforms.
 Current Model Baseline
- `MLPRegressor`
- hidden layers: `(256, 128, 64, 32)`
- activation: `relu`
- solver: `adam`
- max_iter: `1500`
 Reproducibility Notes
- Multi-seed runs are part of standard evaluation.
- Terminal output is mirrored to timestamped logs in `docs_majk_djuk/`.
