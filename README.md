**************** MLP Top-Down Regression Network**************


Python ML thesis project where `MLPRegressor` predicts **Irradiance** from meteorological data.


*Project Structure*
`main.py` - orchestration and entry point
`data_loader.py` - dataset loading
`preprocessing.py` - feature engineering (time cyclic features)
`train.py` - experiment execution, ablation, seed stability
`evaluate.py` - metrics computation
`utils.py` - helpers (stdout tee, parameter counting)
 
*Methodology* (Locked)

- Validation set is for model/feature selection only
- Test set is final checkpoint only
- Multi-seed evaluation is required
- Top-down feature ablation is required
- Unstable removals are candidates, not final conclusions


*Data Files*

Expected in project root:
- `training_meteo_data_cleaned.csv`
- `validation_meteo_data_cleaned.csv`
- `testing_meteo_data_cleaned.csv`


*Setup*

python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn


*Run*

python3 main.py

*Logging*

Run output is mirrored to terminal and timestamped text logs:
- docs_majk_djuk/run_YYYYMMDD_HHMMSS.txt



Notes
Current workflow prioritizes correctness, reproducibility, and fair validation-driven comparison for thesis-quality reporting.
