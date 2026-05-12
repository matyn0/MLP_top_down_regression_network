import sys
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout

from data_loader import load_datasets
from preprocessing import add_time_cyclis_features
from utils import Tee
from train import run_experiment, run_leave_one_feature_out_ablation, run_seed_stability


def main():
    """Exploracny feature-selection skript: baseline, ablation a stability check."""
    train_df, val_df, test_df = load_datasets()
    train_df = add_time_cyclis_features(train_df)
    val_df = add_time_cyclis_features(val_df)
    test_df = add_time_cyclis_features(test_df)

    print(" # DATASET SHAPES #")
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    baseline_remove = ["TiltAngle"]  # aktualny baseline subset uz nepouziva TiltAngle
    next_feature_to_test = "PressureTemp"  # kandidat, ktoreho chceme otestovat ako dalsi removal
    next_remove = baseline_remove + [next_feature_to_test]
    candidate_exp_name = f"without_TiltAngle_and_{next_feature_to_test}"

    non_feature_cols = ["PictureName", "DateTime", "IrradianceNotCompensated", "Irradiance"]

    print("\n# BASELINE without TiltAngle #")
    result = run_experiment(
        train_df,
        val_df,
        test_df,
        features_to_remove=baseline_remove,
        experiment_name="baseline_without_TiltAngle",
        evaluate_test=False,
    )

    baseline_rmse = result["val_rmse"]
    baseline_features = [
        c for c in train_df.columns
        if c not in (non_feature_cols + baseline_remove)
    ]

    print("\n # BASELINE REFERENCE #")
    print(f"baseline_val_rmse={baseline_rmse:.4f}")
    print(f"baseline_feature_count={len(baseline_features)}")

    # Ablation skusi vyhodit vzdy jednu feature z baseline subsetu.
    ablation_df = run_leave_one_feature_out_ablation(
        train_df=train_df,
        val_df=val_df,
        baseline_features=baseline_features,
        baseline_rmse=baseline_rmse,
        baseline_removed_features=baseline_remove,
        random_state=42,
    )

    print("\n # ABLATION RESULTS (sorted by delta_rmse desc) #")
    print(ablation_df.to_string(index=False))

    pressure_row_df = ablation_df.loc[ablation_df["feature_removed"] == next_feature_to_test]

    print(f"\n # FOCUS CANDIDATE: {next_feature_to_test} #")
    if pressure_row_df.empty:
        print(f"{next_feature_to_test} was not found in ablation candidates.")
    else:
        pressure_row = pressure_row_df.iloc[0]
        print(
            f"val_rmse_without={pressure_row['val_rmse_without']:.4f}, "
            f"delta_rmse={pressure_row['delta_rmse']:.4f}"
        )

    best_drop_row = ablation_df.loc[ablation_df["val_rmse_without"].idxmin()]  # najnizsie RMSE po odstraneni jednej feature
    worst_drop_row = ablation_df.loc[ablation_df["val_rmse_without"].idxmax()]  # najhorsie RMSE po odstraneni jednej feature
    most_important_row = ablation_df.loc[ablation_df["delta_rmse"].idxmax()]  # najvacsie zhorsenie po odstraneni
    helpful_drop_df = ablation_df.loc[ablation_df["delta_rmse"] < 0, :].copy()  # odstranenie feature zlepsilo RMSE

    print("\n # FINAL SUMMARY #")
    print(
        f"Baseline model: {result['best_model']} | "
        f"features={result['n_features']} | epochs={result['n_iter_']} | params={result['n_params']}"
    )
    print(f"Baseline metrics: VAL_RMSE={result['val_rmse']:.4f}")

    print("\nBest single-feature drop by VAL_RMSE (lowest):")
    print(
        f"drop='{best_drop_row['feature_removed']}', "
        f"val_rmse_without={best_drop_row['val_rmse_without']:.4f}, "
        f"delta_rmse={best_drop_row['delta_rmse']:.4f}"
    )

    print("\nMost important feature (largest RMSE increase when removed):")
    print(
        f"feature='{most_important_row['feature_removed']}', "
        f"val_rmse_without={most_important_row['val_rmse_without']:.4f}, "
        f"delta_rmse={most_important_row['delta_rmse']:.4f}"
    )

    print("\nWorst single-feature drop by VAL_RMSE (highest):")
    print(
        f"drop='{worst_drop_row['feature_removed']}', "
        f"val_rmse_without={worst_drop_row['val_rmse_without']:.4f}, "
        f"delta_rmse={worst_drop_row['delta_rmse']:.4f}"
    )

    print("\nFeatures with negative delta_rmse (removing helped):")
    if helpful_drop_df.empty:
        print("none")
    else:
        helpful_display_df = helpful_drop_df.loc[:, ["feature_removed", "val_rmse_without", "delta_rmse"]]
        print(helpful_display_df.to_string(index=False))

    seed_list = list(range(30))  # multi-seed check ukaze, ci kandidat nie je nahodne dobry len pre jeden seed

    stability_configs = [
        {"name": "baseline_without_TiltAngle", "remove": baseline_remove},
        {"name": candidate_exp_name, "remove": next_remove},
    ]

    stability_df = run_seed_stability(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        seed_list=seed_list,
        stability_configs=stability_configs,
        evaluate_test=False,
    )

    print("\n # Stability result per SEED #")
    print(stability_df.to_string(index=False))

    print("\n ** val-only summ **")
    exp_summ = (
        stability_df.groupby("experiment", as_index=False)["val_rmse"]
        .agg(mean="mean", std="std", max="max", min="min")
    )

    print("\n Per exp (val rmse stats):")
    print(exp_summ.to_string(index=False))

    # Pivot spravi tabulku seed x experiment, aby sa dala pocitat parova delta.
    paired = stability_df.pivot(
        index="random_state",
        columns="experiment",
        values="val_rmse",
    )

    baseline_col = "baseline_without_TiltAngle"
    candidate_col = candidate_exp_name
    delta_col = f"delta_{next_feature_to_test}_minus_baseline"

    paired[delta_col] = paired[candidate_col] - paired[baseline_col]  # zaporne = kandidat ma nizsie RMSE

    print(f"\n Paired per seed delta ({next_feature_to_test} - baseline): ")
    print(
        paired[[baseline_col, candidate_col, delta_col]]
        .reset_index()
        .to_string(index=False)
    )

    delta_series = paired[delta_col]
    mean_delta = delta_series.mean()
    std_delta = delta_series.std()
    wins = int((delta_series < 0).to_numpy().sum())  # pocet seedov, kde kandidat vyhral nad baseline
    total = int(delta_series.notna().to_numpy().sum())

    print("\n Decision metrics: ")
    print(f"mean_delta = {mean_delta:.4f}")
    print(f"std_delta = {std_delta:.4f}")
    print(f"wins_{next_feature_to_test}_removed = {wins}/{total}")


if __name__ == "__main__":
    log_dir = Path("docs_majk_djuk")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.txt"

    # Vystup ide naraz do terminalu aj do log suboru.
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            main()
