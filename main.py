import sys
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout
from data_loader import load_datasets
from preprocessing import add_time_cyclis_features
from utils import Tee
from train import run_experiment, run_leave_one_feature_out_ablation, run_seed_stability


def main():
    
    train_df, val_df, test_df = load_datasets()

    train_df = add_time_cyclis_features(train_df)
    val_df = add_time_cyclis_features(val_df)
    test_df = add_time_cyclis_features(test_df)

    print(" # DATASET SHAPES #")
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    print("\n# BASELINE with all features #") ##all of this becasue i wanted reorder prints to create result and baseline_rmse earlier
    result = run_experiment(
        train_df,
        val_df,
        test_df,
        features_to_remove = [],
        experiment_name="baseline_all_features",
        evaluate_test=False
    )

    baseline_rmse = result["val_rmse"]
    baseline_features = [
        c for c in train_df.columns
        if c not in ["PictureName", "DateTime", "IrradianceNotCompensated", "Irradiance"]
    ]
    print("\n # BASELINE REFERENCE #")
    print(f"baseline_val_rmse={baseline_rmse:.4f}")
    print(f"baseline_feature_count={len(baseline_features)}")
    
    # Step 2: leave-one-feature-out ablation (validation only)
    ablation_df = run_leave_one_feature_out_ablation(
        train_df=train_df,
        val_df=val_df,
        baseline_features=baseline_features,
        baseline_rmse=baseline_rmse,
        random_state=42,
    )
    print("\n # ABLATION RESULTS (sorted by delta_rmse desc) #")
    print(ablation_df.to_string(index=False))
    
    # Final summary from all tests
    best_drop_row = ablation_df.loc[ablation_df["val_rmse_without"].idxmin()]   # best RMSE after dropping one feature
    worst_drop_row = ablation_df.loc[ablation_df["val_rmse_without"].idxmax()]  # worst RMSE after dropping one feature
    most_important_row = ablation_df.loc[ablation_df["delta_rmse"].idxmax()]    # biggest positive delta
    helpful_drop_df = ablation_df.loc[ablation_df["delta_rmse"] < 0, :].copy()                # dropping helped

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
        f"delta_rmse={best_drop_row['delta_rmse']:.4f}")
    print("\nMost important feature (largest RMSE increase when removed):")
    print(
        f"feature='{most_important_row['feature_removed']}', "
        f"val_rmse_without={most_important_row['val_rmse_without']:.4f}, "
        f"delta_rmse={most_important_row['delta_rmse']:.4f}")
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






    #SEED LIST#
    seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    #STABILITY CONFIGS
    stability_configs = [
        {"name": "baseline_all_features", "remove": []},
        {"name": "baseline_not_TiltAngle_included", "remove": ["TiltAngle"]},
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

    paired = stability_df.pivot(
        index = "random_state",
        columns= "experiment",
        values="val_rmse"

    )

    baseline_col = "baseline_all_features"
    no_tilt_col = "baseline_not_TiltAngle_included"

    paired["delta_no_tilt_minus_baseline"] = paired[no_tilt_col] - paired[baseline_col]

    print("\n Paired per seed delta (no_tilt - baseline): ")
    print(
        paired[[baseline_col, no_tilt_col, "delta_no_tilt_minus_baseline"]]
        .reset_index()
        .to_string(index=False)

    )

    delta_series = paired["delta_no_tilt_minus_baseline"]
    mean_delta = delta_series.mean()
    std_delta = delta_series.std()
    wins = int((delta_series < 0).to_numpy().sum())
    total = int(delta_series.notna().to_numpy().sum())

    print("\n Decision medic: ")
    print(f"mean_delta = {mean_delta:.4f}")
    print(f"std_delta = {std_delta:.4f}")
    print(f"wins_no_tilt = {wins}/{total}")



    

    











    #results = [result]
    ##### ALL ABOUT EXPERIMENT CONFIGS
    #experiment_configs = [
    #    {"name": "reduced_no_tiltangle", "remove": ["TiltAngle"]},
    #    {"name": "reduced_no_hour_sin", "remove": ["hour_sin"]},
    #    {"name": "reduced_no_tiltangle_hour_sin", "remove": ["TiltAngle", "hour_sin"]},
    #]

    #for cfg in experiment_configs:
    #    print(f"\n===== {cfg['name']} | remove={cfg['remove']} =====")
    #    res = run_experiment(
    #        train_df,
    #        val_df,
    #        test_df,
    #        features_to_remove=cfg["remove"],
    #        experiment_name=cfg["name"]
    #    )
    #    results.append(res)

    #comparison_df = pd.DataFrame(results)[
    #    ["experiment", "removed_features", "n_features", "val_rmse", "test_rmse", "test_r2"]
    #].sort_values("val_rmse", ascending=True)

    #print("\n # Feature-Removal Comparison #")
    #print(comparison_df.to_string(index=False))
    


if __name__ == "__main__":
    log_dir = Path("docs_majk_djuk")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.txt"

    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            main()
    

# delta_rmse > 0: removing feature made RMSE worse -> feature is useful/important.
# delta_rmse ~ 0: little effect.
# delta_rmse < 0: removing feature improved RMSE -> feature may be noisy/redundant.
