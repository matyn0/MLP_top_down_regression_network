import sys
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout

import pandas as pd

from data_loader import load_datasets
from preprocessing import add_time_cyclis_features
from train import run_experiment
from utils import Tee


def print_paired_comparison(paired_df, left_col, right_col, delta_col, title):
    """Vypise parove porovnanie dvoch experimentov cez rovnake seedy."""
    comp_df = paired_df[[left_col, right_col]].copy()
    comp_df[delta_col] = comp_df[right_col] - comp_df[left_col]  # zaporne = pravy experiment ma nizsie RMSE

    print(f"\n# {title} #")
    print(comp_df.reset_index().to_string(index=False))

    delta = comp_df[delta_col]
    mean_delta = delta.mean()
    std_delta = delta.std()
    median_delta = delta.median()
    wins = int((delta < 0).to_numpy().sum())  # pocet seedov, kde pravy experiment vyhral
    total = int(delta.notna().to_numpy().sum())
    win_rate = wins / total if total > 0 else 0.0

    print("\nDecision metrics:")
    print(f"mean_delta = {mean_delta:.4f}")
    print(f"std_delta = {std_delta:.4f}")
    print(f"median_delta = {median_delta:.4f}")
    print(f"wins = {wins}/{total}")
    print(f"win_rate = {win_rate:.4f}")


def main():
    """Finalne validation porovnanie vybranych feature subsetov cez viac seedov."""
    train_df, val_df, test_df = load_datasets()
    train_df = add_time_cyclis_features(train_df)
    val_df = add_time_cyclis_features(val_df)
    test_df = add_time_cyclis_features(test_df)

    print("# DATASET SHAPES #")
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    seed_list = list(range(30))  # rovnake seedy pouzivame pre ferove porovnanie subsetov

    model_name = "mlp_256_128_64_32_a1e4"
    model_params = {
        "hidden_layer_sizes": (256, 128, 64, 32),
        "alpha": 1e-4,
    }

    # Porovnavame uz len finalnych kandidatov feature subsetu pri jednom vybranom modeli.
    experiment_configs = [
        {"name": "baseline_all_features", "remove": []},
        {"name": "baseline_without_TiltAngle", "remove": ["TiltAngle"]},
        {
            "name": "without_TiltAngle_and_hour_cos",
            "remove": ["TiltAngle", "hour_cos"],
        },
    ]

    rows = []

    # Kazdy subset pustime cez rovnake seedy, aby rozdiel nebol iba nahodna inicializacia MLP.
    for seed in seed_list:
        print(f"\n# SEED = {seed} #")
        for cfg in experiment_configs:
            print(f"\n# {cfg['name']} | remove={cfg['remove']} | seed={seed} #")
            res = run_experiment(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                features_to_remove=cfg["remove"],
                experiment_name=cfg["name"],
                random_state=seed,
                evaluate_test=False,
                mlp_params=model_params,
                model_name=model_name,
            )

            rows.append(
                {
                    "random_state": seed,
                    "experiment": cfg["name"],
                    "removed_features": res["removed_features"],
                    "n_features": res["n_features"],
                    "val_rmse": res["val_rmse"],
                    "n_iter_": res["n_iter_"],
                    "n_params": res["n_params"],
                }
            )

    stability_df = pd.DataFrame(rows).sort_values(["random_state", "experiment"])

    print("\n# STABILITY RESULT PER SEED #")
    print(stability_df.to_string(index=False))

    # Summary ukazuje priemerny vykon, stabilitu a rozsah RMSE pre kazdy experiment.
    summary_df = (
        stability_df.groupby("experiment", as_index=False)
        .agg(
            mean_rmse=("val_rmse", "mean"),
            std_rmse=("val_rmse", "std"),
            median_rmse=("val_rmse", "median"),
            min_rmse=("val_rmse", "min"),
            max_rmse=("val_rmse", "max"),
            mean_epochs=("n_iter_", "mean"),
            mean_params=("n_params", "mean"),
        )
        .sort_values(["mean_rmse", "std_rmse"], ascending=[True, True])
    )

    print("\n# VAL-ONLY SUMMARY PER EXPERIMENT #")
    print(summary_df.to_string(index=False))

    # Pivot spravi z vysledkov tabulku seed x experiment, aby sa dali pocitat parove delty.
    paired = stability_df.pivot(index="random_state", columns="experiment", values="val_rmse")

    print_paired_comparison(
        paired_df=paired,
        left_col="baseline_all_features",
        right_col="baseline_without_TiltAngle",
        delta_col="delta_no_tilt_minus_all_features",
        title="PAIRED DELTA (no_tilt - all_features)",
    )

    print_paired_comparison(
        paired_df=paired,
        left_col="baseline_without_TiltAngle",
        right_col="without_TiltAngle_and_hour_cos",
        delta_col="delta_no_tilt_hour_cos_minus_no_tilt",
        title="PAIRED DELTA (no_tilt_hour_cos - no_tilt)",
    )

    print_paired_comparison(
        paired_df=paired,
        left_col="baseline_all_features",
        right_col="without_TiltAngle_and_hour_cos",
        delta_col="delta_no_tilt_hour_cos_minus_all_features",
        title="PAIRED DELTA (no_tilt_hour_cos - all_features)",
    )

    print("\n# FINAL RANKING (BEST -> WORST) #")
    ranking_df = summary_df[
        ["experiment", "mean_rmse", "std_rmse", "median_rmse", "min_rmse", "max_rmse"]
    ]
    print(ranking_df.to_string(index=False))


if __name__ == "__main__":
    log_dir = Path("docs_majk_djuk")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"start_for_final_results{timestamp}.txt"

    # Vystup ide naraz do terminalu aj do log suboru.
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            main()
