import sys
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout

import pandas as pd

from data_loader import load_datasets
from preprocessing import add_time_cyclis_features
from train import run_experiment
from utils import Tee


def main():
    """Porovna kandidatov na druhe odstranenie feature cez viac modelov a seedov."""
    train_df, val_df, test_df = load_datasets()
    train_df = add_time_cyclis_features(train_df)
    val_df = add_time_cyclis_features(val_df)
    test_df = add_time_cyclis_features(test_df)

    print("# DATASET SHAPES #")
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    seed_list = list(range(30))  # rovnake seedy pouzivame pre ferove parove porovnanie
    baseline_remove = ["TiltAngle"]  # prvy stabilny removal, od ktoreho testujeme dalsie features

    # Testujeme len top modely, ktore vysli najlepsie z model grid porovnania.
    top_models = [
        {
            "name": "mlp_256_128_64_32_a1e4",
            "mlp_params": {
                "hidden_layer_sizes": (256, 128, 64, 32),
                "alpha": 1e-4,
            },
        },
        {
            "name": "mlp_128_64_32_a1e4",
            "mlp_params": {
                "hidden_layer_sizes": (128, 64, 32),
                "alpha": 1e-4,
            },
        },
    ]

    candidate_features = ["PressureTemp", "hour_cos", "HumidityTemp"]  # kandidati na druhy removal

    rows = []
    for model_cfg in top_models:
        model_name = model_cfg["name"]
        print(f"\n######## MODEL: {model_name} ########")

        for seed in seed_list:
            print(f"\n# SEED = {seed} | BASELINE without TiltAngle #")

            # Baseline pocitame pre kazdy model a seed, aby kandidat mal ferove porovnanie.
            baseline_res = run_experiment(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                features_to_remove=baseline_remove,
                experiment_name="baseline_without_TiltAngle",
                random_state=seed,
                evaluate_test=False,
                mlp_params=model_cfg["mlp_params"],
                model_name=model_name,
            )
            baseline_rmse = baseline_res["val_rmse"]

            for candidate in candidate_features:
                remove_list = baseline_remove + [candidate]
                exp_name = f"without_TiltAngle_and_{candidate}"

                print(f"\n# SEED = {seed} | MODEL={model_name} | CANDIDATE={candidate} #")

                candidate_res = run_experiment(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    features_to_remove=remove_list,
                    experiment_name=exp_name,
                    random_state=seed,
                    evaluate_test=False,
                    mlp_params=model_cfg["mlp_params"],
                    model_name=model_name,
                )

                candidate_rmse = candidate_res["val_rmse"]
                delta = candidate_rmse - baseline_rmse  # zaporne = kandidat je lepsi ako baseline

                rows.append(
                    {
                        "model_name": model_name,
                        "seed": seed,
                        "candidate_feature": candidate,
                        "baseline_rmse": baseline_rmse,
                        "candidate_rmse": candidate_rmse,
                        "delta_candidate_minus_baseline": delta,
                    }
                )

    details_df = pd.DataFrame(rows).sort_values(["model_name", "candidate_feature", "seed"])

    print("\n# RAW PER-SEED COMPARISON #")
    print(details_df.to_string(index=False))

    details_df["is_win"] = (details_df["delta_candidate_minus_baseline"] < 0).astype(int)  # 1 = kandidat vyhral seed

    # Delta statistiky citame takto: mean/median hovoria typicky rozdiel, std stabilitu,
    # min najlepsie zlepsenie a max najhorsie zhorsenie kandidata.
    summary_df = (
        details_df.groupby(["model_name", "candidate_feature"], as_index=False)
        .agg(
            mean_delta=("delta_candidate_minus_baseline", "mean"),
            std_delta=("delta_candidate_minus_baseline", "std"),
            median_delta=("delta_candidate_minus_baseline", "median"),
            min_delta=("delta_candidate_minus_baseline", "min"),
            max_delta=("delta_candidate_minus_baseline", "max"),
            wins=("is_win", "sum"),
            total=("is_win", "count"),
        )
    )
    summary_df["win_rate"] = summary_df["wins"] / summary_df["total"]
    summary_df = summary_df.sort_values(["model_name", "mean_delta", "win_rate"], ascending=[True, True, False])

    print("\n# SUMMARY BY MODEL + CANDIDATE #")
    print(summary_df.to_string(index=False))

    # Overall ranking spaja oba modely; mean/median ukazuju typicky efekt kandidata,
    # win_rate ukazuje, ako casto kandidat realne vyhral oproti baseline.
    overall_df = (
        details_df.groupby("candidate_feature", as_index=False)
        .agg(
            mean_delta=("delta_candidate_minus_baseline", "mean"),
            std_delta=("delta_candidate_minus_baseline", "std"),
            median_delta=("delta_candidate_minus_baseline", "median"),
            wins=("is_win", "sum"),
            total=("is_win", "count"),
        )
    )
    overall_df["win_rate"] = overall_df["wins"] / overall_df["total"]
    overall_df = overall_df.sort_values(["mean_delta", "win_rate"], ascending=[True, False])

    print("\n# OVERALL CANDIDATE RANKING (BOTH MODELS TOGETHER) #")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    log_dir = Path("docs_majk_djuk")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"second_feature_compare_{timestamp}.txt"

    # Vystup ide naraz do terminalu aj do log suboru.
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            main()
