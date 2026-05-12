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
    """Porovna viac MLP konfiguracii cez rovnake seedy a zoradi ich podla validation RMSE."""
    train_df, val_df, test_df = load_datasets()
    train_df = add_time_cyclis_features(train_df)
    val_df = add_time_cyclis_features(val_df)
    test_df = add_time_cyclis_features(test_df)

    print("# DATASET SHAPES #")
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    seed_list = list(range(30))  # kazdy model testujeme na rovnakych seedoch
    features_to_remove = []  # model selection robime najprv na vsetkych features

    # Kazda konfiguracia meni architekturu siete alebo regularizaciu alpha.
    model_configs = [
        {
            "name": "mlp_256_128_64_32_a1e4",
            "mlp_params": {
                "hidden_layer_sizes": (256, 128, 64, 32),
                "alpha": 1e-4,
            },
        },
        {
            "name": "mlp_256_128_64_32_a1e3",
            "mlp_params": {
                "hidden_layer_sizes": (256, 128, 64, 32),
                "alpha": 1e-3,
            },
        },
        {
            "name": "mlp_128_64_32_a1e4",
            "mlp_params": {
                "hidden_layer_sizes": (128, 64, 32),
                "alpha": 1e-4,
            },
        },
        {
            "name": "mlp_128_64_32_a1e3",
            "mlp_params": {
                "hidden_layer_sizes": (128, 64, 32),
                "alpha": 1e-3,
            },
        },
    ]

    rows = []

    # Spustime kazdy model cez kazdy seed, aby porovnanie nestalo na jednej inicializacii vah.
    for seed in seed_list:
        print(f"\n# SEED = {seed} #")
        for cfg in model_configs:
            print(f"\n# MODEL = {cfg['name']} | seed={seed} #")
            res = run_experiment(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                features_to_remove=features_to_remove,
                experiment_name="baseline_all_features",
                random_state=seed,
                evaluate_test=False,
                mlp_params=cfg["mlp_params"],
                model_name=cfg["name"],
            )

            rows.append(
                {
                    "seed": seed,
                    "model_name": cfg["name"],
                    "removed_features": res["removed_features"],
                    "n_features": res["n_features"],
                    "val_mae": res["val_mae"],
                    "val_rmse": res["val_rmse"],
                    "val_r2": res["val_r2"],
                    "n_iter_": res["n_iter_"],
                    "n_params": res["n_params"],
                }
            )

    raw_df = pd.DataFrame(rows).sort_values(["model_name", "seed"])

    print("\n# RAW RESULTS #")
    print(raw_df.to_string(index=False))

    # Summary pouzivame na ranking modelov podla validation RMSE cez vsetky seedy.
    summary_df = (
        raw_df.groupby("model_name", as_index=False)["val_rmse"]
        .agg(mean="mean", std="std", median="median", min="min", max="max")
        .sort_values(["mean", "std", "median"], ascending=[True, True, True])
    )

    print("\n# MODEL GRID SUMMARY (val_rmse) #")
    print(summary_df.to_string(index=False))

    top2_names = summary_df.head(2)["model_name"].tolist()

    print("\n# TOP 2 MODELS #")
    for i, name in enumerate(top2_names, start=1):
        print(f"{i}. {name}")

    print("\n# TOP 2 PARAMS #")
    for cfg in model_configs:
        if cfg["name"] in top2_names:
            print(f"{cfg['name']}: {cfg['mlp_params']}")


if __name__ == "__main__":
    log_dir = Path("docs_majk_djuk")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"model_grid_{timestamp}.txt"

    # Vystup ide naraz do terminalu aj do log suboru.
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            main()
