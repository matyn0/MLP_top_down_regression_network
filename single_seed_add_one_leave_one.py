import pandas as pd
from data_loader import load_datasets
from preprocessing import add_time_cyclis_features
from train import run_experiment


def main():
    """Rychly single-seed screening dalsich kandidatov na odstranenie feature."""

    removed_base = ["TiltAngle"]  # aktualny baseline subset uz nepouziva TiltAngle
    random_state = 42  # jeden fixny seed pre rychly orientacny test

    train_df, val_df, test_df = load_datasets()
    train_df = add_time_cyclis_features(train_df)
    val_df = add_time_cyclis_features(val_df)
    test_df = add_time_cyclis_features(test_df)

    non_feature_cols = ["PictureName", "DateTime", "IrradianceNotCompensated", "Irradiance"]
    all_features = [c for c in train_df.columns if c not in non_feature_cols]

    candidates = [f for f in all_features if f not in removed_base]  # features, ktore este mozeme skusit odstranit
    if not candidates:
        print("No candidates left.")
        return
    

    print("\n ^^ BASELINE CURRENT SUBSET ^^")
    print(f"removed_base={removed_base}")
    baseline = run_experiment(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        features_to_remove=removed_base,
        experiment_name="baseline_current_subset",
        random_state=random_state,
        evaluate_test=False,
    )
    baseline_rmse = baseline["val_rmse"]
    print(f"\n baseline_val_rmse={baseline_rmse:.4f}")

    rows = []

    # Kazdeho kandidata pridame k removed_base a porovname ho s baseline na rovnakom seede.
    for feat in candidates:
        removed_list = removed_base + [feat]
        print(f"\nrunning candidate: {feat}")
        res = run_experiment(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            features_to_remove=removed_list,
            experiment_name=f"candidate_remove_{feat}",
            random_state=random_state,
            evaluate_test=False,
        )

        delta = res["val_rmse"] - baseline_rmse  # zaporne = odstranenie kandidata zlepsilo RMSE
        rows.append(
            {
                "candidate_feature": feat,
                "val_rmse_candidate": res["val_rmse"],
                "delta_candidate_minus_baseline": delta,
            }
        )

    out = pd.DataFrame(rows).sort_values("delta_candidate_minus_baseline", ascending=True)

    print("\n ^^ seed 42 - add one leave one results ^^")
    print(out.to_string(index=False))

    best = out.iloc[0]

    print("\n ^^ best candidate ^^")
    print(
        f"feature = {best['candidate_feature']}, "
        f"val_rmse={best['val_rmse_candidate']:.4f}, "
        f"delta={best['delta_candidate_minus_baseline']:.4f}"
    )

if __name__ == "__main__":
    main()
