import pandas as pd
from data_loader import load_datasets
from preprocessing import add_time_cyclis_features
from train import run_experiment


def main():

    removed_base = ["TiltAngle"]
    random_state = 42

    train_df, val_df, test_df = load_datasets()
    train_df = add_time_cyclis_features(train_df)
    val_df = add_time_cyclis_features(val_df)

    non_feature_cols = ["PictureName", "DateTime", "IrradianceNotCompensated", "Irradiance"]
    all_features = [c for c in train_df.columns if c not in non_feature_cols]

    candidates = [f for f in all_features if f not in removed_base]
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

        delta = res["val_rmse"] - baseline_rmse
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
