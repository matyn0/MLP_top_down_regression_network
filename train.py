import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from evaluate import evaluate_metrics
from utils import count_trainable_params


def run_experiment(train_df, val_df, test_df, features_to_remove=None, experiment_name="baseline", random_state=42, evaluate_test=False):

    if features_to_remove is None:
        features_to_remove = []

    # Target and input features (single final setup)
    target_col = "Irradiance"
    non_feature_cols = ["PictureName", "DateTime", "IrradianceNotCompensated", target_col]

    all_features = [c for c in train_df.columns if c not in non_feature_cols]

    missing_features = [f for f in features_to_remove if f not in all_features]
    if missing_features:
        raise ValueError(f"Neznama feature-a v features_to_remove: {missing_features}")

    feature_cols = [f for f in all_features if f not in features_to_remove]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    if evaluate_test:                    #testing data leakage
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

    # Handle missing values using train-set medians only
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)
    if evaluate_test:
        X_test = X_test.fillna(train_medians)

    print("\n # MAJKL DJUK #")
    print("Same feature columns train/val:", list(X_train.columns) == list(X_val.columns))
    if evaluate_test:
        print("Same feature columns train/test:", list(X_train.columns) == list(X_test.columns))
    print("Number of features:", X_train.shape[1])

    print("NaNs in train:", int(X_train.isna().sum().sum()))
    print("NaNs in val:", int(X_val.isna().sum().sum()))
    if evaluate_test:
        print("NaNs in test:", int(X_test.isna().sum().sum()))

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    if evaluate_test:
        X_test_scaled = scaler.transform(X_test)

    # One selected model only
    mlp = MLPRegressor(
        hidden_layer_sizes=(256,128,64,32),
        activation="relu",
        solver="adam",
        max_iter=1500,
        random_state=random_state,
    )
    mlp.fit(X_train_scaled, y_train)
    epochs_used = mlp.n_iter_
    n_params = count_trainable_params(mlp)

    # Validation
    val_pred = mlp.predict(X_val_scaled)
    val_mae, val_rmse, val_r2 = evaluate_metrics(y_val, val_pred)
    print("\nValidation results:")
    print(f"MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}")
    print(f"Epochs used (n_iter_): {epochs_used}")
    print(f"Model complexity (trainable params): {n_params}")

    # Test
    test_mae, test_rmse, test_r2 = None, None, None

    if evaluate_test:
        test_pred = mlp.predict(X_test_scaled)
        test_mae, test_rmse, test_r2 = evaluate_metrics(y_test, test_pred)
        print("Test results:")
        print(f"MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

    return {
        "best_model": "Model (256,128,64,32)",
        "experiment": experiment_name,
        "removed_features": ",".join(features_to_remove) if features_to_remove else "none",
        "feature_list": feature_cols,
        "n_features": X_train.shape[1],
        "n_iter_": epochs_used,
        "n_params": n_params,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "random_state": random_state

    }


def run_leave_one_feature_out_ablation(
    train_df,
    val_df,
    baseline_features,
    baseline_rmse,
    baseline_removed_features=None,
    random_state=42,
):
    if baseline_removed_features is None:
        baseline_removed_features = []


    ablation_rows = []
    for feature_to_remove in baseline_features:
        feature_cols = [f for f in baseline_features if f != feature_to_remove]

        X_train = train_df[feature_cols]
        y_train = train_df["Irradiance"]
        X_val = val_df[feature_cols]
        y_val = val_df["Irradiance"]

        train_medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(train_medians)
        X_val = X_val.fillna(train_medians)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        mlp = MLPRegressor(
            hidden_layer_sizes=(256,128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=1500,
            random_state=random_state,
        )
        mlp.fit(X_train_scaled, y_train)
        val_pred = mlp.predict(X_val_scaled)
        _, val_rmse_without, _ = evaluate_metrics(y_val, val_pred)
        delta_rmse = val_rmse_without - baseline_rmse

        full_removed_features = baseline_removed_features + [feature_to_remove]

        ablation_rows.append(
            {
                "feature_removed": feature_to_remove,
                "val_rmse_without": val_rmse_without,
                "delta_rmse": delta_rmse,
                "n_iter_": mlp.n_iter_,
                "n_params": count_trainable_params(mlp),
                "removed_features_full": ",".join(full_removed_features),
                "n_features_after_drop": len(feature_cols),
            }
        )

    return pd.DataFrame(ablation_rows).sort_values("delta_rmse", ascending=False)


def run_seed_stability(
    train_df,
    val_df,
    test_df,
    seed_list,
    stability_configs,
    evaluate_test=False,
):
    stability_results = []

    for seed in seed_list:
        print(f"\n # SEED = {seed} #")
        for cfg in stability_configs:
            print(f"\n# {cfg['name']} | remove={cfg['remove']} | seed={seed} #")
            res = run_experiment(
                train_df,
                val_df,
                test_df,
                features_to_remove=cfg["remove"],
                experiment_name=cfg["name"],
                random_state=seed,
                evaluate_test=evaluate_test,
            )
            stability_results.append(res)

    stability_df = pd.DataFrame(stability_results)[
        ["random_state", "experiment", "removed_features", "n_features", "val_rmse"]
    ].sort_values(["random_state", "experiment"])

    return stability_df
