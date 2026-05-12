import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from evaluate import evaluate_metrics
from utils import count_trainable_params


def run_experiment(
    train_df,
    val_df,
    test_df,
    features_to_remove=None,
    experiment_name="baseline",
    random_state=42,
    evaluate_test=False,
    mlp_params=None,
    model_name=None,
):
    """Spusti jeden komplet experiment: vyberie features, pripravi data, natrenuje MLP a vrati metriky."""

    if features_to_remove is None:
        features_to_remove = []

    if mlp_params is None:
        mlp_params = {}


    # Vyberieme vstupne features pre tento experiment.
    target_col = "Irradiance"
    non_feature_cols = ["PictureName", "DateTime", "IrradianceNotCompensated", target_col]
    all_features = [c for c in train_df.columns if c not in non_feature_cols]


    missing_features = [f for f in features_to_remove if f not in all_features]
    if missing_features:
        raise ValueError(f"Neznama feature-a v features_to_remove: {missing_features}")

    feature_cols = [f for f in all_features if f not in features_to_remove]  # finalny feature subset


    # Rozdelime data na vstupy X a cielovu hodnotu y.
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    if evaluate_test:
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

    # Chybajuce hodnoty doplname train medianmi, aby validation/test neovplyvnili preprocessing.
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

    # Scaling sa uci iba na train dat, validation/test iba transformujeme rovnakym scalerom.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    if evaluate_test:
        X_test_scaled = scaler.transform(X_test)


    default_mlp_params = {
        "hidden_layer_sizes": (256, 128, 64, 32),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 1500,
        "random_state": random_state,
    }
    final_mlp_params = {**default_mlp_params, **mlp_params}  # experiment moze prepisat defaultne MLP nastavenia
    final_mlp_params["random_state"] = random_state  # seed drzi porovnania reprodukovatelne

    mlp = MLPRegressor(**final_mlp_params) 
    mlp.fit(X_train_scaled, y_train)
    epochs_used = mlp.n_iter_
    n_params = count_trainable_params(mlp)

    # Validation metriky pouzivame na porovnanie modelov a feature subsetov.
    val_pred = mlp.predict(X_val_scaled)
    val_mae, val_rmse, val_r2 = evaluate_metrics(y_val, val_pred)
    print("\nValidation results:")
    print(f"MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}")
    print(f"Epochs used (n_iter_): {epochs_used}")
    print(f"Model complexity (trainable params): {n_params}")

    test_mae, test_rmse, test_r2 = None, None, None

    if evaluate_test:
        test_pred = mlp.predict(X_test_scaled)
        test_mae, test_rmse, test_r2 = evaluate_metrics(y_test, test_pred)
        print("Test results:")
        print(f"MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

    selected_model_name = model_name or f"MLP_{final_mlp_params['hidden_layer_sizes']}"

    return {
        "best_model": selected_model_name,
        "experiment": experiment_name,
        "removed_features": ",".join(features_to_remove) if features_to_remove else "none",
        "feature_list": feature_cols,
        "n_features": X_train.shape[1],
        "n_iter_": epochs_used,
        "n_params": n_params,
        "model_name": selected_model_name,
        "mlp_params": final_mlp_params,
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
    """Otestuje, co sa stane, ked z baseline subsetu odstranime vzdy jednu feature."""
    if baseline_removed_features is None:
        baseline_removed_features = []


    ablation_rows = []
    for feature_to_remove in baseline_features:
        feature_cols = [f for f in baseline_features if f != feature_to_remove]  # subset bez jednej testovanej feature

        # Kazdy ablation beh pouziva rovnaky train/validation princip ako hlavny experiment.
        X_train = train_df[feature_cols]
        y_train = train_df["Irradiance"]
        X_val = val_df[feature_cols]
        y_val = val_df["Irradiance"]

        # Missing values riesime iba cez train mediany, aby validation neovplyvnil preprocessing.
        train_medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(train_medians)
        X_val = X_val.fillna(train_medians)

        # Scaler sa fituje iba na train dat a potom sa rovnako aplikuje na validation.
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
        delta_rmse = val_rmse_without - baseline_rmse  # kladne = odstranenie zhorsilo model, zaporne = pomohlo

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
    """Spusti viac experimentov cez viac seedov a vrati tabulku stability."""
    stability_results = []

    for seed in seed_list:
        print(f"\n # SEED = {seed} #")
        for cfg in stability_configs:
            print(f"\n# {cfg['name']} | remove={cfg['remove']} | seed={seed} #")

            # Rovnake konfiguracie pustame cez viac seedov, aby sme videli stabilitu vysledkov.
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

    # Z celeho vysledku nechavame iba stlpce potrebne na porovnanie experimentov.
    stability_df = pd.DataFrame(stability_results)[
        ["random_state", "experiment", "removed_features", "n_features", "val_rmse"]
    ].sort_values(["random_state", "experiment"])

    return stability_df
