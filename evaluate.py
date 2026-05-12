from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_metrics(y_true, y_pred):
    """Vypocita regresne metriky pre skutocne a predikovane hodnoty."""
    mae = mean_absolute_error(y_true, y_pred)  # priemerna absolutna chyba
    rmse = mean_squared_error(y_true, y_pred) ** 0.5  # odmocnina z MSE, chyba v povodnej jednotke targetu
    r2 = r2_score(y_true, y_pred)  # podiel vysvetlenej variability cielovej hodnoty

    return mae, rmse, r2
