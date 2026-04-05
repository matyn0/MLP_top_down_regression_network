import pandas as pd
import numpy as np


def add_time_cyclis_features(df):
    df = df.copy()

    dt_str = df["DateTime"].astype(str).str.split("#").str[0]
    dt = pd.to_datetime(dt_str, dayfirst=True, errors="coerce")

    hour = dt.dt.hour
    day_of_year = dt.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df
