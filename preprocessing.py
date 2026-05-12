import pandas as pd
import numpy as np


def add_time_cyclis_features(df):
    """Prida cyklicke casove features z DateTime a vrati upravenu kopiu."""
    df = df.copy()

    dt_str = df["DateTime"].astype(str).str.split("#").str[0]  # ponecha iba realnu casovu cast
    dt = pd.to_datetime(dt_str, dayfirst=True, errors="coerce")  # den je pred mesiacom, neplatne datumy zmenime na NaT

    hour = dt.dt.hour
    day_of_year = dt.dt.dayofyear

    # Cas kodujeme ako cyklus, aby hodnoty ako 23:00 a 00:00 ostali blizko seba.
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Rovnakym sposobom kodujeme rocnu sezonnost.
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df
