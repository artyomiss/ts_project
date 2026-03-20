import pandas as pd


def make_lag_features(df, lags):
    df = df.copy()

    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df.groupby("unique_id")["y"].shift(lag)

    return df.dropna()