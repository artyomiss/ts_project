import numpy as np
import pandas as pd

from src.data_loader import convert_tsf_to_dataframe
from config import RANDOM_STATE


def load_data(path, length):
    df, _, _, _, _ = convert_tsf_to_dataframe(path)
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"]).dt.date

    mask = df.series_value.apply(len).ge(length)
    df = df[mask].reset_index(drop=True)
    return df

def sample_ts(df, n_series):
    rng = np.random.default_rng(RANDOM_STATE)
    ids = rng.choice(range(df.shape[0]), size=n_series, replace=False)
    df = df.iloc[ids]
    return df.reset_index(drop=True)

def to_long(df, length):
    df_long = pd.DataFrame()

    for sid in range(len(df)):
        ts = df.iloc[sid, :]

        values = ts['series_value'][:240] # берём 240 месяцев
        dates =  pd.to_datetime(pd.Series([ts['start_timestamp'] + pd.DateOffset(months=k) for k in range(len(values))]))

        ts_long = pd.DataFrame({
            'unique_id': ts['series_name'],
            'ds': dates.dt.date,
            'y': values
        })

        df_long = pd.concat([df_long, ts_long])
    return df_long


def train_test_split(df, horizon):
    train = []
    test = []

    for uid, group in df.groupby("unique_id"):
        group = group.sort_values("ds")
        train.append(group.iloc[:-horizon])
        test.append(group.iloc[-horizon:])

    return pd.concat(train), pd.concat(test)