import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta


def naive_forecast(train, horizon):
    return np.repeat(train.iloc[-1], horizon)


def seasonal_naive(train, horizon, season=12):
    return np.tile(train.iloc[-season:], int(np.ceil(horizon/season)))[:horizon]


def theta_forecast(train, horizon):
    model = StatsForecast(
        models=[AutoTheta(season_length=12)],
        freq='ME'
    )
    return model.forecast(df=train, h=horizon)


def ets_forecast(train, horizon):
    model = StatsForecast(
        models=[AutoETS(season_length=12)],
        freq='ME'
    )
    return model.forecast(df=train, h=horizon)