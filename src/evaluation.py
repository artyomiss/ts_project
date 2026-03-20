import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

def mase(y_true, y_pred, y_train, freq=12):

    enumerator = np.mean(np.abs(y_true - y_pred))
    denominator = np.mean(np.abs(y_train[:-freq] - y_train[freq:]))

    if denominator == 0:
        return np.inf
    return enumerator / denominator


def evaluate_series(y_true, y_pred, y_train):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train),
    }