DATA_PATH = "/content/m4_monthly_dataset.tsf"  # prepare beforehand
HORIZON = 24
LAGS = 12

N_CLUSTERS = 8
N_SERIES = 100

LENGTH = 240

RANDOM_STATE = 42

TRANSFORMS = ["none", "log", "boxcox", "diff"]

CATBOOST_PARAMS = {
    "iterations": 1000,
    "depth": 6,
    "loss_function": "MAPE",
    "verbose": False
}