from catboost import CatBoostRegressor
import numpy as np
import pandas as pd


class GlobalCatBoost:
    def __init__(self, params, lags):
        self.model = CatBoostRegressor(**params)
        self.lags = lags

    def fit(self, df):
        X = df.drop(columns=["y", "unique_id", "ds"])
        y = df["y"]
        self.model.fit(X, y)

    def _make_features(self, history):
        """
        history: array of last observed values (length >= lags)
        """
        feats = {}
        for i in range(1, self.lags + 1):
            feats[f"lag_{i}"] = history[-i]
        return pd.DataFrame([feats])

    def recursive_forecast(self, cluster, history, horizon):
        """
        history: 1D numpy array (already transformed!)
        """
        history = list(history.copy())
        preds = []

        for _ in range(horizon):
            X = self._make_features(history)

            X['cluster'] = cluster

            y_pred = self.model.predict(X)[0]

            preds.append(y_pred)
            history.append(y_pred)

        return np.array(preds)