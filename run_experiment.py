import pandas as pd
from tqdm import tqdm

from config import *
from src.data import load_data, sample_ts, to_long, train_test_split
from src.transforms import Transformer
from src.features import make_lag_features
from src.clustering import cluster_series
from src.baselines import *
from src.model import GlobalCatBoost
from src.evaluation import evaluate_series

print('=== Data Preprocessing ===')

df = load_data(DATA_PATH, LENGTH)
df_sample = sample_ts(df, N_SERIES)
df = to_long(df_sample, LENGTH)

print(f'\n=== Train-test split: horizon = {HORIZON} ===')

train_df, test_df = train_test_split(df, HORIZON)

train_df.head()

print(f'\n=== Clustering into {N_CLUSTERS} clusters ===')

clusters = cluster_series(train_df, N_CLUSTERS)
train_df["cluster"] = train_df["unique_id"].map(clusters)
test_df["cluster"] = test_df["unique_id"].map(clusters)

results = []

for c in sorted(train_df.cluster.unique()):
    print(f'\n=== Analyzing cluster {c} ===')

    train_df1 = train_df[train_df.cluster.eq(c)]
    test_df1 = test_df[test_df.cluster.eq(c)]

    for transform_name in TRANSFORMS:
        print(f"\n=== Transform: {transform_name} ===")

        transformer_dict = {}
        train_transformed = []

        print('\n=== Transforming train data ===')

        for uid, group in train_df1.groupby("unique_id"):
            transformer = Transformer(transform_name)
            y_t = transformer.fit_transform(group["y"].values)

            transformer_dict[uid] = transformer

            tmp = group.copy()
            tmp["y"] = y_t
            train_transformed.append(tmp)

        train_t_df = pd.concat(train_transformed)
        df_feat = make_lag_features(train_t_df, LAGS)

        print(f'\n=== Fitting CatBoost with params: {CATBOOST_PARAMS} ===')

        model = GlobalCatBoost(CATBOOST_PARAMS, LAGS)
        model.fit(df_feat)

        print('\n=== Predicting ===\n')
        # Построение бейзлайнов и предсказаний CatBoost на каждом ряду
        for uid in tqdm(train_df1["unique_id"].unique()):
            train = train_df1[train_df1.unique_id == uid]
            test = test_df1[test_df1.unique_id == uid]

            # Отделяем сам ряд
            train_series = train["y"].values
            test_series = test["y"].values

            # Трансформируем обучающую выборку
            transformer = transformer_dict[uid]
            history = transformer.fit_transform(train_series)

            # Рекурсивно предсказываем HORIZON точек
            cluster = train.cluster.values[0]
            pred_transformed = model.recursive_forecast(cluster, history, HORIZON)

            # Инвертируем преобразования
            if transform_name == "diff":
                pred = transformer.inverse(pred_transformed, original_first=train_series[-1])
            else:
                pred = transformer.inverse(pred_transformed)

            # Считаем метрики
            metrics = evaluate_series(test_series, pred, train_series)
            results.append([uid, cluster, transform_name, 'catboost', metrics["rmse"], metrics["mae"], metrics["mape"], metrics["smape"], metrics["mase"]])

            # Строим бейзлайны
            preds = {
                "naive": naive_forecast(pd.Series(train_series), HORIZON),
                "seasonal": seasonal_naive(pd.Series(train_series), HORIZON),
                "theta": theta_forecast(train.drop('cluster', axis=1), HORIZON).iloc[:, 2].values,
                "ets": ets_forecast(train.drop('cluster', axis=1), HORIZON).iloc[:, 2].values
            }

            for name, pred in preds.items():
                metrics = evaluate_series(test_series, pred, train_series)
                results.append([uid, cluster, 'none', name, metrics["rmse"], metrics["mae"], metrics["mape"], metrics["smape"], metrics["mase"]])

# Результаты
results_df = (
    pd.DataFrame(
        results,
        columns=["unique_id", "cluster", "transform", "model", "rmse", "mae", "mape", "smape", "mase"]
    )
    .drop_duplicates()
    .sort_values(['cluster', 'unique_id', 'transform', 'model'])
    .reset_index(drop=True)
)

# Смотрим для каждой метрики в каждом кластере как часто та или иная метрика лучше
winners = []

for metric in ['mape', 'smape', 'mase']:
    tmp = results_df.loc[results_df.groupby("unique_id")[metric].idxmin()].copy()
    tmp["metric"] = metric
    winners.append(tmp)

winners_df = pd.concat(winners)

stats = (
    winners_df
    .groupby(["cluster", "metric", "transform"])
    .size()
    .reset_index(name="count")
)