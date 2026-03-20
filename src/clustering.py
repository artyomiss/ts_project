import numpy as np
from tslearn.clustering import TimeSeriesKMeans

from config import N_SERIES, LENGTH, HORIZON


def cluster_series(df, n_clusters):
    groups = df.groupby('unique_id')
    X = np.array(groups['y'].apply(np.array).tolist()).reshape(N_SERIES, LENGTH - HORIZON, 1)

    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        random_state=42
    )

    labels = model.fit_predict(X)

    return dict(zip(groups.indices.keys(), labels))