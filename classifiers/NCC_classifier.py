from __future__ import annotations

import numpy as np
import pandas as pd

from utils.distance_calculator import DistanceCalculator


class NCClassifier:
    @staticmethod
    def calculate_centroids(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        classes = y_train.unique()
        centroids = {}
        for cls in classes:
            class_samples = X_train[y_train == cls]
            centroids[cls] = class_samples.mean()
        return pd.DataFrame(centroids)

    @staticmethod
    def nearest_centroid_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
        centroids = NCClassifier.calculate_centroids(X_train, y_train)
        y_pred = []
        for _, test_sample in X_test.iterrows():
            distances = [
                (
                    class_centroid,
                    DistanceCalculator.calculate_euclidean_distance(
                        centroid.values,
                        test_sample.values,
                    ),
                )
                for class_centroid, centroid in centroids.iterrows()
            ]
            distances.sort(key=lambda x: x[1])
            y_pred.append(distances[0][0])
        return np.array(y_pred)
