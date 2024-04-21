from __future__ import annotations

from typing import Callable, Hashable

import numpy as np
import pandas as pd


class DistanceCalculator:
    @staticmethod
    def calculate_euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))

    @staticmethod
    def calculate_distances(
        X_train: pd.DataFrame,
        test_sample: pd.Series,
        distance_calculator_function: Callable[[np.ndarray, np.ndarray], float],
    ) -> list[tuple[Hashable, float]]:
        distances = []
        for index, train_sample in X_train.iterrows():
            distance = distance_calculator_function(
                train_sample.values,
                test_sample.values,
            )
            distances.append((index, distance))
        return distances
