from typing import Callable

import numpy as np
import pandas as pd

from utils.distance_calculator import DistanceCalculator


class KNNClassifier:
    @staticmethod
    def predict(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        k: int,
        distance_calculator_function: Callable[[np.ndarray, np.ndarray], float],
    ) -> np.ndarray:
        y_pred = []

        for index, test_sample in X_test.iterrows():
            distances = DistanceCalculator.calculate_distances(
                X_train,
                test_sample,
                distance_calculator_function,
            )

            distances.sort(key=lambda x: x[1])
            k_nearest_indexes = [index for index, _ in distances[:k]]
            k_nearest_classes = y_train.loc[k_nearest_indexes]
            mode_class = k_nearest_classes.mode()[0]
            y_pred.append(mode_class)

        return np.array(y_pred)
