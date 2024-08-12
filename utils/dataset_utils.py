import numpy as np
import pandas as pd
from pandas import DataFrame


class DatasetUtils:
    @staticmethod
    def calculate_feature_bounds(
        dataset: pd.DataFrame,
        feature_x: str,
        feature_y: str,
        margin_percentage: float = 0.2,
    ) -> tuple[float, float, float, float]:
        feature_x_max = dataset[feature_x].max() * (1 + margin_percentage)
        feature_x_min = dataset[feature_x].min() * (1 - margin_percentage)

        feature_y_max = dataset[feature_y].max() * (1 + margin_percentage)
        feature_y_min = dataset[feature_y].min() * (1 - margin_percentage)

        return feature_x_max, feature_x_min, feature_y_max, feature_y_min

    @staticmethod
    def create_meshgrid(
        X_train: pd.DataFrame,
        feature_x: str,
        feature_y: str,
        resolution_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        feature_x_max, feature_x_min, feature_y_max, feature_y_min = (
            DatasetUtils.calculate_feature_bounds(
                X_train,
                feature_x,
                feature_y,
            )
        )

        xx, yy = np.meshgrid(
            np.linspace(feature_x_min, feature_x_max, resolution_points),
            np.linspace(feature_y_min, feature_y_max, resolution_points),
        )

        return xx, yy

    @staticmethod
    def create_test_grid(
        X_train: pd.DataFrame,
        feature_x: str,
        feature_y: str,
        resolution_points: int,
        return_shapes: bool,
    ) -> tuple[DataFrame, tuple[int, ...], tuple[int, ...]] | DataFrame:
        xx, yy = DatasetUtils.create_meshgrid(
            X_train,
            feature_x,
            feature_y,
            resolution_points,
        )

        test_grid = pd.DataFrame(
            np.c_[xx.ravel(), yy.ravel()],
            columns=[
                feature_x,
                feature_y,
            ],
        )

        if return_shapes:
            return test_grid, xx.shape, yy.shape

        return test_grid
