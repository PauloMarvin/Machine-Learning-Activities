import numpy as np
import pandas as pd


class ClassifierUtils:
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:

        return np.sum(y_true == y_pred) / len(y_true)

    @staticmethod
    def calculate_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:

        return 1 - ClassifierUtils.calculate_accuracy(y_true, y_pred)

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

        unique_classes = np.unique(np.concatenate((y_true, y_pred)))
        num_classes = len(unique_classes)
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for true_label, pred_label in zip(y_true, y_pred):
            conf_matrix[true_label, pred_label] += 1

        return conf_matrix

    @staticmethod
    def create_holdout_indexes(
        df: pd.DataFrame,
        test_size: float,
    ) -> tuple[list[int], list[int]]:

        df_len = len(df)
        test_size = int(df_len * test_size)
        train_size = df_len - test_size

        indexes = list(df.index)
        train_indexes = np.random.choice(indexes, train_size, replace=False)
        test_indexes = list(set(indexes) - set(train_indexes))

        return list(train_indexes), test_indexes

    @staticmethod
    def create_multiple_holdout_indexes(
        df: pd.DataFrame,
        n_indexes: int,
        test_size: float,
    ) -> list[tuple[list[int], list[int]]]:

        indexes_list = []
        for _ in range(n_indexes):
            indexes_list.append(
                ClassifierUtils.create_holdout_indexes(df, test_size),
            )

        return indexes_list

    @staticmethod
    def separate_train_test(
        df: pd.DataFrame,
        train_indexes: list[int],
        test_indexes: list[int],
        target_column_name: str,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

        train_df = df.loc[train_indexes]
        test_df = df.loc[test_indexes]

        X_train = train_df.drop(columns=[target_column_name])
        y_train = train_df[target_column_name]

        X_test = test_df.drop(columns=[target_column_name])
        y_test = test_df[target_column_name]

        return X_train, y_train, X_test, y_test
