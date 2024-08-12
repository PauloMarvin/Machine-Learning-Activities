import numpy as np
import pandas as pd


class BayesianGaussianClassifier:
    def __init__(self, normalization: bool = False) -> None:
        self.class_names = None
        self.class_means = None
        self.class_covs = None
        self.normalization = normalization

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray | pd.Series) -> None:
        self.class_names = np.unique(y_train)
        self.class_means = {}
        self.class_covs = {}

        for class_name in self.class_names:
            X_train_class = X_train[y_train == class_name]

            mean = np.mean(X_train_class, axis=0)
            if self.normalization:
                cov = self._regularize_covariance_matrix(
                    np.cov(X_train_class, rowvar=False)
                )
            else:
                cov = np.cov(X_train_class, rowvar=False)

            self.class_means[class_name] = mean
            self.class_covs[class_name] = cov

    def multivariate_normal_pdf(
        self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        k = len(mean)
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)

        X_mean = X - mean.reshape(1, -1)
        exponent = -0.5 * np.einsum("ij,ji->i", np.dot(X_mean, inv), X_mean.T)
        coefficient = 1 / np.sqrt((2 * np.pi) ** k * det)
        probabilities = coefficient * np.exp(exponent)

        return probabilities

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        probabilities = self._predict_proba(X_test)
        return np.argmax(probabilities, axis=1)

    def _regularize_covariance_matrix(
        self, covariance_matrix: np.ndarray, lambda_: float = 1e-3
    ) -> np.ndarray:
        regularized_covariance_matrix = covariance_matrix + lambda_ * np.identity(
            covariance_matrix.shape[0]
        )
        return regularized_covariance_matrix

    def _predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        probabilities = []
        for class_name in self.class_names:
            mean = self.class_means[class_name].values
            cov = self.class_covs[class_name]
            prob = self.multivariate_normal_pdf(X_test, mean, cov)
            probabilities.append(prob)
        return np.array(probabilities).T
