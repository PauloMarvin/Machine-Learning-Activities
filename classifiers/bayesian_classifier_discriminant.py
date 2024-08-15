import numpy as np
import pandas as pd


class BayesianGaussianDiscriminant:
    def __init__(self, discriminant_type: str = "linear", normalization: bool = False) -> None:
        self.class_names = None
        self.class_means = None
        self.class_covs = None
        self.class_priors = None
        self.shared_cov = None
        self.discriminant_type = discriminant_type
        self.normalization = normalization

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray | pd.Series) -> None:
        self.class_names = np.unique(y_train)
        self.class_means = {}
        self.class_covs = {}
        self.class_priors = {}

        n_features = X_train.shape[1]
        self.shared_cov = np.zeros((n_features, n_features))

        for class_name in self.class_names:
            X_train_class = X_train[y_train == class_name].to_numpy()

            mean = np.mean(X_train_class, axis=0)
            cov = np.cov(X_train_class, rowvar=False)

            if self.normalization:
                cov = self._regularize_covariance_matrix(cov)

            self.class_means[class_name] = mean
            self.class_covs[class_name] = cov
            self.class_priors[class_name] = X_train_class.shape[0] / X_train.shape[0]

            if self.discriminant_type == "linear":
                self.shared_cov += (X_train_class.shape[0] - 1) * cov

        if self.discriminant_type == "linear":
            if self.normalization:
                self.shared_cov = self._regularize_covariance_matrix(self.shared_cov)
            self.shared_cov /= (X_train.shape[0] - len(self.class_names))

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

    def predict(self, X_test: pd.DataFrame | np.ndarray) -> np.ndarray:
        probabilities = self._predict_proba(X_test)
        return np.argmax(probabilities, axis=1)

    def _predict_proba(self, X_test: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()

        probabilities = []
        for class_name in self.class_names:
            mean = self.class_means[class_name]

            if self.discriminant_type == "linear":
                cov = self.shared_cov
            elif self.discriminant_type == "quadratic":
                cov = self.class_covs[class_name]
            else:
                raise ValueError("Invalid discriminant type. Use 'linear' or 'quadratic'.")

            prior = self.class_priors[class_name]
            prob = self.multivariate_normal_pdf(X_test, mean, cov) * prior
            probabilities.append(prob)

        return np.array(probabilities).T

    def _regularize_covariance_matrix(
            self, covariance_matrix: np.ndarray, lambda_: float = 1e-3
    ) -> np.ndarray:
        regularized_covariance_matrix = covariance_matrix + lambda_ * np.identity(
            covariance_matrix.shape[0]
        )
        return regularized_covariance_matrix
