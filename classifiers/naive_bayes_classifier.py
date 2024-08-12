import numpy as np
import pandas as pd


class NaiveBayesClassifier:
    def __init__(self) -> None:
        self.class_names = None
        self.class_means = None
        self.class_vars = None
        self.class_priors = None
        self.epsilon = 1e-9

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray | pd.Series) -> None:

        self.class_names = np.unique(y_train)
        self.class_means = {}
        self.class_vars = {}
        self.class_priors = {}

        for class_name in self.class_names:
            X_train_class = X_train[y_train == class_name]
            self.class_means[class_name] = np.mean(X_train_class, axis=0)
            self.class_vars[class_name] = np.var(X_train_class, axis=0)
            self.class_priors[class_name] = X_train_class.shape[0] / X_train.shape[0]

    def gaussian_pdf(
        self, x: np.ndarray, mean: np.ndarray, var: np.ndarray
    ) -> np.ndarray:
        var += self.epsilon
        coefficient = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coefficient * exponent

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X_test = X_test.values
        probabilities = self._predict_proba(X_test)
        return np.argmax(probabilities, axis=1)

    def _predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        probabilities = np.zeros((X_test.shape[0], len(self.class_names)))

        for idx, class_name in enumerate(self.class_names):
            mean = self.class_means[class_name]
            var = self.class_vars[class_name]
            prior = self.class_priors[class_name]

            class_prob = np.log(prior)
            for i in range(len(mean)):
                pdf_val = self.gaussian_pdf(X_test[:, i], mean.iloc[i], var.iloc[i])
                pdf_val = np.clip(pdf_val, self.epsilon, None)  # Clip to avoid log(0)
                class_prob += np.log(pdf_val)

            probabilities[:, idx] = class_prob

        return probabilities
