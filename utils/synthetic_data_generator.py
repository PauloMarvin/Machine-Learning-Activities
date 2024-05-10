import numpy as np
import pandas as pd


class SyntheticDatasetGenerator:
    def __init__(self, min_noise=-0.3, max_noise=0.3, seed=None):
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.seed = seed
        np.random.seed(seed)

    def generate_data(self, features_patterns_dict) -> pd.DataFrame:
        synthetic_data = []
        for feature_name, feature_data in features_patterns_dict.items():
            pattern = feature_data["pattern"]
            num_samples = feature_data["num_samples"]
            y = feature_data["y"]
            for _ in range(num_samples):
                pattern_with_noise = self.add_noise(pattern)
                sample = np.concatenate((pattern_with_noise, [[y]]), axis=1)
                synthetic_data.append(sample)

        synthetic_df = pd.DataFrame(
            np.concatenate(synthetic_data),
            columns=[f"x{i}" for i in range(pattern.shape[1])] + ["y"],
        )

        synthetic_df["y"] = synthetic_df["y"].astype(int)

        return synthetic_df

    def add_noise(self, pattern: np.array) -> np.array:
        pattern_shape = pattern.shape
        noise = np.random.uniform(self.min_noise, self.max_noise, size=pattern_shape)
        return pattern + noise
