import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_and_clean, split_and_scale


class TestLoadAndClean:

    @pytest.fixture
    def df(self):
        return load_and_clean()

    def test_returns_dataframe(self, df):
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, df):
        expected = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope",
            "ca", "thal", "target",
        ]
        assert list(df.columns) == expected

    def test_no_missing_values(self, df):
        assert df.isna().sum().sum() == 0, "Dataset should have no NaN after cleaning"

    def test_target_is_binary(self, df):
        unique_targets = set(df["target"].unique())
        assert unique_targets.issubset({0, 1}), f"Target should be binary, got {unique_targets}"

    def test_reasonable_row_count(self, df):
        assert 250 < len(df) < 400, f"UCI Heart has ~303 rows, got {len(df)}"

    def test_age_in_valid_range(self, df):
        assert df["age"].min() >= 0
        assert df["age"].max() <= 120


class TestSplitAndScale:

    @pytest.fixture
    def data(self):
        df = load_and_clean()
        return split_and_scale(df)

    def test_returns_dict_with_expected_keys(self, data):
        expected_keys = {
            "X_train", "X_test", "y_train", "y_test",
            "X_train_scaled", "X_test_scaled",
            "scaler", "class_weights", "feature_names",
        }
        assert set(data.keys()) == expected_keys

    def test_scaled_data_is_numpy(self, data):
        assert isinstance(data["X_train_scaled"], np.ndarray)
        assert isinstance(data["X_test_scaled"], np.ndarray)

    def test_scaled_data_has_correct_shape(self, data):
        n_features = len(data["feature_names"])
        assert data["X_train_scaled"].shape[1] == n_features
        assert data["X_test_scaled"].shape[1] == n_features

    def test_train_test_ratio(self, data):
        total = len(data["X_train"]) + len(data["X_test"])
        test_ratio = len(data["X_test"]) / total
        assert 0.15 <= test_ratio <= 0.25, f"Test ratio should be ~0.2, got {test_ratio:.2f}"

    def test_scaled_mean_near_zero(self, data):
        col_means = np.abs(data["X_train_scaled"].mean(axis=0))
        assert col_means.max() < 0.1, f"Scaled means should be near 0, max is {col_means.max():.3f}"

    def test_class_weights_has_two_classes(self, data):
        assert len(data["class_weights"]) == 2
        assert 0 in data["class_weights"]
        assert 1 in data["class_weights"]

    def test_feature_names_exclude_target(self, data):
        assert "target" not in data["feature_names"]
