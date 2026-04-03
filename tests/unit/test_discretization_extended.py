"""Extended tests for discretization to increase coverage."""

import numpy as np
import pandas as pd
import pytest

from src.features.discretization import LabelDiscretizer


class TestLabelDiscretizerStrategies:
    def test_uniform_strategy(self):
        y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        discretizer = LabelDiscretizer(strategy="uniform", n_bins=3)
        labels = discretizer.fit_transform(y)

        assert discretizer.bins is not None
        assert len(discretizer.bins) == 4  # 3 bins means 4 edges
        assert labels.nunique() == 3
        # Lowest should be 0, highest 2
        assert labels.min() == 0
        assert labels.max() == 2

    def test_fixed_strategy(self):
        y = pd.Series([0, 5, 10, 15, 20])
        discretizer = LabelDiscretizer(strategy="fixed", bins=[0, 10, 20])
        labels = discretizer.fit_transform(y)

        assert discretizer.bins is not None
        assert list(discretizer.bins) == [0, 10, 20]
        assert list(labels) == [0, 0, 0, 1, 1]
        assert len(labels) == 5

    def test_fixed_strategy_missing_bins(self):
        y = pd.Series([1, 2, 3])
        discretizer = LabelDiscretizer(strategy="fixed")  # No bins provided
        with pytest.raises(ValueError, match="Must provide 'bins' argument"):
            discretizer.fit(y)

    def test_unknown_strategy(self):
        y = pd.Series([1, 2, 3])
        discretizer = LabelDiscretizer(strategy="unknown_strategy")
        with pytest.raises(ValueError, match="Unknown strategy: unknown_strategy"):
            discretizer.fit(y)


class TestLabelDiscretizerTransform:
    def test_transform_not_fitted(self):
        y = pd.Series([1, 2, 3])
        discretizer = LabelDiscretizer(strategy="uniform")
        with pytest.raises(ValueError, match="Discretizer not fitted"):
            discretizer.transform(y)

    def test_transform_out_of_bounds_fillna(self):
        y_train = pd.Series([1, 2, 3, 4, 5])
        discretizer = LabelDiscretizer(strategy="fixed", bins=[0, 6])
        discretizer.fit(y_train)

        y_test = pd.Series([-10, 10])  # Out of [0, 6] bounds
        labels = discretizer.transform(y_test)
        # Should be filled with -1
        assert list(labels) == [-1, -1]


class TestLabelDiscretizerInverseTransform:
    def test_inverse_transform_direction(self):
        y = np.array([-5, 0, 5])
        discretizer = LabelDiscretizer(strategy="direction", threshold=1)
        labels = discretizer.fit_transform(y)
        # Direction strategy inverse just returns labels (-1, 0, 1)
        inv = discretizer.inverse_transform(labels)
        np.testing.assert_array_equal(inv, labels)

    def test_inverse_transform_bins_series(self):
        y = pd.Series([1, 2, 3, 4, 5, 6])
        discretizer = LabelDiscretizer(strategy="uniform", n_bins=2)
        labels = discretizer.fit_transform(y)

        inv = discretizer.inverse_transform(labels)
        assert isinstance(inv, pd.Series)
        # It should return bin centers
        assert not inv.isna().all()

    def test_inverse_transform_bins_array(self):
        y = np.array([1, 2, 3, 4, 5, 6])
        discretizer = LabelDiscretizer(strategy="uniform", n_bins=2)
        labels = discretizer.fit_transform(y)

        inv = discretizer.inverse_transform(labels)
        assert isinstance(inv, pd.Series)
        assert not inv.isna().all()

    def test_inverse_transform_invalid_labels(self):
        y = pd.Series([1, 2, 3])
        discretizer = LabelDiscretizer(strategy="uniform", n_bins=2)
        discretizer.fit(y)

        # -1 is invalid/unknown label
        bad_labels = np.array([-1, 100])
        inv = discretizer.inverse_transform(bad_labels)
        # Should be NaN for invalid labels
        assert np.isnan(inv).all()

    def test_inverse_transform_not_fitted_returns_labels(self):
        # A weird edge case where bins is None and strategy is not direction
        y = np.array([0, 1])
        discretizer = LabelDiscretizer(strategy="uniform")
        # Don't fit
        inv = discretizer.inverse_transform(y)
        np.testing.assert_array_equal(inv, y)

