"""Unit tests for data validation utilities."""

import pytest
import pandas as pd
import numpy as np
from src.data.validators import DataValidator


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_calculate_psi_identical_distributions(self):
        """PSI should be near zero for identical distributions."""
        validator = DataValidator()
        data = pd.Series(np.random.normal(0, 1, 1000))
        psi = validator.calculate_psi(data, data)
        assert psi < 0.1, "PSI should be near zero for identical distributions"

    def test_calculate_psi_different_distributions(self):
        """PSI should be higher for different distributions."""
        validator = DataValidator()
        expected = pd.Series(np.random.normal(0, 1, 1000))
        actual = pd.Series(np.random.normal(5, 2, 1000))  # Shifted distribution
        psi = validator.calculate_psi(expected, actual)
        assert psi > 0.1, "PSI should be higher for different distributions"

    def test_get_quality_metrics_returns_expected_keys(self, sample_timeseries_df):
        """Quality metrics should contain expected attributes."""
        validator = DataValidator()
        metrics = validator.get_quality_metrics(sample_timeseries_df)
        assert hasattr(metrics, "null_count")
        assert hasattr(metrics, "unique_count")
        assert hasattr(metrics, "row_count")
        assert hasattr(metrics, "column_count")

    def test_get_quality_metrics_correct_counts(self, sample_df_with_missing):
        """Quality metrics should correctly count nulls."""
        validator = DataValidator()
        metrics = validator.get_quality_metrics(sample_df_with_missing)
        assert metrics.null_count["value1"] == 5
        assert metrics.null_count["value2"] == 5
        assert metrics.row_count == 50
