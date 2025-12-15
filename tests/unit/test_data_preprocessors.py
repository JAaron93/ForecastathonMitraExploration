"""Unit tests for data preprocessing utilities."""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessors import Preprocessor


class TestPreprocessor:
    """Tests for Preprocessor class."""

    def test_handle_missing_values_forward_fill(self, sample_df_with_missing):
        """Forward fill should propagate last valid value."""
        preprocessor = Preprocessor()
        result = preprocessor.handle_missing_values(sample_df_with_missing, "forward_fill")
        # After forward fill, there should be no NaN except possibly at the start
        assert result["value1"].iloc[6:10].notna().all()

    def test_handle_missing_values_mean(self, sample_df_with_missing):
        """Mean fill should replace NaN with column mean."""
        preprocessor = Preprocessor()
        result = preprocessor.handle_missing_values(sample_df_with_missing, "mean")
        assert result.isna().sum().sum() == 0

    def test_handle_missing_values_invalid_strategy(self, sample_df_with_missing):
        """Invalid strategy should raise ValueError."""
        preprocessor = Preprocessor()
        with pytest.raises(ValueError, match="Unknown strategy"):
            preprocessor.handle_missing_values(sample_df_with_missing, "invalid")

    def test_detect_outliers_iqr(self, sample_df_with_outliers):
        """IQR method should detect extreme outliers."""
        preprocessor = Preprocessor()
        outliers = preprocessor.detect_outliers(
            sample_df_with_outliers, method="iqr", columns=["with_outliers"]
        )
        # Should detect at least some of the extreme values
        assert len(outliers) > 0

    def test_detect_outliers_normal_data(self, sample_df_with_outliers):
        """Normal data should have few or no outliers."""
        preprocessor = Preprocessor()
        outliers = preprocessor.detect_outliers(
            sample_df_with_outliers, method="iqr", columns=["normal"]
        )
        # Normal distribution should have very few outliers
        assert len(outliers) < 10
