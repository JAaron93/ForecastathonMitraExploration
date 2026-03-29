"""
Test script to verify that calculate_rolling_stats properly uses config-driven defaults for the stats parameter.
"""

import pandas as pd
import numpy as np
import pytest
from src.features.engineering import FeatureEngineer


def test_calculate_rolling_stats_config_defaults():
    """Test that calculate_rolling_stats uses config-driven defaults for stats parameter."""

    # Create test data
    rng = np.random.default_rng(42)  # For reproducibility
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "close": rng.standard_normal(100).cumsum() + 100,
            "volume": rng.integers(1000, 10000, size=100),
        },
        index=dates,
    )

    # Test 1: No config provided - should use default stats ["mean", "std", "min", "max"]
    result_no_config = FeatureEngineer.calculate_rolling_stats(
        df, columns=["close"], windows=[10]
    )

    # Check that default stats were used
    expected_features_no_config = [
        "close_rolling_mean_10",
        "close_rolling_std_10",
        "close_rolling_min_10",
        "close_rolling_max_10",
    ]

    for feature in expected_features_no_config:
        assert feature in result_no_config.columns, (
            f"Missing expected feature: {feature}"
        )

    # Value-based assertion for rolling mean to catch calculation regressions
    # Use min_periods=1 to match FeatureEngineer implementation
    expected_mean = df["close"].rolling(window=10, min_periods=1).mean()
    pd.testing.assert_series_equal(
        result_no_config["close_rolling_mean_10"],
        expected_mean,
        check_names=False,
        atol=1e-6,
    )

    # Test 2: Config with rolling_stats provided - should use those stats
    config_with_stats = {
        "rolling_stats": ["mean", "median"]  # Only mean and median
    }

    result_with_config = FeatureEngineer.calculate_rolling_stats(
        df, columns=["close"], windows=[10], config=config_with_stats
    )

    # Check that config stats were used
    expected_features_with_config = ["close_rolling_mean_10", "close_rolling_median_10"]

    for feature in expected_features_with_config:
        assert feature in result_with_config.columns, (
            f"Missing expected feature: {feature}"
        )

    # Full-series value assertion for the median column (consistent with Test 1)
    expected_median = df["close"].rolling(window=10, min_periods=1).median()
    pd.testing.assert_series_equal(
        result_with_config["close_rolling_median_10"],
        expected_median,
        check_names=False,
        atol=1e-6,
    )

    # Should NOT have std, min, max since they weren't in config
    unexpected_features = [
        "close_rolling_std_10",
        "close_rolling_min_10",
        "close_rolling_max_10",
    ]
    for feature in unexpected_features:
        assert feature not in result_with_config.columns, (
            f"Unexpected feature found: {feature}"
        )

    # Test 3: Config with empty rolling_stats - should fall back to default
    config_empty_stats = {
        "rolling_stats": []  # Empty list
    }

    result_empty_config = FeatureEngineer.calculate_rolling_stats(
        df, columns=["close"], windows=[10], config=config_empty_stats
    )

    # Should fall back to default stats
    expected_features_empty_config = [
        "close_rolling_mean_10",
        "close_rolling_std_10",
        "close_rolling_min_10",
        "close_rolling_max_10",
    ]

    for feature in expected_features_empty_config:
        assert feature in result_empty_config.columns, (
            f"Missing expected feature: {feature}"
        )

    # Test 4: Config without rolling_stats key - should fall back to default
    config_no_stats_key = {
        "other_setting": "value"
        # No rolling_stats key
    }

    result_no_stats_key = FeatureEngineer.calculate_rolling_stats(
        df, columns=["close"], windows=[10], config=config_no_stats_key
    )

    # Should fall back to default stats
    expected_features_no_stats_key = [
        "close_rolling_mean_10",
        "close_rolling_std_10",
        "close_rolling_min_10",
        "close_rolling_max_10",
    ]

    for feature in expected_features_no_stats_key:
        assert feature in result_no_stats_key.columns, (
            f"Missing expected feature: {feature}"
        )


if __name__ == "__main__":
    test_calculate_rolling_stats_config_defaults()
