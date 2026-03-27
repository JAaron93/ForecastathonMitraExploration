"""
Test script to verify that calculate_rolling_stats properly uses config-driven defaults for the stats parameter.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer


def test_calculate_rolling_stats_config_defaults():
    """Test that calculate_rolling_stats uses config-driven defaults for stats parameter."""

    # Create test data
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    print("Testing calculate_rolling_stats config-driven defaults...")

    # Test 1: No config provided - should use default stats ["mean", "std", "min", "max"]
    print("\nTest 1: No config provided")
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

    print(
        f"✓ No config test passed. Created features: {[f for f in result_no_config.columns if 'rolling' in f]}"
    )

    # Test 2: Config with rolling_stats provided - should use those stats
    print("\nTest 2: Config with rolling_stats provided")
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

    print(
        f"✓ Config with stats test passed. Created features: {[f for f in result_with_config.columns if 'rolling' in f]}"
    )

    # Test 3: Config with empty rolling_stats - should fall back to default
    print("\nTest 3: Config with empty rolling_stats (should fall back to default)")
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

    print(
        f"✓ Empty config stats test passed. Created features: {[f for f in result_empty_config.columns if 'rolling' in f]}"
    )

    # Test 4: Config without rolling_stats key - should fall back to default
    print("\nTest 4: Config without rolling_stats key (should fall back to default)")
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

    print(
        f"✓ No stats key test passed. Created features: {[f for f in result_no_stats_key.columns if 'rolling' in f]}"
    )

    print(
        "\n✅ All tests passed! calculate_rolling_stats properly uses config-driven defaults for stats parameter."
    )


if __name__ == "__main__":
    test_calculate_rolling_stats_config_defaults()
