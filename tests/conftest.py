"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_timeseries_df():
    """Create a sample time series DataFrame for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 200, 100),
        "high": np.random.uniform(150, 250, 100),
        "low": np.random.uniform(50, 150, 100),
        "close": np.random.uniform(100, 200, 100),
        "volume": np.random.uniform(1000, 10000, 100),
    }).set_index("timestamp")


@pytest.fixture
def sample_df_with_missing():
    """Create a DataFrame with missing values for testing."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    np.random.seed(42)
    data = pd.DataFrame({
        "timestamp": dates,
        "value1": np.random.uniform(0, 100, 50),
        "value2": np.random.uniform(0, 100, 50),
    }).set_index("timestamp")
    # Introduce missing values
    data.loc[data.index[5:10], "value1"] = np.nan
    data.loc[data.index[20:25], "value2"] = np.nan
    return data


@pytest.fixture
def sample_df_with_outliers():
    """Create a DataFrame with outliers for testing."""
    np.random.seed(42)
    data = pd.DataFrame({
        "normal": np.random.normal(100, 10, 100),
        "with_outliers": np.concatenate([
            np.random.normal(100, 10, 95),
            np.array([500, 600, -200, -300, 1000])  # Outliers
        ])
    })
    return data


@pytest.fixture
def model_config():
    """Sample model configuration for testing."""
    return {
        "naive_bayes": {
            "discretization_bins": 10,
            "smoothing_alpha": 1.0,
        },
        "xgboost": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
        },
    }
