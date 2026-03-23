import pytest
import pandas as pd
import numpy as np
from src.data.loaders import DataLoader


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory with nested asset files."""
    # Set fixed RNG seed for reproducibility
    np.random.seed(42)
    
    crypto_dir = tmp_path / "crypto"
    crypto_dir.mkdir()
    macro_dir = tmp_path / "macro"
    macro_dir.mkdir()

    # Create dummy crypto data
    df_btc = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
        "open": np.random.randn(5),
        "high": np.random.randn(5),
        "low": np.random.randn(5),
        "close": np.random.randn(5),
        "volume": np.random.randn(5)
    })
    df_btc.to_parquet(crypto_dir / "btc.parquet")

    # Create dummy macro data
    df_sp500 = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
        "sp500_close": np.random.randn(5)
    })
    df_sp500.to_parquet(macro_dir / "sp500.parquet")

    return tmp_path


def test_load_from_config_nested(temp_data_dir):
    loader = DataLoader()
    config = {
        "crypto": {
            "path": str(temp_data_dir / "crypto"),
            "assets": ["btc"],
            "schema": {
                "timestamp": "datetime64[ns]",
                "close": "float64"
            }
        },
        "macro": {
            "path": str(temp_data_dir / "macro"),
            "assets": ["sp500"],
            "schema": {
                "timestamp": "datetime64[ns]",
                "sp500_close": "float64"
            }
        }
    }

    all_data = loader.load_from_config(config)

    assert "crypto" in all_data
    assert "btc" in all_data["crypto"]
    assert len(all_data["crypto"]["btc"]) == 5
    assert "macro" in all_data
    assert "sp500" in all_data["macro"]
    assert len(all_data["macro"]["sp500"]) == 5

    # Assert schema enforcement on returned data
    btc_df = all_data["crypto"]["btc"]
    sp500_df = all_data["macro"]["sp500"]
    
    # Check crypto->btc DataFrame schema
    assert "timestamp" in btc_df.columns
    assert "close" in btc_df.columns
    assert btc_df["timestamp"].dtype == "datetime64[ns]"
    assert btc_df["close"].dtype == "float64"
    
    # Check macro->sp500 DataFrame schema
    assert "sp500_close" in sp500_df.columns
    assert sp500_df["sp500_close"].dtype == "float64"


def test_load_assets_missing_file(temp_data_dir, caplog):
    loader = DataLoader()
    # Should log warning but not raise exception if one asset is missing
    loaded = loader.load_assets(
        str(temp_data_dir / "crypto"), ["btc", "nonexistent"]
    )
    assert "btc" in loaded
    assert "nonexistent" not in loaded
    
    # Assert warning was logged for missing asset
    warning_logged = False
    for record in caplog.records:
        if record.levelname == "WARNING" and "nonexistent" in record.message:
            warning_logged = True
            break
    
    assert warning_logged, (
        "Expected a WARNING log message for missing asset 'nonexistent'"
    )
