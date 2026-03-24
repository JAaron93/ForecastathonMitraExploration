import pytest
import pandas as pd
import numpy as np
import logging
from src.data.loaders import DataLoader


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory with nested asset files."""
    # Set isolated RNG for reproducibility
    rng = np.random.default_rng(42)

    crypto_dir = tmp_path / "crypto"
    crypto_dir.mkdir()
    macro_dir = tmp_path / "macro"
    macro_dir.mkdir()

    # Create dummy crypto data
    df_btc = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
        "open": rng.standard_normal(5),
        "high": rng.standard_normal(5),
        "low": rng.standard_normal(5),
        "close": rng.standard_normal(5),
        "volume": rng.standard_normal(5)
    })
    df_btc.to_parquet(crypto_dir / "btc.parquet")

    # Create dummy macro data
    df_sp500 = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
        "sp500_close": rng.standard_normal(5)
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
    assert "timestamp" in sp500_df.columns
    assert "sp500_close" in sp500_df.columns
    assert sp500_df["timestamp"].dtype == "datetime64[ns]"
    assert sp500_df["sp500_close"].dtype == "float64"


def test_load_assets_missing_file(temp_data_dir, caplog):
    loader = DataLoader()
    # Should log error but not raise exception if one asset is missing
    # and strict=False
    caplog.set_level(logging.ERROR)
    loaded = loader.load_assets(
        str(temp_data_dir / "crypto"), ["btc", "nonexistent"], strict=False
    )
    assert "btc" in loaded
    assert "nonexistent" not in loaded

    # Assert error was logged for missing asset
    assert any(
        record.levelname == "ERROR" and "nonexistent" in record.message
        for record in caplog.records
    ), "Expected an ERROR log message for missing asset 'nonexistent'"


def test_load_assets_strict_missing_file(temp_data_dir):
    loader = DataLoader()
    # Should raise FileNotFoundError if strict=True and an asset is missing
    with pytest.raises(FileNotFoundError):
        loader.load_assets(
            str(temp_data_dir / "crypto"), ["btc", "nonexistent"], strict=True
        )


def test_load_assets_status_tracking(temp_data_dir):
    loader = DataLoader()
    status = {}
    loaded = loader.load_assets(
        str(temp_data_dir / "crypto"), ["btc", "nonexistent"],
        strict=False, status=status
    )

    assert "btc" in loaded
    assert "nonexistent" not in loaded
    assert set(status["loaded_assets"]) == {"btc"}
    assert "nonexistent" in status["failed_assets"]
    assert "Parquet file not found" in status["failed_assets"]["nonexistent"]


def test_load_assets_strict_schema_validation(temp_data_dir):
    loader = DataLoader()
    # Incorrect schema to trigger ValueError
    bad_schema = {"timestamp": "int64"}

    with pytest.raises(ValueError, match="Schema validation failed"):
        loader.load_assets(
            str(temp_data_dir / "crypto"), ["btc"],
            schema=bad_schema, strict=True
        )
