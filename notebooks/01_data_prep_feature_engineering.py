
# 01_data_prep_feature_engineering.py
import marimo

__generated_with = "0.1.0"
app = marimo.App(width="medium")

@app.cell
def __():
    import marimo as mo
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import logging
    from datetime import datetime
    import sys
    import os

    # Add project root to path
    project_root = Path(__file__).parent.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import pipeline components
    from src.data.loaders import DataLoader
    from src.data.preprocessors import Preprocessor
    from src.data.profilers import DataProfiler
    from src.features.engineering import FeatureEngineer
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("notebook_01")
    
    mo.md("# Data Preparation & Feature Engineering")
    return DataLoader, DataProfiler, FeatureEngineer, Path, Preprocessor, datetime, logger, logging, mo, np, pd, sys, os, project_root


@app.cell
def __(mo):
    mo.md("## 1. Configuration")
    return


@app.cell
def __(Path):
    # Configuration
    DATA_CONFIG = {
        "raw_data_path": "data/raw",
        "processed_data_path": "data/processed",
        "reports_path": "data/processed/reports",
        "missing_value_strategy": "forward_fill",
        "resample_freq": "1D",
        "feature_engineering": {
            "lag_periods": [1, 2, 3, 5, 7, 14, 30],
            "rolling_windows": [7, 14, 30, 90],
            "include_holidays": True
        }
    }

    # Ensure directories exist
    Path(DATA_CONFIG["processed_data_path"]).mkdir(parents=True, exist_ok=True)
    Path(DATA_CONFIG["reports_path"]).mkdir(parents=True, exist_ok=True)
    
    print("Configuration loaded and directories verified.")
    return DATA_CONFIG,


@app.cell
def __(mo):
    mo.md("## 2. Load Raw Data")
    return


@app.cell
def __(DATA_CONFIG, DataLoader, Path):
    loader = DataLoader(log_dir="logs/data_validation")
    
    # Define schemas
    btc_schema = {
        "timestamp": "datetime64[ns]",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64"
    }
    
    macro_schema = {
        "timestamp": "datetime64[ns]",
        "interest_rate": "float64",
        "inflation_cpi": "float64",
        "sp500_close": "float64"
    }

    # Load data
    try:
        btc_path = str(Path(DATA_CONFIG["raw_data_path"]) / "btc_ohlcv.parquet")
        btc_df = loader.load_parquet(btc_path, schema=btc_schema)
        
        macro_path = str(Path(DATA_CONFIG["raw_data_path"]) / "macro_data.parquet")
        macro_df = loader.load_parquet(macro_path, schema=macro_schema)
        
        print(f"Loaded BTC data: {len(btc_df)} rows")
        print(f"Loaded Macro data: {len(macro_df)} rows")
        
        # Set index
        btc_df.set_index("timestamp", inplace=True)
        macro_df.set_index("timestamp", inplace=True)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create dummy data if loading fails (for demonstration/testing without files)
        # In production this should stop execution
        raise e
    return btc_df, btc_path, btc_schema, loader, macro_df, macro_path, macro_schema


@app.cell
def __(mo):
    mo.md("## 3. Initial Data Profiling")
    return


@app.cell
def __(DATA_CONFIG, DataProfiler, btc_df):
    profiler = DataProfiler(output_dir=DATA_CONFIG["reports_path"])
    
    # Generate profile for raw BTC data
    # Using minimal=True for speed in this demo, set to False for full analysis
    btc_profile_path = profiler.save_report(
        profiler.generate_profile(btc_df.reset_index(), title="Raw BTC Data", minimal=True),
        dataset_identifier="raw_btc"
    )
    
    print(f"Raw BTC profile saved to: {btc_profile_path}")
    
    # Check data quality
    quality_summary = profiler.get_data_quality_summary(
        profiler.generate_profile(btc_df.reset_index(), title="Raw BTC Data", minimal=True)
    )
    
    print("Data Quality Summary:")
    print(quality_summary)
    return btc_profile_path, profiler, quality_summary


@app.cell
def __(mo):
    mo.md("## 4. Preprocessing")
    return


@app.cell
def __(DATA_CONFIG, Preprocessor, btc_df, macro_df):
    preprocessor = Preprocessor()
    
    # Resample BTC to daily if needed (it was hourly in generator)
    btc_daily = preprocessor.resample_timeseries(
        btc_df, 
        freq=DATA_CONFIG["resample_freq"], 
        agg_method="ohlc"
    )
    
    # Handle missing values
    btc_clean = preprocessor.handle_missing_values(
        btc_daily, 
        strategy=DATA_CONFIG["missing_value_strategy"]
    )
    
    macro_clean = preprocessor.handle_missing_values(
        macro_df,
        strategy=DATA_CONFIG["missing_value_strategy"]
    )
    
    # Outlier detection (just logging for now)
    outliers = preprocessor.detect_outliers(btc_clean, method="iqr")
    print(f"Detected {len(outliers)} outliers in BTC data")
    
    # Merge datasets
    # Align macro data to BTC index
    merged_df = btc_clean.join(macro_clean, how="left")
    merged_df = preprocessor.handle_missing_values(merged_df, strategy="forward_fill")
    
    print(f"Merged dataset shape: {merged_df.shape}")
    return btc_clean, btc_daily, merged_df, outliers, preprocessor, macro_clean


@app.cell
def __(mo):
    mo.md("## 5. Feature Engineering")
    return


@app.cell
def __(DATA_CONFIG, FeatureEngineer, merged_df):
    fe_config = DATA_CONFIG["feature_engineering"]
    engineer = FeatureEngineer(config=fe_config)
    
    # Generate features
    processed_df = engineer.engineer_all_features(
        merged_df,
        price_columns=["close", "volume"], # specific columns to focus on
        include_lags=True,
        include_rolling=True,
        include_calendar=True,
        include_returns=True
    )
    
    # Get definitions
    definitions = engineer.get_feature_definitions()
    
    print(f"Original columns: {len(merged_df.columns)}")
    print(f"Processed columns: {len(processed_df.columns)}")
    print(f"New features created: {len(processed_df.columns) - len(merged_df.columns)}")
    return definitions, engineer, fe_config, processed_df


@app.cell
def __(mo):
    mo.md("## 6. Post-Processing Profiling & Saving")
    return


@app.cell
def __(DATA_CONFIG, Path, processed_df, profiler):
    # Profile processed data
    processed_profile_path = profiler.save_report(
        profiler.generate_profile(processed_df.reset_index(), title="Processed Training Data", minimal=True),
        dataset_identifier="processed_training"
    )
    print(f"Processed profile saved to: {processed_profile_path}")
    
    # Save processed dataset
    output_file = Path(DATA_CONFIG["processed_data_path"]) / "training_data.parquet"
    processed_df.to_parquet(output_file)
    
    print(f"Saved processed data to: {output_file}")
    
    # Validating file exists
    if output_file.exists():
        print("SUCCESS: Pipeline completed and file saved.")
    else:
        print("ERROR: File not found.")
    return output_file, processed_profile_path


if __name__ == "__main__":
    app.run()
