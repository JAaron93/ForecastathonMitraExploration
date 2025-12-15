import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.preprocessors import Preprocessor
from src.features.engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostModel
from src.evaluation.metrics import MetricsCalculator

def test_end_to_end_pipeline(tmp_path, sample_timeseries_df):
    """
    Test the full loop:
    1. Preprocessing
    2. Feature Engineering
    3. Model Training (XGBoost)
    4. Evaluation
    5. Serialization
    """
    # 1. Preprocessing
    preprocessor = Preprocessor()
    sample_df = sample_timeseries_df.copy()
    
    # Resample
    resampled_df = preprocessor.resample_timeseries(
        sample_df, freq="D", agg_method="ohlc"
    )
    
    assert len(resampled_df) > 0
    assert isinstance(resampled_df.index, pd.DatetimeIndex)
    
    # 2. Feature Engineering
    engineer = FeatureEngineer()
    processed_df = engineer.engineer_all_features(
        resampled_df,
        price_columns=["close"],
        include_lags=True,
        include_rolling=True
    )
    
    # Check simple lag feature existence
    # Note: engineer_all_features might name them 'close_lag_1' or similar
    # We check if *some* new columns were added
    assert len(processed_df.columns) > len(resampled_df.columns)
    
    # Drop rows with NaNs created by lags
    processed_df = processed_df.dropna()
    
    # Create target (Next Day Close)
    processed_df["target"] = processed_df["close"].shift(-1)
    processed_df = processed_df.dropna()
    
    # Split features and target
    # Select numeric columns for XGBoost
    X = processed_df.select_dtypes(include=[np.number]).drop(columns=["target"], errors='ignore')
    y = processed_df["target"]
    
    assert not X.empty
    assert not y.empty
    
    # 3. Model Training
    model = XGBoostModel()
    # Use small subset for speed
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Mocking optuna tuning by setting n_trials=1 if possible or just calling fit
    # XGBoostModel.fit() might run tuning if not simplified.
    # Looking at config, it uses default if not tuning.
    # Let's check defaults in test_xgboost or model.
    # Assuming standard fit works.
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    
    # 4. Evaluation
    metrics_calc = MetricsCalculator()
    # Regression metrics
    metrics = metrics_calc.calculate_regression_metrics(y_test.values, preds)
    
    assert "rmse" in metrics
    assert "mae" in metrics
    
    # 5. Serialization
    model_path = tmp_path / "model.json"
    model.save_model(str(model_path))
    assert model_path.exists()
    
    # Load back
    loaded_model = XGBoostModel()
    loaded_model.load_model(str(model_path))
    preds_loaded = loaded_model.predict(X_test)
    
    np.testing.assert_array_almost_equal(preds, preds_loaded)
