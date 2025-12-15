
# 04_model_lstm.py
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
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    # Add project root to path
    project_root = Path(__file__).parent.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import src components
    from src.data.loaders import DataLoader
    from src.models.lstm_model import LSTMModel
    from src.evaluation.metrics import MetricsCalculator
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("notebook_04")
    
    mo.md("# LSTM Sequence Model: Forecasting Returns")
    return DataLoader, LSTMModel, MetricsCalculator, Path, StandardScaler, datetime, logger, logging, mo, np, pd, plt, project_root, sys, os


@app.cell
def __(mo):
    mo.md("## 1. Configuration & Data Loading")
    return


@app.cell
def __(Path, project_root):
    # Configuration
    CONFIG = {
        "data_path": project_root / "data/processed/training_data.parquet",
        "models_dir": project_root / "models/lstm_keras",
        "target_col": "close",
        "seq_len": 30,      # Lookback window (30 days/steps)
        "lookahead": 1,     # Forecast horizon (1 step)
        "test_size": 0.2,
        "experiment_name": "lstm_regression_v1"
    }
    
    Path(CONFIG["models_dir"]).mkdir(parents=True, exist_ok=True)
    return CONFIG,


@app.cell
def __(CONFIG, DataLoader, mo):
    loader = DataLoader()
    
    if not CONFIG["data_path"].exists():
        mo.md(f"**Error**: Data file not found at {CONFIG['data_path']}. Please run notebook 01 first.")
        raise FileNotFoundError(f"Data file not found: {CONFIG['data_path']}")
    
    df = loader.load_parquet(str(CONFIG['data_path']))
    print(f"Loaded {len(df)} rows.")
    return df, loader


@app.cell
def __(mo):
    mo.md("## 2. Preprocessing & Feature Engineering")
    return


@app.cell
def __(CONFIG, StandardScaler, df, np, pd):
    # We are doing Regression: Predicting the numerical return
    
    lookahead = CONFIG["lookahead"]
    target_col = CONFIG["target_col"]
    
    # Target: Forward Return (Price_{t+k} / Price_t) - 1
    future_return = df[target_col].shift(-lookahead) / df[target_col] - 1
    
    data = df.copy()
    data["target_return"] = future_return
    data = data.dropna()
    
    # Feature Selection
    # Drop non-numeric/future cols
    drop_cols = ["target_return", "timestamp"]
    feature_cols = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]
    
    print(f"Features ({len(feature_cols)}): {feature_cols[:5]}...")
    
    # Scaling
    # LSTMs are sensitive to scale. We scale Features AND Target.
    # Ideally fit scaler on Train only to avoid leakage, but for simplicity in baseline notebook we split then scale.
    
    test_size = CONFIG["test_size"]
    split_idx = int(len(data) * (1 - test_size))
    
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    # Scalers
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Fit on Train
    X_train_raw = train_data[feature_cols].values
    y_train_raw = train_data[["target_return"]].values
    
    X_train_scaled = feature_scaler.fit_transform(X_train_raw)
    y_train_scaled = target_scaler.fit_transform(y_train_raw)
    
    # Transform Test
    X_test_raw = test_data[feature_cols].values
    y_test_raw = test_data[["target_return"]].values
    
    X_test_scaled = feature_scaler.transform(X_test_raw)
    
    # We keep y_test_scaled for evaluation of loss, but y_test_raw for real metrics
    y_test_scaled = target_scaler.transform(y_test_raw)
    
    # Pack into DataFrames for the Model wrapper (it expects DFs, outputs np arrays internally)
    # The LSTMModel wrapper splits sequences internally, so we pass continuous DataFrames.
    # However, since we already split Train/Test, we pass them separately.
    
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=train_data.index)
    y_train_series = pd.DataFrame(y_train_scaled, columns=["target"], index=train_data.index)["target"]
    
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=test_data.index)
    
    print(f"Train samples: {len(X_train_df)}")
    print(f"Test samples: {len(X_test_df)}")
    return (X_test_df, X_test_raw, X_test_scaled, X_train_df, X_train_raw, X_train_scaled, 
            data, drop_cols, feature_cols, feature_scaler, future_return, lookahead, 
            split_idx, target_col, target_scaler, test_data, test_size, train_data, 
            y_test_raw, y_test_scaled, y_train_raw, y_train_scaled, y_train_series)


@app.cell
def __(mo):
    mo.md("## 3. Training LSTM")
    return


@app.cell
def __(CONFIG, LSTMModel, X_train_df, y_train_series):
    # Hyperparameters
    params = {
        "seq_len": CONFIG["seq_len"],
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "epochs": 15, # Low for demo
        "batch_size": 32,
        "patience": 5
    }
    
    model = LSTMModel(hyperparameters=params)
    
    # Manual Validation Split for Early Stopping within Train set
    # Or we can let LSTMModel handle it if we passed `validation_data`
    # Let's split X_train_df further for validation
    val_len = int(len(X_train_df) * 0.2)
    X_val = X_train_df.iloc[-val_len:]
    y_val = y_train_series.iloc[-val_len:]
    X_try = X_train_df.iloc[:-val_len]
    y_try = y_train_series.iloc[:-val_len]
    
    print("Training LSTM...")
    model.fit(
        X_try, 
        y_try, 
        validation_data=(X_val, y_val),
        verbose=1 # Show progress in logs
    )
    
    print("Training Complete.")
    print(f"Final Train Loss: {model.training_metrics.get('loss')}")
    print(f"Final Val Loss: {model.validation_metrics.get('loss')}")
    return X_try, X_val, model, params, val_len, y_try, y_val


@app.cell
def __(mo):
    mo.md("## 4. Evaluation & Visualization")
    return


@app.cell
def __(CONFIG, MetricsCalculator, X_test_df, model, np, plt, target_scaler, y_test_raw):
    # Predict (output is scaled)
    # Note: predict() handles sequence creation. Output length will be len(input) - seq_len
    preds_scaled = model.predict(X_test_df)
    
    # Align Truth
    # The model consumes `seq_len` samples to produce predictions for the NEXT step.
    # If X_test has N rows, we get N - seq_len predictions.
    # These predictions correspond to y at indices [seq_len, seq_len+1, ...] of X_test.
    seq_len = CONFIG["seq_len"]
    
    # Prepare True Values aligned with Predictions
    # y_test_raw corresponds to the target at each timestamp of X_test_df
    # We need to slice y_test_raw to match the predictions
    y_true_aligned = y_test_raw[seq_len:]
    
    # Inverse Transform Predictions
    if preds_scaled.ndim == 1:
        preds_scaled = preds_scaled.reshape(-1, 1)
        
    preds_original = target_scaler.inverse_transform(preds_scaled)
    
    # Calculate Metrics
    # Regression metrics
    calculator = MetricsCalculator()
    metrics = calculator.get_all_metrics(
        y_true_aligned.flatten(), 
        preds_original.flatten(), 
        task_type="regression"
    )
    
    print("Evaluation Metrics (Regression):")
    for k, v in metrics.metrics.items():
        print(f"{k}: {v:.6f}")
        
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_aligned, label="Actual Return", alpha=0.7)
    plt.plot(preds_original, label="Predicted Return", alpha=0.7)
    plt.title("LSTM Forecast (Test Set)")
    plt.legend()
    plt.show()
    
    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_aligned, preds_original, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Returns")
    plt.plot([y_true_aligned.min(), y_true_aligned.max()], 
             [y_true_aligned.min(), y_true_aligned.max()], 'r--')
    plt.show()
    
    # Save
    save_path = CONFIG["models_dir"] / "lstm_v1"
    model.save_model(str(save_path))
    print(f"\nModel saved to {save_path}")
    return (calculator, metrics, preds_original, preds_scaled, save_path, seq_len, 
            y_true_aligned)


if __name__ == "__main__":
    app.run()
