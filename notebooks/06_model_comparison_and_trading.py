
# 06_model_comparison_and_trading.py
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

    # Add project root to path
    project_root = Path(__file__).parent.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import src components
    from src.data.loaders import DataLoader
    from src.models.xgboost_model import XGBoostModel
    from src.models.naive_bayes import NaiveBayesModel
    from src.models.lstm_model import LSTMModel
    # We might skip explicit EnsembleModel class if we do flexible manual ensemble
    from src.evaluation.metrics import MetricsCalculator
    from sklearn.preprocessing import StandardScaler
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("notebook_06")
    
    mo.md("# Model Comparison & Trading Simulation")
    return DataLoader, LSTMModel, MetricsCalculator, NaiveBayesModel, Path, StandardScaler, XGBoostModel, datetime, logger, logging, mo, np, pd, plt, project_root, sys, os


@app.cell
def __(mo):
    mo.md("## 1. Configuration & Data Loading")
    return


@app.cell
def __(Path, project_root):
    # Configuration
    CONFIG = {
        "data_path": project_root / "data/processed/training_data.parquet",
        "models_dir": project_root / "models",
        "target_col": "close",
        "test_size": 0.2, # We must use SAME split as training to ensure we look at OOS data
        "risk_free_rate": 0.04, # 4% annual
    }
    
    # Define Model Paths
    MODEL_PATHS = {
        "Naive Bayes": CONFIG["models_dir"] / "naive_bayes/nb_baseline",
        "XGBoost": CONFIG["models_dir"] / "xgboost/xgb_best",
        "LSTM": CONFIG["models_dir"] / "lstm_keras/lstm_v1"
    }
    
    return CONFIG, MODEL_PATHS


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
    mo.md("## 2. Prepare Test Data (Out-of-Sample)")
    return


@app.cell
def __(CONFIG, StandardScaler, df, np, pd):
    # We need to reproduce the feature engineering / preprocessing steps 
    # largely to get the same feature set for the models.
    # Note: Each model might have slightly different feature expectations (e.g. LSTM scaled).
    # We will prep a unified dataset and adapt as needed.
    
    data = df.copy()
    target_col = CONFIG["target_col"]
    
    # Calculate Returns for Evaluation (Actual Buy & Hold)
    # Return at t is (Price_{t+1} / Price_t) - 1. This is what we traded on at T to realize at T+1.
    data["actual_return"] = (data[target_col].shift(-1) / data[target_col]) - 1
    
    # Dropna mainly involves the last row
    data = data.dropna()
    
    # Test Split
    test_size = CONFIG["test_size"]
    split_idx = int(len(data) * (1 - test_size))
    
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    # Common Features
    drop_cols = ["actual_return", "timestamp"]
    feature_cols = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]
    
    X_test = test_data[feature_cols]
    y_test_returns = test_data["actual_return"]
    
    # LSTM Scaling Prep (Fit on Train, Transform Test)
    scaler = StandardScaler()
    X_train_vals = train_data[feature_cols].values
    scaler.fit(X_train_vals)
    X_test_scaled_vals = scaler.transform(X_test[feature_cols].values)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_vals, columns=feature_cols, index=X_test.index)
    
    print(f"Test Set: {len(test_data)} samples represents {test_data.index.min()} to {test_data.index.max()}")
    return (X_test, X_test_scaled_df, X_test_scaled_vals, X_train_vals, data, drop_cols, 
            feature_cols, scaler, split_idx, target_col, test_data, test_size, 
            train_data, y_test_returns)


@app.cell
def __(mo):
    mo.md("## 3. Load Models & Generate Signals")
    return


@app.cell
def __(
    LSTMModel,
    MODEL_PATHS,
    NaiveBayesModel,
    XGBoostModel,
    X_test,
    X_test_scaled_df,
    mo,
    np,
    pd,
    y_test_returns,
):
    # Container for Signals (-1, 0, 1)
    # Index: Test Data Index
    signals_df = pd.DataFrame(index=X_test.index)
    
    # 1. Naive Bayes
    try:
        if MODEL_PATHS["Naive Bayes"].exists():
            nb = NaiveBayesModel()
            nb.load_model(str(MODEL_PATHS["Naive Bayes"]))
            # NB predicts classes (e.g. 0:Down, 1:Neutral, 2:Up depending on discretizer)
            # We assume the discretizer map from Notebook 2 was {-1:0, 0:1, 1:2} or similar.
            # Let's inspect preds to guess or assume simplified handling.
            nb_preds = nb.predict(X_test)
            
            # Map back to Direction if needed. Assuming NB was trained on -1, 0, 1? 
            # Check NB metadata if possible. For now, let's look at unique values.
            uniques = np.unique(nb_preds)
            print(f"NB Unique Preds: {uniques}")
            
            # Heuristic mapping: If [0, 1, 2], assume 0=Down, 1=Neutral, 2=Up
            # If [-1, 0, 1], use as is.
            if set(uniques).issubset({0, 1, 2}) and 2 in uniques:
                 nb_signals = np.where(nb_preds == 2, 1, np.where(nb_preds == 0, -1, 0))
            else:
                 nb_signals = nb_preds
                 
            signals_df["Naive Bayes"] = nb_signals
        else:
             print("Naive Bayes model not found.")
    except Exception as e:
        print(f"Error loading/predicting NB: {e}")

    # 2. XGBoost
    try:
        if MODEL_PATHS["XGBoost"].exists():
            xgb = XGBoostModel()
            xgb.load_model(str(MODEL_PATHS["XGBoost"]))
            # XGB was multiclass 0, 1, 2 (Down, Neutral, Up) in Notebook 3
            xgb_preds = xgb.predict(X_test)
            
            # Map 0->-1, 1->0, 2->1
            xgb_signals = np.where(xgb_preds == 2, 1, np.where(xgb_preds == 0, -1, 0))
            signals_df["XGBoost"] = xgb_signals
        else:
             print("XGBoost model not found.")
    except Exception as e:
        print(f"Error loading/predicting XGB: {e}")

    # 3. LSTM
    try:
        if MODEL_PATHS["LSTM"].exists():
            lstm = LSTMModel()
            lstm.load_model(str(MODEL_PATHS["LSTM"]))
            # LSTM predicts continuous return. We need to threshold it.
            # Use X_test_scaled_df as per Notebook 4
            lstm_preds_scaled = lstm.predict(X_test_scaled_df)
            
            # We need to Inverse Transform if we want magnitude, but for Sign, scaled is fine 
            # IF target_scaler was centered. But standardscaler is centered.
            # Sign of scaled return should roughly match sign of real return if mean > 0 is preserved.
            # However, safer to just use Sign of prediction.
            # NOTE: LSTM outputs shifted predictions (it consumes sequence to predict next).
            # The predict() method in `LSTMModel` handles the sequence generation.
            # It returns predictions for the end of sequences.
            # X_test_df has N rows. LSTM output will be N - seq_len.
            # We need to align.
            seq_len = lstm.hyperparameters.get("seq_len", 30)
            
            # Pad the beginning with 0 (Neutral) so signals align with X_test index
            # Or simpler: We analyze only the intersection.
            # Let's pad.
            lstm_raw_signals = np.sign(lstm_preds_scaled)
            
            # Padding
            pad_width = len(X_test) - len(lstm_raw_signals)
            if pad_width > 0:
                lstm_signals = np.concatenate([np.zeros(pad_width), lstm_raw_signals])
            else:
                lstm_signals = lstm_raw_signals
                
            signals_df["LSTM"] = lstm_signals
        else:
             print("LSTM model not found.")
    except Exception as e:
        print(f"Error loading/predicting LSTM: {e}")
        
    # 4. Ensemble (Simple Majority / Mean)
    # We take available columns
    available_models = signals_df.columns.tolist()
    if available_models:
        print(f"Ensembling models: {available_models}")
        # Mean signal
        mean_signal = signals_df[available_models].mean(axis=1)
        # Threshold: > 0.3 -> 1, < -0.3 -> -1, else 0
        ensemble_signal = np.where(mean_signal > 0.3, 1, np.where(mean_signal < -0.3, -1, 0))
        signals_df["Ensemble"] = ensemble_signal
    
    # Align Signals with Returns
    # Signals computed on X_test (at t) are for Return at t -> t+1.
    # So `signals_df` aligns with `y_test_returns`.
    
    signals_df = signals_df.fillna(0)
    print("Signal Counts:")
    for c in signals_df.columns:
        print(f"{c}: {signals_df[c].value_counts().to_dict()}")
        
    return available_models, ensemble_signal, mean_signal, nb, signals_df, xgb


@app.cell
def __(mo):
    mo.md("## 4. Trading Simulation & Metrics")
    return


@app.cell
def __(CONFIG, MetricsCalculator, np, pd, plt, signals_df, y_test_returns):
    results = []
    calculator = MetricsCalculator()
    
    # Cumulative Returns Plot
    plt.figure(figsize=(12, 6))
    
    # Plot Buy & Hold (Benchmark)
    # Cumulative prod of (1 + r)
    bnh_equity = (1 + y_test_returns).cumprod()
    plt.plot(bnh_equity, label="Buy & Hold", color="black", linestyle="--", linewidth=1.5)
    
    results.append({
        "Model": "Buy & Hold",
        "Total Return": bnh_equity.iloc[-1] - 1,
        "Sharpe": calculator.calculate_trading_metrics(y_test_returns, np.ones(len(y_test_returns)))["sharpe_ratio"]
    })
    
    for model_name in signals_df.columns:
        signals = signals_df[model_name]
        
        # Strategy Returns = Signal * Asset Return
        # We assume execution at Close t (signal generation) for Return t->t+1
        strat_returns = signals * y_test_returns
        
        # Calculation Metrics
        metrics = calculator.calculate_trading_metrics(
            y_test_returns.values, 
            signals.values,
            risk_free_rate=CONFIG["risk_free_rate"]
        )
        
        # Equity Curve
        equity_curve = (1 + strat_returns).cumprod()
        plt.plot(equity_curve, label=model_name)
        
        metrics["Model"] = model_name
        results.append(metrics)
        
    plt.title("Cumulative Returns Strategy Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylabel("Portfolio Value (Start=1.0)")
    plt.show()
    
    # Results Table
    results_df = pd.DataFrame(results).set_index("Model")
    cols_to_show = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"]
    # Handle B&H which might miss some keys if calculated differently
    print("\nPerformance Summary:")
    print(results_df[cols_to_show])
    return (bnh_equity, calculator, cols_to_show, equity_curve, metrics, model_name, 
            results, results_df, signals, strat_returns)


if __name__ == "__main__":
    app.run()
