
# 05_model_mitra_tfm.py
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
    import shutil

    # Add project root to path
    project_root = Path(__file__).parent.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import src components
    from src.data.loaders import DataLoader
    from src.models.mitra_model import MitraModel
    from src.evaluation.metrics import MetricsCalculator
    from src.features.discretization import LabelDiscretizer
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("notebook_05")
    
    mo.md("# Mitra Foundation Model: In-Context Learning")
    return DataLoader, LabelDiscretizer, MetricsCalculator, MitraModel, Path, datetime, logger, logging, mo, np, pd, plt, project_root, shutil, sys, os


@app.cell
def __(mo):
    mo.md("## 1. Configuration & Data Loading")
    return


@app.cell
def __(Path, project_root, shutil):
    # Configuration
    CONFIG = {
        "data_path": project_root / "data/processed/training_data.parquet",
        "models_base_dir": project_root / "models/mitra_icl",
        "target_col": "close",
        "lookahead": 1,
        "context_window": 60,   # Days of history for Support Set
        "query_window": 7,      # Days to predict before re-adapting (Weekly adaptation)
        "test_size": 0.2,
        "n_adaptations": 5,     # Limit number of loops for demo speed
    }
    
    # Clean previous run
    if CONFIG["models_base_dir"].exists():
        shutil.rmtree(CONFIG["models_base_dir"])
    CONFIG["models_base_dir"].mkdir(parents=True, exist_ok=True)
    
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
    mo.md("## 2. Preprocessing")
    return


@app.cell
def __(CONFIG, LabelDiscretizer, df, pd):
    # Regression Target: Forward Return
    lookahead = CONFIG["lookahead"]
    target_col = CONFIG["target_col"]
    
    # Forward return
    # (Price_{t+k} / Price_t) - 1
    future_return = (df[target_col].shift(-lookahead) / df[target_col]) - 1
    
    data = df.copy()
    data["target"] = future_return
    data = data.dropna()
    
    # Features
    drop_cols = ["target", "timestamp"]
    feature_cols = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]
    
    print(f"Feature count: {len(feature_cols)}")
    return data, drop_cols, feature_cols, future_return, lookahead, target_col


@app.cell
def __(mo):
    mo.md("## 3. Regime Adaptation Loop (ICL)")
    return


@app.cell
def __(CONFIG, MetricsCalculator, MitraModel, data, feature_cols, pd):
    # We simulate a rolling window process.
    # Start at beginning of Test set.
    # Use [t - context, t] as Support.
    # Predict [t+1, t+query].
    # Move t -> t + query.
    
    test_size = CONFIG["test_size"]
    test_start_idx = int(len(data) * (1 - test_size))
    
    context_window = CONFIG["context_window"]
    query_window = CONFIG["query_window"]
    max_loops = CONFIG["n_adaptations"]
    
    current_idx = test_start_idx
    all_preds = []
    all_targets = []
    
    print(f"Starting ICL Loop (Max {max_loops} adaptations)...")
    
    for i in range(max_loops):
        if current_idx + query_window >= len(data):
            break
            
        # Define Indices
        support_start = max(0, current_idx - context_window)
        support_end = current_idx
        query_end = current_idx + query_window
        
        # Slices
        support_data = data.iloc[support_start:support_end]
        query_data = data.iloc[support_end:query_end]
        
        X_support = support_data[feature_cols]
        y_support = support_data["target"]
        
        X_query = query_data[feature_cols]
        y_query = query_data["target"]
        
        print(f"Loop {i+1}: Support {len(X_support)} rows, Query {len(X_query)} rows")
        
        # Initialize Mitra Model (New instance per context usually, or re-fit)
        # We use a unique ID to avoid collision in AG folders
        model_id = f"mitra_loop_{i}"
        
        # Note: We must enable 'fine_tune': False for ICL/Zero-shot behavior if supported
        # problem_type='regression' for continuous returns
        model = MitraModel(
            model_id=model_id,
            hyperparameters={"fine_tune": False, "time_limit": 30}, # Short time limit for demo
            problem_type="regression",
            label_column="target",
            eval_metric="rmse"
        )
        
        # Fit on Support (Context)
        # This effectively "loads" the context into the foundation model
        try:
            model.fit(X_support, y_support, validation_data=None, verbosity=0)
            
            # Predict on Query
            preds = model.predict(X_query)
            
            all_preds.extend(preds)
            all_targets.extend(y_query.values)
            
            # Save this model artifact for reference (optional, storing just the last one or all)
            # save_path = CONFIG["models_base_dir"] / model_id
            # model.save_model(str(save_path))
            
        except Exception as e:
            print(f"Skipping loop {i} due to Error: {e}")
            # If AG fails (e.g. not enough data), we just skip or fill
            pass
            
        # Comparison with Baseline (Mean of Support)
        # naive_pred = y_support.mean()
        # baseline_mse = ((y_query - naive_pred)**2).mean()
        # print(f"  > Baseline MSE: {baseline_mse:.6f}")
        
        # Advance Window
        current_idx += query_window
        
    print("Loop Complete.")
    return (X_query, X_support, all_preds, all_targets, context_window, current_idx, 
            i, max_loops, model, model_id, query_data, query_end, query_window, 
            support_data, support_end, support_start, test_size, test_start_idx, 
            y_query, y_support)


@app.cell
def __(mo):
    mo.md("## 4. Evaluation")
    return


@app.cell
def __(MetricsCalculator, all_preds, all_targets, plt):
    if not all_preds:
        print("No predictions generated.")
    else:
        calculator = MetricsCalculator()
        metrics = calculator.get_all_metrics(
            np.array(all_targets), 
            np.array(all_preds), 
            task_type="regression"
        )
        
        print("Aggregate Metrics across all regimes:")
        for k, v in metrics.metrics.items():
            print(f"{k}: {v:.6f}")
            
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(all_targets, label="Actual", alpha=0.7)
        plt.plot(all_preds, label="Mitra Predicted", alpha=0.7)
        plt.title("Mitra Model: In-Context Predictions (Concatenated)")
        plt.legend()
        plt.show()
        
        # Scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(all_targets, all_preds, alpha=0.5)
        plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.show()
    return calculator, metrics


if __name__ == "__main__":
    app.run()
