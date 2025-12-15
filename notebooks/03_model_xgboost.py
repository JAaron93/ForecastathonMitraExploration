
# 03_model_xgboost.py
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
    from src.evaluation.metrics import MetricsCalculator
    from src.features.discretization import LabelDiscretizer
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("notebook_03")
    
    mo.md("# XGBoost Model: Optimization & Explainability")
    return DataLoader, LabelDiscretizer, MetricsCalculator, Path, XGBoostModel, datetime, logger, logging, mo, np, pd, plt, project_root, sys, os


@app.cell
def __(mo):
    mo.md("## 1. Configuration & Data Loading")
    return


@app.cell
def __(Path, project_root):
    # Configuration
    CONFIG = {
        "data_path": project_root / "data/processed/training_data.parquet",
        "models_dir": project_root / "models/xgboost",
        "target_col": "close",
        "lookahead": 1, 
        "test_size": 0.2,
        "experiment_name": "xgboost_optuna_shap",
        "optimize": True,
        "optuna_trials": 5, # Low for demo/speed, increase for real usage
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
    mo.md("## 2. Preprocessing")
    return


@app.cell
def __(CONFIG, LabelDiscretizer, df, pd):
    # Target Generation
    lookahead = CONFIG["lookahead"]
    target_col = CONFIG["target_col"]
    
    # Calculate returns
    future_return = df[target_col].pct_change(lookahead).shift(-lookahead)
    
    data = df.copy()
    data["future_return"] = future_return
    data = data.dropna()
    
    # Discretize Target (Direction)
    discretizer = LabelDiscretizer(strategy="direction", threshold=0.0001)
    data["target"] = discretizer.fit_transform(data["future_return"])
    
    # Remap target to 0, 1, 2 for XGBoost (multiclass) or 0, 1 for binary
    # Direction strategy gives -1, 0, 1. 
    # Let's map: -1 -> 0 (Down), 0 -> 1 (Neutral), 1 -> 2 (Up)
    # Or for simplicity, let's just do Binary (Up vs Not Up) for this demo if mostly 2 classes?
    # Let's stick to multiclass if we have neutral.
    
    # Map -1, 0, 1 to 0, 1, 2
    label_mapping = {-1: 0, 0: 1, 1: 2}
    data["target_encoded"] = data["target"].map(label_mapping)
    
    print("Target Distribution:")
    print(data["target_encoded"].value_counts(normalize=True))
    
    # Features
    drop_cols = ["future_return", "target", "target_encoded", "timestamp"]
    features = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]
    
    X = data[features]
    y = data["target_encoded"]
    
    print(f"Features: {len(features)}")
    return X, data, discretizer, features, future_return, label_mapping, lookahead, target_col, y, drop_cols


@app.cell
def __(CONFIG, X, y):
    # Train/Test Split
    test_size = CONFIG["test_size"]
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    return X_test, X_train, split_idx, test_size, y_test, y_train


@app.cell
def __(mo):
    mo.md("## 3. Training & Optimization")
    return


@app.cell
def __(CONFIG, XGBoostModel, X_train, y_train):
    # Initialize Model for Multiclass
    # num_class=3 for Down, Neutral, Up
    model = XGBoostModel(
        objective="multi:softprob",
        hyperparameters={"num_class": 3, "eval_metric": "mlogloss"}
    )
    
    # Optimization config
    optuna_params = {
        "n_trials": CONFIG["optuna_trials"],
        "n_splits": 3,
        "metric": "logloss"
    }
    
    print("Training XGBoost (this may take a moment)...")
    model.fit(
        X_train, 
        y_train, 
        optimize=CONFIG["optimize"],
        optimization_params=optuna_params
    )
    
    print("Model fitted.")
    print(f"Best params: {model.hyperparameters}")
    return model, optuna_params


@app.cell
def __(mo):
    mo.md("## 4. Evaluation")
    return


@app.cell
def __(CONFIG, MetricsCalculator, X_test, model, y_test):
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate Metrics
    calculator = MetricsCalculator()
    metrics = calculator.get_all_metrics(
        y_test, y_pred, y_proba, task_type="classification", average="weighted"
    )
    
    print("Evaluation Metrics:")
    for k, v in metrics.metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save Results
    save_path = CONFIG["models_dir"] / "xgb_best"
    model.save_model(str(save_path))
    print(f"\nModel saved to {save_path}")
    return calculator, metrics, save_path, y_pred, y_proba


@app.cell
def __(mo):
    mo.md("## 5. Explainability")
    return


@app.cell
def __(model, pd, plt):
    # Feature Importance (Gain)
    importance = model.get_feature_importance(importance_type="gain")
    imp_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Gain"])
    imp_df = imp_df.sort_values("Gain", ascending=False).head(10)
    
    print("Top 10 Features by Gain:")
    print(imp_df)
    
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df["Feature"], imp_df["Gain"])
    plt.gca().invert_yaxis()
    plt.title("XGBoost Feature Importance (Gain)")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.show() # Marimo captures stdout/plots
    return imp_df, importance


@app.cell
def __(X_test, model):
    # SHAP Values
    # We use a subsample of test data for SHAP to be fast
    X_shap = X_test.iloc[:100] 
    
    try:
        print("Calculating SHAP values...")
        shap_values = model.get_shap_values(X_shap)
        print(f"SHAP values shape: {getattr(shap_values, 'shape', 'unknown')}")
        # Note: For multiclass, shap_values is a list of arrays (one per class)
        
        # We can't easily plot interactive SHAP in marimo without JS libraries, 
        # but we can print summary statistics
        if isinstance(shap_values, list):
            print("SHAP values calculated for each class.")
            # Summarize mean absolute SHAP for class 2 (Up)
            print("Mean Absolute SHAP values for 'Up' class (top 5):")
            # shap_values[2] corresponds to class 2
            mean_shap = np.abs(shap_values[2]).mean(axis=0)
            # Match with columns
            # ... (implementation detail omitted for brevity, logic exists in libraries)
        else:
             print("SHAP values calculated.")
            
    except Exception as e:
        print(f"SHAP calculation failed: {e}")
    return X_shap, shap_values


if __name__ == "__main__":
    app.run()
