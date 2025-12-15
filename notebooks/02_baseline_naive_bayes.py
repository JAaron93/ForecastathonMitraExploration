
# 02_baseline_naive_bayes.py
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

    # Import src components
    from src.data.loaders import DataLoader
    from src.models.naive_bayes import NaiveBayesModel
    from src.evaluation.metrics import MetricsCalculator
    from src.features.discretization import LabelDiscretizer
    from src.data.preprocessors import Preprocessor # For any last minute cleaning if needed

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("notebook_02")
    
    mo.md("# Baseline Model: Naive Bayes")
    return DataLoader, LabelDiscretizer, MetricsCalculator, NaiveBayesModel, Path, Preprocessor, datetime, logger, logging, mo, np, pd, project_root, sys


@app.cell
def __(mo):
    mo.md("## 1. Configuration & Data Loading")
    return


@app.cell
def __(Path, project_root):
    # Configuration
    CONFIG = {
        "data_path": project_root / "data/processed/training_data.parquet",
        "models_dir": project_root / "models/naive_bayes",
        "target_col": "close",
        "lookahead": 1, # predict 1 step ahead
        "test_size": 0.2, # last 20% for testing
        "experiment_name": "naive_bayes_baseline"
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
    mo.md("## 2. Target Generation & Feature Selection")
    return


@app.cell
def __(CONFIG, LabelDiscretizer, df, np, pd):
    # Calculate returns for target
    # We want to predict if price goes Up/Down in next period
    # Target = sign(Close_{t+1} - Close_t) or similar
    
    lookahead = CONFIG["lookahead"]
    target_col = CONFIG["target_col"]
    
    # Forward return: (Price_{t+k} / Price_t) - 1
    # We use shift(-k) to align future value to current row
    future_return = (df[target_col].shift(-lookahead) / df[target_col]) - 1
    
    # Drop NaNs created by shifting
    data = df.copy()
    data["future_return"] = future_return
    data = data.dropna()
    
    # Discretize Target
    # Strategy 1: Direction (Up/Down/Neutral)
    discretizer = LabelDiscretizer(strategy="direction", threshold=0.0001) # Small threshold for neutral
    data["target"] = discretizer.fit_transform(data["future_return"])
    
    print("Target Distribution:")
    print(data["target"].value_counts(normalize=True))
    
    # Feature Selection
    # Drop target-related columns and non-numeric cols
    drop_cols = ["future_return", "target", "timestamp"] # timestamp is index usually but checking
    features = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]
    
    # Remove any columns strictly related to future (none here as we used shifted, but good to be careful)
    
    X = data[features]
    y = data["target"]
    
    print(f"Features: {len(features)}")
    return X, data, discretizer, features, future_return, lookahead, target_col, y


@app.cell
def __(mo):
    mo.md("## 3. Train/Test Split")
    return


@app.cell
def __(CONFIG, X, y):
    # Time-series split (no shuffling)
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
    mo.md("## 4. Modeling & Evaluation")
    return


@app.cell
def __(CONFIG, MetricsCalculator, NaiveBayesModel, X_test, X_train, y_test, y_train):
    # Initialize Model
    model = NaiveBayesModel(hyperparameters={"var_smoothing": 1e-9})
    
    # Train
    print("Training Naive Bayes...")
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Evaluate
    calculator = MetricsCalculator()
    metrics = calculator.get_all_metrics(
        y_test, y_pred, y_proba, task_type="classification"
    )
    
    print("\nEvaluation Metrics:")
    for k, v in metrics.metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save Model
    save_path = CONFIG["models_dir"] / "nb_baseline"
    model.save_model(str(save_path))
    print(f"\nModel saved to {save_path}")
    return calculator, metrics, model, save_path, y_pred, y_proba


@app.cell
def __(CONFIG, metrics, pd):
    # Simple report for experiment tracking
    results = {
        "experiment": CONFIG["experiment_name"],
        "lookahead": CONFIG["lookahead"],
        **metrics.metrics
    }
    
    results_df = pd.DataFrame([results])
    print("Experiment Result:")
    print(results_df)
    
    # Save results CSV
    results_path = CONFIG["models_dir"] / "experiment_results.csv"
    mode = 'a' if results_path.exists() else 'w'
    header = not results_path.exists()
    results_df.to_csv(results_path, mode=mode, header=header, index=False)
    print("Experiment logged.")
    return header, mode, results, results_df, results_path


if __name__ == "__main__":
    app.run()
