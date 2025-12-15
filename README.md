# Forecasting Research Pipeline

A modular, notebook-based machine learning system designed for the Autonity/Forecastathon trading competition. The system processes diverse financial time-series data, performs comprehensive feature engineering, trains multiple model families (Naive Bayes, XGBoost, LSTM, Mitra), and generates actionable trading signals.

## Overview

The architecture follows a research-first approach with six specialized Marimo notebooks, each handling a specific aspect of the machine learning pipeline. Marimo provides reactive execution, built-in interactivity, and pure Python files (`.py`) that are git-friendly and reproducible.

### Key Features

*   **Modular Architecture**: Separation of concerns between data, features, models, and evaluation.
*   **Multi-Model Support**: Naive Bayes (Baseline), XGBoost (Tabular), LSTM (Sequential), and Mitra (Foundation Model).
*   **Robust Engineering**: Comprehensive testing (unit & property-based), type-safe interfaces, and centralized logging.
*   **Reproducibility**: Configurable experiments, artifact serialization, and MLflow tracking.
*   **Interactive Analysis**: Interactive `marimo` notebooks for data exploration and result visualization.

## Directory Structure

```
forecasting-research-pipeline/
├── notebooks/                 # Marimo notebooks for the research workflow
│   ├── 01_data_prep_...       # Data loading, cleaning, and feature engineering
│   ├── 02_baseline_...        # Naive Bayes baseline experiments
│   ├── 03_model_xgboost.py    # XGBoost training and tuning
│   ├── 04_model_lstm.py       # LSTM sequence model training
│   ├── 05_model_mitra_tfm.py  # Mitra foundation model experiments
│   └── 06_model_comparison... # Model comparison and trading signals
├── src/                       # Source code for the pipeline
│   ├── data/                  # Data loaders, preprocessors, validators
│   ├── features/              # Feature engineering logic
│   ├── models/                # Model implementations
│   ├── evaluation/            # Metrics and evaluation tools
│   ├── trading/               # Trading signal generation
│   └── utils/                 # Utilities (logging, config, etc.)
├── config/                    # Configuration files
├── data/                      # Data storage (raw, processed)
├── experiments/               # Experiment artifacts and tracking
├── logs/                      # Application logs
└── tests/                     # Unit and integration tests
```

## Setup and Installation

### Prerequisites

*   Python 3.10+
*   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JAaron93/ForecastathonMitraExploration.git
    cd forecasting-research-pipeline
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -e ".[dev]"
    ```
    This installs the package in editable mode along with development dependencies.

## Usage Guide

The workflow is divided into 6 sequential notebooks. You can run them interactively with `marimo` or as scripts.

### Running with Marimo (Recommended)

To launch the interactive notebook interface:

```bash
marimo edit notebooks/01_data_prep_feature_engineering.py
```

### Workflow Steps

1.  **Data Preparation (`01_data_prep...`)**:
    *   Loads raw data from `data/raw/` (configured in `config/data_config.yaml`).
    *   Cleans, validates, and resamples time series.
    *   Generates features (lags, technical indicators).
    *   Saves processed data to `data/processed/`.

2.  **Baseline Model (`02_baseline...`)**:
    *   Trains a Naive Bayes classifier as a baseline.
    *   Establishes performance floor for more complex models.

3.  **XGBoost Model (`03_model_xgboost.py`)**:
    *   Trains Gradient Boosted Trees.
    *   Performs hyperparameter optimization using Optuna.
    *   Analyzes feature importance (SHAP).

4.  **LSTM Model (`04_model_lstm.py`)**:
    *   Trains LSTM sequence models on time-series windows.
    *   Captures temporal dependencies.

5.  **Mitra Model (`05_model_mitra_tfm.py`)**:
    *   Leverages the Mitra Tabular Foundation Model.
    *   Uses in-context learning for regime adaptation.

6.  **Comparison & Trading (`06_model_comparison...`)**:
    *   Loads artifacts from all models.
    *   Compares performance (Sharpe, Returns, Accuracy).
    *   Generates trading signals and portfolio allocations.

## Configuration

Configuration is managed via YAML files in the `config/` directory:

*   `data_config.yaml`: Data sources, schema definitions, and preprocessing settings.
*   `model_config.yaml`: Hyperparameters for all model families (XGBoost, LSTM, Mitra).
*   `experiment_config.yaml`: Experiment tracking settings (MLflow).
*   `logging_config.yaml`: Logging verbosity and output formats.

### Templates

See `config/templates/` for example configurations:
*   `experiment_config_dev.yaml` / `model_config_dev.yaml`: Development/fast iteration (minimal epochs, small n_trials).
*   `experiment_config_prod.yaml` / `model_config_prod.yaml`: Production training (full data, exhaustive tuning).

## Testing

Run the test suite using `pytest`:

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit

# Run property-based tests (slow but thorough)
pytest tests/unit -m hypothesis

# Run integration tests
pytest tests/integration
```

## Quality Assurance

*   **Linting**: `flake8 src tests`
*   **Formatting**: `black src tests`
*   **Type Checking**: `mypy src`

## License

MIT License.