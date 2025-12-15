# AGENTS.md

## Project Overview

This is a **Forecasting Research Pipeline** for the Autonity/Forecastathon trading competition. The system processes financial time-series data, performs feature engineering, trains multiple ML model families, and generates trading signals.

**Architecture**: Modular, Marimo notebook-based ML pipeline with reactive execution and git-friendly pure Python files.

## Tech Stack

- **Language**: Python 3.10+
- **Notebooks**: Marimo (reactive, pure Python .py files)
- **ML Frameworks**: scikit-learn, XGBoost, PyTorch (LSTM), AutoGluon 1.4+ (Mitra)
- **Data**: pandas, numpy, Parquet files
- **Profiling**: ydata-profiling
- **Experiment Tracking**: MLflow
- **Explainability**: SHAP
- **Testing**: pytest, hypothesis (property-based testing)
- **CI**: GitHub Actions

## Directory Structure

```
├── notebooks/           # Marimo notebooks (01-06, not yet created)
├── src/
│   ├── data/           # Data loading, preprocessing, validation, profiling, splitting
│   ├── features/       # Feature engineering, technical indicators, regime detection
│   ├── models/         # Model implementations (Naive Bayes, XGBoost, LSTM, Mitra)
│   ├── evaluation/     # Metrics, calibration, explainability
│   ├── trading/        # Signal generation, portfolio utilities
│   └── utils/          # Experiment tracking, logging, visualization
├── config/             # YAML configuration files
├── data/               # raw/, processed/, external/
├── models/             # Saved model artifacts by type
├── experiments/        # MLflow runs
├── logs/               # Structured logs (validation, training, errors)
└── tests/              # unit/, integration/, reports/
```

## Current Implementation Status

### Completed (Tasks 1-3.1)
- [x] Project structure and core infrastructure
- [x] Testing framework with pytest + hypothesis
- [x] CI pipeline (GitHub Actions)
- [x] Data processing: `DataLoader`, `Preprocessor`, `DataValidator`, `DataProfiler`, `TimeSeriesSplitter`
- [x] Feature engineering: lag features, rolling stats, technical indicators (RSI, MACD, Bollinger), regime detection, calendar features
- [x] Property tests for data validation, time series operations, feature engineering

### In Progress (Task 4+)
- [ ] Base model interface and evaluation framework
- [ ] Model implementations (Naive Bayes, XGBoost, LSTM, Mitra)
- [ ] Trading signal generation
- [ ] Error handling and monitoring infrastructure
- [ ] Marimo notebooks (01-06)

## Key Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `DataLoader` | `src/data/loaders.py` | Parquet loading, schema validation |
| `Preprocessor` | `src/data/preprocessors.py` | Missing values, outliers, resampling |
| `DataValidator` | `src/data/validators.py` | Schema validation, PSI calculation |
| `DataProfiler` | `src/data/profilers.py` | ydata-profiling integration |
| `TimeSeriesSplitter` | `src/data/splitters.py` | Rolling/expanding window splits |
| `FeatureEngineer` | `src/features/engineering.py` | Lag, rolling, cross-asset features |
| `TechnicalIndicators` | `src/features/technical_indicators.py` | RSI, MACD, Bollinger Bands |
| `RegimeDetector` | `src/features/regime_detection.py` | HMM, volatility regime detection |

## Configuration Files

- `config/data_config.yaml` - Data sources, preprocessing, profiling settings
- `config/model_config.yaml` - Model hyperparameters for all model families
- `config/experiment_config.yaml` - MLflow and experiment settings
- `config/logging_config.yaml` - Structured logging configuration

## Testing Strategy

**Coverage Target**: 80% minimum

**Property-Based Tests** (hypothesis, 100+ iterations):
1. Data serialization round-trip consistency
2. Time series temporal ordering preservation
3. Feature engineering mathematical correctness
4. Data validation and preprocessing consistency
5. Model training and evaluation correctness
6. Mitra in-context learning adaptation
7. Ensemble and comparison consistency
8. Trading signal generation correctness
9. Error handling and logging completeness
10. Experiment tracking and monitoring accuracy
11. Configuration flexibility and validation

Run tests: `pytest tests/ -v --cov=src --cov-report=html`

## Planned Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_prep_feature_engineering.py` | Data loading, preprocessing, feature engineering, profiling |
| `02_baseline_naive_bayes.py` | Fast iteration baseline experiments |
| `03_model_xgboost.py` | XGBoost with hyperparameter optimization |
| `04_model_lstm.py` | LSTM sequence modeling |
| `05_model_mitra_tfm.py` | Mitra foundation model with in-context learning |
| `06_model_comparison_and_trading.py` | Model comparison, ensemble, trading signals |

## Model Families

1. **Naive Bayes** - Fast baseline for data/feature validation
2. **XGBoost** - Gradient boosting with Optuna hyperparameter tuning
3. **LSTM** - Sequence model for temporal dependencies
4. **Mitra** - Tabular foundation model via AutoGluon, in-context learning for regime adaptation

## Key Design Decisions

- **Marimo over Jupyter**: Git-friendly, reactive execution, built-in interactivity
- **Property-based testing**: Ensures correctness across edge cases
- **Time-series aware splitting**: Prevents data leakage
- **Mitra for regime adaptation**: Adapts to market changes without retraining
- **Structured logging**: JSON format to `logs/` directories with retention policies

## Error Handling

- Retry with exponential backoff (max 5 retries, 2s initial, 2x multiplier, 60s ceiling)
- Pipeline state checkpointing for recovery
- Structured error logs with full context to `logs/errors/`

## Development Commands

```bash
# Install dependencies
pip install -e .

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run property tests only
pytest tests/ -v -k "property"

# Type checking
mypy src/

# Linting
pre-commit run --all-files
```

## Contributing Guidelines

1. All new code requires tests (unit + property-based where applicable)
2. Maintain 80%+ test coverage
3. Use type hints throughout
4. Follow existing module patterns in `src/`
5. Configuration changes go in `config/*.yaml`
6. Document assumptions in notebook markdown cells
