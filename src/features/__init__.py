"""Feature engineering utilities for forecasting pipeline.

This module provides comprehensive feature engineering capabilities including:
- Lag features and rolling statistics
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Volatility measures (realized, Parkinson, Garman-Klass, Yang-Zhang)
- Regime detection algorithms
- Calendar features
- Cross-asset correlation and spread calculations
"""

from src.features.engineering import (
    FeatureEngineer,
    FeatureDefinitions,
)
from src.features.technical_indicators import TechnicalIndicators
from src.features.regime_detection import (
    VolatilityCalculator,
    RegimeDetector,
    MarketRegime,
)

__all__ = [
    "FeatureEngineer",
    "FeatureDefinitions",
    "TechnicalIndicators",
    "VolatilityCalculator",
    "RegimeDetector",
    "MarketRegime",
]
