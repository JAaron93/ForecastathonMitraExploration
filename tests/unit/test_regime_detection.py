import pytest
import pandas as pd
import numpy as np

from src.features.regime_detection import RegimeDetector, MarketRegime


@pytest.fixture
def detector():
    """Returns a basic RegimeDetector instance."""
    return RegimeDetector(
        low_vol_threshold=0.10,
        high_vol_threshold=0.25,
        crisis_threshold=0.40
    )


def test_regime_detection_empty_series(detector):
    """Test RegimeDetector behavior with an empty Series."""
    empty_series = pd.Series([], dtype=float)
    
    # Should return an empty series without crashing
    regime = detector.detect_volatility_regime(empty_series)
    assert len(regime) == 0
    assert isinstance(regime, pd.Series)


def test_regime_detection_single_element(detector):
    """Test RegimeDetector with a single-element Series."""
    single_series = pd.Series([0.15])
    
    regime = detector.detect_volatility_regime(single_series)
    assert len(regime) == 1
    assert regime.iloc[0] == MarketRegime.NORMAL.value


def test_regime_detection_nan_handling(detector):
    """Test how RegimeDetector handles NaN values."""
    # NaN should fall through conditions and result in the default regime ("normal")
    nan_series = pd.Series([np.nan, 0.30, np.nan])
    
    regime = detector.detect_volatility_regime(nan_series)
    assert len(regime) == 3
    assert regime.iloc[0] == MarketRegime.NORMAL.value
    assert regime.iloc[1] == MarketRegime.HIGH_VOLATILITY.value
    assert regime.iloc[-1] == MarketRegime.NORMAL.value


def test_regime_detection_threshold_boundaries(detector):
    """Test inputs exactly at the defined volatility thresholds."""
    # Detector defaults: low=0.10, high=0.25, crisis=0.40
    boundary_series = pd.Series([0.10, 0.25, 0.40])
    
    regime = detector.detect_volatility_regime(boundary_series)
    
    # Check that boundaries are inclusive according to conditions: <= 0.10, >= 0.25, >= 0.40
    assert regime.iloc[0] == MarketRegime.LOW_VOLATILITY.value
    assert regime.iloc[1] == MarketRegime.HIGH_VOLATILITY.value
    assert regime.iloc[2] == MarketRegime.CRISIS.value


def test_volatility_regime_transitions(detector):
    """Test transitions between different volatility regimes."""
    # Create an array of volatility values that transition through regimes
    # low_vol (0.05) -> normal (0.15) -> high_vol (0.30) -> crisis (0.50) -> low_vol (0.08)
    volatility_values = [0.05, 0.15, 0.30, 0.50, 0.08]
    dates = pd.date_range("2023-01-01", periods=len(volatility_values))
    volatility = pd.Series(volatility_values, index=dates)

    regime = detector.detect_volatility_regime(volatility)

    # Correct states based on thresholds
    expected_states = [
        MarketRegime.LOW_VOLATILITY.value,
        MarketRegime.NORMAL.value,
        MarketRegime.HIGH_VOLATILITY.value,
        MarketRegime.CRISIS.value,
        MarketRegime.LOW_VOLATILITY.value,
    ]

    np.testing.assert_array_equal(regime.values, expected_states)


def test_trend_regime_transitions(detector):
    """Test transitions between trend regimes: sideways -> uptrend -> downtrend."""
    # We will spoof the rolling functionality of short (2) and long (4)
    # Using specific prices to manipulate short_ma and long_ma
    
    # Let's say we have prices.
    # We only need enough prices to produce diffs over the long window.
    prices = pd.Series(
        [100, 100, 100, 100, 100, # Sideways for stabilization
         110, 120, 130,           # Uptrend: short MA will rise faster than long
         90, 80, 70]              # Downtrend: short MA drops faster than long
    )

    regime = detector.detect_trend_regime(prices, short_window=2, long_window=4)

    # Validate that we correctly see the expected regimes
    # Early values will be NaN, which np.select handles based on default
    
    # Sideways validation
    assert regime.iloc[4] == "sideways"
    
    # Uptrend validation (by periods 6, 7)
    assert regime.iloc[6] == "uptrend"
    assert regime.iloc[7] == "uptrend"
    
    # Downtrend validation (by periods 9, 10)
    assert regime.iloc[9] == "downtrend"
    assert regime.iloc[10] == "downtrend"


def test_momentum_regime_transitions(detector):
    """Test transitions between momentum regimes."""
    # Using small windows: window=2, threshold=0.5
    returns = pd.Series(
        [0.0, 0.0, 0.0, 0.0,  # Neutral
         0.1, 0.2,            # Positive momentum
        -0.1, -0.2]           # Negative momentum
    )

    # Note: momentum regime standardizes using rolling window*2 (4)
    # So we need at least 4 periods for std to be non-zero
    regime = detector.detect_momentum_regime(returns, window=2, threshold=0.5)

    # Verify exact temporal placement of regimes rather than mere existence
    # Indices 0-3 are neutral (0.0 returns)
    assert all(r == "neutral" for r in regime.values[0:4])
    
    # Indices 4-6 capture the positive momentum wave
    assert all(r == "strong_positive" for r in regime.values[4:7])
    
    # Index 7 captures the sharp drop triggering negative momentum
    assert regime.values[-1] == "strong_negative"


def test_regime_duration_calculation(detector):
    """Test calculation of consecutive days in same regime."""
    # Transitions: normal(1), normal(2), high(1), high(2), high(3), low(1)
    states = ["normal", "normal", "high_volatility", "high_volatility", "high_volatility", "low_volatility"]
    regime_series = pd.Series(states)

    duration = detector._calculate_regime_duration(regime_series)

    expected_duration = [1, 2, 1, 2, 3, 1]
    np.testing.assert_array_equal(duration.values, expected_duration)


def test_calculate_regime_features(detector):
    """Test the main method to ensure integrations correctly calculate all regimes and features."""
    dates = pd.date_range("2023-01-01", periods=50)
    
    # Using some dummy price data to run without crashing
    df = pd.DataFrame({
        "close": np.linspace(100, 150, 50),
        "volatility": np.full(50, 0.20)  # Constant normal volatility
    }, index=dates)

    features_df = detector.calculate_regime_features(df, close_col="close", volatility_col="volatility")

    # Check for core columns
    assert "volatility_regime" in features_df.columns
    assert "trend_regime" in features_df.columns
    assert "momentum_regime" in features_df.columns
    assert "vol_regime_duration" in features_df.columns
    assert "trend_regime_duration" in features_df.columns

    # Verify duration actually increments due to constant trend and vol
    # Note: The first elements might be NaN depending on the default values and how long it has been in that state
    # But duration will always be calculated based on the string labels
    
    # Volatility duration check: use a robust range check instead of strict equality
    # because rolling window preprocessing could introduce leading NaNs.
    final_duration = features_df["vol_regime_duration"].iloc[-1]
    assert not pd.isna(final_duration), "Final duration should not be NaN"
    assert final_duration >= 48, f"Expected duration >= 48, got {final_duration}"

