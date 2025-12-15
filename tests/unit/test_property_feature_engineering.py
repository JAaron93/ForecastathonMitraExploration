"""
Property-based tests for feature engineering correctness.

**Feature: forecasting-research-pipeline, Property 3: Feature engineering
mathematical correctness**

Tests that for any input time series, generated features (lags, rolling
statistics, technical indicators, volatility measures, correlations) are
mathematically correct according to their definitions.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**
"""

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from src.features.engineering import FeatureEngineer
from src.features.technical_indicators import TechnicalIndicators
from src.features.regime_detection import VolatilityCalculator, RegimeDetector


# Custom strategies for generating test data
@st.composite
def ohlcv_dataframe(draw, min_rows=30, max_rows=200):
    """Generate a DataFrame with OHLCV data and DatetimeIndex."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate dates
    start_date = pd.Timestamp("2020-01-01")
    dates = pd.date_range(start=start_date, periods=n_rows, freq="D")

    # Generate base price
    base_price = draw(st.floats(min_value=10.0, max_value=1000.0))

    # Generate price changes (small percentage changes)
    changes = draw(
        st.lists(
            st.floats(min_value=-0.05, max_value=0.05),
            min_size=n_rows,
            max_size=n_rows,
        )
    )

    # Build price series
    prices = [base_price]
    for change in changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Ensure positive prices

    close = np.array(prices)

    # Generate OHLC from close
    high_mult = draw(
        st.lists(
            st.floats(min_value=1.0, max_value=1.05),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    low_mult = draw(
        st.lists(
            st.floats(min_value=0.95, max_value=1.0),
            min_size=n_rows,
            max_size=n_rows,
        )
    )

    high = close * np.array(high_mult)
    low = close * np.array(low_mult)

    # Open is between low and high
    open_prices = (high + low) / 2

    # Volume
    volume = draw(
        st.lists(
            st.floats(min_value=1000, max_value=1000000),
            min_size=n_rows,
            max_size=n_rows,
        )
    )

    df = pd.DataFrame({
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    return df


@st.composite
def numeric_series(draw, min_size=30, max_size=200):
    """Generate a numeric Series with positive values."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    values = draw(
        st.lists(
            st.floats(min_value=1.0, max_value=1000.0, allow_nan=False,
                      allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    return pd.Series(values, index=dates)


@st.composite
def lag_periods(draw):
    """Generate valid lag periods."""
    n_lags = draw(st.integers(min_value=1, max_value=5))
    lags = sorted(set(draw(
        st.lists(
            st.integers(min_value=1, max_value=20),
            min_size=n_lags,
            max_size=n_lags,
        )
    )))
    return lags


@st.composite
def rolling_windows(draw):
    """Generate valid rolling window sizes."""
    n_windows = draw(st.integers(min_value=1, max_value=4))
    windows = sorted(set(draw(
        st.lists(
            st.integers(min_value=2, max_value=30),
            min_size=n_windows,
            max_size=n_windows,
        )
    )))
    return windows


class TestLagFeatureCorrectness:
    """
    Property tests for lag feature mathematical correctness.

    **Feature: forecasting-research-pipeline, Property 3: Feature engineering
    mathematical correctness**
    **Validates: Requirements 2.1**
    """

    @given(df=ohlcv_dataframe(), lags=lag_periods())
    @settings(max_examples=100, deadline=None)
    def test_lag_values_match_shifted_original(self, df, lags):
        """
        Property: For any lag feature, the value at time t should equal
        the original value at time t-lag.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.1**
        """
        engineer = FeatureEngineer()
        result = engineer.create_lag_features(df, columns=["close"], lags=lags)

        for lag in lags:
            feature_name = f"close_lag_{lag}"
            assert feature_name in result.columns

            # Check values match shifted original (excluding NaN positions)
            expected = df["close"].shift(lag)
            actual = result[feature_name]

            # Compare non-NaN values
            valid_mask = ~expected.isna() & ~actual.isna()
            if valid_mask.any():
                np.testing.assert_array_almost_equal(
                    actual[valid_mask].values,
                    expected[valid_mask].values,
                    decimal=10,
                    err_msg=f"Lag {lag} values don't match shifted original"
                )

    @given(df=ohlcv_dataframe(), lags=lag_periods())
    @settings(max_examples=100, deadline=None)
    def test_lag_features_have_correct_nan_count(self, df, lags):
        """
        Property: A lag-n feature should have exactly n NaN values at the start.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.1**
        """
        engineer = FeatureEngineer()
        result = engineer.create_lag_features(df, columns=["close"], lags=lags)

        for lag in lags:
            feature_name = f"close_lag_{lag}"
            nan_count = result[feature_name].isna().sum()

            assert nan_count == lag, (
                f"Lag {lag} should have {lag} NaN values, got {nan_count}"
            )


class TestRollingStatisticsCorrectness:
    """
    Property tests for rolling statistics mathematical correctness.

    **Feature: forecasting-research-pipeline, Property 3: Feature engineering
    mathematical correctness**
    **Validates: Requirements 2.1**
    """

    @given(df=ohlcv_dataframe(), windows=rolling_windows())
    @settings(max_examples=100, deadline=None)
    def test_rolling_mean_is_correct(self, df, windows):
        """
        Property: Rolling mean should equal the arithmetic mean of the window.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.1**
        """
        engineer = FeatureEngineer()
        result = engineer.calculate_rolling_stats(
            df, columns=["close"], windows=windows, stats=["mean"]
        )

        for window in windows:
            feature_name = f"close_rolling_mean_{window}"
            assert feature_name in result.columns

            # Verify rolling mean calculation
            expected = df["close"].rolling(window=window, min_periods=1).mean()
            actual = result[feature_name]

            np.testing.assert_array_almost_equal(
                actual.values,
                expected.values,
                decimal=10,
                err_msg=f"Rolling mean (window={window}) incorrect"
            )

    @given(df=ohlcv_dataframe(), windows=rolling_windows())
    @settings(max_examples=100, deadline=None)
    def test_rolling_std_is_non_negative(self, df, windows):
        """
        Property: Rolling standard deviation should always be non-negative.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.1**
        """
        engineer = FeatureEngineer()
        result = engineer.calculate_rolling_stats(
            df, columns=["close"], windows=windows, stats=["std"]
        )

        for window in windows:
            feature_name = f"close_rolling_std_{window}"
            valid_values = result[feature_name].dropna()

            assert (valid_values >= 0).all(), (
                f"Rolling std (window={window}) has negative values"
            )

    @given(df=ohlcv_dataframe(), windows=rolling_windows())
    @settings(max_examples=100, deadline=None)
    def test_rolling_min_max_bounds(self, df, windows):
        """
        Property: Rolling min should be <= rolling mean <= rolling max.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.1**
        """
        engineer = FeatureEngineer()
        result = engineer.calculate_rolling_stats(
            df, columns=["close"], windows=windows, stats=["mean", "min", "max"]
        )

        for window in windows:
            min_col = f"close_rolling_min_{window}"
            mean_col = f"close_rolling_mean_{window}"
            max_col = f"close_rolling_max_{window}"

            # Get valid indices (where all three are not NaN)
            valid_mask = (
                ~result[min_col].isna() &
                ~result[mean_col].isna() &
                ~result[max_col].isna()
            )

            if valid_mask.any():
                assert (result.loc[valid_mask, min_col] <=
                        result.loc[valid_mask, mean_col] + 1e-10).all(), (
                    f"Rolling min > mean for window={window}"
                )
                assert (result.loc[valid_mask, mean_col] <=
                        result.loc[valid_mask, max_col] + 1e-10).all(), (
                    f"Rolling mean > max for window={window}"
                )


class TestTechnicalIndicatorsCorrectness:
    """
    Property tests for technical indicators mathematical correctness.

    **Feature: forecasting-research-pipeline, Property 3: Feature engineering
    mathematical correctness**
    **Validates: Requirements 2.1, 2.2**
    """

    @given(series=numeric_series())
    @settings(max_examples=100, deadline=None)
    def test_rsi_bounds(self, series):
        """
        Property: RSI should always be between 0 and 100.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.1**
        """
        indicators = TechnicalIndicators()
        rsi = indicators.calculate_rsi(series, period=14)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all(), "RSI has values below 0"
        assert (valid_rsi <= 100).all(), "RSI has values above 100"

    @given(series=numeric_series())
    @settings(max_examples=100, deadline=None)
    def test_macd_signal_relationship(self, series):
        """
        Property: MACD histogram should equal MACD line minus signal line.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.1**
        """
        indicators = TechnicalIndicators()
        macd_line, signal_line, histogram = indicators.calculate_macd(series)

        expected_histogram = macd_line - signal_line
        valid_mask = ~macd_line.isna() & ~signal_line.isna()

        if valid_mask.any():
            np.testing.assert_array_almost_equal(
                histogram[valid_mask].values,
                expected_histogram[valid_mask].values,
                decimal=10,
                err_msg="MACD histogram != MACD line - signal line"
            )

    @given(series=numeric_series())
    @settings(max_examples=100, deadline=None)
    def test_bollinger_bands_ordering(self, series):
        """
        Property: Lower band <= Middle band <= Upper band.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.1**
        """
        indicators = TechnicalIndicators()
        upper, middle, lower = indicators.calculate_bollinger_bands(series)

        valid_mask = ~upper.isna() & ~middle.isna() & ~lower.isna()

        if valid_mask.any():
            assert (lower[valid_mask] <= middle[valid_mask] + 1e-10).all(), (
                "Lower band > middle band"
            )
            assert (middle[valid_mask] <= upper[valid_mask] + 1e-10).all(), (
                "Middle band > upper band"
            )

    @given(df=ohlcv_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_atr_is_non_negative(self, df):
        """
        Property: ATR (Average True Range) should always be non-negative.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.2**
        """
        indicators = TechnicalIndicators()
        atr = indicators.calculate_atr(
            df["high"], df["low"], df["close"], period=14
        )

        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all(), "ATR has negative values"

    @given(df=ohlcv_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_stochastic_bounds(self, df):
        """
        Property: Stochastic %K and %D should be between 0 and 100.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.1**
        """
        indicators = TechnicalIndicators()
        k, d = indicators.calculate_stochastic(
            df["high"], df["low"], df["close"]
        )

        valid_k = k.dropna()
        valid_d = d.dropna()

        assert (valid_k >= -1e-10).all(), "Stochastic %K below 0"
        assert (valid_k <= 100 + 1e-10).all(), "Stochastic %K above 100"
        assert (valid_d >= -1e-10).all(), "Stochastic %D below 0"
        assert (valid_d <= 100 + 1e-10).all(), "Stochastic %D above 100"

    @given(df=ohlcv_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_williams_r_bounds(self, df):
        """
        Property: Williams %R should be between -100 and 0.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.1**
        """
        indicators = TechnicalIndicators()
        williams_r = indicators.calculate_williams_r(
            df["high"], df["low"], df["close"]
        )

        valid_wr = williams_r.dropna()
        assert (valid_wr >= -100 - 1e-10).all(), "Williams %R below -100"
        assert (valid_wr <= 1e-10).all(), "Williams %R above 0"


class TestVolatilityMeasuresCorrectness:
    """
    Property tests for volatility measures mathematical correctness.

    **Feature: forecasting-research-pipeline, Property 3: Feature engineering
    mathematical correctness**
    **Validates: Requirements 2.2**
    """

    @given(df=ohlcv_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_realized_volatility_non_negative(self, df):
        """
        Property: Realized volatility should always be non-negative.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.2**
        """
        vol_calc = VolatilityCalculator()
        returns = np.log(df["close"] / df["close"].shift(1))
        vol = vol_calc.calculate_realized_volatility(returns, window=20)

        valid_vol = vol.dropna()
        assert (valid_vol >= 0).all(), "Realized volatility has negative values"

    @given(df=ohlcv_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_parkinson_volatility_non_negative(self, df):
        """
        Property: Parkinson volatility should always be non-negative.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.2**
        """
        vol_calc = VolatilityCalculator()
        vol = vol_calc.calculate_parkinson_volatility(
            df["high"], df["low"], window=20
        )

        valid_vol = vol.dropna()
        assert (valid_vol >= 0).all(), "Parkinson volatility has negative values"

    @given(df=ohlcv_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_garman_klass_volatility_non_negative(self, df):
        """
        Property: Garman-Klass volatility should always be non-negative.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.2**
        """
        vol_calc = VolatilityCalculator()
        vol = vol_calc.calculate_garman_klass_volatility(
            df["open"], df["high"], df["low"], df["close"], window=20
        )

        valid_vol = vol.dropna()
        assert (valid_vol >= 0).all(), (
            "Garman-Klass volatility has negative values"
        )


class TestCalendarFeaturesCorrectness:
    """
    Property tests for calendar features correctness.

    **Feature: forecasting-research-pipeline, Property 3: Feature engineering
    mathematical correctness**
    **Validates: Requirements 2.4**
    """

    @given(df=ohlcv_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_day_of_week_bounds(self, df):
        """
        Property: Day of week should be between 0 (Monday) and 6 (Sunday).

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.4**
        """
        engineer = FeatureEngineer()
        result = engineer.create_calendar_features(df)

        assert (result["day_of_week"] >= 0).all(), "Day of week below 0"
        assert (result["day_of_week"] <= 6).all(), "Day of week above 6"

    @given(df=ohlcv_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_month_bounds(self, df):
        """
        Property: Month should be between 1 and 12.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.4**
        """
        engineer = FeatureEngineer()
        result = engineer.create_calendar_features(df)

        assert (result["month"] >= 1).all(), "Month below 1"
        assert (result["month"] <= 12).all(), "Month above 12"

    @given(df=ohlcv_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_cyclical_encoding_bounds(self, df):
        """
        Property: Cyclical sin/cos encodings should be between -1 and 1.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.4**
        """
        engineer = FeatureEngineer()
        result = engineer.create_calendar_features(df)

        for col in ["day_of_week_sin", "day_of_week_cos",
                    "month_sin", "month_cos"]:
            assert (result[col] >= -1 - 1e-10).all(), f"{col} below -1"
            assert (result[col] <= 1 + 1e-10).all(), f"{col} above 1"

    @given(df=ohlcv_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_is_weekend_binary(self, df):
        """
        Property: is_weekend should be binary (0 or 1).

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.4**
        """
        engineer = FeatureEngineer()
        result = engineer.create_calendar_features(df)

        assert set(result["is_weekend"].unique()).issubset({0, 1}), (
            "is_weekend has non-binary values"
        )

    @given(df=ohlcv_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_weekend_matches_day_of_week(self, df):
        """
        Property: is_weekend should be 1 iff day_of_week is 5 or 6.

        **Feature: forecasting-research-pipeline, Property 3: Feature
        engineering mathematical correctness**
        **Validates: Requirements 2.4**
        """
        engineer = FeatureEngineer()
        result = engineer.create_calendar_features(df)

        expected_weekend = (result["day_of_week"] >= 5).astype(int)
        assert (result["is_weekend"] == expected_weekend).all(), (
            "is_weekend doesn't match day_of_week >= 5"
        )
