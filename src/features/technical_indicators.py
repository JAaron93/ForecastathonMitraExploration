"""Technical indicators for financial time series analysis.

Implements RSI, MACD, Bollinger Bands, and other common technical indicators
used in trading and forecasting applications.
"""

from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for financial time series."""

    def calculate_rsi(
        self,
        series: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over the period

        Args:
            series: Price series (typically close prices)
            period: Lookback period (default 14)

        Returns:
            Series with RSI values (0-100)
        """
        delta = series.diff()
        
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        # Use exponential moving average for smoothing
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = rsi.fillna(50)  # Neutral RSI when undefined
        
        return rsi

    def calculate_macd(
        self,
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).

        MACD Line = Fast EMA - Slow EMA
        Signal Line = EMA of MACD Line
        Histogram = MACD Line - Signal Line

        Args:
            series: Price series (typically close prices)
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(
        self,
        series: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Middle Band = SMA(period)
        Upper Band = Middle Band + (num_std * std)
        Lower Band = Middle Band - (num_std * std)

        Args:
            series: Price series (typically close prices)
            period: Moving average period (default 20)
            num_std: Number of standard deviations (default 2.0)

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle_band = series.rolling(window=period).mean()
        rolling_std = series.rolling(window=period).std()
        
        upper_band = middle_band + (num_std * rolling_std)
        lower_band = middle_band - (num_std * rolling_std)
        
        return upper_band, middle_band, lower_band

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
        ATR = EMA of True Range

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default 14)

        Returns:
            Series with ATR values
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr

    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA of %K

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period (default 14)
            d_period: %D smoothing period (default 3)

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # Avoid division by zero
        range_diff = highest_high - lowest_low
        range_diff = range_diff.replace(0, np.nan)
        
        k = ((close - lowest_low) / range_diff) * 100
        d = k.rolling(window=d_period).mean()
        
        return k, d

    def calculate_williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Williams %R.

        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Lookback period (default 14)

        Returns:
            Series with Williams %R values (-100 to 0)
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        range_diff = highest_high - lowest_low
        range_diff = range_diff.replace(0, np.nan)
        
        williams_r = ((highest_high - close) / range_diff) * -100
        
        return williams_r

    def calculate_cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).

        Typical Price = (High + Low + Close) / 3
        CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: CCI period (default 20)

        Returns:
            Series with CCI values
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        
        # Mean deviation
        mean_dev = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        
        # Avoid division by zero
        mean_dev = mean_dev.replace(0, np.nan)
        
        cci = (typical_price - sma_tp) / (0.015 * mean_dev)
        
        return cci

    def calculate_obv(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).

        OBV increases when close > previous close
        OBV decreases when close < previous close

        Args:
            close: Close price series
            volume: Volume series

        Returns:
            Series with OBV values
        """
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        
        obv = (direction * volume).cumsum()
        
        return obv

    def calculate_momentum(
        self,
        series: pd.Series,
        period: int = 10
    ) -> pd.Series:
        """
        Calculate Price Momentum.

        Momentum = Current Price - Price n periods ago

        Args:
            series: Price series
            period: Lookback period (default 10)

        Returns:
            Series with momentum values
        """
        return series - series.shift(period)

    def calculate_roc(
        self,
        series: pd.Series,
        period: int = 10
    ) -> pd.Series:
        """
        Calculate Rate of Change (ROC).

        ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100

        Args:
            series: Price series
            period: Lookback period (default 10)

        Returns:
            Series with ROC values (percentage)
        """
        prev_price = series.shift(period)
        roc = ((series - prev_price) / prev_price.replace(0, np.nan)) * 100
        
        return roc
