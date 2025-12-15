"""Volatility measures and regime detection for financial time series.

Implements realized volatility, GARCH-style measures, and regime detection
algorithms for market state identification.
"""

from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    LOW_VOLATILITY = "low_volatility"
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"


class VolatilityCalculator:
    """Calculate various volatility measures for financial time series."""

    def calculate_realized_volatility(
        self,
        returns: pd.Series,
        window: int = 20,
        annualize: bool = True,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate realized volatility (rolling standard deviation of returns).

        Args:
            returns: Return series (log or simple returns)
            window: Rolling window size
            annualize: Whether to annualize the volatility
            trading_days: Number of trading days per year

        Returns:
            Series with realized volatility values
        """
        vol = returns.rolling(window=window).std()
        
        if annualize:
            vol = vol * np.sqrt(trading_days)
        
        return vol

    def calculate_parkinson_volatility(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int = 20,
        annualize: bool = True,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate Parkinson volatility estimator.

        Uses high-low range to estimate volatility, more efficient than
        close-to-close volatility.

        Parkinson = sqrt(1/(4*ln(2)) * (ln(H/L))^2)

        Args:
            high: High price series
            low: Low price series
            window: Rolling window size
            annualize: Whether to annualize
            trading_days: Number of trading days per year

        Returns:
            Series with Parkinson volatility values
        """
        log_hl = np.log(high / low)
        squared_log_hl = log_hl ** 2
        
        # Parkinson constant: 1 / (4 * ln(2))
        parkinson_const = 1 / (4 * np.log(2))
        
        vol = np.sqrt(parkinson_const * squared_log_hl.rolling(window=window).mean())
        
        if annualize:
            vol = vol * np.sqrt(trading_days)
        
        return vol

    def calculate_garman_klass_volatility(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        annualize: bool = True,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate Garman-Klass volatility estimator.

        Uses OHLC data for more efficient volatility estimation.

        Args:
            open_price: Open price series
            high: High price series
            low: Low price series
            close: Close price series
            window: Rolling window size
            annualize: Whether to annualize
            trading_days: Number of trading days per year

        Returns:
            Series with Garman-Klass volatility values
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_price)
        
        # Garman-Klass formula
        term1 = 0.5 * (log_hl ** 2)
        term2 = (2 * np.log(2) - 1) * (log_co ** 2)
        
        gk_var = (term1 - term2).rolling(window=window).mean()
        vol = np.sqrt(gk_var.clip(lower=0))  # Ensure non-negative
        
        if annualize:
            vol = vol * np.sqrt(trading_days)
        
        return vol

    def calculate_yang_zhang_volatility(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        annualize: bool = True,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate Yang-Zhang volatility estimator.

        Combines overnight and intraday volatility for more accurate estimation.

        Args:
            open_price: Open price series
            high: High price series
            low: Low price series
            close: Close price series
            window: Rolling window size
            annualize: Whether to annualize
            trading_days: Number of trading days per year

        Returns:
            Series with Yang-Zhang volatility values
        """
        # Overnight volatility (close to open)
        log_oc = np.log(open_price / close.shift(1))
        overnight_var = log_oc.rolling(window=window).var()
        
        # Open to close volatility
        log_co = np.log(close / open_price)
        open_close_var = log_co.rolling(window=window).var()
        
        # Rogers-Satchell volatility
        log_ho = np.log(high / open_price)
        log_lo = np.log(low / open_price)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)
        
        rs_var = (log_ho * log_hc + log_lo * log_lc).rolling(window=window).mean()
        
        # Yang-Zhang combination
        if window < 2:
            raise ValueError("window must be at least 2 for Yang-Zhang volatility")
        
        # Yang-Zhang combination
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var
        
        vol = np.sqrt(yz_var.clip(lower=0))
        
        if annualize:
            vol = vol * np.sqrt(trading_days)
        
        return vol

    def calculate_ewma_volatility(
        self,
        returns: pd.Series,
        span: int = 20,
        annualize: bool = True,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) volatility.

        Args:
            returns: Return series
            span: EWMA span parameter
            annualize: Whether to annualize
            trading_days: Number of trading days per year

        Returns:
            Series with EWMA volatility values
        """
        vol = returns.ewm(span=span).std()
        
        if annualize:
            vol = vol * np.sqrt(trading_days)
        
        return vol

    def add_all_volatility_measures(
        self,
        df: pd.DataFrame,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Add all volatility measures to DataFrame.

        Args:
            df: DataFrame with OHLC data
            open_col: Column name for open prices
            high_col: Column name for high prices
            low_col: Column name for low prices
            close_col: Column name for close prices
            windows: List of window sizes

        Returns:
            DataFrame with volatility measures added
        """
        result = df.copy()
        
        if windows is None:
            windows = [5, 10, 20, 50]

        # Calculate returns
        close = df[close_col]
        returns = np.log(close / close.shift(1))

        for window in windows:
            # Realized volatility
            result[f"realized_vol_{window}"] = self.calculate_realized_volatility(
                returns, window=window
            )
            
            # Parkinson volatility
            if high_col in df.columns and low_col in df.columns:
                result[f"parkinson_vol_{window}"] = self.calculate_parkinson_volatility(
                    df[high_col], df[low_col], window=window
                )
            
            # Garman-Klass volatility
            if all(c in df.columns for c in [open_col, high_col, low_col]):
                result[f"gk_vol_{window}"] = self.calculate_garman_klass_volatility(
                    df[open_col], df[high_col], df[low_col], close, window=window
                )

        # EWMA volatility
        result["ewma_vol_20"] = self.calculate_ewma_volatility(returns, span=20)
        result["ewma_vol_60"] = self.calculate_ewma_volatility(returns, span=60)

        logger.info(f"Added {len(result.columns) - len(df.columns)} volatility measures")
        return result


class RegimeDetector:
    """Detect market regimes based on volatility and other indicators."""

    def __init__(
        self,
        low_vol_threshold: float = 0.10,
        high_vol_threshold: float = 0.25,
        crisis_threshold: float = 0.40
    ):
        """
        Initialize RegimeDetector with volatility thresholds.

        Args:
            low_vol_threshold: Annualized vol below this is low volatility
            high_vol_threshold: Annualized vol above this is high volatility
            crisis_threshold: Annualized vol above this is crisis
        """
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.crisis_threshold = crisis_threshold

    def detect_volatility_regime(
        self,
        volatility: pd.Series
    ) -> pd.Series:
        """
        Detect regime based on volatility levels.

        Args:
            volatility: Annualized volatility series

        Returns:
            Series with regime labels
        """
        conditions = [
            volatility >= self.crisis_threshold,
            volatility >= self.high_vol_threshold,
            volatility <= self.low_vol_threshold,
        ]
        choices = [
            MarketRegime.CRISIS.value,
            MarketRegime.HIGH_VOLATILITY.value,
            MarketRegime.LOW_VOLATILITY.value,
        ]
        
        regime = np.select(conditions, choices, default=MarketRegime.NORMAL.value)
        return pd.Series(regime, index=volatility.index, name="volatility_regime")

    def detect_trend_regime(
        self,
        prices: pd.Series,
        short_window: int = 20,
        long_window: int = 50
    ) -> pd.Series:
        """
        Detect trend regime based on moving average crossovers.

        Args:
            prices: Price series
            short_window: Short-term MA window
            long_window: Long-term MA window

        Returns:
            Series with trend regime labels ('uptrend', 'downtrend', 'sideways')
        """
        short_ma = prices.rolling(window=short_window).mean()
        long_ma = prices.rolling(window=long_window).mean()
        
        # Calculate MA difference as percentage
        ma_diff_pct = (short_ma - long_ma) / long_ma
        
        conditions = [
            ma_diff_pct > 0.02,   # Short MA significantly above long MA
            ma_diff_pct < -0.02,  # Short MA significantly below long MA
        ]
        choices = ["uptrend", "downtrend"]
        
        regime = np.select(conditions, choices, default="sideways")
        return pd.Series(regime, index=prices.index, name="trend_regime")

    def detect_momentum_regime(
        self,
        returns: pd.Series,
        window: int = 20,
        threshold: float = 0.5
    ) -> pd.Series:
        """
        Detect momentum regime based on cumulative returns.

        Args:
            returns: Return series
            window: Lookback window for cumulative returns
            threshold: Threshold for strong momentum (in std units)

        Returns:
            Series with momentum regime labels
        """
        cum_returns = returns.rolling(window=window).sum()
        cum_returns_std = cum_returns.rolling(window=window * 2).std()
        
        # Standardize cumulative returns
        z_score = cum_returns / cum_returns_std.replace(0, np.nan)
        
        conditions = [
            z_score > threshold,
            z_score < -threshold,
        ]
        choices = ["strong_positive", "strong_negative"]
        
        regime = np.select(conditions, choices, default="neutral")
        return pd.Series(regime, index=returns.index, name="momentum_regime")

    def calculate_regime_features(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
        volatility_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate all regime-related features.

        Args:
            df: DataFrame with price data
            close_col: Column name for close prices
            volatility_col: Column name for volatility (calculated if not provided)

        Returns:
            DataFrame with regime features added
        """
        result = df.copy()
        close = df[close_col]
        
        # Calculate returns
        returns = np.log(close / close.shift(1))
        
        # Calculate volatility if not provided
        if volatility_col and volatility_col in df.columns:
            volatility = df[volatility_col]
        else:
            vol_calc = VolatilityCalculator()
            volatility = vol_calc.calculate_realized_volatility(returns, window=20)
            result["realized_vol_20"] = volatility

        # Detect regimes
        result["volatility_regime"] = self.detect_volatility_regime(volatility)
        result["trend_regime"] = self.detect_trend_regime(close)
        result["momentum_regime"] = self.detect_momentum_regime(returns)
        
        # One-hot encode regimes for model input
        for regime_col in ["volatility_regime", "trend_regime", "momentum_regime"]:
            dummies = pd.get_dummies(result[regime_col], prefix=regime_col)
            result = pd.concat([result, dummies], axis=1)

        # Regime duration (consecutive days in same regime)
        result["vol_regime_duration"] = self._calculate_regime_duration(
            result["volatility_regime"]
        )
        result["trend_regime_duration"] = self._calculate_regime_duration(
            result["trend_regime"]
        )

        logger.info("Added regime detection features")
        return result

    def _calculate_regime_duration(self, regime_series: pd.Series) -> pd.Series:
        """Calculate consecutive days in the same regime."""
        # Create groups where regime changes
        regime_change = regime_series != regime_series.shift(1)
        regime_groups = regime_change.cumsum()
        
        # Count consecutive occurrences
        duration = regime_series.groupby(regime_groups).cumcount() + 1
        
        return duration

    def get_regime_statistics(
        self,
        df: pd.DataFrame,
        regime_col: str,
        return_col: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each regime.

        Args:
            df: DataFrame with regime and return data
            regime_col: Column name for regime labels
            return_col: Column name for returns

        Returns:
            Dictionary with statistics per regime
        """
        stats = {}
        
        for regime in df[regime_col].unique():
            regime_data = df[df[regime_col] == regime][return_col]
            
            stats[regime] = {
                "count": len(regime_data),
                "mean_return": regime_data.mean(),
                "std_return": regime_data.std(),
                "sharpe": (
                    regime_data.mean() / regime_data.std() * np.sqrt(252)
                    if regime_data.std() > 0 else 0
                ),
                "min_return": regime_data.min(),
                "max_return": regime_data.max(),
                "pct_positive": (regime_data > 0).mean(),
            }
        
        return stats
