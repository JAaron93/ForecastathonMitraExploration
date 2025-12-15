"""Core feature engineering utilities for time series data.

Provides lag features, rolling statistics, calendar features, and cross-asset
calculations for financial time series forecasting.
"""

from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinitions:
    """Container for feature definitions and metadata."""
    lag_features: List[str]
    rolling_features: List[str]
    calendar_features: List[str]
    cross_asset_features: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lag_features": self.lag_features,
            "rolling_features": self.rolling_features,
            "calendar_features": self.calendar_features,
            "cross_asset_features": self.cross_asset_features,
            "metadata": self.metadata,
        }


class FeatureEngineer:
    """Handles feature engineering for time series data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize FeatureEngineer with optional configuration."""
        self.config = config or {}
        self._feature_definitions: Optional[FeatureDefinitions] = None

    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create lag features for specified columns.

        Args:
            df: DataFrame with time series data
            columns: Columns to create lags for (defaults to all numeric)
            lags: List of lag periods (e.g., [1, 2, 3, 5, 10])

        Returns:
            DataFrame with original and lag features
        """
        result = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if lags is None:
            lags = self.config.get("lag_periods", [1, 2, 3, 5, 10, 20])

        lag_feature_names = []
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            for lag in lags:
                feature_name = f"{col}_lag_{lag}"
                result[feature_name] = df[col].shift(lag)
                lag_feature_names.append(feature_name)

        logger.info(f"Created {len(lag_feature_names)} lag features")
        return result

    def calculate_rolling_stats(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
        stats: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling statistics for specified columns.

        Args:
            df: DataFrame with time series data
            columns: Columns to calculate stats for (defaults to all numeric)
            windows: List of window sizes (e.g., [5, 10, 20, 50])
            stats: Statistics to calculate ('mean', 'std', 'min', 'max', 
                   'quantile_25', 'quantile_75', 'median')

        Returns:
            DataFrame with original and rolling statistics
        """
        result = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if windows is None:
            windows = self.config.get("rolling_windows", [5, 10, 20, 50])
        
        if stats is None:
            stats = ["mean", "std", "min", "max"]

        rolling_feature_names = []
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            for window in windows:
                rolling = df[col].rolling(window=window, min_periods=1)
                
                for stat in stats:
                    feature_name = f"{col}_rolling_{stat}_{window}"
                    
                    if stat == "mean":
                        result[feature_name] = rolling.mean()
                    elif stat == "std":
                        result[feature_name] = rolling.std()
                    elif stat == "min":
                        result[feature_name] = rolling.min()
                    elif stat == "max":
                        result[feature_name] = rolling.max()
                    elif stat == "median":
                        result[feature_name] = rolling.median()
                    elif stat == "quantile_25":
                        result[feature_name] = rolling.quantile(0.25)
                    elif stat == "quantile_75":
                        result[feature_name] = rolling.quantile(0.75)
                    elif stat == "sum":
                        result[feature_name] = rolling.sum()
                    else:
                        logger.warning(f"Unknown statistic: {stat}")
                        continue
                    
                    rolling_feature_names.append(feature_name)

        logger.info(f"Created {len(rolling_feature_names)} rolling features")
        return result

    def create_calendar_features(
        self,
        df: pd.DataFrame,
        include_holidays: bool = True,
        holiday_country: str = "US"
    ) -> pd.DataFrame:
        """
        Create calendar-based features from DatetimeIndex.

        Args:
            df: DataFrame with DatetimeIndex
            include_holidays: Whether to include holiday indicators
            holiday_country: Country code for holiday calendar

        Returns:
            DataFrame with calendar features added
        """
        result = df.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Day of week (0=Monday, 6=Sunday)
        result["day_of_week"] = df.index.dayofweek
        
        # Day of month
        result["day_of_month"] = df.index.day
        
        # Month
        result["month"] = df.index.month
        
        # Quarter
        result["quarter"] = df.index.quarter
        
        # Year
        result["year"] = df.index.year
        
        # Week of year
        result["week_of_year"] = df.index.isocalendar().week.values
        
        # Is weekend
        result["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        
        # Is month start/end
        result["is_month_start"] = df.index.is_month_start.astype(int)
        result["is_month_end"] = df.index.is_month_end.astype(int)
        
        # Is quarter start/end
        result["is_quarter_start"] = df.index.is_quarter_start.astype(int)
        result["is_quarter_end"] = df.index.is_quarter_end.astype(int)

        # Cyclical encoding for day of week
        result["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        result["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Cyclical encoding for month
        result["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        result["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

        if include_holidays:
            result = self._add_holiday_features(result, holiday_country)

        logger.info("Created calendar features")
        return result

    def _add_holiday_features(
        self,
        df: pd.DataFrame,
        country: str = "US"
    ) -> pd.DataFrame:
        """Add holiday indicator features."""
        result = df.copy()
        
        # Simple holiday detection (major US holidays)
        # For production, use a proper holiday library
        dates = df.index
        
        # Initialize holiday column
        result["is_holiday"] = 0
        
        # Check for common holidays (simplified)
        for date in dates:
            # New Year's Day
            if date.month == 1 and date.day == 1:
                result.loc[date, "is_holiday"] = 1
            # Independence Day (US)
            elif date.month == 7 and date.day == 4:
                result.loc[date, "is_holiday"] = 1
            # Christmas
            elif date.month == 12 and date.day == 25:
                result.loc[date, "is_holiday"] = 1
            # Thanksgiving (4th Thursday of November - approximation)
            elif date.month == 11 and date.weekday() == 3 and 22 <= date.day <= 28:
                result.loc[date, "is_holiday"] = 1

        # Days until/since holiday (simplified - just weekend proximity)
        result["days_to_weekend"] = (5 - df.index.dayofweek) % 7
        result["days_from_weekend"] = df.index.dayofweek % 7

        return result

    def calculate_cross_asset_features(
        self,
        dfs: Dict[str, pd.DataFrame],
        price_column: str = "close",
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate cross-asset correlation and spread features.

        Args:
            dfs: Dictionary of DataFrames keyed by asset name
            price_column: Column name for price data
            windows: Rolling windows for correlation calculation

        Returns:
            DataFrame with cross-asset features
        """
        if len(dfs) < 2:
            raise ValueError("Need at least 2 assets for cross-asset features")
        
        if windows is None:
            windows = self.config.get("rolling_windows", [5, 10, 20, 50])

        # Align all DataFrames to common index
        asset_names = list(dfs.keys())
        aligned_prices = pd.DataFrame()
        
        for name, df in dfs.items():
            if price_column in df.columns:
                aligned_prices[name] = df[price_column]
            else:
                logger.warning(f"Price column {price_column} not found in {name}")

        # Forward fill to handle missing data
        aligned_prices = aligned_prices.ffill()
        
        result = aligned_prices.copy()

        # Calculate pairwise correlations
        for i, asset1 in enumerate(asset_names):
            for asset2 in asset_names[i+1:]:
                if asset1 not in aligned_prices.columns:
                    continue
                if asset2 not in aligned_prices.columns:
                    continue
                    
                # Spread (difference)
                result[f"{asset1}_{asset2}_spread"] = (
                    aligned_prices[asset1] - aligned_prices[asset2]
                )
                
                # Ratio
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = aligned_prices[asset1] / aligned_prices[asset2]
                    ratio = ratio.replace([np.inf, -np.inf], np.nan)
                    result[f"{asset1}_{asset2}_ratio"] = ratio
                
                # Rolling correlations
                for window in windows:
                    corr = aligned_prices[asset1].rolling(window).corr(
                        aligned_prices[asset2]
                    )
                    result[f"{asset1}_{asset2}_corr_{window}"] = corr

        logger.info(f"Created cross-asset features for {len(asset_names)} assets")
        return result

    def calculate_returns(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        periods: Optional[List[int]] = None,
        log_returns: bool = True
    ) -> pd.DataFrame:
        """
        Calculate returns for specified columns.

        Args:
            df: DataFrame with price data
            columns: Columns to calculate returns for
            periods: Return periods (e.g., [1, 5, 10] for 1-day, 5-day returns)
            log_returns: Whether to calculate log returns (True) or simple (False)

        Returns:
            DataFrame with return features added
        """
        result = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if periods is None:
            periods = [1, 5, 10, 20]

        for col in columns:
            if col not in df.columns:
                continue
            
            for period in periods:
                if log_returns:
                    # Log returns: ln(P_t / P_{t-n})
                    with np.errstate(divide='ignore', invalid='ignore'):
                        returns = np.log(df[col] / df[col].shift(period))
                        returns = returns.replace([np.inf, -np.inf], np.nan)
                    feature_name = f"{col}_log_return_{period}"
                else:
                    # Simple returns: (P_t - P_{t-n}) / P_{t-n}
                    returns = df[col].pct_change(periods=period)
                    feature_name = f"{col}_return_{period}"
                
                result[feature_name] = returns

        return result

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        price_columns: Optional[List[str]] = None,
        volume_columns: Optional[List[str]] = None,
        include_lags: bool = True,
        include_rolling: bool = True,
        include_calendar: bool = True,
        include_returns: bool = True
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Args:
            df: DataFrame with time series data
            price_columns: Columns to use for price-based features (returns)
            volume_columns: Columns to use for volume-based features (no returns)
            include_lags: Whether to include lag features
            include_rolling: Whether to include rolling statistics
            include_calendar: Whether to include calendar features
            include_returns: Whether to include return features

        Returns:
            DataFrame with all engineered features
        """
        result = df.copy()
        
        if price_columns is None:
            price_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        if volume_columns is None:
            volume_columns = []
            
        # Combine for operations that apply to both
        all_numeric_columns = list(set(price_columns + volume_columns))

        if include_returns:
            result = self.calculate_returns(result, columns=price_columns)
        
        if include_lags:
            result = self.create_lag_features(result, columns=all_numeric_columns)
        
        if include_rolling:
            result = self.calculate_rolling_stats(result, columns=all_numeric_columns)
        
        if include_calendar and isinstance(df.index, pd.DatetimeIndex):
            result = self.create_calendar_features(result)

        # Store feature definitions
        self._feature_definitions = self._create_feature_definitions(
            df, result, price_columns
        )

        logger.info(f"Total features created: {len(result.columns) - len(df.columns)}")
        return result

    def _create_feature_definitions(
        self,
        original_df: pd.DataFrame,
        result_df: pd.DataFrame,
        price_columns: List[str]
    ) -> FeatureDefinitions:
        """Create feature definitions metadata."""
        new_cols = set(result_df.columns) - set(original_df.columns)
        
        lag_features = [c for c in new_cols if "_lag_" in c]
        rolling_features = [c for c in new_cols if "_rolling_" in c]
        calendar_features = [
            c for c in new_cols 
            if c in [
                "day_of_week", "day_of_month", "month", "quarter", "year",
                "week_of_year", "is_weekend", "is_month_start", "is_month_end",
                "is_quarter_start", "is_quarter_end", "day_of_week_sin",
                "day_of_week_cos", "month_sin", "month_cos", "is_holiday",
                "days_to_weekend", "days_from_weekend"
            ]
        ]
        cross_asset_features = [
            c for c in new_cols 
            if "_spread" in c or "_ratio" in c or "_corr_" in c
        ]

        return FeatureDefinitions(
            lag_features=lag_features,
            rolling_features=rolling_features,
            calendar_features=calendar_features,
            cross_asset_features=cross_asset_features,
            metadata={
                "original_columns": list(original_df.columns),
                "price_columns": price_columns,
                "total_features": len(result_df.columns),
                "new_features": len(new_cols),
            }
        )

    def get_feature_definitions(self) -> Optional[FeatureDefinitions]:
        """Get the feature definitions from the last engineering run."""
        return self._feature_definitions
