"""Data preprocessing utilities for time series data."""

from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """Handles data preprocessing including missing values and outliers."""

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "forward_fill"
    ) -> pd.DataFrame:
        """
        Handle missing values using specified strategy.

        Args:
            df: DataFrame with potential missing values
            strategy: One of 'forward_fill', 'backward_fill', 'mean',
                     'median', 'interpolate'

        Returns:
            DataFrame with missing values handled

        Raises:
            ValueError: If strategy is unknown
        """
        result = df.copy()

        if strategy == "forward_fill":
            return result.ffill()
        elif strategy == "backward_fill":
            return result.bfill()
        elif strategy == "mean":
            return result.fillna(result.mean(numeric_only=True))
        elif strategy == "median":
            return result.fillna(result.median(numeric_only=True))
        elif strategy == "interpolate":
            return result.interpolate(method="linear")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        columns: Optional[List[str]] = None,
        threshold: float = 1.5
    ) -> List[int]:
        """
        Detect outliers using specified method.

        Args:
            df: DataFrame to check for outliers
            method: Detection method ('iqr' or 'zscore')
            columns: Columns to check (defaults to all numeric)
            threshold: Threshold for outlier detection
                      (1.5 for IQR, 3.0 for z-score)

        Returns:
            List of row indices containing outliers
        """
        if method not in ["iqr", "zscore"]:
            raise ValueError(f"Unknown method: {method}. Supported methods are: ['iqr', 'zscore']")


        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_indices: set = set()

        for col in columns:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if len(series) == 0:
                continue

            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_indices.update(df[mask].index.tolist())

            elif method == "zscore":
                mean = series.mean()
                std = series.std()
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    mask = z_scores > threshold
                    outlier_indices.update(df[mask].index.tolist())

        return list(outlier_indices)

    def treat_outliers(
        self,
        df: pd.DataFrame,
        method: str = "winsorize",
        columns: Optional[List[str]] = None,
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Treat outliers using specified method.

        Args:
            df: DataFrame with outliers
            method: Treatment method ('winsorize', 'clip', 'remove')
            columns: Columns to treat (defaults to all numeric)
            threshold: Threshold for outlier bounds (IQR multiplier)

        Returns:
            DataFrame with outliers treated
        """
        result = df.copy()

        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in result.columns:
                continue

            series = result[col].dropna()
            if len(series) == 0:
                continue

            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            if method == "winsorize":
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == "clip":
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == "remove":
                mask = (result[col] >= lower_bound) & (result[col] <= upper_bound)
                result = result[mask | result[col].isna()]

        return result

    def resample_timeseries(
        self,
        df: pd.DataFrame,
        freq: str,
        agg_method: str = "mean"
    ) -> pd.DataFrame:
        """
        Resample time series to specified frequency.

        Args:
            df: DataFrame with DatetimeIndex
            freq: Target frequency (e.g., '1D', '1H', '5min')
            agg_method: Aggregation method ('mean', 'sum', 'last', 'first')

        Returns:
            Resampled DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for resampling")

        resampler = df.resample(freq)

        if agg_method == "mean":
            return resampler.mean()
        elif agg_method == "sum":
            return resampler.sum()
        elif agg_method == "last":
            return resampler.last()
        elif agg_method == "first":
            return resampler.first()
        elif agg_method == "ohlc":
            # For OHLCV data, use appropriate aggregations
            agg_map = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Build the actual aggregation dictionary based on what's in df
            final_agg = {}
            for col in df.columns:
                # Check case-insensitive match for OHLC columns if standard names aren't guaranteed,
                # but user specified keys 'open', 'high' etc. Let's stick to exact keys first 
                # or maybe safer to handle them purely as requested. 
                # The user prompt: "sets 'open'->'first', 'high'->'max'..." 
                # so I will assume strict keys but will check if `col` is in `agg_map`.
                if col in agg_map:
                    final_agg[col] = agg_map[col]
                elif pd.api.types.is_numeric_dtype(df[col]):
                    final_agg[col] = "mean"
            
            return resampler.agg(final_agg)
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")

    def get_preprocessing_summary(
        self,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate summary of preprocessing changes.

        Args:
            original_df: Original DataFrame before preprocessing
            processed_df: DataFrame after preprocessing

        Returns:
            Dictionary with preprocessing summary
        """
        summary = {
            "original_rows": len(original_df),
            "processed_rows": len(processed_df),
            "rows_removed": len(original_df) - len(processed_df),
            "original_nulls": int(original_df.isna().sum().sum()),
            "processed_nulls": int(processed_df.isna().sum().sum()),
            "nulls_filled": int(
                original_df.isna().sum().sum() - processed_df.isna().sum().sum()
            ),
        }
        return summary
