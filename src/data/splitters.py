"""Time series alignment and splitting utilities."""

from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SplitIndices:
    """Container for train/validation/test split indices with metadata."""
    train_indices: List[int]
    validation_indices: List[int]
    test_indices: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "train_indices": self.train_indices,
            "validation_indices": self.validation_indices,
            "test_indices": self.test_indices,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitIndices":
        """Create from dictionary."""
        return cls(
            train_indices=data["train_indices"],
            validation_indices=data["validation_indices"],
            test_indices=data["test_indices"],
            metadata=data.get("metadata", {}),
        )


class TimeSeriesAligner:
    """Aligns multiple time series to a consistent time grid."""

    def align_timeseries(
        self,
        dataframes: List[pd.DataFrame],
        freq: Optional[str] = None,
        method: str = "outer"
    ) -> pd.DataFrame:
        """
        Align multiple time series DataFrames to a consistent time grid.

        Args:
            dataframes: List of DataFrames with DatetimeIndex
            freq: Target frequency (inferred if None)
            method: Join method ('outer', 'inner')

        Returns:
            Single aligned DataFrame with all columns
        """
        if not dataframes:
            raise ValueError("No dataframes provided for alignment")

        # Validate all have DatetimeIndex
        for i, df in enumerate(dataframes):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    f"DataFrame {i} must have DatetimeIndex for alignment"
                )

        # Infer frequency if not provided
        if freq is None:
            freq = self._infer_frequency(dataframes)

        # Create common time grid
        if method == "outer":
            min_time = min(df.index.min() for df in dataframes)
            max_time = max(df.index.max() for df in dataframes)
        else:  # inner
            min_time = max(df.index.min() for df in dataframes)
            max_time = min(df.index.max() for df in dataframes)

        common_index = pd.date_range(start=min_time, end=max_time, freq=freq)

        # Reindex all dataframes to common grid
        aligned_dfs = []
        for df in dataframes:
            # Resample to target frequency first
            resampled = df.resample(freq).mean()
            # Reindex to common grid
            aligned = resampled.reindex(common_index)
            aligned_dfs.append(aligned)

        # Concatenate all aligned dataframes
        result = pd.concat(aligned_dfs, axis=1)

        # Handle duplicate column names
        if result.columns.duplicated().any():
            result.columns = self._make_unique_columns(result.columns)

        return result

    def _infer_frequency(self, dataframes: List[pd.DataFrame]) -> str:
        """Infer the most common frequency from dataframes."""
        freqs = []
        for df in dataframes:
            if len(df) > 1:
                inferred = pd.infer_freq(df.index)
                if inferred:
                    freqs.append(inferred)

        if freqs:
            # Return most common frequency
            from collections import Counter
            return Counter(freqs).most_common(1)[0][0]

        # Default to daily
        return "D"

    def _make_unique_columns(self, columns: pd.Index) -> List[str]:
        """Make column names unique by adding suffixes."""
        seen: Dict[str, int] = {}
        result = []
        for col in columns:
            if col in seen:
                seen[col] += 1
                result.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                result.append(col)
        return result


class TimeSeriesSplitter:
    """Time-series aware train/validation/test splitting."""

    def __init__(self, save_dir: Optional[str] = None):
        """Initialize splitter with optional save directory."""
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_val_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> SplitIndices:
        """
        Split time series data into train/validation/test sets.

        Ensures temporal ordering is preserved - no future data leaks.

        Args:
            df: DataFrame with DatetimeIndex (must be sorted)
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing

        Returns:
            SplitIndices with indices for each split
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Get indices (preserving original index values)
        indices = list(range(n))
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        metadata = {
            "split_type": "sequential",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "total_samples": n,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "test_samples": len(test_indices),
            "created_at": datetime.now().isoformat(),
        }

        if isinstance(df.index, pd.DatetimeIndex):
            metadata["train_start"] = str(df.index[train_indices[0]])
            metadata["train_end"] = str(df.index[train_indices[-1]])
            metadata["val_start"] = str(df.index[val_indices[0]])
            metadata["val_end"] = str(df.index[val_indices[-1]])
            metadata["test_start"] = str(df.index[test_indices[0]])
            metadata["test_end"] = str(df.index[test_indices[-1]])

        return SplitIndices(
            train_indices=train_indices,
            validation_indices=val_indices,
            test_indices=test_indices,
            metadata=metadata,
        )

    def rolling_window_split(
        self,
        df: pd.DataFrame,
        train_size: int,
        val_size: int,
        test_size: int,
        step_size: int = 1
    ) -> Iterator[SplitIndices]:
        """
        Generate rolling window splits for time series cross-validation.

        Args:
            df: DataFrame to split
            train_size: Number of samples in training window
            val_size: Number of samples in validation window
            test_size: Number of samples in test window
            step_size: Step size between windows

        Yields:
            SplitIndices for each window position
        """
        n = len(df)
        window_size = train_size + val_size + test_size

        if window_size > n:
            raise ValueError(
                f"Window size ({window_size}) exceeds data length ({n})"
            )

        fold = 0
        for start in range(0, n - window_size + 1, step_size):
            train_end = start + train_size
            val_end = train_end + val_size
            test_end = val_end + test_size

            train_indices = list(range(start, train_end))
            val_indices = list(range(train_end, val_end))
            test_indices = list(range(val_end, test_end))

            metadata = {
                "split_type": "rolling_window",
                "fold": fold,
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
                "step_size": step_size,
                "window_start": start,
                "created_at": datetime.now().isoformat(),
            }

            yield SplitIndices(
                train_indices=train_indices,
                validation_indices=val_indices,
                test_indices=test_indices,
                metadata=metadata,
            )
            fold += 1

    def expanding_window_split(
        self,
        df: pd.DataFrame,
        initial_train_size: int,
        val_size: int,
        test_size: int,
        step_size: int = 1
    ) -> Iterator[SplitIndices]:
        """
        Generate expanding window splits for time series cross-validation.

        Training window grows with each fold while val/test remain fixed.

        Args:
            df: DataFrame to split
            initial_train_size: Initial training window size
            val_size: Number of samples in validation window
            test_size: Number of samples in test window
            step_size: Step size between windows

        Yields:
            SplitIndices for each window position
        """
        n = len(df)
        min_size = initial_train_size + val_size + test_size

        if min_size > n:
            raise ValueError(
                f"Minimum window size ({min_size}) exceeds data length ({n})"
            )

        fold = 0
        train_end = initial_train_size

        while train_end + val_size + test_size <= n:
            val_end = train_end + val_size
            test_end = val_end + test_size

            train_indices = list(range(0, train_end))
            val_indices = list(range(train_end, val_end))
            test_indices = list(range(val_end, test_end))

            metadata = {
                "split_type": "expanding_window",
                "fold": fold,
                "train_size": len(train_indices),
                "val_size": val_size,
                "test_size": test_size,
                "step_size": step_size,
                "created_at": datetime.now().isoformat(),
            }

            yield SplitIndices(
                train_indices=train_indices,
                validation_indices=val_indices,
                test_indices=test_indices,
                metadata=metadata,
            )

            train_end += step_size
            fold += 1

    def save_split_indices(
        self,
        split: SplitIndices,
        name: str
    ) -> Path:
        """
        Save split indices to JSON file.

        Args:
            split: SplitIndices to save
            name: Name for the split file

        Returns:
            Path to saved file
        """
        if not self.save_dir:
            raise ValueError("No save directory configured")

        file_path = self.save_dir / f"{name}_split.json"
        with open(file_path, "w") as f:
            json.dump(split.to_dict(), f, indent=2)

        logger.info(f"Split indices saved to {file_path}")
        return file_path

    def load_split_indices(self, name: str) -> SplitIndices:
        """
        Load split indices from JSON file.

        Args:
            name: Name of the split file

        Returns:
            SplitIndices loaded from file
        """
        if not self.save_dir:
            raise ValueError("No save directory configured")

        file_path = self.save_dir / f"{name}_split.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        return SplitIndices.from_dict(data)

    def apply_split(
        self,
        df: pd.DataFrame,
        split: SplitIndices
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply split indices to get train/val/test DataFrames.

        Args:
            df: DataFrame to split
            split: SplitIndices with indices

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df = df.iloc[split.train_indices]
        val_df = df.iloc[split.validation_indices]
        test_df = df.iloc[split.test_indices]

        return train_df, val_df, test_df

    def validate_no_leakage(
        self,
        df: pd.DataFrame,
        split: SplitIndices
    ) -> Tuple[bool, List[str]]:
        """
        Validate that split has no temporal data leakage.

        Args:
            df: DataFrame with DatetimeIndex
            split: SplitIndices to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues: List[str] = []

        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("DataFrame does not have DatetimeIndex")
            return False, issues

        # Get timestamps for each split
        train_times = df.index[split.train_indices]
        val_times = df.index[split.validation_indices]
        test_times = df.index[split.test_indices]

        # Check train < val
        if len(train_times) > 0 and len(val_times) > 0:
            if train_times.max() >= val_times.min():
                issues.append(
                    f"Training data ({train_times.max()}) overlaps with "
                    f"validation data ({val_times.min()})"
                )

        # Check val < test
        if len(val_times) > 0 and len(test_times) > 0:
            if val_times.max() >= test_times.min():
                issues.append(
                    f"Validation data ({val_times.max()}) overlaps with "
                    f"test data ({test_times.min()})"
                )

        # Check train < test
        if len(train_times) > 0 and len(test_times) > 0:
            if train_times.max() >= test_times.min():
                issues.append(
                    f"Training data ({train_times.max()}) overlaps with "
                    f"test data ({test_times.min()})"
                )

        is_valid = len(issues) == 0
        return is_valid, issues
