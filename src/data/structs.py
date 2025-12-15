"""Core data structures for the forecasting pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd

@dataclass
class TimeSeriesData:
    """
    Container for time series data with metadata and split information.
    
    Attributes:
        timestamp: DatetimeIndex for the time series
        features: DataFrame of features
        targets: Series or DataFrame of targets
        metadata: Dictionary of metadata (frequency, source, etc.)
        split_indices: Dictionary of split indices (train/val/test)
    """
    timestamp: pd.DatetimeIndex
    features: pd.DataFrame
    targets: Any  # pd.Series or pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    split_indices: Dict[str, List[int]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate consistency after initialization."""
        if len(self.timestamp) != len(self.features):
            raise ValueError(
                f"Length mismatch: timestamp ({len(self.timestamp)}) vs features ({len(self.features)})"
            )
        if len(self.timestamp) != len(self.targets):
            raise ValueError(
                f"Length mismatch: timestamp ({len(self.timestamp)}) vs targets ({len(self.targets)})"
            )
