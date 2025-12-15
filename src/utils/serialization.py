"""
Serialization utilities for the forecasting pipeline.
Handles JSON, Parquet, Pickle, and custom struct serialization.
"""

import json
import pickle
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from src.data.structs import TimeSeriesData

logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime and numpy types."""
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_json(data: Any, path: Union[str, Path], **kwargs) -> None:
    """Save data to JSON with datetime support."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, cls=DateTimeEncoder, indent=2, **kwargs)
    logger.debug(f"Saved JSON to {path}")

def load_json(path: Union[str, Path]) -> Any:
    """Load data from JSON."""
    with open(path, 'r') as f:
        return json.load(f)

def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Save object to pickle."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    logger.debug(f"Saved pickle to {path}")

def load_pickle(path: Union[str, Path]) -> Any:
    """Load object from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """Save DataFrame to Parquet."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, **kwargs)
    logger.debug(f"Saved Parquet to {path}")

def load_parquet(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load DataFrame from Parquet."""
    return pd.read_parquet(path, **kwargs)

def save_timeseries_data(data: TimeSeriesData, path: Union[str, Path]) -> None:
    """
    Save TimeSeriesData to a directory structure.
    
    Structure:
    - path/
        - features.parquet
        - targets.parquet
        - timestamp.parquet (as a single column df)
        - metadata.json
        - split_indices.json
    """
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_parquet(data.features, save_dir / "features.parquet")
    
    # Handle targets (Series or DataFrame)
    if isinstance(data.targets, pd.Series):
        data.targets.to_frame(name="target").to_parquet(save_dir / "targets.parquet")
    else:
        save_parquet(data.targets, save_dir / "targets.parquet")
        
    # Save timestamp
    pd.DataFrame({"timestamp": data.timestamp}).to_parquet(save_dir / "timestamp.parquet")
    
    save_json(data.metadata, save_dir / "metadata.json")
    save_json(data.split_indices, save_dir / "split_indices.json")
    
    logger.info(f"Saved TimeSeriesData to {save_dir}")

def load_timeseries_data(path: Union[str, Path]) -> TimeSeriesData:
    """Load TimeSeriesData from a directory."""
    load_dir = Path(path)
    if not load_dir.exists():
        raise FileNotFoundError(f"TimeSeriesData directory not found: {path}")

    features = load_parquet(load_dir / "features.parquet")
    targets_df = load_parquet(load_dir / "targets.parquet")
    
    # If standard name 'target' and single column, convert to Series? 
    # For robust loading, if we saved a series as 'target', we'll get a DF with 'target' col.
    # Let's verify if original was series or DF from metadata? 
    # MVP: Return DataFrame if multiclass, Series if single 'target' column?
    # Simple Heuristic: If 1 col named 'target', convert to series.
    if len(targets_df.columns) == 1 and targets_df.columns[0] == "target":
        targets = targets_df["target"]
    else:
        targets = targets_df

    timestamp_df = load_parquet(load_dir / "timestamp.parquet")
    timestamp = pd.DatetimeIndex(timestamp_df["timestamp"])
    
    metadata = load_json(load_dir / "metadata.json")
    split_indices = load_json(load_dir / "split_indices.json")
    
    return TimeSeriesData(
        timestamp=timestamp,
        features=features,
        targets=targets,
        metadata=metadata,
        split_indices=split_indices
    )
