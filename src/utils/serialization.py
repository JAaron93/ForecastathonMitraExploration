"""
Serialization utilities for the forecasting pipeline.
Handles JSON, Parquet, Pickle, and custom struct serialization.
"""

import hmac
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

import joblib
import numpy as np
import pandas as pd

from src.data.structs import TimeSeriesData

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime and numpy types."""

    def default(self, o):
        if isinstance(o, (datetime, pd.Timestamp)):
            return o.isoformat()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(data: Any, path: Union[str, Path], **kwargs) -> None:
    """Save data to JSON with datetime support."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=DateTimeEncoder, indent=2, **kwargs)
    logger.debug(f"Saved JSON to {path}")


def load_json(
    path: Union[str, Path],
) -> Union[Dict[str, Any], List[Any], str, int, float, bool, None]:
    """Load data from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def save_joblib(obj: Any, path: Union[str, Path]) -> None:
    """Save object to joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.debug(f"Saved joblib to {path}")


T = TypeVar("T")


def load_joblib(
    path: Union[str, Path],
    signature: Optional[str] = None,
    hmac_key: Optional[bytes] = None,
) -> T:
    """
    Load object from joblib with optional HMAC signature verification.

    The TypeVar `T` allows for generic typing, but its inference behavior varies by
    static type checker. Some may infer it as `object`, others as `Any` when context
    doesn't provide enough information. For precise typing, provide an explicit type
    hint (e.g., `model: MyType = load_joblib(...)`) or use overload signatures for
    common types.

    Args:
        path: Path to the joblib file
        signature: Optional HMAC signature to verify (hex string)
        hmac_key: Optional HMAC key for verification (bytes)

    Returns:
        Deserialized object

    Raises:
        ValueError: If signature verification fails
        FileNotFoundError: If file doesn't exist
        Exception: For other deserialization errors
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Joblib file not found: {path}")

    # Read file contents once for both verification and loading
    with open(path, "rb") as f:
        file_data = f.read()

    # If signature verification is requested
    if signature is not None or hmac_key is not None:
        if signature is None or hmac_key is None:
            raise ValueError(
                "Both 'signature' and 'hmac_key' must be provided for verification"
            )

        # Verify HMAC
        expected_signature = hmac.new(hmac_key, file_data, "sha256").hexdigest()
        if not hmac.compare_digest(expected_signature, signature):
            raise ValueError(f"HMAC signature verification failed for {path}")

    # If no signature verification or verification passed, proceed with loading from in-memory data
    try:
        return joblib.load(io.BytesIO(file_data))
    except Exception as e:
        raise ValueError(f"Failed to load joblib file: {str(e)}")


def save_parquet(
    df: Union[pd.DataFrame, pd.Series], path: Union[str, Path], **kwargs
) -> None:
    """Save DataFrame or Series to Parquet."""
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError(f"Expected pandas DataFrame or Series, got {type(df)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, **kwargs)
    logger.debug(f"Saved Parquet to {path}")


def load_parquet(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load DataFrame from Parquet."""
    return pd.read_parquet(path, **kwargs)


def save_timeseries_data(data: TimeSeriesData, path: Union[str, Path]) -> None:
    """
    Save TimeSeriesData to a directory structure.
    """
    if not isinstance(data, TimeSeriesData):
        raise TypeError(f"Expected TimeSeriesData object, got {type(data)}")

    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_parquet(data.features, save_dir / "features.parquet")

    # Handle targets (Series or DataFrame)
    target_meta = {"is_series": False, "name": None}

    if isinstance(data.targets, pd.Series):
        # Preserve original name (which can be None), but use safe
        # string for Parquet column
        original_name = data.targets.name
        # JSON serializer handles None -> null. If name is complex
        # object, use str()
        # But for exact preservation, we assume it's JSON-compatible
        # or None for now.
        # If it's a number, JSON is fine.
        target_meta = {"is_series": True, "name": original_name}

        target_col_name = str(original_name) if original_name is not None else "target"
        data.targets.to_frame(name=target_col_name).to_parquet(
            save_dir / "targets.parquet"
        )
    else:
        save_parquet(data.targets, save_dir / "targets.parquet")

    save_json(target_meta, save_dir / "target_meta.json")

    # Save timestamp
    pd.DataFrame({"timestamp": data.timestamp}).to_parquet(
        save_dir / "timestamp.parquet"
    )

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

    # Load target metadata if exists (backward compatibility)
    target_meta_path = load_dir / "target_meta.json"
    if target_meta_path.exists():
        target_meta = load_json(target_meta_path)
        # Ensure target_meta is a dictionary for safe .get() access
        if not isinstance(target_meta, dict):
            target_meta = {
                "is_series": len(targets_df.columns) == 1
                and targets_df.columns[0] == "target",
                "name": "target",
            }
    else:
        # Fallback heuristic
        target_meta = {
            "is_series": len(targets_df.columns) == 1
            and targets_df.columns[0] == "target",
            "name": "target",
        }

    if target_meta.get("is_series"):
        original_name = target_meta.get("name")

        # Determine likely column name in Parquet
        # If original_name is None, we saved it as "target"
        expected_col = str(original_name) if original_name is not None else "target"

        if expected_col in targets_df.columns:
            targets = targets_df[expected_col]
        elif "target" in targets_df.columns:
            targets = targets_df["target"]
        else:
            # Fallback: take first column
            targets = targets_df.iloc[:, 0]

        targets.name = original_name
    else:
        targets = targets_df

    timestamp_df = load_parquet(load_dir / "timestamp.parquet")
    timestamp = pd.DatetimeIndex(timestamp_df["timestamp"])

    metadata = load_json(load_dir / "metadata.json")
    # Ensure metadata is a dict for TimeSeriesData
    if not isinstance(metadata, dict):
        metadata = {}

    split_indices = load_json(load_dir / "split_indices.json")
    # Ensure split_indices is a dict of str to list of ints for TimeSeriesData
    if not isinstance(split_indices, dict):
        split_indices = {}
    else:
        # Filter to only include entries with str keys and list-like of int-like values
        filtered_split_indices = {}
        for key, value in split_indices.items():
            # Check key type
            if not isinstance(key, str):
                logger.debug(
                    f"Filtered invalid split_indices entry: key {key!r} is not a string "
                    f"(type: {type(key).__name__})"
                )
                continue

            # Check value type (allow list or np.ndarray)
            if not isinstance(value, (list, np.ndarray)):
                logger.debug(
                    f"Filtered invalid split_indices entry for key {key!r}: "
                    f"value is not a list or ndarray (type: {type(value).__name__})"
                )
                continue

            # Convert to list for consistent internal representation if it's an ndarray
            if isinstance(value, np.ndarray):
                value = value.tolist()

            # Verify all elements are integer-like
            if all(isinstance(i, (int, np.integer)) for i in value):
                filtered_split_indices[key] = value
            else:
                invalid_types = {
                    type(i).__name__
                    for i in value
                    if not isinstance(i, (int, np.integer))
                }
                logger.debug(
                    f"Filtered invalid split_indices entry for key {key!r}: "
                    f"contains non-integer types ({', '.join(invalid_types)})"
                )
        split_indices = filtered_split_indices

    return TimeSeriesData(
        timestamp=timestamp,
        features=features,
        targets=targets,
        metadata=metadata,
        split_indices=split_indices,
    )
