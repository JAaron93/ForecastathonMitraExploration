"""Data loading utilities for Parquet files and schema validation."""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    schema_violations: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "schema_violations": self.schema_violations,
        }


class DataLoader:
    """Handles loading data from various sources with schema validation."""

    def __init__(self, log_dir: Optional[str] = None):
        """Initialize DataLoader with optional logging directory."""
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def load_parquet(
        self,
        path: str,
        schema: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Load data from a Parquet file with optional schema validation.

        Args:
            path: Path to the Parquet file
            schema: Expected schema as {column_name: dtype_string}

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If schema validation fails
            pd.errors.ParserError: If there's an error parsing the Parquet file
            OSError: If there's an OS-level error reading the file
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        try:
            df = pd.read_parquet(path)
            logger.info(f"Loaded {len(df)} rows from {path}")
        except pd.errors.ParserError as e:
            error_msg = f"Error parsing Parquet file {path}: {e}"
            raise pd.errors.ParserError(error_msg) from e
        except OSError as e:
            error_msg = f"OS error reading Parquet file {path}: {e}"
            raise OSError(error_msg) from e

        if schema:
            result = self.validate_schema(df, schema)
            if not result.is_valid:
                error_msg = "; ".join(result.errors)
                raise ValueError(f"Schema validation failed: {error_msg}")

        return df

    def load_assets(
        self,
        base_path: str,
        assets: List[str],
        schema: Optional[Dict[str, str]] = None,
        extension: str = "parquet",
        strict: bool = False,
        status: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple assets from a base directory.

        Args:
            base_path: Base directory containing asset files
            assets: List of asset names (file names without extension)
            schema: Expected schema for all assets
            extension: File extension (default: "parquet")
            strict: If True, any error will abort loading and re-raise.
                   If False, errors are logged and loading continues.
            status: Optional dictionary to store loading status/metadata.
                   If provided, it must be a dictionary.
                   - 'loaded_assets' (if exists) must be a list.
                   - 'failed_assets' (if exists) must be a dict.
                   These keys will be initialized if they don't exist.

        Returns:
            Dictionary mapping asset name to its DataFrame. If strict is False,
            this may contain only a subset of requested assets.

        Raises:
            FileNotFoundError: If base_dir doesn't exist, or if strict=True
                              and an asset file is missing.
            ValueError: If strict=True and schema validation fails.
            Exception: Any other error if strict=True.
        """
        base_dir = Path(base_path)
        if not base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {base_path}")

        loaded_assets = {}
        if status is not None:
            if not isinstance(status, dict):
                raise TypeError(
                    f"status must be a dict, got {type(status).__name__}"
                )

            if "loaded_assets" in status:
                if not isinstance(status["loaded_assets"], list):
                    raise TypeError(
                        f"status['loaded_assets'] must be a list, "
                        f"got {type(status['loaded_assets']).__name__}"
                    )
            else:
                status["loaded_assets"] = []

            if "failed_assets" in status:
                if not isinstance(status["failed_assets"], dict):
                    raise TypeError(
                        f"status['failed_assets'] must be a dict, "
                        f"got {type(status['failed_assets']).__name__}"
                    )
            else:
                status["failed_assets"] = {}

        for asset in assets:
            file_path = base_dir / f"{asset}.{extension}"
            try:
                df = self.load_parquet(str(file_path), schema=schema)
                loaded_assets[asset] = df
                if status is not None:
                    status["loaded_assets"].append(asset)
                logger.info(f"Successfully loaded asset: {asset}")
            except FileNotFoundError as e:
                if strict:
                    error_msg = (
                        f"Strict mode: Aborting load due to missing file for asset {asset}: {e}"
                    )
                    logger.error(error_msg)
                    raise
                logger.error(f"File not found for asset {asset}: {e}")
                if status is not None:
                    error_info = {
                        "type": "FileNotFoundError",
                        "message": str(e),
                    }
                    status["failed_assets"][asset] = error_info
            except pd.errors.ParserError as e:
                if strict:
                    error_msg = (
                        f"Strict mode: Aborting load due to parsing error for asset {asset}: {e}"
                    )
                    logger.error(error_msg)
                    raise
                logger.error(f"Parsing error for asset {asset}: {e}")
                if status is not None:
                    error_info = {
                        "type": "ParserError",
                        "message": str(e),
                    }
                    status["failed_assets"][asset] = error_info
            except OSError as e:
                if strict:
                    error_msg = (
    f"Strict mode: Aborting load due to OS error for asset {asset}: {e}"
)
                    logger.error(error_msg)
                    raise
                logger.error(f"OS error for asset {asset}: {e}")
                if status is not None:
                    error_info = {
                        "type": "OSError",
                        "message": str(e),
                    }
                    status["failed_assets"][asset] = error_info
            except ValueError as e:
                if strict:
                    error_msg = f"Strict mode: Aborting load due to schema validation error for asset {asset}: {e}"
                    logger.error(error_msg)
                    raise
                logger.error(f"Schema validation error for asset {asset}: {e}")
                if status is not None:
                    error_info = {
                        "type": "ValueError",
                        "message": str(e),
                    }
                    status["failed_assets"][asset] = error_info
            except Exception as e:
                if strict:
                    error_msg = f"Strict mode: Aborting load due to unexpected error for asset {asset}: {e}"
                    logger.error(error_msg)
                    raise
                logger.error(f"Unexpected error for asset {asset}: {e}")
                if status is not None:
                    error_info = {
                        "type": type(e).__name__,
                        "message": str(e),
                    }
                    status["failed_assets"][asset] = error_info
                continue

        return loaded_assets

    def load_from_config(
        self,
        data_sources_config: Dict[str, Any]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load all data sources based on configuration.

        Args:
            data_sources_config: The 'data_sources' section of data_config.yaml

        Returns:
            Nested dictionary: {category: {asset_name: DataFrame}}
        """
        all_data = {}
        for category, config in data_sources_config.items():
            logger.info(f"Loading category: {category}")
            if not isinstance(config, dict):
                error_msg = f"Invalid config for category '{category}': expected dict, got {type(config).__name__}"
                logger.warning(error_msg)
                continue
            path = config.get("path")
            assets = config.get("assets", [])
            schema = config.get("schema")
            if not path or not assets:
                error_msg = f"Missing path or assets for category: {category}"
                logger.warning(error_msg)
                continue
            loaded_category = self.load_assets(path, assets, schema=schema)
            all_data[category] = loaded_category
        return all_data

    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: Dict[str, str]
    ) -> ValidationResult:
        """
        Validate DataFrame against expected schema.

        Args:
            df: DataFrame to validate
            schema: Expected schema as {column_name: dtype_string}

        Returns:
            ValidationResult with validation details
        """
        errors: List[str] = []
        warnings: List[str] = []
        schema_violations: Dict[str, str] = {}

        # Check for missing columns
        for col in schema.keys():
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
                schema_violations[col] = "missing"

        # Check for extra columns (warning only)
        extra_cols = set(df.columns) - set(schema.keys())
        if extra_cols:
            warnings.append(f"Extra columns found: {extra_cols}")

        # Check data types
        for col, expected_dtype in schema.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not self._dtype_compatible(actual_dtype, expected_dtype):
                    errors.append(
                        f"Column '{col}' has dtype '{actual_dtype}', "
                        f"expected '{expected_dtype}'"
                    )
                    schema_violations[col] = (
                        f"dtype_mismatch: {actual_dtype} != {expected_dtype}"
                    )

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            schema_violations=schema_violations,
        )

    def _dtype_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual dtype is compatible with expected dtype."""
        # Normalize dtype strings
        actual_norm = actual.lower().replace(" ", "")
        expected_norm = expected.lower().replace(" ", "")

        # Direct match
        if actual_norm == expected_norm:
            return True

        # Float compatibility
        float_types = {"float64", "float32", "float", "float16"}
        if actual_norm in float_types and expected_norm in float_types:
            return True

        # Int compatibility
        int_types = {"int64", "int32", "int", "int16", "int8"}
        if actual_norm in int_types and expected_norm in int_types:
            return True

        # Datetime compatibility
        datetime_types = {"datetime64[ns]", "datetime64", "<m8[ns]"}
        if actual_norm in datetime_types and expected_norm in datetime_types:
            return True

        return False

    def log_validation_result(
        self,
        result: ValidationResult,
        run_id: str,
        source_path: str
    ) -> None:
        """Log validation result to file."""
        if not self.log_dir:
            logger.warning("No log directory configured")
            return

        log_data = {
            "run_id": run_id,
            "source_path": source_path,
            "timestamp": datetime.now().isoformat(),
            "validation_result": result.to_dict(),
        }

        log_path = self.log_dir / f"{run_id}_validation_report.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Validation report saved to {log_path}")
