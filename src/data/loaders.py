"""Data loading utilities for Parquet files and schema validation."""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
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
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} rows from {path}")

        if schema:
            result = self.validate_schema(df, schema)
            if not result.is_valid:
                error_msg = "; ".join(result.errors)
                raise ValueError(f"Schema validation failed: {error_msg}")

        return df

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
