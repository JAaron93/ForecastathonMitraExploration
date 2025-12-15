"""Data validation utilities with PSI calculation and quality metrics."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Data quality metrics for a DataFrame."""
    null_count: Dict[str, int]
    unique_count: Dict[str, int]
    row_count: int
    column_count: int
    value_ranges: Dict[str, Dict[str, float]]
    schema_violations: List[str] = field(default_factory=list)
    psi_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "null_count": self.null_count,
            "unique_count": self.unique_count,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "value_ranges": self.value_ranges,
            "schema_violations": self.schema_violations,
            "psi_scores": self.psi_scores,
        }


class DataValidator:
    """Validates data quality and calculates distribution metrics."""

    def __init__(self, log_dir: Optional[str] = None):
        """Initialize DataValidator with optional logging directory."""
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def calculate_psi(
        self,
        expected: pd.Series,
        actual: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index between two distributions.

        PSI measures how much a distribution has shifted over time.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate change
        PSI >= 0.25: Significant change

        Args:
            expected: Baseline distribution (e.g., training data)
            actual: Current distribution (e.g., new data)
            bins: Number of bins for histogram

        Returns:
            PSI value (float)
        """
        # Remove NaN values
        expected_clean = expected.dropna()
        actual_clean = actual.dropna()

        if len(expected_clean) == 0 or len(actual_clean) == 0:
            return 0.0

        # Create bins based on expected distribution
        min_val = min(expected_clean.min(), actual_clean.min())
        max_val = max(expected_clean.max(), actual_clean.max())

        # Handle edge case where all values are the same
        if min_val == max_val:
            return 0.0

        bin_edges = np.linspace(min_val, max_val, bins + 1)

        # Calculate percentages in each bin
        expected_counts = np.histogram(expected_clean, bins=bin_edges)[0]
        actual_counts = np.histogram(actual_clean, bins=bin_edges)[0]

        expected_percents = expected_counts / len(expected_clean)
        actual_percents = actual_counts / len(actual_clean)

        # Avoid division by zero and log(0)
        epsilon = 1e-10
        expected_percents = np.where(
            expected_percents == 0, epsilon, expected_percents
        )
        actual_percents = np.where(
            actual_percents == 0, epsilon, actual_percents
        )

        # Calculate PSI
        psi = np.sum(
            (actual_percents - expected_percents) *
            np.log(actual_percents / expected_percents)
        )

        return float(psi)

    def get_quality_metrics(
        self,
        df: pd.DataFrame,
        baseline_df: Optional[pd.DataFrame] = None
    ) -> QualityMetrics:
        """
        Calculate comprehensive data quality metrics for a DataFrame.

        Args:
            df: DataFrame to analyze
            baseline_df: Optional baseline for PSI calculation

        Returns:
            QualityMetrics object with all metrics
        """
        # Null counts per column
        null_count = df.isnull().sum().to_dict()

        # Unique counts per column
        unique_count = df.nunique().to_dict()

        # Value ranges for numeric columns
        value_ranges: Dict[str, Dict[str, float]] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            value_ranges[col] = {
                "min": float(df[col].min()) if not df[col].isna().all() else 0.0,
                "max": float(df[col].max()) if not df[col].isna().all() else 0.0,
                "mean": float(df[col].mean()) if not df[col].isna().all() else 0.0,
                "std": float(df[col].std()) if not df[col].isna().all() else 0.0,
            }

        # PSI scores if baseline provided
        psi_scores: Dict[str, float] = {}
        if baseline_df is not None:
            common_cols = set(df.columns) & set(baseline_df.columns)
            for col in common_cols:
                if col in numeric_cols:
                    psi_scores[col] = self.calculate_psi(
                        baseline_df[col], df[col]
                    )

        return QualityMetrics(
            null_count=null_count,
            unique_count=unique_count,
            row_count=len(df),
            column_count=len(df.columns),
            value_ranges=value_ranges,
            schema_violations=[],
            psi_scores=psi_scores,
        )

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        max_null_ratio: float = 0.3,
        min_unique_ratio: float = 0.01
    ) -> tuple[bool, List[str]]:
        """
        Validate data quality against thresholds.

        Args:
            df: DataFrame to validate
            max_null_ratio: Maximum allowed null ratio per column
            min_unique_ratio: Minimum required unique ratio per column

        Returns:
            Tuple of (is_valid, list of issues)
        """
        if len(df) == 0:
            return True, []  # Empty DataFrame passes validation with no issues

        issues: List[str] = []

        for col in df.columns:
            # Check null ratio
            null_ratio = df[col].isna().sum() / len(df)
            if null_ratio > max_null_ratio:
                issues.append(
                    f"Column '{col}' has {null_ratio:.1%} nulls "
                    f"(max: {max_null_ratio:.1%})"
                )

            # Check unique ratio for non-datetime columns
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < min_unique_ratio and len(df) > 10:
                    issues.append(
                        f"Column '{col}' has low variance "
                        f"({unique_ratio:.1%} unique values)"
                    )

        is_valid = len(issues) == 0
        return is_valid, issues

    def log_quality_metrics(
        self,
        metrics: QualityMetrics,
        run_id: str,
        source_name: str = "unknown"
    ) -> None:
        """
        Log quality metrics to JSON file.

        Args:
            metrics: QualityMetrics to log
            run_id: Unique identifier for this run
            source_name: Name of the data source
        """
        if not self.log_dir:
            logger.warning("No log directory configured for metrics logging")
            return

        log_data = {
            "run_id": run_id,
            "source_name": source_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.to_dict(),
        }

        log_path = self.log_dir / f"{run_id}_validation_report.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Quality metrics saved to {log_path}")

    def check_distribution_shift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        psi_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Check for distribution shift between baseline and current data.

        Args:
            baseline_df: Baseline/training data
            current_df: Current/new data
            psi_threshold: PSI threshold for significant shift

        Returns:
            Dictionary with shift analysis results
        """
        results = {
            "has_significant_shift": False,
            "shifted_columns": [],
            "psi_scores": {},
        }

        numeric_cols = baseline_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        for col in numeric_cols:
            if col in current_df.columns:
                psi = self.calculate_psi(baseline_df[col], current_df[col])
                results["psi_scores"][col] = psi

                if psi >= psi_threshold:
                    results["shifted_columns"].append(col)
                    results["has_significant_shift"] = True

        return results
