"""Data profiling utilities using ydata-profiling for EDA reports."""

from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy import to avoid import errors if ydata-profiling is not installed
ProfileReport = None


def _get_profile_report_class():
    """Lazy import of ProfileReport to handle optional dependency."""
    global ProfileReport
    if ProfileReport is None:
        try:
            from ydata_profiling import ProfileReport as PR
            ProfileReport = PR
        except ImportError:
            raise ImportError(
                "ydata-profiling is required for data profiling. "
                "Install it with: pip install ydata-profiling"
            )
    return ProfileReport


class DataProfiler:
    """
    Generates comprehensive data profiling reports using ydata-profiling.

    Provides automated exploratory data analysis with statistics,
    distributions, correlations, missing value analysis, and data
    quality insights.
    """

    def __init__(self, output_dir: str = "data/processed/reports"):
        """
        Initialize DataProfiler with output directory for reports.

        Args:
            output_dir: Directory path where HTML reports will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_profile(
        self,
        df: pd.DataFrame,
        title: str = "Data Profile Report",
        minimal: bool = False,
        time_series_config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Generate a profile report for the given DataFrame.

        Args:
            df: DataFrame to profile
            title: Title for the report
            minimal: If True, generate minimal report (faster for large data)
            time_series_config: Optional config for time series analysis
                Keys can include:
                - 'tsmode': bool - Enable time series mode
                - 'sortby': str - Column name to sort by for time series

        Returns:
            ProfileReport object from ydata-profiling
        """
        PR = _get_profile_report_class()

        # Build configuration
        config = {
            "title": title,
            "explorative": not minimal,
        }

        # Apply minimal mode settings
        if minimal:
            config.update({
                "samples": {"head": 10, "tail": 10},
                "correlations": {
                    "pearson": {"calculate": True},
                    "spearman": {"calculate": False},
                    "kendall": {"calculate": False},
                    "phi_k": {"calculate": False},
                },
                "interactions": {"continuous": False},
                "missing_diagrams": {
                    "bar": True,
                    "matrix": False,
                    "heatmap": False
                },
            })
        else:
            # Full analysis configuration
            config.update({
                "correlations": {
                    "pearson": {"calculate": True},
                    "spearman": {"calculate": True},
                    "kendall": {"calculate": False},
                    "phi_k": {"calculate": False},
                },
                "interactions": {"continuous": True},
                "missing_diagrams": {
                    "bar": True,
                    "matrix": True,
                    "heatmap": True
                },
            })

        # Apply time series configuration if provided
        if time_series_config:
            if time_series_config.get("tsmode", False):
                config["tsmode"] = True
            if "sortby" in time_series_config:
                config["sortby"] = time_series_config["sortby"]

        logger.info(
            f"Generating profile report: {title} (minimal={minimal})"
        )

        report = PR(df, **config)

        logger.info(
            f"Profile report generated for {len(df)} rows, "
            f"{len(df.columns)} columns"
        )

        return report

    def save_report(
        self,
        report: Any,
        output_path: Optional[str] = None,
        dataset_identifier: Optional[str] = None
    ) -> str:
        """
        Save the profile report to an HTML file.

        Args:
            report: ProfileReport object to save
            output_path: Full path for the output file. If None,
                auto-generates based on timestamp and dataset_identifier
            dataset_identifier: Identifier for the dataset (used in
                auto-generated filename)

        Returns:
            Path to the saved HTML report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            identifier = dataset_identifier or "dataset"
            filename = f"{identifier}_{timestamp}_profile.html"
            output_path = str(self.output_dir / filename)

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        report.to_file(output_path)

        logger.info(f"Profile report saved to {output_path}")

        return output_path

    def get_data_quality_summary(self, report: Any) -> Dict[str, Any]:
        """
        Extract data quality summary from a profile report.

        Args:
            report: ProfileReport object

        Returns:
            Dictionary containing data quality metrics including:
            - row_count: Number of rows
            - column_count: Number of columns
            - missing_cells: Total missing cells
            - missing_cells_pct: Percentage of missing cells
            - duplicate_rows: Number of duplicate rows
            - duplicate_rows_pct: Percentage of duplicate rows
            - memory_size: Memory usage in bytes
            - variable_types: Count of each variable type
            - alerts: List of data quality alerts/warnings
        """
        # Get the description from the report
        description = report.get_description()

        # Extract table-level statistics (description.table is a dict)
        table_stats = getattr(description, "table", {}) or {}

        # Extract variable types from table stats (types key contains counts)
        variable_types = table_stats.get("types", {})

        # Extract alerts (description.alerts is a list)
        alerts = []
        alerts_data = getattr(description, "alerts", []) or []
        for alert in alerts_data:
            if hasattr(alert, "column_name") and hasattr(alert, "alert_type"):
                alerts.append({
                    "column": alert.column_name,
                    "type": str(alert.alert_type),
                    "message": str(alert) if hasattr(alert, "__str__") else ""
                })
            elif isinstance(alert, dict):
                alerts.append(alert)
            else:
                alerts.append({"message": str(alert)})

        # Extract duplicates info if available
        duplicates = getattr(description, "duplicates", None)
        n_duplicates = 0
        p_duplicates = 0.0
        if duplicates is not None:
            if isinstance(duplicates, dict):
                n_duplicates = duplicates.get("n_duplicates", 0)
                p_duplicates = duplicates.get("p_duplicates", 0.0)
            elif hasattr(duplicates, "__len__"):
                # duplicates might be a DataFrame of duplicate rows
                n_duplicates = len(duplicates)

        # Extract values from table_stats dict
        summary = {
            "row_count": table_stats.get("n", 0),
            "column_count": table_stats.get("n_var", 0),
            "missing_cells": table_stats.get("n_cells_missing", 0),
            "missing_cells_pct": table_stats.get("p_cells_missing", 0.0) * 100,
            "duplicate_rows": n_duplicates,
            "duplicate_rows_pct": p_duplicates * 100,
            "memory_size": table_stats.get("memory_size", 0),
            "variable_types": variable_types,
            "alerts_count": len(alerts),
            "alerts": alerts,
        }

        return summary

    def compare_profiles(
        self,
        report1: Any,
        report2: Any,
        title: str = "Profile Comparison Report"
    ) -> Any:
        """
        Compare two profile reports to analyze differences.

        Useful for before/after preprocessing analysis or comparing
        different datasets.

        Args:
            report1: First ProfileReport (e.g., raw data)
            report2: Second ProfileReport (e.g., processed data)
            title: Title for the comparison report

        Returns:
            Comparison ProfileReport object
        """
        comparison = report1.compare(report2)
        comparison.config.title = title

        logger.info(f"Generated comparison report: {title}")

        return comparison

    def generate_and_save_profile(
        self,
        df: pd.DataFrame,
        dataset_identifier: str,
        title: Optional[str] = None,
        minimal: bool = False,
        time_series_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to generate a profile and save it in one call.

        Args:
            df: DataFrame to profile
            dataset_identifier: Identifier for the dataset (used in filename)
            title: Title for the report (defaults to dataset_identifier)
            minimal: If True, generate a minimal report
            time_series_config: Optional time series configuration

        Returns:
            Dictionary containing:
            - report_path: Path to saved HTML report
            - quality_summary: Data quality summary dictionary
        """
        if title is None:
            title = f"{dataset_identifier} Profile Report"

        report = self.generate_profile(
            df=df,
            title=title,
            minimal=minimal,
            time_series_config=time_series_config
        )

        report_path = self.save_report(
            report=report,
            dataset_identifier=dataset_identifier
        )

        quality_summary = self.get_data_quality_summary(report)

        return {
            "report_path": report_path,
            "quality_summary": quality_summary
        }
