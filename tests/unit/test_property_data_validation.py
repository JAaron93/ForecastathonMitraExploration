"""
Property-based tests for data loading and validation.

**Feature: forecasting-research-pipeline, Property 4: Data validation and
preprocessing consistency**

Tests that for any input dataset with schema violations, missing values,
or outliers, the preprocessing pipeline handles them according to
configuration while preserving data integrity and logging appropriate metrics.

**Validates: Requirements 1.1, 1.2, 1.3, 1.5**
"""

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from src.data.preprocessors import Preprocessor
from src.data.validators import DataValidator
from src.data.loaders import DataLoader


# Custom strategies for generating test data
@st.composite
def numeric_dataframe(draw, min_rows=5, max_rows=100, min_cols=1, max_cols=5):
    """Generate a DataFrame with numeric columns."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    data = {}
    for i in range(n_cols):
        col_name = f"col_{i}"
        values = draw(
            st.lists(
                st.floats(
                    min_value=-1e6,
                    max_value=1e6,
                    allow_nan=False,
                    allow_infinity=False
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        data[col_name] = values

    return pd.DataFrame(data)


@st.composite
def dataframe_with_missing(draw, min_rows=10, max_rows=50):
    """Generate a DataFrame with some missing values."""
    df = draw(numeric_dataframe(min_rows=min_rows, max_rows=max_rows))

    # Introduce missing values in random positions
    n_missing = draw(st.integers(min_value=1, max_value=min(10, len(df) // 2)))

    for _ in range(n_missing):
        row_idx = draw(st.integers(min_value=0, max_value=len(df) - 1))
        col_idx = draw(st.integers(min_value=0, max_value=len(df.columns) - 1))
        df.iloc[row_idx, col_idx] = np.nan

    return df


@st.composite
def timeseries_dataframe(draw, min_rows=10, max_rows=100):
    """Generate a DataFrame with DatetimeIndex."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate dates
    start_date = pd.Timestamp("2020-01-01")
    dates = pd.date_range(start=start_date, periods=n_rows, freq="D")

    # Generate numeric columns
    n_cols = draw(st.integers(min_value=1, max_value=3))
    data = {}
    for i in range(n_cols):
        col_name = f"value_{i}"
        values = draw(
            st.lists(
                st.floats(
                    min_value=-1000,
                    max_value=1000,
                    allow_nan=False,
                    allow_infinity=False
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        data[col_name] = values

    df = pd.DataFrame(data, index=dates)
    return df


class TestDataValidationProperty:
    """
    Property tests for data validation and preprocessing consistency.

    **Feature: forecasting-research-pipeline, Property 4: Data validation
    and preprocessing consistency**
    **Validates: Requirements 1.1, 1.2, 1.3, 1.5**
    """

    @given(df=numeric_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_missing_value_handling_preserves_shape(self, df):
        """
        Property: For any DataFrame, missing value handling should preserve
        the number of rows and columns.

        **Feature: forecasting-research-pipeline, Property 4: Data validation
        and preprocessing consistency**
        **Validates: Requirements 1.1, 1.2**
        """
        preprocessor = Preprocessor()

        # Introduce some missing values
        df_with_missing = df.copy()
        if len(df) > 0 and len(df.columns) > 0:
            df_with_missing.iloc[0, 0] = np.nan

        for strategy in ["forward_fill", "backward_fill", "mean", "median"]:
            result = preprocessor.handle_missing_values(
                df_with_missing, strategy=strategy
            )
            assert result.shape == df_with_missing.shape, (
                f"Shape changed after {strategy}: "
                f"{df_with_missing.shape} -> {result.shape}"
            )

    @given(df=dataframe_with_missing())
    @settings(max_examples=100, deadline=None)
    def test_missing_value_strategies_reduce_nulls(self, df):
        """
        Property: For any DataFrame with missing values, applying a fill
        strategy should reduce or maintain the number of nulls.

        **Feature: forecasting-research-pipeline, Property 4: Data validation
        and preprocessing consistency**
        **Validates: Requirements 1.1, 1.2**
        """
        preprocessor = Preprocessor()
        original_nulls = df.isna().sum().sum()

        assume(original_nulls > 0)  # Only test if there are nulls

        for strategy in ["mean", "median"]:
            result = preprocessor.handle_missing_values(df, strategy=strategy)
            result_nulls = result.isna().sum().sum()
            assert result_nulls <= original_nulls, (
                f"Strategy {strategy} increased nulls: "
                f"{original_nulls} -> {result_nulls}"
            )

    @given(df=numeric_dataframe(min_rows=20))
    @settings(max_examples=100, deadline=None)
    def test_outlier_detection_returns_valid_indices(self, df):
        """
        Property: For any DataFrame, outlier detection should return
        indices that exist in the DataFrame.

        **Feature: forecasting-research-pipeline, Property 4: Data validation
        and preprocessing consistency**
        **Validates: Requirements 1.3**
        """
        preprocessor = Preprocessor()

        outliers = preprocessor.detect_outliers(df, method="iqr")

        # All returned indices should be valid
        for idx in outliers:
            assert idx in df.index, f"Invalid outlier index: {idx}"

    @given(df=numeric_dataframe(min_rows=20))
    @settings(max_examples=100, deadline=None)
    def test_outlier_treatment_preserves_non_outliers(self, df):
        """
        Property: For any DataFrame, outlier treatment with winsorize
        should not change values that are not outliers.

        **Feature: forecasting-research-pipeline, Property 4: Data validation
        and preprocessing consistency**
        **Validates: Requirements 1.3**
        """
        preprocessor = Preprocessor()

        # Get outlier indices before treatment
        outliers = set(preprocessor.detect_outliers(df, method="iqr"))

        # Apply treatment
        result = preprocessor.treat_outliers(df, method="winsorize")

        # Non-outlier values should be unchanged
        non_outlier_mask = ~df.index.isin(outliers)
        if non_outlier_mask.any():
            for col in df.columns:
                original_vals = df.loc[non_outlier_mask, col]
                result_vals = result.loc[non_outlier_mask, col]
                # Use allclose for floating point comparison
                assert np.allclose(
                    original_vals.values,
                    result_vals.values,
                    equal_nan=True
                ), f"Non-outlier values changed in column {col}"

    @given(df=numeric_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_quality_metrics_consistency(self, df):
        """
        Property: For any DataFrame, quality metrics should accurately
        reflect the DataFrame's properties.

        **Feature: forecasting-research-pipeline, Property 4: Data validation
        and preprocessing consistency**
        **Validates: Requirements 1.1, 1.5**
        """
        validator = DataValidator()
        metrics = validator.get_quality_metrics(df)

        # Row count should match
        assert metrics.row_count == len(df), (
            f"Row count mismatch: {metrics.row_count} != {len(df)}"
        )

        # Column count should match
        assert metrics.column_count == len(df.columns), (
            f"Column count mismatch: {metrics.column_count} != {len(df.columns)}"
        )

        # Null counts should match actual nulls
        for col in df.columns:
            actual_nulls = df[col].isna().sum()
            reported_nulls = metrics.null_count.get(col, 0)
            assert actual_nulls == reported_nulls, (
                f"Null count mismatch for {col}: "
                f"{reported_nulls} != {actual_nulls}"
            )

    @given(
        df1=numeric_dataframe(min_rows=50, max_rows=100),
        df2=numeric_dataframe(min_rows=50, max_rows=100)
    )
    @settings(max_examples=50, deadline=None)
    def test_psi_symmetry_and_bounds(self, df1, df2):
        """
        Property: PSI should be non-negative and PSI(A, A) should be
        approximately zero.

        **Feature: forecasting-research-pipeline, Property 4: Data validation
        and preprocessing consistency**
        **Validates: Requirements 1.5**
        """
        validator = DataValidator()

        # Ensure we have at least one column
        assume(len(df1.columns) > 0 and len(df2.columns) > 0)

        col1 = df1.columns[0]
        col2 = df2.columns[0]

        # PSI should be non-negative
        psi = validator.calculate_psi(df1[col1], df2[col2])
        assert psi >= 0, f"PSI should be non-negative, got {psi}"

        # PSI of identical distributions should be near zero
        psi_same = validator.calculate_psi(df1[col1], df1[col1])
        assert psi_same < 0.1, (
            f"PSI of identical distributions should be near zero, got {psi_same}"
        )

    @given(df=timeseries_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_resample_preserves_temporal_order(self, df):
        """
        Property: For any time series DataFrame, resampling should
        preserve temporal ordering.

        **Feature: forecasting-research-pipeline, Property 4: Data validation
        and preprocessing consistency**
        **Validates: Requirements 1.2**
        """
        preprocessor = Preprocessor()

        # Resample to weekly
        result = preprocessor.resample_timeseries(df, freq="W")

        # Check temporal ordering is preserved
        if len(result) > 1:
            time_diffs = result.index.to_series().diff()[1:]
            assert (time_diffs > pd.Timedelta(0)).all(), (
                "Temporal ordering not preserved after resampling"
            )

    @given(df=numeric_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_schema_validation_detects_missing_columns(self, df):
        """
        Property: Schema validation should correctly identify missing columns.

        **Feature: forecasting-research-pipeline, Property 4: Data validation
        and preprocessing consistency**
        **Validates: Requirements 1.1**
        """
        loader = DataLoader()

        # Create schema with an extra required column
        schema = {col: "float64" for col in df.columns}
        schema["nonexistent_column"] = "float64"

        result = loader.validate_schema(df, schema)

        assert not result.is_valid, "Should detect missing column"
        assert "nonexistent_column" in result.schema_violations, (
            "Should report missing column in violations"
        )
