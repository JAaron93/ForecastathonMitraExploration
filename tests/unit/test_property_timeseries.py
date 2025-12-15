"""
Property-based tests for time series operations.

**Feature: forecasting-research-pipeline, Property 2: Time series temporal
ordering preservation**

Tests that for any time series data splitting or sequence generation operation,
the resulting splits or sequences never contain future information in training
sets and maintain strict temporal ordering.

**Validates: Requirements 3.1, 3.2, 6.1, 6.3**
"""

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from src.data.splitters import TimeSeriesSplitter, TimeSeriesAligner, SplitIndices


# Custom strategies for generating test data
@st.composite
def timeseries_dataframe(draw, min_rows=20, max_rows=200):
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


@st.composite
def split_ratios(draw):
    """Generate valid train/val/test split ratios that sum to 1.0."""
    train = draw(st.floats(min_value=0.5, max_value=0.8))
    remaining = 1.0 - train
    val_ratio = draw(st.floats(min_value=0.3, max_value=0.7))
    val = remaining * val_ratio
    test = remaining - val
    return train, val, test


@st.composite
def window_sizes(draw, max_total=50):
    """Generate valid window sizes for rolling/expanding splits."""
    train_size = draw(st.integers(min_value=5, max_value=max_total // 2))
    remaining = max_total - train_size
    val_size = draw(st.integers(min_value=2, max_value=max(2, remaining // 2)))
    test_size = draw(st.integers(min_value=2, max_value=max(2, remaining - val_size)))
    return train_size, val_size, test_size


class TestTimeSeriesTemporalOrdering:
    """
    Property tests for time series temporal ordering preservation.

    **Feature: forecasting-research-pipeline, Property 2: Time series temporal
    ordering preservation**
    **Validates: Requirements 3.1, 3.2, 6.1, 6.3**
    """

    @given(df=timeseries_dataframe(), ratios=split_ratios())
    @settings(max_examples=100, deadline=None)
    def test_sequential_split_no_future_leakage(self, df, ratios):
        """
        Property: For any time series split, training data should never
        contain timestamps later than validation or test data.

        **Feature: forecasting-research-pipeline, Property 2: Time series
        temporal ordering preservation**
        **Validates: Requirements 3.1, 3.2**
        """
        train_ratio, val_ratio, test_ratio = ratios
        splitter = TimeSeriesSplitter()

        split = splitter.train_val_test_split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        # Get timestamps for each split
        train_times = df.index[split.train_indices]
        val_times = df.index[split.validation_indices]
        test_times = df.index[split.test_indices]

        # Train max should be < val min
        if len(train_times) > 0 and len(val_times) > 0:
            assert train_times.max() < val_times.min(), (
                f"Training data ({train_times.max()}) overlaps with "
                f"validation data ({val_times.min()})"
            )

        # Val max should be < test min
        if len(val_times) > 0 and len(test_times) > 0:
            assert val_times.max() < test_times.min(), (
                f"Validation data ({val_times.max()}) overlaps with "
                f"test data ({test_times.min()})"
            )

    @given(df=timeseries_dataframe(min_rows=50), sizes=window_sizes())
    @settings(max_examples=100, deadline=None)
    def test_rolling_window_no_future_leakage(self, df, sizes):
        """
        Property: For any rolling window split, each fold should maintain
        strict temporal ordering with no future information in training.

        **Feature: forecasting-research-pipeline, Property 2: Time series
        temporal ordering preservation**
        **Validates: Requirements 3.1, 3.2**
        """
        train_size, val_size, test_size = sizes
        total_size = train_size + val_size + test_size

        assume(total_size <= len(df))

        splitter = TimeSeriesSplitter()

        for split in splitter.rolling_window_split(
            df,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            step_size=5
        ):
            train_times = df.index[split.train_indices]
            val_times = df.index[split.validation_indices]
            test_times = df.index[split.test_indices]

            # Check no overlap between splits
            assert train_times.max() < val_times.min(), (
                "Rolling window: train overlaps with validation"
            )
            assert val_times.max() < test_times.min(), (
                "Rolling window: validation overlaps with test"
            )

    @given(df=timeseries_dataframe(min_rows=50), sizes=window_sizes())
    @settings(max_examples=100, deadline=None)
    def test_expanding_window_no_future_leakage(self, df, sizes):
        """
        Property: For any expanding window split, training data should
        grow monotonically while maintaining temporal ordering.

        **Feature: forecasting-research-pipeline, Property 2: Time series
        temporal ordering preservation**
        **Validates: Requirements 3.1, 3.2**
        """
        train_size, val_size, test_size = sizes
        total_size = train_size + val_size + test_size

        assume(total_size <= len(df))

        splitter = TimeSeriesSplitter()

        prev_train_size = 0
        for split in splitter.expanding_window_split(
            df,
            initial_train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            step_size=5
        ):
            # Training size should grow or stay same
            current_train_size = len(split.train_indices)
            assert current_train_size >= prev_train_size, (
                "Expanding window: training size should grow monotonically"
            )
            prev_train_size = current_train_size

            # Check temporal ordering
            train_times = df.index[split.train_indices]
            val_times = df.index[split.validation_indices]
            test_times = df.index[split.test_indices]

            assert train_times.max() < val_times.min(), (
                "Expanding window: train overlaps with validation"
            )
            assert val_times.max() < test_times.min(), (
                "Expanding window: validation overlaps with test"
            )

    @given(df=timeseries_dataframe(), ratios=split_ratios())
    @settings(max_examples=100, deadline=None)
    def test_split_indices_are_disjoint(self, df, ratios):
        """
        Property: For any split, train/val/test indices should be
        mutually exclusive (no overlap).

        **Feature: forecasting-research-pipeline, Property 2: Time series
        temporal ordering preservation**
        **Validates: Requirements 3.1, 3.2**
        """
        train_ratio, val_ratio, test_ratio = ratios
        splitter = TimeSeriesSplitter()

        split = splitter.train_val_test_split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        train_set = set(split.train_indices)
        val_set = set(split.validation_indices)
        test_set = set(split.test_indices)

        # Check no overlap
        assert len(train_set & val_set) == 0, "Train and val indices overlap"
        assert len(val_set & test_set) == 0, "Val and test indices overlap"
        assert len(train_set & test_set) == 0, "Train and test indices overlap"

    @given(df=timeseries_dataframe(), ratios=split_ratios())
    @settings(max_examples=100, deadline=None)
    def test_split_covers_all_data(self, df, ratios):
        """
        Property: For any split, the union of train/val/test indices
        should cover all data points.

        **Feature: forecasting-research-pipeline, Property 2: Time series
        temporal ordering preservation**
        **Validates: Requirements 3.1, 3.2**
        """
        train_ratio, val_ratio, test_ratio = ratios
        splitter = TimeSeriesSplitter()

        split = splitter.train_val_test_split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        all_indices = (
            set(split.train_indices) |
            set(split.validation_indices) |
            set(split.test_indices)
        )

        expected_indices = set(range(len(df)))
        assert all_indices == expected_indices, (
            f"Split does not cover all data: "
            f"missing {expected_indices - all_indices}"
        )

    @given(df=timeseries_dataframe(), ratios=split_ratios())
    @settings(max_examples=100, deadline=None)
    def test_validate_no_leakage_function(self, df, ratios):
        """
        Property: The validate_no_leakage function should correctly
        identify valid splits as having no leakage.

        **Feature: forecasting-research-pipeline, Property 2: Time series
        temporal ordering preservation**
        **Validates: Requirements 3.1, 3.2**
        """
        train_ratio, val_ratio, test_ratio = ratios
        splitter = TimeSeriesSplitter()

        split = splitter.train_val_test_split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        is_valid, issues = splitter.validate_no_leakage(df, split)

        assert is_valid, f"Valid split reported as having leakage: {issues}"
        assert len(issues) == 0, f"Unexpected issues: {issues}"

    @given(df=timeseries_dataframe(), ratios=split_ratios())
    @settings(max_examples=100, deadline=None)
    def test_split_metadata_accuracy(self, df, ratios):
        """
        Property: Split metadata should accurately reflect the split
        configuration and results.

        **Feature: forecasting-research-pipeline, Property 2: Time series
        temporal ordering preservation**
        **Validates: Requirements 3.3**
        """
        train_ratio, val_ratio, test_ratio = ratios
        splitter = TimeSeriesSplitter()

        split = splitter.train_val_test_split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        # Check metadata accuracy
        assert split.metadata["total_samples"] == len(df)
        assert split.metadata["train_samples"] == len(split.train_indices)
        assert split.metadata["val_samples"] == len(split.validation_indices)
        assert split.metadata["test_samples"] == len(split.test_indices)

    @given(df=timeseries_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_apply_split_preserves_data(self, df):
        """
        Property: Applying a split should preserve all original data
        without modification.

        **Feature: forecasting-research-pipeline, Property 2: Time series
        temporal ordering preservation**
        **Validates: Requirements 3.1, 6.1**
        """
        splitter = TimeSeriesSplitter()

        split = splitter.train_val_test_split(df)
        train_df, val_df, test_df = splitter.apply_split(df, split)

        # Concatenate back and compare
        reconstructed = pd.concat([train_df, val_df, test_df])
        reconstructed = reconstructed.sort_index()

        # Should have same data
        assert len(reconstructed) == len(df)
        for col in df.columns:
            assert np.allclose(
                df[col].values,
                reconstructed[col].values,
                equal_nan=True
            ), f"Data changed in column {col}"

    @given(df=timeseries_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_split_indices_within_bounds(self, df):
        """
        Property: All split indices should be valid indices for the
        original DataFrame.

        **Feature: forecasting-research-pipeline, Property 2: Time series
        temporal ordering preservation**
        **Validates: Requirements 3.1, 6.3**
        """
        splitter = TimeSeriesSplitter()

        split = splitter.train_val_test_split(df)

        all_indices = (
            split.train_indices +
            split.validation_indices +
            split.test_indices
        )

        for idx in all_indices:
            assert 0 <= idx < len(df), f"Index {idx} out of bounds for df of length {len(df)}"
