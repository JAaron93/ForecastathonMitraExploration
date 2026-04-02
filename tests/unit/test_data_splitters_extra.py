"""Additional unit tests for TimeSeriesSplitter and TimeSeriesAligner."""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.splitters import (
    SplitIndices,
    TimeSeriesAligner,
    TimeSeriesSplitter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_df():
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": np.random.rand(n), "volume": np.random.rand(n)}, index=idx)


# ---------------------------------------------------------------------------
# SplitIndices serialization
# ---------------------------------------------------------------------------

class TestSplitIndicesSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        s = SplitIndices(
            train_indices=list(range(100)),
            validation_indices=list(range(100, 140)),
            test_indices=list(range(140, 200)),
            metadata={"split_type": "sequential"},
        )
        restored = SplitIndices.from_dict(s.to_dict())
        assert restored.train_indices == s.train_indices
        assert restored.validation_indices == s.validation_indices
        assert restored.test_indices == s.test_indices
        assert restored.metadata == s.metadata


# ---------------------------------------------------------------------------
# TimeSeriesSplitter.train_val_test_split
# ---------------------------------------------------------------------------

class TestTrainValTestSplit:
    def test_basic_split_proportions(self, daily_df):
        splitter = TimeSeriesSplitter()
        split = splitter.train_val_test_split(daily_df, 0.7, 0.15, 0.15)
        n = len(daily_df)
        assert split.train_indices[-1] < split.validation_indices[0]
        assert split.validation_indices[-1] < split.test_indices[0]
        assert (
            len(split.train_indices) + len(split.validation_indices) + len(split.test_indices)
            == n
        )

    def test_ratios_must_sum_to_one(self, daily_df):
        splitter = TimeSeriesSplitter()
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            splitter.train_val_test_split(daily_df, 0.5, 0.3, 0.3)

    def test_metadata_contains_datetime_bounds(self, daily_df):
        splitter = TimeSeriesSplitter()
        split = splitter.train_val_test_split(daily_df)
        assert "train_start" in split.metadata
        assert "val_start" in split.metadata
        assert "test_start" in split.metadata


# ---------------------------------------------------------------------------
# TimeSeriesSplitter.rolling_window_split
# ---------------------------------------------------------------------------

class TestRollingWindowSplit:
    def test_yields_correct_indices(self, daily_df):
        splitter = TimeSeriesSplitter()
        folds = list(splitter.rolling_window_split(daily_df, train_size=100, val_size=30, test_size=20))
        assert len(folds) > 0
        first = folds[0]
        assert len(first.train_indices) == 100
        assert len(first.validation_indices) == 30
        assert len(first.test_indices) == 20

    def test_no_overlap_between_splits(self, daily_df):
        splitter = TimeSeriesSplitter()
        for fold in splitter.rolling_window_split(daily_df, 80, 30, 20, step_size=10):
            train_set = set(fold.train_indices)
            val_set = set(fold.validation_indices)
            test_set = set(fold.test_indices)
            assert train_set.isdisjoint(val_set)
            assert val_set.isdisjoint(test_set)
            assert train_set.isdisjoint(test_set)

    def test_window_exceeds_data_raises(self):
        splitter = TimeSeriesSplitter()
        tiny_df = pd.DataFrame({"a": range(10)})
        with pytest.raises(ValueError, match="Window size"):
            list(splitter.rolling_window_split(tiny_df, train_size=5, val_size=4, test_size=4))


# ---------------------------------------------------------------------------
# TimeSeriesSplitter.expanding_window_split
# ---------------------------------------------------------------------------

class TestExpandingWindowSplit:
    def test_train_grows_with_each_fold(self, daily_df):
        splitter = TimeSeriesSplitter()
        folds = list(splitter.expanding_window_split(daily_df, initial_train_size=100, val_size=20, test_size=10))
        assert len(folds) > 1
        for i in range(1, len(folds)):
            assert len(folds[i].train_indices) >= len(folds[i - 1].train_indices)

    def test_minimum_exceeds_data_raises(self):
        splitter = TimeSeriesSplitter()
        tiny_df = pd.DataFrame({"a": range(5)})
        with pytest.raises(ValueError, match="Minimum window size"):
            list(splitter.expanding_window_split(tiny_df, initial_train_size=4, val_size=2, test_size=1))


# ---------------------------------------------------------------------------
# TimeSeriesSplitter.save/load split indices
# ---------------------------------------------------------------------------

class TestSaveLoadSplitIndices:
    def test_save_and_load_roundtrip(self, tmp_path, daily_df):
        splitter = TimeSeriesSplitter(save_dir=str(tmp_path))
        split = splitter.train_val_test_split(daily_df)
        splitter.save_split_indices(split, "test_split")

        loaded = splitter.load_split_indices("test_split")
        assert loaded.train_indices == split.train_indices
        assert loaded.validation_indices == split.validation_indices

    def test_save_raises_without_save_dir(self, daily_df):
        splitter = TimeSeriesSplitter()
        split = splitter.train_val_test_split(daily_df)
        with pytest.raises(ValueError, match="No save directory"):
            splitter.save_split_indices(split, "any_name")

    def test_load_raises_file_not_found(self, tmp_path):
        splitter = TimeSeriesSplitter(save_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            splitter.load_split_indices("nonexistent")

    def test_load_raises_without_save_dir(self):
        splitter = TimeSeriesSplitter()
        with pytest.raises(ValueError, match="No save directory"):
            splitter.load_split_indices("any")


# ---------------------------------------------------------------------------
# TimeSeriesSplitter.apply_split
# ---------------------------------------------------------------------------

class TestApplySplit:
    def test_apply_split_returns_three_dataframes(self, daily_df):
        splitter = TimeSeriesSplitter()
        split = splitter.train_val_test_split(daily_df)
        train, val, test = splitter.apply_split(daily_df, split)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert len(train) + len(val) + len(test) == len(daily_df)


# ---------------------------------------------------------------------------
# TimeSeriesSplitter.validate_no_leakage
# ---------------------------------------------------------------------------

class TestValidateNoLeakage:
    def test_valid_split_passes(self, daily_df):
        splitter = TimeSeriesSplitter()
        split = splitter.train_val_test_split(daily_df)
        is_valid, issues = splitter.validate_no_leakage(daily_df, split)
        assert is_valid
        assert issues == []

    def test_non_datetime_index_fails(self):
        df = pd.DataFrame({"a": range(100)})
        splitter = TimeSeriesSplitter()
        split = SplitIndices(
            train_indices=list(range(70)),
            validation_indices=list(range(70, 85)),
            test_indices=list(range(85, 100)),
        )
        is_valid, issues = splitter.validate_no_leakage(df, split)
        assert not is_valid
        assert len(issues) > 0


# ---------------------------------------------------------------------------
# TimeSeriesAligner
# ---------------------------------------------------------------------------

class TestTimeSeriesAligner:
    def test_align_two_series_outer(self):
        aligner = TimeSeriesAligner()
        idx1 = pd.date_range("2023-01-01", periods=5, freq="D")
        idx2 = pd.date_range("2023-01-03", periods=5, freq="D")
        df1 = pd.DataFrame({"a": range(5)}, index=idx1)
        df2 = pd.DataFrame({"b": range(5)}, index=idx2)
        result = aligner.align_timeseries([df1, df2], method="outer")
        assert "a" in result.columns
        assert "b" in result.columns
        # outer union means we span both ranges
        assert result.index.min() == idx1.min()
        assert result.index.max() == idx2.max()

    def test_align_empty_raises(self):
        aligner = TimeSeriesAligner()
        with pytest.raises(ValueError, match="No dataframes provided"):
            aligner.align_timeseries([])

    def test_align_non_datetime_index_raises(self):
        aligner = TimeSeriesAligner()
        df = pd.DataFrame({"a": range(5)})  # RangeIndex, not DatetimeIndex
        with pytest.raises(ValueError, match="DatetimeIndex"):
            aligner.align_timeseries([df])

    def test_align_inner_method(self):
        aligner = TimeSeriesAligner()
        idx1 = pd.date_range("2023-01-01", periods=5, freq="D")
        idx2 = pd.date_range("2023-01-03", periods=5, freq="D")
        df1 = pd.DataFrame({"a": range(5)}, index=idx1)
        df2 = pd.DataFrame({"b": range(5)}, index=idx2)
        result = aligner.align_timeseries([df1, df2], method="inner")
        # inner intersection starts at the later of the two starts
        assert result.index.min() >= idx2.min()
