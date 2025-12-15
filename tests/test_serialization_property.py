
import os
import shutil
import tempfile
import json
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from datetime import datetime, timedelta

from src.data.structs import TimeSeriesData
from src.utils.serialization import (
    save_json, load_json, 
    save_parquet, load_parquet,
    save_pickle, load_pickle,
    save_timeseries_data, load_timeseries_data
)

# --- Strategies ---

# DataFrame Strategy (Numeric + datetime index)
@st.composite
def time_series_dataframe(draw):
    n_rows = draw(st.integers(min_value=5, max_value=50))
    n_cols = draw(st.integers(min_value=1, max_value=5))
    
    start_time = datetime(2023, 1, 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_rows)]
    
    data = {
        f"col_{i}": draw(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=n_rows, max_size=n_rows))
        for i in range(n_cols)
    }
    
    df = pd.DataFrame(data, index=timestamps)
    return df

# Helper to compare DataFrames allowing for floating point diffs
def assert_frame_equal_res(df1, df2):
    pd.testing.assert_frame_equal(df1, df2, check_freq=False)

class TestSerializationProperties:
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.test_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_dir)

    @given(st.dictionaries(st.text(), st.integers() | st.floats(allow_nan=False) | st.text()))
    @settings(max_examples=50, deadline=None)
    def test_json_roundtrip(self, data):
        """Property: JSON serialization round-trip consistency."""
        path = os.path.join(self.test_dir, "test.json")
        save_json(data, path)
        loaded = load_json(path)
        assert data == loaded

    @given(time_series_dataframe())
    @settings(max_examples=20, deadline=None)
    def test_parquet_roundtrip(self, df):
        """Property: Parquet serialization round-trip consistency."""
        path = os.path.join(self.test_dir, "test.parquet")
        save_parquet(df, path)
        loaded = load_parquet(path)
        assert_frame_equal_res(df, loaded)

    @given(time_series_dataframe(), st.one_of(st.none(), st.text(min_size=1)), st.booleans(), st.integers(min_value=0, max_value=9999))
    @settings(max_examples=30, deadline=None)
    def test_timeseries_data_roundtrip(self, features, target_name, as_dataframe, ts_seed):
        """Property: TimeSeriesData serialization round-trip consistency."""
        # Setup targets
        if as_dataframe:
            # Create a multi-column DataFrame target
            targets = pd.DataFrame({
                "t1": np.random.randn(len(features)),
                "t2": np.random.randn(len(features))
            }, index=features.index)
        else:
            # Create a Series target with potentially custom name
            targets = pd.Series(
                np.random.randn(len(features)), 
                index=features.index, 
                name=target_name
            )
            
        timestamp = features.index
        
        ts_data = TimeSeriesData(
            timestamp=timestamp,
            features=features,
            targets=targets,
            metadata={"source": "hypothesis", "version": 1},
            split_indices={"train": [0, 1], "test": [2]}
        )
        
        path = os.path.join(self.test_dir, f"ts_data_{ts_seed}")
        save_timeseries_data(ts_data, path)
        loaded_ts_data = load_timeseries_data(path)
        
        # Verify components
        pd.testing.assert_index_equal(ts_data.timestamp, loaded_ts_data.timestamp, check_names=False)
        assert_frame_equal_res(ts_data.features, loaded_ts_data.features)
        
        if as_dataframe:
            assert isinstance(loaded_ts_data.targets, pd.DataFrame)
            assert_frame_equal_res(ts_data.targets, loaded_ts_data.targets)
        else:
            assert isinstance(loaded_ts_data.targets, pd.Series)
            pd.testing.assert_series_equal(ts_data.targets, loaded_ts_data.targets)
            # Explicitly check name preservation
            assert loaded_ts_data.targets.name == target_name
                 
        assert ts_data.metadata == loaded_ts_data.metadata
        assert ts_data.split_indices == loaded_ts_data.split_indices

if __name__ == "__main__":
    pytest.main([__file__])
