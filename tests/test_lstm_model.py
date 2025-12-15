"""
Tests for LSTM model (Keras version).
"""

import os
# We use JAX for testing in this environment (Python 3.13) as TF is not available
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import numpy as np
import pandas as pd
import keras
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes

from src.models.lstm_model import LSTMModel

# --- Fixtures ---

@pytest.fixture
def sample_ts_data():
    X = pd.DataFrame({
        'feature1': np.linspace(0, 10, 100),
        'feature2': np.sin(np.linspace(0, 10, 100))
    })
    y = pd.Series(np.linspace(0, 10, 100) + 1, name='target')
    return X, y

# --- Unit Tests ---

def test_lstm_sequence_generation(sample_ts_data):
    X, y = sample_ts_data
    model = LSTMModel()
    seq_len = 5
    
    X_seq, y_seq = model._create_sequences(X.values, y.values, seq_len=seq_len)
    
    expected_samples = len(X) - seq_len
    assert X_seq.shape == (expected_samples, seq_len, 2)
    assert y_seq.shape == (expected_samples,) # 1D target because y passed as 1D
    
    assert y_seq[0] == y.iloc[seq_len]

def test_lstm_fit_predict(sample_ts_data):
    X, y = sample_ts_data
    model = LSTMModel(hyperparameters={"epochs": 2, "batch_size": 16, "seq_len": 5, "hidden_size": 8})
    
    model.fit(X, y)
    assert model.is_fitted
    
    preds = model.predict(X)
    
    expected_len = len(X) - 5
    assert len(preds) == expected_len
    assert isinstance(preds, np.ndarray)

def test_lstm_save_load(sample_ts_data, tmp_path):
    X, y = sample_ts_data
    model = LSTMModel(hyperparameters={"epochs": 1, "seq_len": 5})
    model.fit(X, y)
    
    save_path = tmp_path / "lstm_test"
    model.save_model(save_path)
    
    loaded_model = LSTMModel()
    loaded_model.load_model(save_path)
    
    assert loaded_model.is_fitted
    assert loaded_model.hyperparameters["seq_len"] == 5
    
    # Check predictions match
    p1 = model.predict(X)
    p2 = loaded_model.predict(X)
    np.testing.assert_allclose(p1, p2, atol=1e-5)

# --- Property Test ---

@settings(max_examples=10, deadline=None)
@given(data=st.data())
def test_lstm_sequence_preservation(data):
    """
    Property 2: Time series temporal ordering preservation
    """
    n_rows = 50
    seq_len = 5
    
    X_df = pd.DataFrame({
        'time_idx': np.arange(n_rows, dtype=float)
    })
    y_series = pd.Series(np.arange(n_rows, dtype=float), name='target')
    
    model = LSTMModel()
    X_seq, y_seq = model._create_sequences(X_df.values, y_series.values, seq_len=seq_len)
    
    n_samples = X_seq.shape[0]
    assert n_samples == n_rows - seq_len
    
    for i in range(n_samples):
        sequence = X_seq[i, :, 0] # feature 0 is time_idx
        expected_sequence = np.arange(i, i + seq_len, dtype=float)
        np.testing.assert_array_equal(sequence, expected_sequence)
        
        target = y_seq[i]
        assert target == i + seq_len

@settings(max_examples=5, deadline=None) # Very few examples as training is slow
@given(data=st.data())
def test_lstm_training_correctness(data):
    """
    Property 5: Model training and evaluation correctness
    """
    # Features - finite values, reasonable scale
    X_df = data.draw(data_frames(
        columns=[column(name=f"c{i}", dtype=float, elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-5, max_value=5)) for i in range(2)],
        index=range_indexes(min_size=20, max_size=50) 
    ))
    
    # Target
    y_series = data.draw(st.builds(
        pd.Series,
        st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-5, max_value=5), min_size=len(X_df), max_size=len(X_df)),
        index=st.just(X_df.index)
    ))
    
    seq_len = 5
    
    # Keras/JAX can be chatty or slow initialization
    model = LSTMModel(hyperparameters={"epochs": 1, "seq_len": seq_len, "hidden_size": 4, "batch_size": 8})
    
    model.fit(X_df, y_series)
    assert model.is_fitted
    
    preds = model.predict(X_df)
    assert len(preds) == len(X_df) - seq_len
    assert not np.isnan(preds).any()
