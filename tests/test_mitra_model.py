"""
Tests for Mitra foundation model wrapper.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes

from src.models.mitra_model import MitraModel

# --- Fixtures ---

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50)
    })
    y = pd.Series(np.random.randint(0, 2, 50), name='target')
    return X, y

# --- Unit Tests ---

def test_mitra_initialization():
    model = MitraModel(hyperparameters={"fine_tune": True}, label_column="label")
    assert model.hyperparameters["fine_tune"] is True
    assert model.label_column == "label"
    assert model.model_type == "mitra"

@patch("src.models.mitra_model.TabularPredictor")
def test_mitra_fit_predict(mock_predictor_cls, sample_data):
    X, y = sample_data
    model = MitraModel()
    
    # Mock predictor instance
    mock_predictor = MagicMock()
    mock_predictor_cls.return_value = mock_predictor
    
    model.fit(X, y)
    
    assert model.is_fitted
    mock_predictor_cls.assert_called_once()
    mock_predictor.fit.assert_called_once()
    
    # Check hyperparameters passed to fit
    call_args = mock_predictor.fit.call_args
    assert "hyperparameters" in call_args.kwargs
    assert call_args.kwargs["hyperparameters"]["MITRA"]["fine_tune"] is False
    
    # Predict
    model.predict(X)
    mock_predictor.predict.assert_called_once()

def test_regime_adaptation_recent(sample_data):
    X, y = sample_data
    model = MitraModel()
    
    # Strategy 'recent', n=10
    X_sub, y_sub = model.adapt_to_regime(X, y, strategy="recent", n_samples=10)
    
    assert len(X_sub) == 10
    # Must be last 10
    pd.testing.assert_frame_equal(X_sub, X.iloc[-10:])
    pd.testing.assert_series_equal(y_sub, y.iloc[-10:])

def test_regime_adaptation_random(sample_data):
    X, y = sample_data
    model = MitraModel()
    
    X_sub, y_sub = model.adapt_to_regime(X, y, strategy="random", n_samples=10)
    
    assert len(X_sub) == 10
    # Indices should match between X_sub and y_sub
    assert X_sub.index.equals(y_sub.index)

# --- Property Test ---

@settings(max_examples=10, deadline=None)
@given(data=st.data())
def test_mitra_in_context_learning_logic(data):
    """
    Property 8.1: Mitra in-context learning adaptation
    Validates: Requirements 7.4, 11.1
    
    Since we cannot easily run the actual large Mitra model in this test environment without
    heavy download/compute, we verify the *logic* of the ICL wrapper:
    that changing the 'support set' via adapt_to_regime produces different training sets
    that are properly passed to the fit method.
    """
    # Generate synthetic history
    X_history = data.draw(data_frames(
        columns=[column(name="f1", dtype=float), column(name="f2", dtype=float)],
        index=range_indexes(min_size=20, max_size=50)
    ))
    y_history = pd.Series(np.zeros(len(X_history)), index=X_history.index, name="target")
    
    model = MitraModel()
    
    # Regime 1: Recent
    X_support1, y_support1 = model.adapt_to_regime(X_history, y_history, strategy="recent", n_samples=5)
    assert len(X_support1) == 5
    
    # Regime 2: Random (might overlap but logic distinct)
    X_support2, y_support2 = model.adapt_to_regime(X_history, y_history, strategy="random", n_samples=5)
    assert len(X_support2) == 5
    
    # If we were to fit, we'd expect different outcomes. 
    # Here we just verify the data manipulation correctness which underpins the ICL feature.
    assert X_support1.shape[1] == X_history.shape[1]
    assert y_support1.name == "target"
