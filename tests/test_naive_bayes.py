"""
Tests for Naive Bayes model and utilities.
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes

from src.models.naive_bayes import NaiveBayesModel
from src.features.discretization import LabelDiscretizer
from src.features.selection import FeatureSelector

# --- Fixtures ---

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    y = pd.Series(np.random.choice([0, 1], size=100), name='target')
    return X, y

# --- Unit Tests: LabelDiscretizer ---

def test_label_discretizer_direction():
    y = np.array([-0.5, 0.5, 0.0, 1.0, -1.0])
    disc = LabelDiscretizer(strategy="direction", threshold=0.1)
    labels = disc.fit_transform(y)
    
    # -0.5 < -0.1 -> -1 (Down)
    # 0.5 > 0.1 -> 1 (Up)
    # 0.0 -> 0 (Neutral)
    # 1.0 > 0.1 -> 1 (Up)
    # -1.0 < -0.1 -> -1 (Down)
    expected = np.array([-1, 1, 0, 1, -1])
    np.testing.assert_array_equal(labels, expected)

def test_label_discretizer_quantile():
    y = np.linspace(0, 1, 100)
    disc = LabelDiscretizer(strategy="quantile", n_bins=4)
    labels = disc.fit_transform(y)
    
    assert len(np.unique(labels)) == 4
    # Should be roughly equal counts
    counts = pd.Series(labels).value_counts()
    assert (counts >= 24).all() and (counts <= 26).all() # approximate

# --- Unit Tests: FeatureSelector ---

def test_feature_selector_correlation():
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [5, 4, 3, 2, 1], # Perfectly neg corr
        'c': [1, 1, 0, 0, 1]  # Random-ish
    })
    target = pd.Series([1, 2, 3, 4, 5])
    
    selected = FeatureSelector.select_by_correlation(df, target, n_features=1)
    # 'a' has corr 1.0, 'b' has corr -1.0. abs corr are equal.
    # implementation might pick either, but likely 'a' or 'b'
    assert len(selected) == 1
    assert selected[0] in ['a', 'b']

# --- Unit Tests: NaiveBayesModel ---

def test_naive_bayes_fit_predict(sample_data):
    X, y = sample_data
    model = NaiveBayesModel()
    model.fit(X, y)
    
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})
    
    probs = model.predict_proba(X)
    assert probs.shape == (len(y), 2)

def test_naive_bayes_feature_importance(sample_data):
    X, y = sample_data
    model = NaiveBayesModel()
    model.fit(X, y)
    
    importances = model.calculate_permutation_importance(X, y, n_repeats=2)
    assert len(importances) == 2
    assert "feature1" in importances
    assert "feature2" in importances

# --- Property Test ---

@given(data=st.data())
def test_naive_bayes_training_correctness(data):
    """
    **Property 5: Model training and evaluation correctness (Naive Bayes component)**
    Validates: Requirements 4.1, 4.3
    """
    # Generate random training data
    # Features
    X_df = data.draw(data_frames(
        columns=[column(name=f"c{i}", dtype=float, elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e5, max_value=1e5)) for i in range(3)],
        index=range_indexes(min_size=10, max_size=50)
    ))
    
    # Target (2 or 3 classes) - ensure at least 2 classes are present
    y_list = data.draw(st.lists(st.integers(min_value=0, max_value=1), min_size=len(X_df), max_size=len(X_df)))
    if len(set(y_list)) < 2:
        # Force at least two classes if mostly random gen failed to do so (unlikely but possible with small N)
        y_list[0] = 0
        y_list[1] = 1
        
    y_series = pd.Series(y_list, index=X_df.index, name='target')
    
    # Use higher var_smoothing to avoid divide-by-zero validation errors on synthetic data
    model = NaiveBayesModel(hyperparameters={"var_smoothing": 1e-5})
    
    # Train
    try:
        model.fit(X_df, y_series)
    except Exception as e:
        # Should not fail for valid input
        pytest.fail(f"Training failed with valid input: {e}")
        
    assert model.is_fitted
    
    # Predict
    preds = model.predict(X_df)
    assert len(preds) == len(y_series)
    
    # Verify probabilities sum to 1
    proba = model.predict_proba(X_df)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    
    # Verify feature importance produces valid results
    imps = model.calculate_permutation_importance(X_df, y_series, n_repeats=2)
    assert len(imps) == X_df.shape[1]
    
