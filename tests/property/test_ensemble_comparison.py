"""Property tests for model ensemble and comparison."""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from src.models.base_model import BaseModel
from src.models.ensemble import EnsembleModel
from src.evaluation.comparison import ModelComparator


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, constant_pred: float, model_type: str = "Mock"):
        super().__init__(model_id=f"{model_type}_{constant_pred}")
        self._model_type = model_type
        self.constant_pred = constant_pred
        self.is_fitted = True
        
    @property
    def model_type(self) -> str:
        return self._model_type
        
    def fit(self, X, y, **kwargs):
        return self
        
    def predict(self, X):
        return np.full(len(X), self.constant_pred)
        
    def predict_proba(self, X):
        # Mock proba: return [1-p, p] where p is constant_pred (clamped 0-1)
        p = np.clip(self.constant_pred, 0, 1)
        # Create array of shape (n, 2)
        return np.array([[1-p, p]] * len(X))

    def get_feature_importance(self):
        return {"feat1": 0.5}


@given(
    # Generate list of weights (potentially unnormalized)
    st.lists(st.floats(min_value=0.1, max_value=1.0), min_size=2, max_size=5),
    # Generate predictions for each model
    st.lists(st.floats(min_value=0, max_value=100), min_size=2, max_size=5),
    # Generate random dataframe
    data_frames(columns=[column('A', dtype=float)], index=range_indexes(min_size=1, max_size=10))
)
@settings(max_examples=50, deadline=None)
def test_property_7_ensemble_weighted_average(weights, preds, X):
    """
    Property 7: Ensemble consistency (Weighted Average).
    
    Verify that weighted average ensemble produces strictly the 
    weighted average of component model predictions.
    """
    # Ensure dimensions match
    min_len = min(len(weights), len(preds))
    weights = weights[:min_len]
    preds = preds[:min_len]
    
    # Create models
    models = [MockModel(p) for p in preds]
    
    # Create ensemble
    ensemble = EnsembleModel(models=models, weights=weights, method="weighted_average")
    
    # Predict
    result = ensemble.predict(X)
    
    # Expected: weighted average of constants
    # Weights are normalized inside EnsembleModel
    norm_weights = np.array(weights) / sum(weights)
    expected_val = np.average(preds, weights=norm_weights)
    
    # Check all predictions match expected value
    assert np.allclose(result, expected_val)


@given(
    st.lists(st.integers(min_value=0, max_value=2), min_size=3, max_size=5),
    data_frames(columns=[column('A', dtype=float)], index=range_indexes(min_size=1, max_size=10))
)
@settings(max_examples=50, deadline=None)
def test_property_7_ensemble_voting(preds, X):
    """
    Property 7: Ensemble consistency (Voting).
    
    Verify that voting ensemble returns the majority class.
    """
    # Create models with integer predictions (classes)
    models = [MockModel(p, model_type="Classifier") for p in preds]
    
    ensemble = EnsembleModel(models=models, method="voting")
    
    result = ensemble.predict(X)
    
    # Expected: Mode of predictions
    values, counts = np.unique(preds, return_counts=True)
    expected_mode = values[np.argmax(counts)]
    
    assert np.all(result == expected_mode)


def test_model_comparator_metrics():
    """
    Test ModelComparator metric aggregation.
    """
    # Setup
    m1 = MockModel(1.0)
    m1.validation_metrics = {"rmse": 0.5, "mae": 0.4}
    m1.training_metrics = {"rmse": 0.1}
    
    m2 = MockModel(2.0)
    m2.validation_metrics = {"rmse": 0.6, "mae": 0.5}
    m2.training_metrics = {"rmse": 0.2}
    
    comparator = ModelComparator()
    comparator.add_model(m1, "m1")
    comparator.add_model(m2, "m2")
    
    # Compare
    df = comparator.compare_metrics()
    
    assert "rmse" in df.columns
    assert "mae" in df.columns
    assert "train_rmse" in df.columns
    assert df.loc["m1", "rmse"] == 0.5
    assert df.loc["m2", "train_rmse"] == 0.2


# Simple integration test for robustness
def test_robustness_analysis():
    # Mock data
    X = pd.DataFrame({
        "feature": np.random.randn(100),
        "regime": ["bull"]*50 + ["bear"]*50
    })
    y = pd.Series(np.random.randn(100))
    
    m1 = MockModel(0.0) # Flat predictor
    
    comparator = ModelComparator()
    comparator.add_model(m1, "baseline")
    
    # Because MockModel.predict returns constant, and y is random, 
    # calculate_regression_metrics will run fine.
    
    results = comparator.analyze_robustness(X, y, regime_col="regime")
    
    assert "bull" in results
    assert "bear" in results
    assert not results["bull"].empty
    assert "mse" in results["bull"].columns or "rmse" in results["bull"].columns

def test_instantiate_model_factory():
    """
    Test _instantiate_model factory logic.
    """
    comparator = ModelComparator()
    
    # CASE 1: EnsembleModel should get models=[] automatically
    meta_ensemble = {"model_type": "Ensemble", "hyperparameters": {"method": "voting"}}
    model = comparator._instantiate_model(EnsembleModel, meta_ensemble)
    assert isinstance(model, EnsembleModel)
    assert model.models == []
    assert model.method == "voting"
    
    # CASE 2: Model with required args should fail with clear message if args missing
    # MockModel requires constant_pred
    meta_mock = {"model_type": "Mock"}
    
    with pytest.raises(ValueError) as excinfo:
        comparator._instantiate_model(MockModel, meta_mock)
    
    assert "Failed to instantiate MockModel" in str(excinfo.value)
    assert "Ensure metadata contains all required constructor arguments" in str(excinfo.value)
