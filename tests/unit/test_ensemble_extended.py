import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.models.ensemble import EnsembleModel
from src.models.base_model import BaseModel

@pytest.fixture
def mock_models():
    m1 = MagicMock(spec=BaseModel)
    m1.model_type = "Model1"
    m1.is_fitted = True
    m1.predict.return_value = np.array([1.0, 2.0, 3.0])
    m1.predict_proba.return_value = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
    m1.get_feature_importance.return_value = {"feat1": 0.5, "feat2": 0.5}
    m1.training_metrics = {"mae": 0.1}
    m1.validation_metrics = {"mae": 0.12}

    m2 = MagicMock(spec=BaseModel)
    m2.model_type = "Model2"
    m2.is_fitted = True
    m2.predict.return_value = np.array([2.0, 3.0, 4.0])
    m2.predict_proba.return_value = np.array([[0.4, 0.6], [0.5, 0.5], [0.6, 0.4]])
    m2.get_feature_importance.return_value = {"feat1": 0.7, "feat2": 0.3}
    m2.training_metrics = {"mae": 0.2}
    m2.validation_metrics = {"mae": 0.22}

    return [m1, m2]

def test_ensemble_weighted_average_predict(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    ensemble = EnsembleModel(models=mock_models, weights=[0.7, 0.3], method="weighted_average")
    
    preds = ensemble.predict(X)
    # 0.7 * [1, 2, 3] + 0.3 * [2, 3, 4] = [1.3, 2.3, 3.3]
    expected = np.array([1.3, 2.3, 3.3])
    np.testing.assert_allclose(preds, expected)

def test_ensemble_voting_predict(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    # Mock hard labels for voting
    mock_models[0].predict.return_value = np.array([0, 1, 0])
    mock_models[1].predict.return_value = np.array([1, 1, 1])
    
    ensemble = EnsembleModel(models=mock_models, method="voting")
    preds = ensemble.predict(X)
    # Voting: [0, 1] -> 0 or 1 (depending on tie breaking), but here [0, 1], [1, 1], [0, 1]
    # np.unique returns sorted unique values. 
    # For sample 0: [0, 1], counts [1, 1], argmax [0] -> 0
    # For sample 1: [1, 1], counts [2], argmax [1] -> 1
    # For sample 2: [0, 1], counts [1, 1], argmax [0] -> 0
    expected = np.array([0, 1, 0])
    np.testing.assert_array_equal(preds, expected)

def test_ensemble_predict_proba(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    ensemble = EnsembleModel(models=mock_models, weights=[0.6, 0.4], method="weighted_average")
    
    probas = ensemble.predict_proba(X)
    # 0.6 * [[0.1, 0.9], ...] + 0.4 * [[0.4, 0.6], ...]
    expected = 0.6 * mock_models[0].predict_proba.return_value + 0.4 * mock_models[1].predict_proba.return_value
    np.testing.assert_allclose(probas, expected)

def test_ensemble_predict_proba_no_support(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    mock_models[0].predict_proba.side_effect = NotImplementedError
    mock_models[1].predict_proba.side_effect = NotImplementedError
    
    ensemble = EnsembleModel(models=mock_models)
    with pytest.raises(NotImplementedError, match="No base models support predict_proba"):
        ensemble.predict_proba(X)

def test_ensemble_feature_importance(mock_models):
    ensemble = EnsembleModel(models=mock_models)
    importance = ensemble.get_feature_importance()
    
    # Mean of {"feat1": 0.5, "feat2": 0.5} and {"feat1": 0.7, "feat2": 0.3}
    expected = {"feat1": 0.6, "feat2": 0.4}
    assert importance == expected

def test_ensemble_aggregate_metrics(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    y = pd.Series([1, 0, 1])
    ensemble = EnsembleModel(models=mock_models)
    
    ensemble._aggregate_metrics()
    
    assert ensemble.training_metrics["mean_mae"] == pytest.approx(0.15)
    assert ensemble.validation_metrics["mean_mae"] == pytest.approx(0.17)

def test_ensemble_not_fitted_error(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3]})
    mock_models[0].is_fitted = False
    mock_models[1].is_fitted = False
    
    ensemble = EnsembleModel(models=mock_models)
    # Ensemble not fitted, and sub-models not all fitted
    with pytest.raises(ValueError, match="Model is not fitted"):
        ensemble.predict(X)

def test_ensemble_unknown_method(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3]})
    ensemble = EnsembleModel(models=mock_models, method="unknown")
    with pytest.raises(ValueError, match="Unknown ensemble method: unknown"):
        ensemble.predict(X)

def test_ensemble_init_weight_mismatch(mock_models):
    with pytest.raises(ValueError, match="Number of weights must match number of models"):
        EnsembleModel(models=mock_models, weights=[0.5], method="weighted_average")

def test_ensemble_init_default_weighted_average(mock_models):
    ensemble = EnsembleModel(models=mock_models, method="weighted_average")
    assert ensemble.weights == [0.5, 0.5]

def test_ensemble_fit(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3]})
    y = pd.Series([1, 0, 1])
    for m in mock_models:
        m.is_fitted = False
        m.training_time = 1.5

    ensemble = EnsembleModel(models=mock_models)
    assert not ensemble.is_fitted
    ensemble.fit(X, y)
    assert ensemble.is_fitted
    assert ensemble.training_time == pytest.approx(3.0)
    for m in mock_models:
        m.fit.assert_called_once()

def test_ensemble_predict_average(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3]})
    ensemble = EnsembleModel(models=mock_models, method="average")
    preds = ensemble.predict(X)
    # Mean of [1.0, 2.0, 3.0] and [2.0, 3.0, 4.0]
    expected = np.array([1.5, 2.5, 3.5])
    np.testing.assert_array_equal(preds, expected)

def test_ensemble_predict_weighted_average_no_weights(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3]})
    ensemble = EnsembleModel(models=mock_models, method="weighted_average")
    ensemble.weights = None # Force internal fallback 
    preds = ensemble.predict(X)
    expected = np.array([1.5, 2.5, 3.5])
    np.testing.assert_array_equal(preds, expected)

def test_ensemble_predict_auto_set_is_fitted(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3]})
    ensemble = EnsembleModel(models=mock_models)
    ensemble.is_fitted = False # Models are fitted, ensemble is not
    ensemble.predict(X)
    assert ensemble.is_fitted

def test_ensemble_predict_not_fitted_exception(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3]})
    mock_models[0].is_fitted = False
    ensemble = EnsembleModel(models=mock_models)
    ensemble.is_fitted = False
    with pytest.raises(ValueError, match="Model is not fitted"):
        ensemble.predict(X)

def test_ensemble_predict_proba_average(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3]})
    ensemble = EnsembleModel(models=mock_models, method="average")
    probas = ensemble.predict_proba(X)
    expected = 0.5 * mock_models[0].predict_proba.return_value + 0.5 * mock_models[1].predict_proba.return_value
    np.testing.assert_allclose(probas, expected)

def test_ensemble_predict_proba_weighted_average_no_weights(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3]})
    ensemble = EnsembleModel(models=mock_models, method="weighted_average")
    ensemble.weights = None # Force fallback
    probas = ensemble.predict_proba(X)
    expected = 0.5 * mock_models[0].predict_proba.return_value + 0.5 * mock_models[1].predict_proba.return_value
    np.testing.assert_allclose(probas, expected)

def test_ensemble_predict_proba_unknown_method(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3]})
    ensemble = EnsembleModel(models=mock_models)
    ensemble.method = "unknown"
    with pytest.raises(ValueError, match="Unknown ensemble method: unknown"):
        ensemble.predict_proba(X)

def test_ensemble_predict_proba_auto_set_is_fitted(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3]})
    ensemble = EnsembleModel(models=mock_models)
    ensemble.is_fitted = False
    ensemble.predict_proba(X)
    assert ensemble.is_fitted

def test_ensemble_predict_proba_not_fitted_exception(mock_models):
    X = pd.DataFrame({"feat1": [1, 2, 3]})
    mock_models[0].is_fitted = False
    ensemble = EnsembleModel(models=mock_models)
    ensemble.is_fitted = False
    with pytest.raises(ValueError, match="Model is not fitted"):
        ensemble.predict_proba(X)

def test_ensemble_get_feature_importance_auto_set_is_fitted(mock_models):
    ensemble = EnsembleModel(models=mock_models)
    ensemble.is_fitted = False
    ensemble.get_feature_importance()
    assert ensemble.is_fitted

def test_ensemble_get_feature_importance_not_fitted(mock_models):
    mock_models[0].is_fitted = False
    ensemble = EnsembleModel(models=mock_models)
    ensemble.is_fitted = False
    importance = ensemble.get_feature_importance()
    assert importance == {}

def test_ensemble_get_feature_importance_empty(mock_models):
    mock_models[0].get_feature_importance.return_value = {}
    mock_models[1].get_feature_importance.return_value = {}
    ensemble = EnsembleModel(models=mock_models)
    importance = ensemble.get_feature_importance()
    assert importance == {}
