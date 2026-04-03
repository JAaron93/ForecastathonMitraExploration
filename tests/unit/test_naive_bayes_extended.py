"""Extended unit tests for NaiveBayesModel to increase coverage."""
import pytest
import numpy as np
import pandas as pd

from src.models.naive_bayes import NaiveBayesModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_data():
    np.random.seed(0)
    n = 150
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n) * 2,
        "f3": np.random.randn(n) + 1,
    })
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int), name="label")
    return X, y


@pytest.fixture
def fitted_model(classification_data):
    X, y = classification_data
    model = NaiveBayesModel()
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_hyperparameters(self):
        model = NaiveBayesModel()
        assert model.model_object is not None
        assert model._feature_importances == {}

    def test_custom_var_smoothing(self):
        model = NaiveBayesModel(hyperparameters={"var_smoothing": 1e-5})
        assert model.model_object.var_smoothing == pytest.approx(1e-5)


# ---------------------------------------------------------------------------
# fit / predict / predict_proba
# ---------------------------------------------------------------------------

class TestFitPredict:
    def test_fit_succeeds(self, classification_data):
        X, y = classification_data
        model = NaiveBayesModel()
        returned = model.fit(X, y)
        assert returned is model
        assert model.is_fitted

    def test_fit_with_float_y_converts_to_int(self, classification_data):
        X, y = classification_data
        y_float = y.astype(float)  # e.g. 0.0, 1.0
        model = NaiveBayesModel()
        model.fit(X, y_float)
        assert model.is_fitted

    def test_fit_with_validation_data(self, classification_data):
        X, y = classification_data
        split = len(X) // 5
        X_val, y_val = X.iloc[:split], y.iloc[:split]
        X_train, y_train = X.iloc[split:], y.iloc[split:]
        model = NaiveBayesModel()
        model.fit(X_train, y_train, validation_data=(X_val, y_val))
        assert model.is_fitted

    def test_predict_not_fitted_raises(self):
        model = NaiveBayesModel()
        X = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)

    def test_predict_proba_not_fitted_raises(self):
        model = NaiveBayesModel()
        X = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict_proba(X)

    def test_predict_returns_array(self, fitted_model):
        model, X, _ = fitted_model
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)

    def test_predict_proba_shape(self, fitted_model):
        model, X, _ = fitted_model
        probas = model.predict_proba(X)
        assert probas.shape == (len(X), 2)  # binary classification

    def test_predict_with_reordered_columns(self, fitted_model):
        model, X, _ = fitted_model
        # Get baseline predictions with original column order
        baseline_preds = model.predict(X)
        
        # Reorder columns
        X_reordered = X[["f3", "f1", "f2"]]
        preds = model.predict(X_reordered)
        
        # Verify predictions are the same regardless of column order
        assert len(preds) == len(X)
        np.testing.assert_allclose(preds, baseline_preds)

    def test_predict_with_missing_feature_raises(self, fitted_model):
        model, X, _ = fitted_model
        X_missing = X[["f1", "f2"]]  # drop f3
        with pytest.raises(ValueError, match="Missing features"):
            model.predict(X_missing)


# ---------------------------------------------------------------------------
# feature importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_get_feature_importance_before_permutation(self, fitted_model):
        model, X, _ = fitted_model
        # Should return empty dict until calculate_permutation_importance is called
        importance = model.get_feature_importance()
        assert importance == {}

    def test_calculate_permutation_importance(self, fitted_model):
        model, X, y = fitted_model
        importance = model.calculate_permutation_importance(X, y, n_repeats=3)
        assert set(importance.keys()) == set(X.columns)

    def test_permutation_importance_not_fitted_raises(self, classification_data):
        X, y = classification_data
        model = NaiveBayesModel()
        with pytest.raises(ValueError, match="Model not fitted"):
            model.calculate_permutation_importance(X, y)

    def test_get_feature_importance_after_permutation(self, fitted_model):
        model, X, y = fitted_model
        model.calculate_permutation_importance(X, y, n_repeats=2)
        importance = model.get_feature_importance()
        assert len(importance) == 3  # f1, f2, f3

