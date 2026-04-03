"""Extended unit tests for XGBoostModel coverages including optuna."""

import logging
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_model import XGBoostModel


@pytest.fixture
def regression_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n) * 2})
    y = pd.Series(X["f1"] + X["f2"] * 0.5 + np.random.randn(n) * 0.1, name="target")
    return X, y


@pytest.fixture
def classification_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n) * 2})
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int), name="target")
    return X, y


class TestXGBoostInit:
    def test_init_with_custom_hyperparameters(self):
        model = XGBoostModel(
            hyperparameters={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05}
        )
        assert model.hyperparameters["n_estimators"] == 50
        assert model.hyperparameters["max_depth"] == 3
        assert model.hyperparameters["learning_rate"] == 0.05


class TestXGBoostGuards:
    def test_predict_not_fitted(self):
        model = XGBoostModel()
        X = pd.DataFrame({"f1": [1, 2]})
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)

    def test_predict_proba_not_fitted(self):
        model = XGBoostModel()
        X = pd.DataFrame({"f1": [1, 2]})
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict_proba(X)

    def test_predict_proba_not_classification(self, regression_data):
        X, y = regression_data
        model = XGBoostModel(objective="reg:squarederror")
        model.fit(X, y)
        with pytest.raises(
            NotImplementedError, match="predict_proba only supported for classification"
        ):
            model.predict_proba(X)

    def test_get_feature_importance_not_fitted(self):
        model = XGBoostModel()
        assert model.get_feature_importance() == {}

    def test_get_shap_values_not_fitted(self):
        model = XGBoostModel()
        X = pd.DataFrame({"f1": [1, 2]})
        with pytest.raises(ValueError, match="Model not fitted"):
            model.get_shap_values(X)


class TestXGBoostFit:
    def test_fit_with_validation_data(self, regression_data):
        X, y = regression_data
        model = XGBoostModel(hyperparameters={"n_estimators": 2})
        model.fit(X, y, validation_data=(X, y))
        assert model.is_fitted

    @patch("shap.TreeExplainer")
    def test_fit_shap_initialization_fails_gracefully(
        self, mock_explainer, regression_data
    ):
        mock_explainer.side_effect = Exception("Mocked SHAP failure")
        X, y = regression_data
        model = XGBoostModel(hyperparameters={"n_estimators": 2})
        model.fit(X, y)
        assert model.is_fitted
        assert model._explainer is None


class TestXGBoostOptimize:
    def test_optimize_regression_rmse(self, regression_data):
        X, y = regression_data
        model = XGBoostModel(objective="reg:squarederror")
        # Fast optimization
        model.fit(
            X,
            y,
            optimize=True,
            optimization_params={"n_trials": 2, "n_splits": 2, "metric": "rmse"},
        )
        assert model.is_fitted
        assert "max_depth" in model.hyperparameters

    def test_optimize_classification_accuracy(self, classification_data):
        X, y = classification_data
        model = XGBoostModel(objective="binary:logistic")
        model.fit(
            X,
            y,
            optimize=True,
            optimization_params={"n_trials": 2, "n_splits": 2, "metric": "accuracy"},
        )
        assert model.is_fitted
        assert "max_depth" in model.hyperparameters

    def test_optimize_classification_logloss(self, classification_data):
        X, y = classification_data
        model = XGBoostModel(objective="binary:logistic")
        model.fit(
            X,
            y,
            optimize=True,
            optimization_params={"n_trials": 2, "n_splits": 2, "metric": "logloss"},
        )
        assert model.is_fitted
        assert "max_depth" in model.hyperparameters

    def test_optimize_fallback_metric(self, regression_data, caplog):
        X, y = regression_data
        model = XGBoostModel(objective="reg:squarederror")
        # Unknown metric falls back to rmse
        with caplog.at_level(logging.WARNING):
            model.fit(
                X,
                y,
                optimize=True,
                optimization_params={"n_trials": 1, "n_splits": 2, "metric": "unknown"},
            )
        assert model.is_fitted
        assert (
            "Unknown metric 'unknown' provided. Falling back to 'rmse'." in caplog.text
        )

    def test_optimize_regression_logloss_raises(self, regression_data):
        X, y = regression_data
        model = XGBoostModel(objective="reg:squarederror")
        with pytest.raises(
            ValueError,
            match="does not support 'predict_proba', but metric 'logloss' was requested",
        ):
            model.fit(
                X,
                y,
                optimize=True,
                optimization_params={"n_trials": 1, "n_splits": 2, "metric": "logloss"},
            )

    @patch("optuna.create_study")
    def test_optimize_time_limit(self, mock_create_study, regression_data):
        X, y = regression_data
        model = XGBoostModel(objective="reg:squarederror")
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study

        test_time_limit = 10

        model.fit(
            X,
            y,
            optimize=True,
            optimization_params={
                "n_trials": 1,
                "n_splits": 2,
                "time_limit": test_time_limit,
            },
        )

        mock_study.optimize.assert_called_once_with(
            ANY, n_trials=1, timeout=test_time_limit
        )
