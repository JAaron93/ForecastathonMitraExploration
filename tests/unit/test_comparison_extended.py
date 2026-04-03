"""Extended unit tests for ModelComparator to improve coverage."""

import json
import os
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.comparison import ModelComparator
from src.evaluation.metrics import MetricsCalculator
from src.models.base_model import BaseModel, ModelArtifact


class MockModel(BaseModel):
    def __init__(self, is_fitted=True, metadata=None, **kwargs):
        super().__init__(**kwargs)
        self.is_fitted = is_fitted
        self.metadata = metadata or {}

    @property
    def model_type(self) -> str:
        return "mock"

    def fit(self, X, y, **kwargs):
        self.is_fitted = True
        return self

    def predict(self, X):
        return np.array([1, 0, 1] * (len(X) // 3 + 1))[: len(X)]

    def predict_proba(self, X):
        return np.array([[0.5, 0.5]] * len(X))

    def get_feature_importance(self):
        return {"feat": 0.5}


@pytest.fixture
def comparator():
    tracker = MagicMock()
    tracker.artifact_location = "/tmp/artifacts"
    return ModelComparator(experiment_tracker=tracker)


def test_compare_metrics_filter(comparator):
    m1 = MockModel()
    m1.training_metrics = {"rmse": 1.0, "mae": 0.5}
    m1.validation_metrics = {"rmse": 1.2, "mae": 0.6}

    comparator.add_model(m1, "model1")

    df = comparator.compare_metrics(metric_names=["rmse"])
    assert "train_rmse" in df.columns
    assert "rmse" in df.columns
    assert "mae" not in df.columns
    assert "train_mae" not in df.columns


def test_compare_metrics_empty(comparator):
    df = comparator.compare_metrics()
    assert df.empty


# --- analyze_robustness ---


def test_analyze_robustness_empty_subset(comparator):
    m1 = MockModel()
    comparator.add_model(m1, "m1")

    X = pd.DataFrame({"feat": [1, 2], "regime": ["A", "B"]})
    y = pd.Series([0, 1])

    # If we filter by 'C', it will be empty and should continue
    # We will test an empty regime condition by patching the loop or just passing an empty DF
    # But analyze_robustness splits by X[regime_col].unique(), so empty subsets only happen if
    # the index alignment leads to it or similar. Let's force an empty X.
    X_empty = pd.DataFrame()
    y_empty = pd.Series(dtype=float)
    results = comparator.analyze_robustness(X_empty, y_empty, regime_col="regime")
    assert results == {}


def test_analyze_robustness_metadata_inference(comparator):
    X = pd.DataFrame({"feat": [1, 2, 3]})

    # 1. Regression via metadata
    m_reg = MockModel(metadata={"task_type": "regression"})
    m_reg.predict = MagicMock(return_value=np.array([0.1, 0.5, 0.9]))
    y_reg = pd.Series([0.2, 0.4, 0.8])

    comparator.add_model(m_reg, "reg_model")
    results = comparator.analyze_robustness(X, y_reg)
    assert "global" in results
    assert "rmse" in results["global"].columns

    # 2. Classification via metadata
    comparator.loaded_models.clear()
    m_clf = MockModel(metadata={"task_type": "classification"})
    m_clf.predict = MagicMock(return_value=np.array([0, 1, 0]))
    y_clf = pd.Series([0, 1, 0])

    comparator.add_model(m_clf, "clf_model")
    results = comparator.analyze_robustness(X, y_clf)
    assert "accuracy" in results["global"].columns

    # 3. Fallback to model_type strings
    class FallbackModel(BaseModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.is_fitted = False
            self.objective = ""

        @property
        def model_type(self):
            return "xgboost"

        def fit(self, X, y, **kwargs):
            return self

        def predict(self, X):
            return np.array([0.1, 0.5, 0.9])

        def predict_proba(self, X):
            return np.zeros((len(X), 2))

        def get_feature_importance(self):
            return {}

    comparator.loaded_models.clear()
    m_fallback = FallbackModel()
    m_fallback.is_fitted = True
    m_fallback.objective = "reg:squarederror"

    comparator.add_model(m_fallback, "fb_model")
    results = comparator.analyze_robustness(
        X, y_clf
    )  # Pass discrete y but metadata says regression
    assert "rmse" in results["global"].columns


def test_analyze_robustness_exception_handling(comparator):
    X = pd.DataFrame({"feat": [1, 2, 3]})
    y = pd.Series([0, 1, 0])

    m_fail = MockModel()
    m_fail.predict = MagicMock(side_effect=ValueError("Prediction failed"))

    comparator.add_model(m_fail, "m_fail")
    results = comparator.analyze_robustness(X, y)
    assert results == {}  # Handled exception, no metrics added


# --- load_artifacts ---


def test_load_artifacts_no_tracker():
    comp = ModelComparator(experiment_tracker=None)
    with pytest.raises(ValueError, match="ExperimentTracker not provided"):
        comp.load_artifacts(["run1"])


def test_load_artifacts_run_not_found(comparator):
    comparator.tracker.get_run.return_value = None
    comparator.load_artifacts(["run1"])
    assert "run1" not in comparator.loaded_models


@patch("os.path.exists")
def test_load_artifacts_missing_metadata(mock_exists, comparator):
    comparator.tracker.get_run.return_value = {"id": "run1"}
    mock_exists.return_value = False
    comparator.load_artifacts(["run1"])
    assert "run1" not in comparator.loaded_models


@patch("os.path.exists")
@patch("builtins.open")
def test_load_artifacts_empty_metadata(open_mock, mock_exists, comparator):
    comparator.tracker.get_run.return_value = {"id": "run1"}
    mock_exists.return_value = True
    # Return empty JSON string or {}
    open_mock.side_effect = mock_open(read_data="{}")
    comparator.load_artifacts(["run1"])
    assert "run1" not in comparator.loaded_models


@patch("os.path.exists")
@patch("builtins.open")
def test_load_artifacts_missing_model_type(open_mock, mock_exists, comparator):
    comparator.tracker.get_run.return_value = {"id": "run1"}
    mock_exists.return_value = True
    meta = {"some_other_key": "val"}
    open_mock.side_effect = mock_open(read_data=json.dumps(meta))
    comparator.load_artifacts(["run1"])
    assert "run1" not in comparator.loaded_models


@patch("os.path.exists")
@patch("builtins.open")
def test_load_artifacts_json_decode_error(open_mock, mock_exists, comparator):
    comparator.tracker.get_run.return_value = {"id": "run1"}
    mock_exists.return_value = True
    open_mock.side_effect = mock_open(read_data="invalid json")
    comparator.load_artifacts(["run1"])
    assert "run1" not in comparator.loaded_models


# --- _instantiate_model ---


def test_instantiate_model_type_error_unrecognized_kwargs():
    comp = ModelComparator()

    class StrictModel(BaseModel):
        def __init__(self, **kwargs):
            if "invalid" in kwargs:
                raise TypeError(
                    "__init__() got an unexpected keyword argument 'invalid'"
                )

        @property
        def model_type(self) -> str:
            return "strict"

        def fit(self, X, y, **kwargs):
            return self

        def predict(self, X):
            return np.zeros(len(X))

        def predict_proba(self, X):
            return np.zeros((len(X), 2))

        def get_feature_importance(self):
            return {}

    meta = {"model_id": "test", "hyperparameters": {}, "invalid": "kwarg"}

    # Tests the fallback block
    model = comp._instantiate_model(StrictModel, meta)
    assert isinstance(model, StrictModel)


def test_instantiate_model_type_error_missing_kwargs():
    comp = ModelComparator()

    class MissingKwargModel(BaseModel):
        def __init__(self, required_arg, **kwargs):
            pass

        @property
        def model_type(self) -> str:
            return "missing"

        def fit(self, X, y, **kwargs):
            return self

        def predict(self, X):
            return np.zeros(len(X))

        def predict_proba(self, X):
            return np.zeros((len(X), 2))

        def get_feature_importance(self):
            return {}

    meta = {"model_id": "test"}
    with pytest.raises(ValueError, match="Failed to instantiate MissingKwargModel"):
        comp._instantiate_model(MissingKwargModel, meta)


def test_instantiate_ensemble():
    comp = ModelComparator()
    from src.models.ensemble import EnsembleModel

    meta = {
        "model_type": "Ensemble",
        "hyperparameters": {"method": "voting", "weights": [0.5, 0.5]},
        "models": [MockModel(), MockModel()],
    }

    model = comp._instantiate_model(EnsembleModel, meta)
    assert isinstance(model, EnsembleModel)
    assert model.method == "voting"
    assert model.weights == [0.5, 0.5]
