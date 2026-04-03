"""Extended unit tests for MitraModel covering save/load and adapt_to_regime."""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.mitra_model import MitraModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_context():
    """Returns a (X_context, y_context) pair large enough for all strategies."""
    np.random.seed(42)
    n = 300
    index = pd.date_range("2023-01-01", periods=n, freq="D")
    X = pd.DataFrame(
        {"open": np.random.rand(n), "close": np.random.rand(n)}, index=index
    )
    y = pd.Series(np.random.randn(n), index=index, name="returns")
    return X, y


@pytest.fixture
def small_context():
    """Fewer rows than n_samples so the function returns all data."""
    np.random.seed(42)
    n = 20
    index = pd.date_range("2023-01-01", periods=n, freq="D")
    X = pd.DataFrame({"open": np.random.rand(n)}, index=index)
    y = pd.Series(np.random.randn(n), index=index)
    return X, y


# ---------------------------------------------------------------------------
# adapt_to_regime tests
# ---------------------------------------------------------------------------


class TestAdaptToRegime:
    def test_returns_all_when_insufficient_data(self, small_context):
        model = MitraModel()
        X, y = small_context
        X_s, y_s = model.adapt_to_regime(X, y, strategy="recent", n_samples=100)
        # Should return full data since len(X) <= n_samples
        assert len(X_s) == len(X)

    def test_recent_strategy_returns_last_n(self, sample_context):
        model = MitraModel()
        X, y = sample_context
        X_s, y_s = model.adapt_to_regime(X, y, strategy="recent", n_samples=50)
        assert len(X_s) == 50
        pd.testing.assert_frame_equal(X_s, X.iloc[-50:])

    def test_random_strategy_returns_n_sorted(self, sample_context):
        model = MitraModel()
        X, y = sample_context
        X_s, y_s = model.adapt_to_regime(X, y, strategy="random", n_samples=50)
        assert len(X_s) == 50
        # Indices must be in sorted order to preserve temporal ordering
        assert list(X_s.index) == sorted(X_s.index)

    def test_volatility_matching_fallback_no_target(self, sample_context):
        model = MitraModel()
        X, y = sample_context
        X_s, y_s = model.adapt_to_regime(
            X, y, strategy="volatility_matching", n_samples=50, target_volatility=None
        )
        # Falls back to recent
        assert len(X_s) == 50
        pd.testing.assert_frame_equal(X_s, X.iloc[-50:])

    def test_volatility_matching_with_target(self, sample_context):
        model = MitraModel()
        X, y = sample_context
        X_s, y_s = model.adapt_to_regime(
            X,
            y,
            strategy="volatility_matching",
            n_samples=50,
            target_volatility=0.5,
            volatility_window=10,
        )
        assert len(X_s) == 50
        # Temporal order preserved
        assert list(X_s.index) == sorted(X_s.index)

    def test_volatility_matching_insufficient_valid_samples(self, sample_context):
        """When volatility_window > available rows after dropna, falls back to recent."""
        model = MitraModel()
        X, y = sample_context
        # Enormous window leaves very few valid rolling std values
        X_s, y_s = model.adapt_to_regime(
            X,
            y,
            strategy="volatility_matching",
            n_samples=200,
            target_volatility=0.5,
            volatility_window=290,  # leaves only ~10 valid rows
        )
        # Falls back to recent
        assert len(X_s) == 200

    def test_invalid_strategy_raises(self, sample_context):
        model = MitraModel()
        X, y = sample_context
        with pytest.raises(ValueError, match="Unknown adaptation strategy"):
            model.adapt_to_regime(X, y, strategy="bogus")

    def test_invalid_volatility_window_raises(self, sample_context):
        model = MitraModel()
        X, y = sample_context
        with pytest.raises(ValueError, match="volatility_window must be positive"):
            model.adapt_to_regime(
                X,
                y,
                strategy="volatility_matching",
                target_volatility=0.5,
                volatility_window=0,
            )

    def test_negative_target_volatility_raises(self, sample_context):
        model = MitraModel()
        X, y = sample_context
        with pytest.raises(ValueError, match="target_volatility cannot be negative"):
            model.adapt_to_regime(
                X,
                y,
                strategy="volatility_matching",
                target_volatility=-1.0,
                volatility_window=5,
            )


# ---------------------------------------------------------------------------
# predict / predict_proba guard tests (no AutoGluon needed)
# ---------------------------------------------------------------------------


class TestPredictGuards:
    def test_predict_raises_when_not_fitted(self):
        model = MitraModel()
        X = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)

    def test_predict_proba_raises_when_not_fitted(self):
        model = MitraModel()
        X = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict_proba(X)

    def test_get_feature_importance_returns_empty_when_not_fitted(self):
        model = MitraModel()
        assert model.get_feature_importance() == {}

    def test_save_model_raises_when_not_fitted(self, tmp_path):
        model = MitraModel()
        with pytest.raises(ValueError, match="Cannot save unfitted model"):
            model.save_model(str(tmp_path))


# ---------------------------------------------------------------------------
# save_model / load_model with mocked predictor
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_model_creates_metadata_json(self, tmp_path):
        model = MitraModel(model_id="test-save", label_column="target")
        mock_predictor = MagicMock()
        mock_predictor.path = str(tmp_path / "ag_model_src")
        model.predictor = mock_predictor
        model.is_fitted = True
        model.feature_names = ["open", "close"]

        # Mock clone to create the destination directory
        def mock_clone(path):
            Path(path).mkdir(parents=True, exist_ok=True)

        mock_predictor.clone.side_effect = mock_clone

        # Mock get_artifact so we can control the JSON output
        with patch.object(model, "get_artifact") as mock_artifact:
            mock_artifact.return_value.to_dict.return_value = {
                "model_id": "test-save",
                "hyperparameters": {"fine_tune": False},
                "label_column": "target",
                "problem_type": "auto",
                "eval_metric": None,
                "feature_names": ["open", "close"],
                "training_metrics": {},
            }
            save_dir = tmp_path / "saved_model"
            model.save_model(str(save_dir))

        assert (save_dir / "metadata.json").exists()
        with open(save_dir / "metadata.json") as f:
            data = json.load(f)
        assert data["model_id"] == "test-save"

    def test_load_model_restores_state(self, tmp_path):
        save_dir = tmp_path / "mitra_model"
        ag_model_dir = save_dir / "ag_model"
        ag_model_dir.mkdir(parents=True)

        metadata = {
            "model_id": "test-load",
            "hyperparameters": {"fine_tune": True},
            "label_column": "target",
            "problem_type": "regression",
            "eval_metric": "rmse",
            "feature_names": ["f1", "f2"],
            "training_metrics": {"rmse": 0.05},
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        model = MitraModel()
        mock_predictor = MagicMock()
        with patch("src.models.mitra_model.TabularPredictor") as MockAG:
            MockAG.load.return_value = mock_predictor
            model.load_model(str(save_dir))

        assert model.model_id == "test-load"
        assert model.feature_names == ["f1", "f2"]
        assert model.problem_type == "regression"
        assert model.eval_metric == "rmse"
        assert model.is_fitted is True
        assert model.training_metrics["rmse"] == pytest.approx(0.05)
