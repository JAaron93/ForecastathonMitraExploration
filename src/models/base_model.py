"""Base model interface for all forecasting models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifact:
    """Container for model artifacts and metadata."""
    model_id: str
    model_type: str
    model_object: Any
    hyperparameters: Dict[str, Any]
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    training_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact metadata to dictionary (excludes model object)."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "feature_importance": self.feature_importance,
            "feature_names": self.feature_names,
            "training_time": self.training_time,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class BaseModel(ABC):
    """Abstract base class for all forecasting models."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base model.

        Args:
            model_id: Unique identifier for the model
            hyperparameters: Model hyperparameters
        """
        self.model_id = model_id or self._generate_model_id()
        self.hyperparameters = hyperparameters or {}
        self.model_object: Any = None
        self.is_fitted: bool = False
        self.feature_names: List[str] = []
        self.training_metrics: Dict[str, float] = {}
        self.validation_metrics: Dict[str, float] = {}
        self.training_time: float = 0.0
        self._created_at: datetime = datetime.now()

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier."""
        pass

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple] = None,
        **kwargs
    ) -> "BaseModel":
        """
        Fit the model to training data.

        Args:
            X: Feature DataFrame
            y: Target Series
            validation_data: Optional tuple of (X_val, y_val)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for input data.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions for classification models.

        Args:
            X: Feature DataFrame

        Returns:
            Array of probability predictions

        Raises:
            NotImplementedError: If model doesn't support probability predictions
        """
        raise NotImplementedError(
            f"{self.model_type} does not support probability predictions"
        )

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass

    def save_model(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model artifacts
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model object
        model_path = save_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model_object, f)

        # Save metadata
        artifact = self.get_artifact()
        metadata_path = save_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(artifact.to_dict(), f, indent=2)

        logger.info(f"Model saved to {save_dir}")

    def load_model(self, path: str) -> "BaseModel":
        """
        Load model from disk.

        Args:
            path: Directory path containing model artifacts

        Returns:
            Self with loaded model
        """
        load_dir = Path(path)

        # Load model object
        model_path = load_dir / "model.pkl"
        with open(model_path, "rb") as f:
            self.model_object = pickle.load(f)

        # Load metadata
        metadata_path = load_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.model_id = metadata["model_id"]
        self.hyperparameters = metadata["hyperparameters"]
        self.training_metrics = metadata["training_metrics"]
        self.validation_metrics = metadata["validation_metrics"]
        self.feature_names = metadata["feature_names"]
        self.training_time = metadata["training_time"]
        self._created_at = datetime.fromisoformat(metadata["created_at"])
        self.is_fitted = True

        logger.info(f"Model loaded from {load_dir}")
        return self

    def get_artifact(self) -> ModelArtifact:
        """
        Get model artifact containing all metadata.

        Returns:
            ModelArtifact instance
        """
        return ModelArtifact(
            model_id=self.model_id,
            model_type=self.model_type,
            model_object=self.model_object,
            hyperparameters=self.hyperparameters,
            training_metrics=self.training_metrics,
            validation_metrics=self.validation_metrics,
            feature_importance=self.get_feature_importance() if self.is_fitted else {},
            feature_names=self.feature_names,
            training_time=self.training_time,
            created_at=self._created_at,
        )

    def _generate_model_id(self) -> str:
        """Generate a unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.model_type}_{timestamp}"

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.empty:
            raise ValueError("X cannot be empty")
        if X.isnull().any().any():
            logger.warning("Input contains NaN values")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_id='{self.model_id}', "
            f"is_fitted={self.is_fitted})"
        )
