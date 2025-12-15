"""Ensemble model implementation."""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble model that combines predictions from multiple base models.
    Supports weighted averaging for regression/probabilities and voting for classification.
    """

    def __init__(
        self,
        models: List[BaseModel],
        weights: Optional[List[float]] = None,
        method: str = "average",
        model_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ensemble model.

        Args:
            models: List of base models to ensemble
            weights: Optional weights for each model (must sum to 1.0)
            method: Ensemble method ('average', 'weighted_average', 'voting')
            model_id: Unique identifier for the model
            hyperparameters: Model hyperparameters
        """
        super().__init__(model_id=model_id, hyperparameters=hyperparameters)
        self.models = models
        self.weights = weights
        self.method = method
        
        if self.weights is not None:
             if len(self.weights) != len(self.models):
                 raise ValueError("Number of weights must match numberOf models")
             if not np.isclose(sum(self.weights), 1.0):
                 # Normalize weights if they don't sum to 1
                 total = sum(self.weights)
                 self.weights = [w / total for w in self.weights]
                 logger.warning("Weights normalized to sum to 1.0")
        elif self.method == "weighted_average":
            # Default to equal weights if not provided but requested
            self.weights = [1.0 / len(self.models)] * len(self.models)

        # Update hyperparameters
        self.hyperparameters = self.hyperparameters or {}
        self.hyperparameters.update({
            "method": method,
            "n_models": len(models),
            "model_types": [m.model_type for m in models],
            "weights": self.weights
        })

    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return "Ensemble"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple] = None,
        **kwargs
    ) -> "EnsembleModel":
        """
        Fit all base models.
        
        Note: In many cases, ensembles are created from already fitted models.
        This method is provided for cases where training from scratch is needed.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            validation_data: Optional tuple of (X_val, y_val)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        self._validate_input(X)
        self.feature_names = X.columns.tolist()
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.model_type}")
            model.fit(X, y, validation_data=validation_data, **kwargs)
            
        self.is_fitted = True
        self.training_time = sum(m.training_time for m in self.models)
        
        # Aggregate metrics if available
        self._aggregate_metrics()
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using ensemble method.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        if not self.is_fitted and not all(m.is_fitted for m in self.models):
             # If ensemble itself isn't marked fitted, but all sub-models are, we proceed.
             # This allows creating ensembles from pre-trained models.
             if all(m.is_fitted for m in self.models):
                 self.is_fitted = True
             else:
                raise ValueError("Model is not fitted")
                
        self._validate_input(X)
        
        # Collect predictions from all models
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
            
        predictions = np.array(predictions)
        
        if self.method == "average":
            return np.mean(predictions, axis=0)
            
        elif self.method == "weighted_average":
            if self.weights is None:
                self.weights = [1.0 / len(self.models)] * len(self.models)
            return np.average(predictions, axis=0, weights=self.weights)
            
        elif self.method == "voting":
            # Majority voting for classification
            # Transpose to get (n_samples, n_models)
            preds_t = predictions.T
            final_preds = []
            for sample_preds in preds_t:
                # Count occurrences of each class
                values, counts = np.unique(sample_preds, return_counts=True)
                # Get the value with max count
                final_preds.append(values[np.argmax(counts)])
            return np.array(final_preds)
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of probability predictions
        """
        if not self.is_fitted and not all(m.is_fitted for m in self.models):
             if all(m.is_fitted for m in self.models):
                 self.is_fitted = True
             else:
                raise ValueError("Model is not fitted")

        self._validate_input(X)
        
        # Collect proba predictions
        probas = []
        for model in self.models:
            try:
                probas.append(model.predict_proba(X))
            except NotImplementedError:
                logger.warning(f"Model {model.model_type} does not support predict_proba, skipping in probability ensemble")
        
        if not probas:
            raise NotImplementedError("No base models support predict_proba")
            
        probas = np.array(probas)
        
        # For probabilities, we usually average (weighted or not)
        # Voting applies to hard labels, not probabilities usually
        if self.method in ["average", "voting"]:  # Treat voting as average for probabilities
            return np.mean(probas, axis=0)
            
        elif self.method == "weighted_average":
             # We need to filter weights to match models that supported proba if some were skipped
             # But for now assuming all models support it if calling predict_proba
            if self.weights is None:
                eff_weights = [1.0 / len(probas)] * len(probas)
            else:
                eff_weights = self.weights
            return np.average(probas, axis=0, weights=eff_weights)
            
        else:
             raise ValueError(f"Unknown ensemble method: {self.method}")

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get averaged feature importance.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
             if all(m.is_fitted for m in self.models):
                 self.is_fitted = True
             else:
                return {}
        
        # Aggregate importance from all models
        total_importance = {}
        count = 0
        
        for model in self.models:
            importance = model.get_feature_importance()
            if importance:
                for feat, score in importance.items():
                    total_importance[feat] = total_importance.get(feat, 0.0) + score
                count += 1
                
        if count == 0:
            return {}
            
        # Average
        return {k: v / count for k, v in total_importance.items()}

    def _aggregate_metrics(self):
        """Aggregate metrics from base models."""
        self.training_metrics = {}
        self.validation_metrics = {}
        
        # Simple averaging of metrics for now
        # Ideally we would evaluate the ensemble itself, but that requires X and y
        # This is just a summary of component performance
        for metric_type in ["training_metrics", "validation_metrics"]:
             target_dict = getattr(self, metric_type)
             
             # Collect all keys
             all_keys = set()
             for model in self.models:
                 all_keys.update(getattr(model, metric_type).keys())
                 
             for key in all_keys:
                 values = [getattr(m, metric_type).get(key) for m in self.models if key in getattr(m, metric_type)]
                 if values:
                     target_dict[f"mean_{key}"] = float(np.mean(values))
