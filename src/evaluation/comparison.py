"""Model comparison and evaluation framework."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.base_model import BaseModel, ModelArtifact

# Circular import avoidance: we assume base_model is independent
from src.utils.experiment_tracking import ExperimentTracker

logger = logging.getLogger(__name__)

# Model type sets for task detection
REGRESSION_MODEL_TYPES = {"LSTM", "lstm_keras", "XGBoost"}
CLASSIFICATION_MODEL_TYPES = {"NaiveBayes", "Mitra", "Ensemble"}


class ModelComparator:
    """
    Framework for systematically comparing different models.
    """

    def __init__(self, experiment_tracker: Optional[ExperimentTracker] = None):
        """

        Args:
            experiment_tracker: Optional experiment tracker for loading models
        """
        self.experiment_tracker = experiment_tracker
        self.loaded_models: Dict[str, BaseModel] = {}

    def add_model(self, model: BaseModel, name: str):
        """
        Manually add a model instance for comparison.

        Args:
            model: Fitted BaseModel instance
            name: Identifier for the model
        """
        self.loaded_models[name] = model

    def compare_metrics(self, metric_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare metrics across loaded models.

        Args:
            metric_names: List of metrics to compare (e.g. 'rmse', 'accuracy')
                         If None, show all available.

        Returns:
            DataFrame with models as rows and metrics as columns
        """
        from src.evaluation.metrics import MetricsCalculator

        calc = MetricsCalculator()

        data = []
        indices = []

        for name, model in self.loaded_models.items():
            try:
                artifact = model.get_artifact()
                metrics = artifact.validation_metrics.copy()
                # Also include training metrics with prefix
                for k, v in artifact.training_metrics.items():
                    metrics[f"train_{k}"] = v

                # Filter if requested
                if metric_names:
                    filtered = {
                        k: v
                        for k, v in metrics.items()
                        if any(m in k for m in metric_names)
                    }
                    data.append(filtered)
                else:
                    data.append(metrics)
                indices.append(name)
            except Exception as e:
                logger.warning(f"Error getting metrics for model {name}: {e}")
                continue

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, index=indices)
        return df

    def load_models_from_runs(self, run_ids: List[str]) -> None:
        """
        Load models from experiment tracker runs.

        Args:
            run_ids: List of run IDs to load models from
        """
        if not self.experiment_tracker:
            logger.warning("No experiment tracker provided, skipping model loading")
            return

        for run_id in run_ids:
            run = self.experiment_tracker.get_run(run_id)
            if not run:
                logger.warning(f"Run {run_id} not found")
                continue

            artifact_path = (
                f"{self.experiment_tracker.artifact_location}/{run_id}/model"
            )
            meta_path = f"{artifact_path}/metadata.json"

            # Check if metadata exists before attempting to load
            if not os.path.exists(meta_path):
                logger.warning(
                    f"No metadata found for run_id '{run_id}' at '{meta_path}'. Skipping."
                )
                continue

            try:
                # Load and parse metadata
                with open(meta_path, "r") as f:
                    metadata = json.load(f)

                model_type = metadata.get("model_type")
                if not model_type:
                    logger.warning(
                        f"No model_type found in metadata for run {run_id}. Skipping."
                    )
                    continue

                # Dynamically import the model class
                if model_type in ("LSTM", "lstm_keras"):
                    from src.models.lstm_model import LSTMModel

                    model_class = LSTMModel
                elif model_type == "XGBoost":
                    from src.models.xgboost_model import XGBoostModel

                    model_class = XGBoostModel
                elif model_type == "NaiveBayes":
                    from src.models.naive_bayes import NaiveBayesModel

                    model_class = NaiveBayesModel
                elif model_type == "Mitra":
                    from src.models.mitra_model import MitraModel

                    model_class = MitraModel
                elif model_type == "Ensemble":
                    from src.models.ensemble import EnsembleModel

                    model_class = EnsembleModel
                else:
                    logger.warning(
                        f"Unknown model type '{model_type}' for run {run_id}. Skipping."
                    )
                    continue

                # Load the model
                model = model_class.load(artifact_path)
                self.loaded_models[run_id] = model
                logger.info(f"Loaded model {model_type} from run {run_id}")

            except Exception as e:
                logger.error(f"Failed to load model from run {run_id}: {e}")
                continue

    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        run_ids: List[str],
        regime_col: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compare models across different market regimes.

        Args:
            X: Feature DataFrame
            y: Target Series
            run_ids: List of run IDs to compare
            regime_col: Column name in X defining regimes (e.g. 'bull', 'bear').
                       If None, one global regime is assumed.

        Returns:
            Dictionary mapping regime names to DataFrames of model metrics
        """
        # Load models first
        self.load_models_from_runs(run_ids)

        # Then compare them
        return self.analyze_robustness(X, y, regime_col)

    def analyze_robustness(
        self, X: pd.DataFrame, y: pd.Series, regime_col: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Evaluate models across different market regimes (or simply subsets).

        Args:
            X: Feature DataFrame containing validation data
            y: Target Series
            regime_col: Column name in X defining regimes (e.g. 'bull', 'bear').
                        If None, one global regime is assumed.

        Returns:
            Dictionary mapping regime names to DataFrames of model metrics
        """
        from src.evaluation.metrics import MetricsCalculator

        calc = MetricsCalculator()

        results = {}

        if regime_col and regime_col in X.columns:
            regimes = X[regime_col].unique()
        else:
            regimes = ["global"]

        for regime in regimes:
            if regime == "global":
                X_sub = X
                y_sub = y
            else:
                mask = X[regime_col] == regime
                X_sub = X[mask]
                y_sub = y[mask]

            if X_sub.empty:
                continue

            regime_metrics = []
            model_names = []

            for name, model in self.loaded_models.items():
                try:
                    preds = model.predict(X_sub)

                    # Determine task type (regression vs classification)
                    # Simple heuristic: floats vs ints or few unique values
                    # Better: check model type or metadata

                    # Ensure predictions are numpy array
                    preds = np.array(preds)

                    # Calculate both if unsure, or strictly one.
                    # For this pipeline, let's assume specific metrics based on model type metadata if available
                    # Or just calc all applicable

                    metrics = {}

                    # Determine task type (regression vs classification)
                    # Priority: y_sub properties -> model metadata -> model attributes -> default
                    is_regression = False
                    if (
                        pd.api.types.is_numeric_dtype(y_sub)
                        and y_sub.nunique(dropna=False) > 20
                    ):
                        is_regression = True
                    else:
                        # Fallback: Check model metadata for explicit task type
                        # This avoids misclassifying classification probabilities as regression.
                        task_meta = model.metadata.get("task_type", "").lower()
                        if "regression" in task_meta:
                            is_regression = True
                        elif "classification" in task_meta:
                            is_regression = False
                        # Secondary fallback: Check model-specific indicators
                        m_type = getattr(model, "model_type", "").lower()
                        obj = getattr(model, "objective", "").lower()

                        if "reg:" in obj or m_type in {
                            t.lower() for t in REGRESSION_MODEL_TYPES
                        }:
                            is_regression = True
                        elif (
                            "binary:" in obj
                            or "multi:" in obj
                            or m_type in {t.lower() for t in CLASSIFICATION_MODEL_TYPES}
                        ):
                            is_regression = False
                        else:
                            # Default to classification for discrete or low-variance targets
                            is_regression = False

                    # Regression metrics
                    if is_regression:
                        metrics = calc.calculate_regression_metrics(y_sub.values, preds)
                    else:
                        # Classification
                        # Ensure y is int/str for classification if needed, but sklearn handles it if consistent
                        metrics = calc.calculate_classification_metrics(
                            y_sub.values, preds
                        )

                    regime_metrics.append(metrics)
                    model_names.append(name)
                except Exception as e:
                    logger.warning(f"Error evaluating model {name}: {e}")
                    continue

            if regime_metrics:
                comparison_results = {
                    "models": model_names,
                    "metrics": regime_metrics,
                }
                results[regime] = self._metrics_to_dataframe(comparison_results)

        return results

    def _metrics_to_dataframe(self, comparison_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert comparison results to a DataFrame.

        Args:
            comparison_results: Dictionary with 'models' and 'metrics' keys

        Returns:
            DataFrame with models as index and metrics as columns
        """
        model_names = comparison_results["models"]
        metrics_list = comparison_results["metrics"]

        if not metrics_list:
            return pd.DataFrame()

        # Extract all unique metric names
        metric_names = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())

        # Build data rows
        data = []
        for i, metrics in enumerate(metrics_list):
            row = {metric: metrics.get(metric, np.nan) for metric in metric_names}
            data.append(row)

        df = pd.DataFrame(data, index=model_names)
        return df
