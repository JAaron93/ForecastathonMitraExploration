"""Model comparison and evaluation framework."""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.base_model import BaseModel, ModelArtifact
# Circular import avoidance: we assume base_model is independent
from src.utils.experiment_tracking import ExperimentTracker

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Framework for systematically comparing different models.
    """

    def __init__(self, experiment_tracker: Optional[ExperimentTracker] = None):
        """
        Initialize comparator.

        Args:
            experiment_tracker: Optional tracker to load artifacts from
        """
        self.tracker = experiment_tracker
        self.loaded_models: Dict[str, BaseModel] = {}
        self.artifacts: Dict[str, ModelArtifact] = {}

    def load_artifacts(self, run_ids: List[str]) -> None:
        """
        Load model artifacts from experiment runs.

        Args:
            run_ids: List of experiment run IDs
        """
        if not self.tracker:
            raise ValueError("ExperimentTracker not provided")

        for run_id in run_ids:
            run = self.tracker.get_run(run_id)
            if not run:
                logger.warning(f"Run {run_id} not found")
                continue
                
            # Logic to find model artifacts path from run
            # For this implementation, we assume we can manually load them 
            # or the tracker provides a path. 
            # Since ExperimentTracker.get_run returns a valid object if found,
            # we need a way to actually reconstruct the model.
            
            # This part depends on how artifacts are stored. 
            # Assuming standard structure <artifact_loc>/<run_id>/model/
            artifact_path = f"{self.tracker.artifact_location}/{run_id}/model"
            
            # We need to know which class to instantiate. 
            # Usually we pick a generic loader or metadata tells us.
            # For now, we'll try to load metadata first to find type
            import json
            import os
            
            try:
                # Helper to instantiate correct class
                # We need to import all model types here or use a factory
                from src.models.naive_bayes import NaiveBayesModel
                from src.models.xgboost_model import XGBoostModel
                from src.models.lstm_model import LSTMModel
                from src.models.mitra_model import MitraModel
                from src.models.ensemble import EnsembleModel
                
                # Check if metadata exists
                meta_path = f"{artifact_path}/metadata.json"
                if not os.path.exists(meta_path):
                    logger.warning(f"No metadata found for {run_id}")
                    continue
                    
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    
                model_type = meta.get("model_type")
                
                model_map = {
                    "NaiveBayes": NaiveBayesModel,
                    "XGBoost": XGBoostModel,
                    "LSTM": LSTMModel,
                    "Mitra": MitraModel,
                    "Ensemble": EnsembleModel
                }
                
                model_cls = model_map.get(model_type)
                if not model_cls:
                    # Fallback or generic
                    logger.warning(f"Unknown model type {model_type} for {run_id}")
                    continue
                    
                # Instantiate and load
                model = model_cls()
                model.load_model(artifact_path)
                
                self.loaded_models[run_id] = model
                self.artifacts[run_id] = model.get_artifact()
                logger.info(f"Loaded model {model_type} from {run_id}")
                
            except Exception as e:
                logger.error(f"Failed to load artifact for {run_id}: {e}")

    def add_model(self, model: BaseModel, name: str):
        """
        Manually add a model instance for comparison.
        
        Args:
            model: Fitted BaseModel instance
            name: Identifier for the model
        """
        self.loaded_models[name] = model
        if model.is_fitted:
            self.artifacts[name] = model.get_artifact()

    def compare_metrics(self, metric_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare metrics across loaded models.

        Args:
            metric_names: List of metrics to compare (e.g. 'rmse', 'accuracy')
                         If None, show all available.

        Returns:
            DataFrame with models as rows and metrics as columns
        """
        data = []
        indices = []
        
        for name, artifact in self.artifacts.items():
            metrics = artifact.validation_metrics.copy()
            # Also include training metrics with prefix
            for k, v in artifact.training_metrics.items():
                metrics[f"train_{k}"] = v
            
            # Filter if requested
            if metric_names:
                filtered = {k: v for k, v in metrics.items() if any(m in k for m in metric_names)}
                data.append(filtered)
            else:
                data.append(metrics)
            indices.append(name)
            
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data, index=indices)
        return df

    def analyze_robustness(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_col: Optional[str] = None
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
                    
                    # Heuristic to detect task type
                    is_regression = False
                    if pd.api.types.is_numeric_dtype(y_sub) and len(np.unique(y_sub)) > 20:
                        is_regression = True
                    elif pd.api.types.is_numeric_dtype(preds) and len(np.unique(preds)) > 20: 
                         is_regression = True
                         
                    # Regression metrics
                    if is_regression: 
                        metrics = calc.calculate_regression_metrics(y_sub.values, preds)
                    else:
                        # Classification
                        # Ensure y is int/str for classification if needed, but sklearn handles it if consistent
                        metrics = calc.calculate_classification_metrics(y_sub.values, preds)
                        
                    regime_metrics.append(metrics)
                    model_names.append(name)
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {name} on regime {regime}: {e}")
            
            if regime_metrics:
                results[str(regime)] = pd.DataFrame(regime_metrics, index=model_names)
                
        return results
