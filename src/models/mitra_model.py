"""
Mitra Foundation Model integration using AutoGluon.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import json
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from autogluon.tabular import TabularPredictor

from ..models.base_model import BaseModel, logger

class MitraModel(BaseModel):
    """
    Wrapper for AutoGluon's Mitra tabular foundation model.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        label_column: str = "target",
        problem_type: str = "auto",  # 'binary', 'multiclass', 'regression'
        eval_metric: Optional[str] = None
    ):
        """
        Initialize Mitra model.
        
        Args:
            hyperparameters:
                - fine_tune: bool (default False). If True, fine-tunes model weights. 
                             If False, uses zero-shot/in-context learning.
                - time_limit: int (seconds, default None)
            label_column: Name of target column
            problem_type: AutoGluon problem type
            eval_metric: AutoGluon evaluation metric
        """
        super().__init__(model_id, hyperparameters)
        self.label_column = label_column
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.predictor: Optional[TabularPredictor] = None
        
        # Default hyperparameters
        if "fine_tune" not in self.hyperparameters:
            self.hyperparameters["fine_tune"] = False

    @property
    def model_type(self) -> str:
        return "mitra"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> "MitraModel":
        """
        Fit (or configure context for) the Mitra model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            validation_data: (X_val, y_val) - strictly used for validation scoring in AG
        """
        self._validate_input(X)
        self.feature_names = X.columns.tolist()
        
        # AutoGluon expects a single DataFrame with target column
        train_data = X.copy()
        train_data[self.label_column] = y
        
        tuning_data = None
        if validation_data:
            X_val, y_val = validation_data
            tuning_data = X_val.copy()
            tuning_data[self.label_column] = y_val

        # Initialize Predictor
        # We perform a fresh fit every time fit() is called as AG Predictors aren't easily re-trainable 
        # without loading existing.
        # We use a temporary path or model_id based path.
        if self.model_id is None:
            import uuid
            self.model_id = str(uuid.uuid4())[:8]
        path = f"autogluon_models/{self.model_id}"
        
        self.predictor = TabularPredictor(
            label=self.label_column,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            path=path,
            verbosity=2
        )
        
        # Configure for Mitra
        # 'MITRA' key in hyperparameters enables it.
        # fine_tune=False means we use zero-shot/ICL mode basically.
        ag_hyperparams = {
            "MITRA": {
                "fine_tune": self.hyperparameters["fine_tune"]
            }
        }
        
        time_limit = self.hyperparameters.get("time_limit")
        
        self.predictor.fit(
            train_data=train_data,
            tuning_data=tuning_data,
            hyperparameters=ag_hyperparams,
            time_limit=time_limit,
            **kwargs
        )
        
        self.is_fitted = True
        
        # Capture metrics if available from leaderboards
        try:
            lb = self.predictor.leaderboard(silent=True)
            # Assuming Mitra is the model trained (might be named differently internally like 'Transformer')
            # But since we only asked for MITRA, it should be there.
            # We take the best score.
            best_model = lb.iloc[0]
            metric_name = self.eval_metric or "score_val"
            self.training_metrics[metric_name] = best_model.get("score_val", 0.0)
        except Exception as e:
            logger.warning(f"Could not retrieve leaderboard metrics: {e}")
            
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        self._validate_input(X)
        return self.predictor.predict(X).values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        self._validate_input(X)
        # AG predict_proba returns DataFrame with columns for classes
        return self.predictor.predict_proba(X).values

    def adapt_to_regime(
        self, 
        X_context: pd.DataFrame, 
        y_context: pd.Series, 
        strategy: str = "recent", 
        n_samples: int = 100
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select a subset of history to serve as the 'support set' for in-context learning.
        
        Args:
            X_context: Full available historical features
            y_context: Full available historical targets
            strategy: 'recent' (last N), 'random' (random N)
            n_samples: Number of samples to select
            
        Returns:
            X_support, y_support
        """
        available = len(X_context)
        if available <= n_samples:
            return X_context, y_context
            
        if strategy == "recent":
            return X_context.iloc[-n_samples:], y_context.iloc[-n_samples:]
        elif strategy == "random":
            indices = np.random.choice(available, n_samples, replace=False)
            return X_context.iloc[indices], y_context.iloc[indices]
        else:
            raise ValueError(f"Unknown adaptation strategy: {strategy}")

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        try:
            # feature_importance() is expensive in AG (permutation based)
            # We allow it, but note it might take time.
            fi = self.predictor.feature_importance(
                data=None, # uses validation data by default if present
                silent=True
            )
            return fi["importance"].to_dict()
        except Exception:
            return {}
            
    def save_model(self, path: str) -> None:
        """
        AutoGluon models are directories. We can move/copy the predictor directory.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        save_dir = Path(path)
        # Predictor is already saved at self.predictor.path
        # We need to move/copy it to `path` or ensure `path` points to it.
        # But Base model expectation is `path` is a directory where we dump stuff.
        # Let's clone the AG directory into `path/ag_model`.
        
        ag_dest = save_dir / "ag_model"
        if ag_dest.exists():
            shutil.rmtree(ag_dest)
            
        # clone_for_deployment or just copy?
        # predictor.save() saves to its current path.
        # We can use predictor.clone(path=...)
        self.predictor.save() # Ensure current state saved
        self.predictor.clone(path=str(ag_dest))
        
        # Save metadata
        metadata_path = save_dir / "metadata.json"
        artifact = self.get_artifact()
        
        with open(metadata_path, "w") as f:
            # json dump artifact
            # Note: artifact.model_object will be the MitraModel instance, which isn't serializable easily 
            # if it contains the predictor object (which might modify state).
            # Base model pickle dumps model_object.
            # We should probably exclude the predictor from the pickled object or ensure it reloads.
            # AG Predictor is visible to pickle as a path usually.
            
            # For metadata, we convert to dict.
            data = artifact.to_dict()
            json.dump(data, f, indent=2)
            
        logger.info(f"Mitra model saved to {save_dir}")
    def load_model(self, path: str) -> "MitraModel":
        load_dir = Path(path)
        ag_path = load_dir / "ag_model"
        
        self.predictor = TabularPredictor.load(str(ag_path))
        
        metadata_path = load_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            
        self.model_id = metadata["model_id"]
        self.hyperparameters = metadata["hyperparameters"]
        self.label_column = metadata.get("label_column", "target")
        self.problem_type = metadata.get("problem_type", "auto")
        self.eval_metric = metadata.get("eval_metric")
        self.feature_names = metadata.get("feature_names", [])
        self.training_metrics = metadata.get("training_metrics", {})
        self.is_fitted = True
        return self
