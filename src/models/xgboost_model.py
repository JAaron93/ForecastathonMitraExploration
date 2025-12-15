"""
XGBoost model implementation with Optuna hyperparameter optimization and SHAP explainability.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
import json
import pickle
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss

from ..models.base_model import BaseModel, logger

class XGBoostModel(BaseModel):
    """
    XGBoost model wrapper with integral hyperparameter optimization and explainability.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        objective: str = "reg:squarederror"
    ):
        """
        Initialize XGBoost model.
        
        Args:
            model_id: Unique identifier
            hyperparameters: Initial hyperparameters (can be updated via optimization)
            objective: XGBoost objective function (e.g., 'reg:squarederror', 'binary:logistic')
        """
        super().__init__(model_id, hyperparameters)
        self.objective = objective
        self.model_object: Optional[xgb.XGBModel] = None
        self._explainer: Optional[shap.Explainer] = None
        
        # Default params
        if "n_estimators" not in self.hyperparameters:
            self.hyperparameters["n_estimators"] = 100
        if "max_depth" not in self.hyperparameters:
            self.hyperparameters["max_depth"] = 6
        if "learning_rate" not in self.hyperparameters:
            self.hyperparameters["learning_rate"] = 0.1
            
    @property
    def model_type(self) -> str:
        return "xgboost"
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        optimize: bool = False,
        optimization_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "XGBoostModel":
        """
        Fit the XGBoost model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            validation_data: Optional (X_val, y_val) for early stopping
            optimize: Whether to run hyperparameter optimization before final fit
            optimization_params: Parameters for Optuna (n_trials, cv_folds, etc.)
            **kwargs: Additional args passed to xgb.fit
        """
        self._validate_input(X)
        self.feature_names = X.columns.tolist()
        
        if optimize:
            logger.info("Starting hyperparameter optimization...")
            best_params = self.optimize_hyperparameters(
                X, y, 
                params=optimization_params or {}
            )
            logger.info(f"Optimization complete. Best params: {best_params}")
            self.hyperparameters.update(best_params)
            
        # Instantiate model with current hyperparameters
        model_cls = xgb.XGBClassifier if "binary" in self.objective or "multi" in self.objective else xgb.XGBRegressor
        
        self.model_object = model_cls(
            objective=self.objective,
            **self.hyperparameters
        )
        
        eval_set = None
        if validation_data:
            eval_set = [validation_data]
            
        # Fit model
        self.model_object.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
            **kwargs
        )
        self.is_fitted = True
        
        # Initialize SHAP explainer
        # For tree models, TreeExplainer is fast
        try:
            self._explainer = shap.TreeExplainer(self.model_object)
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        self._validate_input(X)
        X = X[self.feature_names] # Ensure alignment
        return self.model_object.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        if not isinstance(self.model_object, xgb.XGBClassifier):
            raise NotImplementedError("predict_proba only supported for classification")
            
        self._validate_input(X)
        X = X[self.feature_names]
        return self.model_object.predict_proba(X)
        
    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        Get feature importance.
        
        Args:
            importance_type: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
        """
        if not self.is_fitted:
            return {}
            
        # get_booster().get_score returns dict {feature: score}
        booster = self.model_object.get_booster()
        scores = booster.get_score(importance_type=importance_type)
        
        # Normalize? Usually raw scores are returned. 
        # Missing features have 0 importance.
        importance_dict = {feat: scores.get(feat, 0.0) for feat in self.feature_names}
        return importance_dict
        
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for X."""
        if not self.is_fitted or self._explainer is None:
            raise ValueError("Model not fitted or SHAP explainer not initialized")
        
        X = X[self.feature_names]
        return self._explainer.shap_values(X)
        
    def optimize_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Optuna optimization.
        
        Args:
            X, y: Training data
            params: Optimization config
                - n_trials: Number of trials (default: 20)
                - n_splits: CV splits (default: 5)
                - time_limit: Max time in seconds
                - metric: 'rmse', 'logloss', 'accuracy'
                
        Returns:
            Best hyperparameters
        """
        n_trials = params.get("n_trials", 20)
        n_splits = params.get("n_splits", 3) # Using 3 for speed in default
        metric = params.get("metric", "rmse")
        
        def objective(trial):
            # Define search space
            param = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "objective": self.objective
            }
            
            # TimeSeries CV
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model_cls = xgb.XGBClassifier if "binary" in self.objective or "multi" in self.objective else xgb.XGBRegressor
                model = model_cls(**param, n_jobs=1) # Single job per trial to avoid oversubscription
                
                model.fit(X_train, y_train, verbose=False)
                preds = model.predict(X_val)
                
                if metric == "rmse":
                    score = np.sqrt(mean_squared_error(y_val, preds))
                elif metric == "accuracy":
                    score = accuracy_score(y_val, preds)
                    score = 1 - score # Optuna minimizes
                elif metric == "logloss":
                    # For logloss we need probas
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_val)
                        score = log_loss(y_val, proba)
                    else:
                        score = 0.0 # Should fail for reg
                else:
                    score = np.sqrt(mean_squared_error(y_val, preds))
                    
                scores.append(score)
                
            return np.mean(scores)
            
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
