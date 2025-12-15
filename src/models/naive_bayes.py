"""
Naive Bayes model implementation.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance

from ..models.base_model import BaseModel

class NaiveBayesModel(BaseModel):
    """
    Naive Bayes model wrapper.
    Uses Gaussian Naive Bayes for continuous features.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_id, hyperparameters)
        self.hyperparameters = hyperparameters or {}
        
        # Extract specific NB params
        var_smoothing = self.hyperparameters.get("var_smoothing", 1e-9)
        
        self.model_object = GaussianNB(var_smoothing=var_smoothing)
        self._feature_importances: Dict[str, float] = {}
        
    @property
    def model_type(self) -> str:
        return "naive_bayes"
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple] = None,
        **kwargs
    ) -> "NaiveBayesModel":
        """
        Fit the Naive Bayes model.
        
        Args:
            X: Feature DataFrame
            y: Target Series (must be discrete/categorical for NB classification)
            validation_data: Optional (X_val, y_val) - NB doesn't use early stopping but we can log metrics
            **kwargs: Additional args
            
        Returns:
            Self
        """
        self._validate_input(X)
        self.feature_names = X.columns.tolist()
        
        # Ensure y is proper type
        y_clean = y
        if not pd.api.types.is_integer_dtype(y) and not pd.api.types.is_object_dtype(y):
             # Try to convert to int if it looks like float-integer (e.g. 1.0, 0.0)
             try:
                 y_clean = y.astype(int)
             except ValueError:
                 # If truly continuous, user should have discretized it first
                 # But we fit anyway, sklearn might warn or fail if too many classes
                 pass
        
        self.model_object.fit(X, y_clean)
        self.is_fitted = True
        
        # Calculate training metrics if needed
        # self.training_metrics = ...
        
        # If validation data provided, calculate validation metrics
        if validation_data:
            X_val, y_val = validation_data
            # predictions = self.predict(X_val)
            # metrics = classification_metrics(y_val, predictions)
            # self.validation_metrics = metrics
            pass

        # Calculate feature importance via permutation on training data (optional, can be slow)
        # We'll rely on explicit get_feature_importance call to avoid overhead during fit
            
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        self._validate_input(X)
        
        # Ensure columns match
        if X.columns.tolist() != self.feature_names:
            # Basic check - in production might want alignment
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")
            X = X[self.feature_names] # Reorder if needed
            
        return self.model_object.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        self._validate_input(X)
        
        if X.columns.tolist() != self.feature_names:
             X = X[self.feature_names]
             
        return self.model_object.predict_proba(X)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance using permutation importance.
        Note: GaussianNB doesn't have intrinsic feature importance coeff like Linear models.
        We can approximate or use permutation.
        Here we'll return cached importances if available, or empty dict.
        Calculating permutation importance requires X, y which we don't store.
        
        To actually get importance, one should call `calculate_permutation_importance(X, y)`
        method which we can add, or we accept that lightweight NB objects might not have it by default.
        """
        return self._feature_importances

    def calculate_permutation_importance(self, X: pd.DataFrame, y: pd.Series, n_repeats=5) -> Dict[str, float]:
        """
        Calculate and cache permutation importance.
        """
        if not self.is_fitted:
             raise ValueError("Model not fitted")
             
        result = permutation_importance(
            self.model_object, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
        )
        
        importances = {}
        for i, name in enumerate(self.feature_names):
            importances[name] = result.importances_mean[i]
            
        self._feature_importances = importances
        return importances
