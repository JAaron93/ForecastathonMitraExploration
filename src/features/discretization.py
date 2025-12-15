"""
Label discretization utilities for converting continuous targets into classification labels.
Use this for classification tasks like Naive Bayes.
"""

from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class DiscretizationResult:
    """Container for discretization results."""
    labels: pd.Series
    bins: np.ndarray
    label_map: Dict[int, str]

class LabelDiscretizer:
    """
    Discretizes continuous variables into discrete labels.
    """
    
    def __init__(self, strategy: str = "direction", **kwargs):
        """
        Initialize the discretizer.
        
        Args:
            strategy: Discretization strategy. Options:
                - "direction": 3 classes (Up, Down, Neutral)
                - "quantile": Equal frequency bins
                - "fixed": Custom fixed bin edges
                - "uniform": Equal width bins
            **kwargs: Strategy-specific arguments:
                - threshold (float): For "direction" strategy (default: 0.0)
                - n_bins (int): For "quantile" and "uniform" (default: 5)
                - bins (List[float]): For "fixed" strategy
        """
        self.strategy = strategy
        self.kwargs = kwargs
        self.bins: Optional[np.ndarray] = None
        self.label_map: Dict[int, str] = {}
        
    def fit(self, y: Union[pd.Series, np.ndarray]) -> "LabelDiscretizer":
        """
        Fit the discretizer to the data.
        
        Args:
            y: Continuous target variable
            
        Returns:
            Self
        """
        y_arr = np.array(y)
        
        if self.strategy == "direction":
            threshold = self.kwargs.get("threshold", 0.0)
            # Strategy doesn't need fitting in the traditional sense, 
            # but we define label map here
            self.label_map = {-1: "Down", 0: "Neutral", 1: "Up"}
            
        elif self.strategy == "quantile":
            n_bins = self.kwargs.get("n_bins", 5)
            # Use qcut to determine bins
            _, self.bins = pd.qcut(y_arr, q=n_bins, retbins=True, duplicates='drop')
            self._create_bin_labels()
            
        elif self.strategy == "uniform":
            n_bins = self.kwargs.get("n_bins", 5)
            _, self.bins = pd.cut(y_arr, bins=n_bins, retbins=True)
            self._create_bin_labels()
            
        elif self.strategy == "fixed":
            bins = self.kwargs.get("bins")
            if bins is None:
                raise ValueError("Must provide 'bins' argument for fixed strategy")
            self.bins = np.array(bins)
            self._create_bin_labels()
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        return self
    
    def transform(self, y: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Discretize the data.
        
        Args:
            y: Continuous target variable
            
        Returns:
            Discretized labels
        """
        y_arr = np.array(y)
        index = y.index if isinstance(y, pd.Series) else None
        
        if self.strategy == "direction":
            threshold = self.kwargs.get("threshold", 0.0)
            labels = np.zeros_like(y_arr, dtype=int)
            labels[y_arr > threshold] = 1
            labels[y_arr < -threshold] = -1
            # Neutral is 0 (already set by zeros_like)
            
            return pd.Series(labels, index=index, name="label")
            
        else:
            if self.bins is None:
                raise ValueError("Discretizer not fitted")
                
            # Use pd.cut with fitted bins
            labels_cat = pd.cut(y_arr, bins=self.bins, labels=False, include_lowest=True)
            
            # Handle potential NaNs from out of bounds if fitting data didn't cover range
            # ideally bins should handle it or we fillna
            return pd.Series(labels_cat, index=index, name="label").fillna(-1).astype(int)

    def fit_transform(self, y: Union[pd.Series, np.ndarray]) -> pd.Series:
        """Fit and transform in one step."""
        return self.fit(y).transform(y)
        
    def _create_bin_labels(self):
        """Create string labels for bins."""
        if self.bins is None:
            return
            
        n_bins = len(self.bins) - 1
        self.label_map = {}
        for i in range(n_bins):
            # Format: [low, high)
            low = self.bins[i]
            high = self.bins[i+1]
            self.label_map[i] = f"[{low:.4f}, {high:.4f})"
            
    def inverse_transform(self, labels: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Convert labels back to representative values (e.g., bin centers).
        Note: For 'direction', this returns -1, 0, 1.
        """
        labels_arr = np.array(labels)
        
        if self.strategy == "direction":
            return labels_arr # Already -1, 0, 1
            
        elif self.bins is not None:
            # Return bin centers
            centers = (self.bins[:-1] + self.bins[1:]) / 2
            
            # Map labels to centers
            # Handle -1 (unknown) -> NaN or 0? Let's go with NaN for now
            result = np.full(labels_arr.shape, np.nan)
            
            valid_mask = (labels_arr >= 0) & (labels_arr < len(centers))
            result[valid_mask] = centers[labels_arr[valid_mask].astype(int)]
            
            if isinstance(labels, pd.Series):
                return pd.Series(result, index=labels.index)
            return result
            
        return labels_arr
