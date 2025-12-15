"""
Feature selection utilities for the forecasting pipeline.
"""

from typing import List, Union, Optional
import pandas as pd
import numpy as np

class FeatureSelector:
    """
    Utilities for selecting subsets of features.
    """
    
    @staticmethod
    def select_by_correlation(
        df: pd.DataFrame, 
        target: pd.Series, 
        n_features: int = 10,
        method: str = 'pearson'
    ) -> List[str]:
        """
        Select top features absolute correlation with target.
        
        Args:
            df: Feature DataFrame
            target: Target Series
            n_features: Number of features to select
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            List of selected feature names
        """
        # Align index
        common_idx = df.index.intersection(target.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping indices between features and target")
            
        df_aligned = df.loc[common_idx]
        target_aligned = target.loc[common_idx]
        
        corrs = df_aligned.corrwith(target_aligned, method=method).abs()
        return corrs.nlargest(n_features).index.tolist()
        
    @staticmethod
    def select_by_variance(
        df: pd.DataFrame,
        threshold: float = 0.0
    ) -> List[str]:
        """
        Select features with variance above threshold.
        
        Args:
            df: Feature DataFrame
            threshold: Variance threshold
            
        Returns:
            List of selected feature names
        """
        variances = df.var()
        return variances[variances > threshold].index.tolist()
    
    @staticmethod
    def drop_collinear(
        df: pd.DataFrame,
        threshold: float = 0.95
    ) -> List[str]:
        """
        Drop highly correlated features. Keep the first one found.
        
        Args:
            df: Feature DataFrame
            threshold: Correlation threshold
            
        Returns:
            List of selected feature names (kept features)
        """
        # specific implementation to drop collinear features
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        return [c for c in df.columns if c not in to_drop]
