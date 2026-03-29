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
