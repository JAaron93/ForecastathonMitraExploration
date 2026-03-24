import pytest
import pandas as pd
import numpy as np
from src.models.mitra_model import MitraModel

def test_volatility_matching_strategy():
    np.random.seed(42)
    model = MitraModel()
    
    # Create synthetic data with varying volatility
    # First 50: Low vol, Last 50: High vol
    low_vol = np.random.normal(0, 0.1, 50)
    high_vol = np.random.normal(0, 1.0, 50)
    y = pd.Series(np.concatenate([low_vol, high_vol]))
    X = pd.DataFrame({"feature": np.random.randn(100)})
    
    # Target high volatility (1.0)
    X_support, y_support = model.adapt_to_regime(
        X, y, strategy="volatility_matching", n_samples=10, target_volatility=1.0
    )

    assert len(X_support) == 10
    assert len(y_support) == 10
    # Should pick from the high volatility section (indices 50-99)
    # The first valid volatility index is at index 19 (volatility_window=20),
    # but the high vol section starts at 50.
    assert all(X_support.index >= 50)
    # Ensure y_support corresponds to high volatility values
    assert y_support.std() > 0.5  # Should be close to 1.0, definitely higher than 0.1
    assert y_support.index.equals(X_support.index)

def test_volatility_matching_fallback():
    model = MitraModel()
    X = pd.DataFrame({"f": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    
    # No target_volatility provided -> should fallback to 'recent'
    X_sub, y_sub = model.adapt_to_regime(X, y, strategy="volatility_matching", n_samples=2)
    assert len(X_sub) == 2
    assert X_sub.index.tolist() == [1, 2]
