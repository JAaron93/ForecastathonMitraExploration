import numpy as np
import pandas as pd
from src.evaluation.comparison import ModelComparator
from src.models.base_model import BaseModel

class StandaloneMockModel(BaseModel):
    """Local mock model for debugging ModelComparator without test dependencies."""
    def __init__(self, is_fitted=True, metadata=None, **kwargs):
        super().__init__(**kwargs)
        self.is_fitted = is_fitted
        self.metadata = metadata or {}
        
    @property
    def model_type(self) -> str:
        return "StandaloneMock"
        
    def fit(self, X, y, **kwargs):
        self.is_fitted = True
        return self
        
    def predict(self, X):
        return np.array([1, 0, 1] * (len(X) // 3 + 1))[:len(X)]
        
    def predict_proba(self, X):
        return np.array([[0.5, 0.5]] * len(X))
        
    def get_feature_importance(self):
        return {"feat": 0.5}

def main():
    print("Running ModelComparator debug script...")
    c = ModelComparator()
    
    m1 = StandaloneMockModel()
    m1.training_metrics = {"rmse": 1.0, "mae": 0.5}
    m1.validation_metrics = {"rmse": 1.2, "mae": 0.6}
    
    print(f"Model fitted: {m1.is_fitted}")
    art = m1.get_artifact()
    print(f"Artifact training metrics: {art.training_metrics}")
    print(f"Artifact validation metrics: {art.validation_metrics}")
    
    c.add_model(m1, "model1")
    print(f"Comparator artifacts: {list(c.artifacts.keys())}")
    
    df = c.compare_metrics(["rmse"])
    print("\nComparison DataFrame (filtered for 'rmse'):")
    print(df)

if __name__ == "__main__":
    main()
