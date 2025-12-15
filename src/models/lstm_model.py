import os
# Set backend to JAX before importing keras as we are in Python 3.13 env
os.environ["KERAS_BACKEND"] = "jax"

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import keras
from typing import Dict, Any, Optional, Tuple, List, Union
import os
import numpy as np
import pandas as pd
import keras
from keras import layers, callbacks
import json

from ..models.base_model import BaseModel, logger

class LSTMModel(BaseModel):
    """
    LSTM model for time-series forecasting using Keras (JAX backend).
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LSTM model.
        
        Hyperparameters:
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers (stacked)
            dropout: Dropout probability
            seq_len: Input sequence length (window size)
            batch_size: Training batch size
            learning_rate: Learning rate
            epochs: Max training epochs
            patience: Early stopping patience
        """
        super().__init__(model_id, hyperparameters)
        self.model_object: Optional[keras.Model] = None
        
        # Defaults
        self.defaults = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "seq_len": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 20,
            "patience": 5
        }
        for k, v in self.defaults.items():
            if k not in self.hyperparameters:
                self.hyperparameters[k] = v

    @property
    def model_type(self) -> str:
        return "lstm_keras"
        
    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None, seq_len: int = 10) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sliding window sequences.
        """
        xs, ys = [], []
        n_samples = len(X)
        
        if n_samples <= seq_len:
            raise ValueError(f"Input data length ({n_samples}) must be greater than sequence length ({seq_len})")
            
        for i in range(n_samples - seq_len):
            # Window: [i, i+1, ... i+seq_len-1]
            x_window = X[i:(i + seq_len)]
            xs.append(x_window)
            if y is not None:
                # Target: y[i+seq_len] (next step)
                ys.append(y[i + seq_len])
                
        return np.array(xs), np.array(ys) if y is not None else None

    def _build_model(self, input_shape: Tuple[int, int], output_shape: int):
        """Build Keras LSTM model."""
        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape))
        
        num_layers = self.hyperparameters["num_layers"]
        hidden_size = self.hyperparameters["hidden_size"]
        dropout = self.hyperparameters["dropout"]
        
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1) # True for all but last layer
            model.add(layers.LSTM(
                hidden_size, 
                return_sequences=return_sequences,
                dropout=dropout
            ))
            
        model.add(layers.Dense(output_shape))
        
        optimizer = keras.optimizers.Adam(learning_rate=self.hyperparameters["learning_rate"])
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> "LSTMModel":
        """
        Fit LSTM model.
        """
        self._validate_input(X)
        self.feature_names = X.columns.tolist()
        
        # Data Prep
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.float32)
        if len(y_np.shape) == 1:
            y_np = y_np.reshape(-1, 1)
            
        seq_len = self.hyperparameters["seq_len"]
        X_seq, y_seq = self._create_sequences(X_np, y_np, seq_len)
        
        val_data_seq = None
        if validation_data:
            X_val, y_val = validation_data
            X_val_np = X_val.values.astype(np.float32)
            y_val_np = y_val.values.astype(np.float32)
            if len(y_val_np.shape) == 1:
                y_val_np = y_val_np.reshape(-1, 1)
                
            if len(X_val_np) > seq_len:
                X_val_seq, y_val_seq = self._create_sequences(X_val_np, y_val_np, seq_len)
                val_data_seq = (X_val_seq, y_val_seq)

        # Build Model
        input_shape = (seq_len, X_np.shape[1])
        output_shape = y_np.shape[1]
        
        self.model_object = self._build_model(input_shape, output_shape)
        
        # Training
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if val_data_seq else 'loss',
            patience=self.hyperparameters["patience"],
            restore_best_weights=True
        )
        
        history = self.model_object.fit(
            X_seq, y_seq,
            validation_data=val_data_seq,
            epochs=self.hyperparameters["epochs"],
            batch_size=self.hyperparameters["batch_size"],
            callbacks=[early_stopping],
            verbose=0,
            **kwargs
        )
        
        # Store metrics
        hist = history.history
        self.training_metrics["loss"] = hist["loss"][-1]
        if "val_loss" in hist:
            self.validation_metrics["loss"] = hist["val_loss"][-1]
            
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        self._validate_input(X)
        X_df = X[self.feature_names]
        X_np = X_df.values.astype(np.float32)
        seq_len = self.hyperparameters["seq_len"]
        
        if len(X_np) <= seq_len:
             raise ValueError(f"Input length {len(X_np)} too short for sequence length {seq_len}")
             
        X_seq, _ = self._create_sequences(X_np, None, seq_len)
        
        preds = self.model_object.predict(X_seq, verbose=0)
        
        # Determine format (if scalar predictions, flatten)
        if preds.shape[1] == 1:
            return preds.flatten()
        return preds

    def get_feature_importance(self) -> Dict[str, float]:
        return {}
        
    def save_model(self, path: str) -> None:
        """Override save to use keras save format."""
        if not self.is_fitted:
             raise ValueError("Cannot save unfitted model")
             
        from pathlib import Path
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save metadata (standard)
        # We need to temporarily unset model_object so artifact creation doesn't crash on pickle
        # or we exclude it. Keras models aren't cleanly picklable usually.
        # But get_artifact puts model_object in it.
        # base_model.save_model pickles the object.
        # We are overriding, so we don't use base_model.save_model logic for the object.
        
        # Metadata
        metadata_path = save_dir / "metadata.json"
        artifact = self.get_artifact()
        # We don't want to serialize the actual keras object in the artifact dict
        # The artifact.to_dict() already excludes model_object.
        
        # Save Keras model
        model_path = save_dir / "model.keras"
        self.model_object.save(model_path)
        
        with open(metadata_path, "w") as f:
            json.dump(artifact.to_dict(), f, indent=2)
            
    def load_model(self, path: str) -> "LSTMModel":
        """Override load to use keras load format."""
        from pathlib import Path
        load_dir = Path(path)
        
        model_path = load_dir / "model.keras"
        self.model_object = keras.models.load_model(model_path)
        
        metadata_path = load_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            
        self.model_id = metadata["model_id"]
        self.hyperparameters = metadata["hyperparameters"]
        self.training_metrics = metadata["training_metrics"]
        self.validation_metrics = metadata["validation_metrics"]
        self.feature_names = metadata["feature_names"]
        # etc... (copy from base logic)
        self.is_fitted = True
        return self
