"""Trading signal generation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class TradingSignal:
    """Container for trading signals."""
    timestamp: datetime
    asset: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    position_size: float = 0.0
    model_source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalGenerator:
    """Generates trading signals from model predictions."""

    def __init__(self, strategy: str = "threshold", **kwargs):
        """
        Initialize signal generator.

        Args:
            strategy: Strategy name ('threshold', 'directional')
            **kwargs: Strategy parameters
        """
        self.strategy = strategy
        self.params = kwargs

    def generate_signals(
        self,
        predictions: Union[np.ndarray, pd.Series],
        dates: Union[pd.DatetimeIndex, List[datetime]],
        asset_name: str,
        model_name: str = "model",
    ) -> List[TradingSignal]:
        """
        Generate signals from predictions.

        Args:
            predictions: Model predictions (probabilities or regression values)
            dates: Timestamps corresponding to predictions
            asset_name: Name of the asset
            model_name: Name of the source model

        Returns:
            List of TradingSignal objects
        """
        signals = []
        
        # Ensure aligned inputs
        if len(predictions) != len(dates):
            raise ValueError("Predictions and dates must have same length")

        # Convert to numpy for consistent handling
        preds_arr = np.array(predictions)
        
        for i, (pred, date) in enumerate(zip(preds_arr, dates)):
            signal_type = "hold"
            confidence = 0.0
            
            if self.strategy == "threshold":
                # Expecting probabilities (0-1) or scores
                buy_threshold = self.params.get("buy_threshold", 0.6)
                sell_threshold = self.params.get("sell_threshold", 0.4)
                
                if pred >= buy_threshold:
                    signal_type = "buy"
                    confidence = float(pred)
                elif pred <= sell_threshold:
                    signal_type = "sell"
                    confidence = float(1.0 - pred) if 0 <= pred <= 1 else float(abs(pred))
                else:
                    signal_type = "hold"
                    # Confidence for hold could be "closeness to center" or just 0
                    confidence = 0.0

            elif self.strategy == "directional":
                # Expecting returns or direction (-1 to 1, or raw return)
                threshold = self.params.get("threshold", 0.0)
                
                if pred > threshold:
                    signal_type = "buy"
                    confidence = float(abs(pred)) # Magnitude as confidence
                elif pred < -threshold:
                    signal_type = "sell"
                    confidence = float(abs(pred))
                else:
                    signal_type = "hold"
                    confidence = 0.0
            
            else:
                # Default / Pass-through
                pass

            signals.append(TradingSignal(
                timestamp=date,
                asset=asset_name,
                signal_type=signal_type,
                confidence=confidence,
                model_source=model_name,
                metadata={"raw_prediction": float(pred)}
            ))
            
        return signals
