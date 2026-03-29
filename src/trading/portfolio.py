"""Portfolio management and analysis utilities."""

from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from src.trading.signals import TradingSignal


class PortfolioUtility:
    """Utilities for portfolio management and backtesting."""

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital

    def calculate_position_size(
        self,
        signal: TradingSignal,
        current_capital: float,
        max_position_size: float = 0.2
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            signal: Trading signal
            current_capital: Available capital
            risk_per_trade: Fraction of capital to risk
            max_position_size: Maximum fraction of capital per position
            
        Returns:
            Position size amount (currency)
        """
        if signal.signal_type not in ["buy", "sell"]:
            return 0.0
            
        # Basic Fixed Fractional Position Sizing
        # Size = Capital * Risk / Confidence (scaled) or just simple percent
        
        # Here we implement a simple confidence-weighted sizing
        # bounded by max_position_size
        
        raw_size_pct = max_position_size * signal.confidence
        size_pct = min(raw_size_pct, max_position_size)
        
        return current_capital * size_pct

    def backtest_signals(
        self,
        signals: List[TradingSignal],
        price_data: pd.DataFrame,
        price_col: str = "close"
    ) -> pd.DataFrame:
        """
        Simple vector backtest of signals.
        
        Args:
            signals: List of trading signals
            price_data: DataFrame with timestamps index and price column
            price_col: Name of price column
            
        Returns:
            DataFrame with portfolio value over time
        """
        # Convert signals to DataFrame
        sig_data = []
        for s in signals:
            sig_data.append({
                "timestamp": s.timestamp,
                "signal": 1 if s.signal_type == "buy" else (-1 if s.signal_type == "sell" else 0),
                "confidence": s.confidence
            })
        
        if not sig_data:
            return pd.DataFrame()
            
        df_sig = pd.DataFrame(sig_data).set_index("timestamp")
        
        # Merge with price data
        df = price_data[[price_col]].copy()
        df = df.join(df_sig, how="left").fillna(0)
        
        # Simple strategy: Hold position = signal
        # (Assuming signals are target positions, or we accumulate? 
        #  Let's assume signal indicates desired direction for period)
        
        df["position"] = df["signal"] # Simple: 1 for long, -1 for short
        df["returns"] = df[price_col].pct_change()
        df["strategy_returns"] = df["position"].shift(1) * df["returns"]
        
        df["cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
        df["portfolio_value"] = self.initial_capital * df["cumulative_returns"]
        
        return df

