"""Property tests for trading components."""

from datetime import datetime
import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from src.trading.signals import SignalGenerator, TradingSignal
from src.trading.portfolio import PortfolioUtility


@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=5, max_size=20),
    st.floats(min_value=0.5, max_value=0.9),  # buy threshold
    st.floats(min_value=0.1, max_value=0.4)   # sell threshold
)
@settings(max_examples=50, deadline=None)
def test_property_8_signal_correctness_threshold(preds, buy_thresh, sell_thresh):
    """
    Property 8: Signal generation correctness (Threshold strategy).
    
    Verify signals match the threshold logic:
    - pred >= buy_thresh => Buy
    - pred <= sell_thresh => Sell
    - Else => Hold
    """
    # Create dates
    dates = [datetime.now() for _ in range(len(preds))]
    
    gen = SignalGenerator(strategy="threshold", buy_threshold=buy_thresh, sell_threshold=sell_thresh)
    
    signals = gen.generate_signals(preds, dates, "TEST_ASSET")
    
    for i, sig in enumerate(signals):
        p = preds[i]
        expected_type = "hold"
        if p >= buy_thresh:
            expected_type = "buy"
        elif p <= sell_thresh:
            expected_type = "sell"
            
        assert sig.signal_type == expected_type
        assert sig.asset == "TEST_ASSET"
        assert sig.timestamp == dates[i]


@given(
    st.lists(st.floats(min_value=-0.2, max_value=0.2), min_size=5, max_size=20),
    st.floats(min_value=0.01, max_value=0.05)
)
@settings(max_examples=50, deadline=None)
def test_property_8_signal_correctness_directional(preds, threshold):
    """
    Property 8: Signal generation correctness (Directional strategy).
    """
    dates = [datetime.now() for _ in range(len(preds))]
    
    gen = SignalGenerator(strategy="directional", threshold=threshold)
    
    signals = gen.generate_signals(preds, dates, "TEST_ASSET")
    
    for i, sig in enumerate(signals):
        p = preds[i]
        expected_type = "hold"
        if p > threshold:
            expected_type = "buy"
        elif p < -threshold:
            expected_type = "sell"
            
        assert sig.signal_type == expected_type


def test_position_sizing_logic():
    """Verify position sizing calculations."""
    portfolio = PortfolioUtility(initial_capital=10000.0)
    
    # 1. Buy signal with high confidence
    sig_buy = TradingSignal(
        timestamp=datetime.now(),
        asset="A",
        signal_type="buy",
        confidence=1.0,
        metadata={}
    )
    
    # Max size = 0.2, Confidence = 1.0 => Size = 0.2 * 10000 = 2000
    size = portfolio.calculate_position_size(sig_buy, 10000.0, max_position_size=0.2)
    assert size == 2000.0
    
    # 2. Buy signal with low confidence
    sig_weak = TradingSignal(
        timestamp=datetime.now(),
        asset="A",
        signal_type="buy",
        confidence=0.5,
        metadata={}
    )
    # Size = 0.2 * 0.5 = 0.1 * 10000 = 1000
    size = portfolio.calculate_position_size(sig_weak, 10000.0, max_position_size=0.2)
    assert size == 1000.0
    
    # 3. Hold signal
    sig_hold = TradingSignal(
        timestamp=datetime.now(),
        asset="A",
        signal_type="hold",
        confidence=0.0,
        metadata={}
    )
    size = portfolio.calculate_position_size(sig_hold, 10000.0)
    assert size == 0.0


def test_backtest_mechanics():
    """Verify simple backtest mechanics."""
    # Create simple price path: 100, 110, 121 (10% up, 10% up)
    dates = pd.date_range("2023-01-01", periods=3)
    prices = pd.DataFrame({"close": [100.0, 110.0, 121.0]}, index=dates)
    
    # Signals: Buy at t0, Hold at t1 (so position carries over if logic allows, 
    # but our simple backtest takes signal as position for THAT period/following period?
    # Logic in backtest: df["position"] = df["signal"].
    # df["strategy_returns"] = df["position"].shift(1) * df["returns"]
    # So signal at t0 affects returns from t0 to t1.
    
    signals = [
        TradingSignal(dates[0], "A", "buy", 1.0, metadata={}),  # Position 1 for t0->t1 return
        TradingSignal(dates[1], "A", "buy", 1.0, metadata={}),  # Position 1 for t1->t2 return
        TradingSignal(dates[2], "A", "sell", 1.0, metadata={})  # Position -1 for t2->t3 (not in data)
    ]
    
    portfolio = PortfolioUtility(initial_capital=100.0)
    res = portfolio.backtest_signals(signals, prices)
    
    # Returns:
    # t0: NaN
    # t1: (110-100)/100 = 0.1. Position(t0) = 1. Strat Ret = 1 * 0.1 = 0.1
    # t2: (121-110)/110 = 0.1. Position(t1) = 1. Strat Ret = 1 * 0.1 = 0.1
    
    # Cum Ret:
    # t1: 1.1
    # t2: 1.1 * 1.1 = 1.21
    
    # Portfolio Value:
    # t2: 100 * 1.21 = 121.0
    
    assert np.isclose(res.iloc[-1]["portfolio_value"], 121.0)
