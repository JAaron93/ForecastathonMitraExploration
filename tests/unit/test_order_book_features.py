import pytest
import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer

def test_calculate_order_book_features():
    fe = FeatureEngineer()
    df = pd.DataFrame({
        "bid_price": [100.0, 101.0],
        "ask_price": [102.0, 103.0],
        "bid_size": [10.0, 20.0],
        "ask_size": [15.0, 5.0]
    })
    
    result = fe.calculate_order_book_features(df)
    
    # Mid Price: (100+102)/2 = 101, (101+103)/2 = 102
    assert result["mid_price"].iloc[0] == 101.0
    assert result["mid_price"].iloc[1] == 102.0
    
    # Spread: 102-100 = 2, 103-101 = 2
    assert result["bid_ask_spread"].iloc[0] == 2.0
    assert result["bid_ask_spread"].iloc[1] == 2.0
    
    # WAP: (bid_p * ask_s + ask_p * bid_s) / (bid_s + ask_s)
    # Row 0: (100 * 15 + 102 * 10) / (10 + 15) = (1500 + 1020) / 25 = 2520 / 25 = 100.8
    # Row 1: (101 * 5 + 103 * 20) / (20 + 5) = (505 + 2060) / 25 = 2565 / 25 = 102.6
    assert result["wap"].iloc[0] == pytest.approx(100.8)
    assert result["wap"].iloc[1] == pytest.approx(102.6)

def test_calculate_order_book_features_missing_cols():
    fe = FeatureEngineer()
    df = pd.DataFrame({"some_col": [1, 2]})
    result = fe.calculate_order_book_features(df)
    assert "mid_price" not in result.columns
    assert "bid_ask_spread" not in result.columns
    assert "wap" not in result.columns

def test_calculate_order_book_features_div_by_zero():
    fe = FeatureEngineer()
    df = pd.DataFrame({
        "bid_price": [100.0, 101.0],
        "ask_price": [102.0, 103.0],
        "bid_size": [0.0, 0.0],
        "ask_size": [0.0, 0.0]
    })
    result = fe.calculate_order_book_features(df)
    
    # Mid Price: (100+102)/2 = 101, (101+103)/2 = 102
    assert result["mid_price"].iloc[0] == 101.0
    assert result["mid_price"].iloc[1] == 102.0

    # Spread: 102-100 = 2, 103-101 = 2 (unaffected by zero sizes)
    assert result["bid_ask_spread"].iloc[0] == 2.0
    assert result["bid_ask_spread"].iloc[1] == 2.0

    # WAP: Total size is 0, should fallback to mid_price
    assert result["wap"].iloc[0] == 101.0
    assert result["wap"].iloc[1] == 102.0

def test_calculate_order_book_features_partial_missing_cols():
    fe = FeatureEngineer()
    # Only bid_price present
    df = pd.DataFrame({
        "bid_price": [100.0, 101.0],
        "bid_size": [10.0, 20.0]
    })
    result = fe.calculate_order_book_features(df)
    
    # Should not crash, but also not calculate features requiring ask_price
    assert "mid_price" not in result.columns
    assert "bid_ask_spread" not in result.columns
    assert "wap" not in result.columns
    assert "bid_price" in result.columns
