"""
Property-based tests for model evaluation metrics.

**Feature: forecasting-research-pipeline, Property 5: Model training and
evaluation correctness (metrics component)**

Tests that evaluation metrics are mathematically correct, consistent,
and handle edge cases appropriately across classification, regression,
and trading scenarios.

**Validates: Requirements 4.3, 5.2**
"""

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from src.evaluation.metrics import MetricsCalculator, MetricsResult
from src.evaluation.calibration import CalibrationAnalyzer, CalibrationCurve


# Custom strategies for generating test data
@st.composite
def binary_classification_data(draw, min_samples=20, max_samples=200):
    """Generate binary classification predictions and labels."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    
    # Generate true labels (0 or 1)
    y_true = draw(
        st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=n_samples,
            max_size=n_samples,
        )
    )
    y_true = np.array(y_true)
    
    # Ensure we have both classes
    assume(len(np.unique(y_true)) == 2)
    
    # Generate predicted labels
    y_pred = draw(
        st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=n_samples,
            max_size=n_samples,
        )
    )
    y_pred = np.array(y_pred)
    
    # Generate probabilities (correlated with predictions for realism)
    y_proba = np.zeros((n_samples, 2))
    for i in range(n_samples):
        if y_pred[i] == 1:
            p = draw(st.floats(min_value=0.5, max_value=1.0))
        else:
            p = draw(st.floats(min_value=0.0, max_value=0.5))
        y_proba[i, 1] = p
        y_proba[i, 0] = 1 - p
    
    return y_true, y_pred, y_proba


@st.composite
def multiclass_classification_data(draw, min_samples=30, max_samples=150, n_classes=3):
    """Generate multiclass classification predictions and labels."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    
    # Generate true labels
    y_true = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_classes - 1),
            min_size=n_samples,
            max_size=n_samples,
        )
    )
    y_true = np.array(y_true)
    
    # Ensure we have all classes represented
    assume(len(np.unique(y_true)) == n_classes)
    
    # Generate predicted labels
    y_pred = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_classes - 1),
            min_size=n_samples,
            max_size=n_samples,
        )
    )
    y_pred = np.array(y_pred)
    
    return y_true, y_pred


@st.composite
def regression_data(draw, min_samples=20, max_samples=200):
    """Generate regression predictions and true values."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    
    # Generate true values
    y_true = draw(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=n_samples,
            max_size=n_samples,
        )
    )
    y_true = np.array(y_true)
    
    # Generate predictions with some noise
    noise_scale = draw(st.floats(min_value=0.1, max_value=10.0))
    noise = np.array([
        draw(st.floats(min_value=-noise_scale, max_value=noise_scale))
        for _ in range(n_samples)
    ])
    y_pred = y_true + noise
    
    return y_true, y_pred


@st.composite
def trading_data(draw, min_periods=50, max_periods=200):
    """Generate trading returns and signals."""
    n_periods = draw(st.integers(min_value=min_periods, max_value=max_periods))
    
    # Generate returns (small daily returns)
    returns = draw(
        st.lists(
            st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
            min_size=n_periods,
            max_size=n_periods,
        )
    )
    returns = np.array(returns)
    
    # Generate signals (-1, 0, 1)
    signals = draw(
        st.lists(
            st.integers(min_value=-1, max_value=1),
            min_size=n_periods,
            max_size=n_periods,
        )
    )
    signals = np.array(signals)
    
    # Ensure we have some non-zero signals
    assume(np.sum(signals != 0) > 5)
    
    return returns, signals


@st.composite
def probability_data(draw, min_samples=30, max_samples=150):
    """Generate probability predictions for calibration testing."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    
    # Generate true labels
    y_true = draw(
        st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=n_samples,
            max_size=n_samples,
        )
    )
    y_true = np.array(y_true)
    
    # Ensure both classes present
    assume(len(np.unique(y_true)) == 2)
    
    # Generate probabilities spread across [0, 1]
    y_proba = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=0.99),
            min_size=n_samples,
            max_size=n_samples,
        )
    )
    y_proba = np.array(y_proba)
    
    return y_true, y_proba


class TestClassificationMetricsProperty:
    """
    Property tests for classification metrics.

    **Feature: forecasting-research-pipeline, Property 5: Model training
    and evaluation correctness**
    **Validates: Requirements 4.3, 5.2**
    """

    @given(data=binary_classification_data())
    @settings(max_examples=100, deadline=None)
    def test_accuracy_bounds(self, data):
        """
        Property: Accuracy should always be between 0 and 1.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 4.3**
        """
        y_true, y_pred, y_proba = data
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_classification_metrics(y_true, y_pred, y_proba)
        
        assert 0 <= metrics["accuracy"] <= 1, (
            f"Accuracy out of bounds: {metrics['accuracy']}"
        )

    @given(data=binary_classification_data())
    @settings(max_examples=100, deadline=None)
    def test_precision_recall_f1_bounds(self, data):
        """
        Property: Precision, recall, and F1 should be between 0 and 1.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 4.3**
        """
        y_true, y_pred, y_proba = data
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_classification_metrics(y_true, y_pred, y_proba)
        
        for metric_name in ["precision", "recall", "f1"]:
            assert 0 <= metrics[metric_name] <= 1, (
                f"{metric_name} out of bounds: {metrics[metric_name]}"
            )

    @given(data=binary_classification_data())
    @settings(max_examples=100, deadline=None)
    def test_brier_score_bounds(self, data):
        """
        Property: Brier score should be between 0 and 1.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 4.3**
        """
        y_true, y_pred, y_proba = data
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_classification_metrics(y_true, y_pred, y_proba)
        
        assert 0 <= metrics["brier_score"] <= 1, (
            f"Brier score out of bounds: {metrics['brier_score']}"
        )

    @given(data=binary_classification_data())
    @settings(max_examples=100, deadline=None)
    def test_perfect_predictions_accuracy(self, data):
        """
        Property: Perfect predictions should yield accuracy of 1.0.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 4.3**
        """
        y_true, _, _ = data
        calculator = MetricsCalculator()
        
        # Perfect predictions
        metrics = calculator.calculate_classification_metrics(y_true, y_true)
        
        assert metrics["accuracy"] == 1.0, (
            f"Perfect predictions should have accuracy 1.0, got {metrics['accuracy']}"
        )

    @given(data=multiclass_classification_data())
    @settings(max_examples=100, deadline=None)
    def test_multiclass_metrics_bounds(self, data):
        """
        Property: Multiclass metrics should be between 0 and 1.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 4.3**
        """
        y_true, y_pred = data
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_classification_metrics(
            y_true, y_pred, average="macro"
        )
        
        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            assert 0 <= metrics[metric_name] <= 1, (
                f"Multiclass {metric_name} out of bounds: {metrics[metric_name]}"
            )


class TestRegressionMetricsProperty:
    """
    Property tests for regression metrics.

    **Feature: forecasting-research-pipeline, Property 5: Model training
    and evaluation correctness**
    **Validates: Requirements 5.2**
    """

    @given(data=regression_data())
    @settings(max_examples=100, deadline=None)
    def test_mse_non_negative(self, data):
        """
        Property: MSE should always be non-negative.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 5.2**
        """
        y_true, y_pred = data
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)
        
        assert metrics["mse"] >= 0, f"MSE should be non-negative: {metrics['mse']}"
        assert metrics["rmse"] >= 0, f"RMSE should be non-negative: {metrics['rmse']}"
        assert metrics["mae"] >= 0, f"MAE should be non-negative: {metrics['mae']}"

    @given(data=regression_data())
    @settings(max_examples=100, deadline=None)
    def test_rmse_equals_sqrt_mse(self, data):
        """
        Property: RMSE should equal sqrt(MSE).

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 5.2**
        """
        y_true, y_pred = data
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)
        
        expected_rmse = np.sqrt(metrics["mse"])
        assert np.isclose(metrics["rmse"], expected_rmse, rtol=1e-10), (
            f"RMSE {metrics['rmse']} != sqrt(MSE) {expected_rmse}"
        )

    @given(data=regression_data())
    @settings(max_examples=100, deadline=None)
    def test_perfect_predictions_zero_error(self, data):
        """
        Property: Perfect predictions should yield zero MSE, RMSE, MAE.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 5.2**
        """
        y_true, _ = data
        calculator = MetricsCalculator()
        
        # Perfect predictions
        metrics = calculator.calculate_regression_metrics(y_true, y_true)
        
        assert metrics["mse"] == 0, f"Perfect MSE should be 0, got {metrics['mse']}"
        assert metrics["rmse"] == 0, f"Perfect RMSE should be 0, got {metrics['rmse']}"
        assert metrics["mae"] == 0, f"Perfect MAE should be 0, got {metrics['mae']}"

    @given(data=regression_data())
    @settings(max_examples=100, deadline=None)
    def test_r2_perfect_predictions(self, data):
        """
        Property: Perfect predictions should yield R² of 1.0.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 5.2**
        """
        y_true, _ = data
        
        # Skip if all values are the same (R² undefined)
        assume(np.std(y_true) > 1e-10)
        
        calculator = MetricsCalculator()
        metrics = calculator.calculate_regression_metrics(y_true, y_true)
        
        assert np.isclose(metrics["r2"], 1.0, rtol=1e-10), (
            f"Perfect R² should be 1.0, got {metrics['r2']}"
        )

    @given(data=regression_data())
    @settings(max_examples=100, deadline=None)
    def test_mae_less_than_or_equal_rmse(self, data):
        """
        Property: MAE should be less than or equal to RMSE.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 5.2**
        """
        y_true, y_pred = data
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)
        
        # MAE <= RMSE always holds (by Cauchy-Schwarz inequality)
        assert metrics["mae"] <= metrics["rmse"] + 1e-10, (
            f"MAE {metrics['mae']} should be <= RMSE {metrics['rmse']}"
        )


class TestTradingMetricsProperty:
    """
    Property tests for trading metrics.

    **Feature: forecasting-research-pipeline, Property 5: Model training
    and evaluation correctness**
    **Validates: Requirements 5.2**
    """

    @given(data=trading_data())
    @settings(max_examples=100, deadline=None)
    def test_max_drawdown_bounds(self, data):
        """
        Property: Max drawdown should be between -1 and 0.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 5.2**
        """
        returns, signals = data
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_trading_metrics(returns, signals)
        
        assert metrics["max_drawdown"] <= 0, (
            f"Max drawdown should be <= 0: {metrics['max_drawdown']}"
        )
        assert metrics["max_drawdown"] >= -1, (
            f"Max drawdown should be >= -1: {metrics['max_drawdown']}"
        )

    @given(data=trading_data())
    @settings(max_examples=100, deadline=None)
    def test_hit_rate_bounds(self, data):
        """
        Property: Hit rate should be between 0 and 1.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 5.2**
        """
        returns, signals = data
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_trading_metrics(returns, signals)
        
        assert 0 <= metrics["hit_rate"] <= 1, (
            f"Hit rate out of bounds: {metrics['hit_rate']}"
        )

    @given(data=trading_data())
    @settings(max_examples=100, deadline=None)
    def test_win_rate_bounds(self, data):
        """
        Property: Win rate should be between 0 and 1.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 5.2**
        """
        returns, signals = data
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_trading_metrics(returns, signals)
        
        assert 0 <= metrics["win_rate"] <= 1, (
            f"Win rate out of bounds: {metrics['win_rate']}"
        )

    @given(data=trading_data())
    @settings(max_examples=100, deadline=None)
    def test_profit_factor_non_negative(self, data):
        """
        Property: Profit factor should be non-negative.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 5.2**
        """
        returns, signals = data
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_trading_metrics(returns, signals)
        
        assert metrics["profit_factor"] >= 0, (
            f"Profit factor should be >= 0: {metrics['profit_factor']}"
        )

    @given(data=trading_data())
    @settings(max_examples=100, deadline=None)
    def test_zero_signals_zero_return(self, data):
        """
        Property: All-zero signals should yield zero total return.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 5.2**
        """
        returns, _ = data
        calculator = MetricsCalculator()
        
        # All zero signals
        zero_signals = np.zeros_like(returns)
        metrics = calculator.calculate_trading_metrics(returns, zero_signals)
        
        assert metrics["total_return"] == 0, (
            f"Zero signals should yield zero return: {metrics['total_return']}"
        )


class TestCalibrationMetricsProperty:
    """
    Property tests for calibration metrics.

    **Feature: forecasting-research-pipeline, Property 5: Model training
    and evaluation correctness**
    **Validates: Requirements 4.3, 5.2**
    """

    @given(data=probability_data())
    @settings(max_examples=100, deadline=None)
    def test_ece_bounds(self, data):
        """
        Property: ECE should be between 0 and 1.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 4.3**
        """
        y_true, y_proba = data
        analyzer = CalibrationAnalyzer(n_bins=10)
        
        ece = analyzer.compute_ece(y_true, y_proba)
        
        assert 0 <= ece <= 1, f"ECE out of bounds: {ece}"

    @given(data=probability_data())
    @settings(max_examples=100, deadline=None)
    def test_mce_bounds(self, data):
        """
        Property: MCE should be between 0 and 1.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 4.3**
        """
        y_true, y_proba = data
        analyzer = CalibrationAnalyzer(n_bins=10)
        
        mce = analyzer.compute_mce(y_true, y_proba)
        
        assert 0 <= mce <= 1, f"MCE out of bounds: {mce}"

    @given(data=probability_data())
    @settings(max_examples=100, deadline=None)
    def test_ece_less_than_or_equal_mce(self, data):
        """
        Property: ECE should be less than or equal to MCE.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 4.3**
        """
        y_true, y_proba = data
        analyzer = CalibrationAnalyzer(n_bins=10)
        
        ece = analyzer.compute_ece(y_true, y_proba)
        mce = analyzer.compute_mce(y_true, y_proba)
        
        # ECE is weighted average, MCE is max, so ECE <= MCE
        assert ece <= mce + 1e-10, f"ECE {ece} should be <= MCE {mce}"

    @given(data=probability_data())
    @settings(max_examples=100, deadline=None)
    def test_calibration_curve_bin_counts_sum(self, data):
        """
        Property: Calibration curve bin counts should sum to total samples.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 4.3**
        """
        y_true, y_proba = data
        analyzer = CalibrationAnalyzer(n_bins=10)
        
        curve = analyzer.compute_calibration_curve(y_true, y_proba)
        
        assert curve.bin_counts.sum() == len(y_true), (
            f"Bin counts {curve.bin_counts.sum()} != total samples {len(y_true)}"
        )

    @given(data=probability_data())
    @settings(max_examples=100, deadline=None)
    def test_calibration_summary_consistency(self, data):
        """
        Property: Calibration summary should be internally consistent.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 4.3**
        """
        y_true, y_proba = data
        analyzer = CalibrationAnalyzer(n_bins=10)
        
        summary = analyzer.get_calibration_summary(y_true, y_proba)
        
        # Check consistency
        assert summary["n_samples"] == len(y_true)
        assert summary["n_bins"] == 10
        assert "ece" in summary
        assert "mce" in summary
        assert "curve" in summary


class TestMetricsResultProperty:
    """
    Property tests for MetricsResult container.

    **Feature: forecasting-research-pipeline, Property 5: Model training
    and evaluation correctness**
    **Validates: Requirements 4.3, 5.2**
    """

    @given(data=binary_classification_data())
    @settings(max_examples=100, deadline=None)
    def test_get_all_metrics_classification(self, data):
        """
        Property: get_all_metrics should return valid MetricsResult for classification.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 4.3**
        """
        y_true, y_pred, y_proba = data
        calculator = MetricsCalculator()
        
        result = calculator.get_all_metrics(
            y_true, y_pred, y_proba, task_type="classification"
        )
        
        assert isinstance(result, MetricsResult)
        assert result.metric_type == "classification"
        assert result.metadata["n_samples"] == len(y_true)
        assert "accuracy" in result.metrics

    @given(data=regression_data())
    @settings(max_examples=100, deadline=None)
    def test_get_all_metrics_regression(self, data):
        """
        Property: get_all_metrics should return valid MetricsResult for regression.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 5.2**
        """
        y_true, y_pred = data
        calculator = MetricsCalculator()
        
        result = calculator.get_all_metrics(
            y_true, y_pred, task_type="regression"
        )
        
        assert isinstance(result, MetricsResult)
        assert result.metric_type == "regression"
        assert result.metadata["n_samples"] == len(y_true)
        assert "mse" in result.metrics
        assert "rmse" in result.metrics

    @given(data=binary_classification_data())
    @settings(max_examples=100, deadline=None)
    def test_metrics_result_to_dict(self, data):
        """
        Property: MetricsResult.to_dict should produce valid dictionary.

        **Feature: forecasting-research-pipeline, Property 5**
        **Validates: Requirements 4.3**
        """
        y_true, y_pred, y_proba = data
        calculator = MetricsCalculator()
        
        result = calculator.get_all_metrics(
            y_true, y_pred, y_proba, task_type="classification"
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "metrics" in result_dict
        assert "metric_type" in result_dict
        assert "metadata" in result_dict
