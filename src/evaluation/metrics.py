"""Evaluation metrics for classification, regression, and trading."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    metrics: Dict[str, float]
    metric_type: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "metric_type": self.metric_type,
            "metadata": self.metadata,
        }


class MetricsCalculator:
    """Calculate evaluation metrics for different model types."""

    def __init__(self):
        """Initialize MetricsCalculator."""
        self._classification_metrics = [
            "accuracy", "precision", "recall", "f1",
            "brier_score", "log_loss", "roc_auc"
        ]
        self._regression_metrics = ["mse", "rmse", "mae", "r2", "mape"]
        self._trading_metrics = [
            "sharpe_ratio", "max_drawdown", "hit_rate",
            "profit_factor", "win_rate"
        ]

    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = "binary",
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            average: Averaging method for multiclass ('binary', 'micro', 'macro', 'weighted')

        Returns:
            Dictionary of metric names to values
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        metrics: Dict[str, float] = {}

        # Basic classification metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

        # Handle multiclass vs binary
        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) <= 2

        if is_binary:
            metrics["precision"] = float(
                precision_score(y_true, y_pred, zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y_true, y_pred, zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(y_true, y_pred, zero_division=0)
            )
        else:
            metrics["precision"] = float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y_true, y_pred, average=average, zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(y_true, y_pred, average=average, zero_division=0)
            )

        # Probability-based metrics
        if y_proba is not None:
            y_proba = np.asarray(y_proba)

            # Handle 2D probability arrays
            if y_proba.ndim == 2:
                if y_proba.shape[1] == 2:
                    # Binary classification - use positive class probability
                    proba_for_metrics = y_proba[:, 1]
                else:
                    # Multiclass - use full probability matrix
                    proba_for_metrics = y_proba
            else:
                proba_for_metrics = y_proba

            if is_binary or y_proba.ndim == 1:
                metrics["brier_score"] = float(
                    brier_score_loss(y_true, proba_for_metrics)
                )
                try:
                    metrics["log_loss"] = float(
                        log_loss(y_true, proba_for_metrics)
                    )
                except ValueError:
                    metrics["log_loss"] = np.nan

                try:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, proba_for_metrics)
                    )
                except ValueError:
                    metrics["roc_auc"] = np.nan
            else:
                try:
                    metrics["log_loss"] = float(
                        log_loss(y_true, proba_for_metrics)
                    )
                except ValueError:
                    metrics["log_loss"] = np.nan

                try:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, proba_for_metrics, multi_class="ovr")
                    )
                except ValueError:
                    metrics["roc_auc"] = np.nan

        # Directional hit rate (for trading applications)
        metrics["hit_rate"] = float(accuracy_score(y_true, y_pred))

        return metrics

    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metric names to values
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        metrics: Dict[str, float] = {}

        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics["mape"] = float(mape)
        else:
            metrics["mape"] = np.nan

        return metrics

    def calculate_trading_metrics(
        self,
        returns: np.ndarray,
        signals: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> Dict[str, float]:
        """
        Calculate trading-specific metrics.

        Args:
            returns: Asset returns
            signals: Trading signals (-1, 0, 1 for short, hold, long)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of trading periods per year

        Returns:
            Dictionary of metric names to values
        """
        returns = np.asarray(returns)
        signals = np.asarray(signals)

        # Strategy returns
        strategy_returns = returns * signals

        metrics: Dict[str, float] = {}

        # Sharpe Ratio
        if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
            excess_returns = strategy_returns - risk_free_rate / periods_per_year
            sharpe = (
                np.mean(excess_returns) / np.std(excess_returns)
            ) * np.sqrt(periods_per_year)
            metrics["sharpe_ratio"] = float(sharpe)
        else:
            metrics["sharpe_ratio"] = 0.0

        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        metrics["max_drawdown"] = float(np.min(drawdowns))

        # Hit Rate (percentage of correct direction predictions)
        if len(returns) > 0:
            correct_direction = (np.sign(returns) == np.sign(signals)) | (signals == 0)
            metrics["hit_rate"] = float(np.mean(correct_direction))
        else:
            metrics["hit_rate"] = 0.0

        # Win Rate (percentage of profitable trades)
        trades = strategy_returns[signals != 0]
        if len(trades) > 0:
            metrics["win_rate"] = float(np.mean(trades > 0))
        else:
            metrics["win_rate"] = 0.0

        # Profit Factor
        gains = strategy_returns[strategy_returns > 0].sum()
        losses = abs(strategy_returns[strategy_returns < 0].sum())
        if losses > 0:
            metrics["profit_factor"] = float(gains / losses)
        else:
            metrics["profit_factor"] = float("inf") if gains > 0 else 0.0

        # Total Return
        metrics["total_return"] = float(np.prod(1 + strategy_returns) - 1)

        # Sortino Ratio (using downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                sortino = (
                    np.mean(strategy_returns) / downside_std
                ) * np.sqrt(periods_per_year)
                metrics["sortino_ratio"] = float(sortino)
            else:
                metrics["sortino_ratio"] = 0.0
        else:
            metrics["sortino_ratio"] = 0.0

        return metrics

    def calculate_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Calculate probability calibration metrics.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration analysis

        Returns:
            Dictionary of calibration metrics
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        # Handle 2D probability arrays
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]

        metrics: Dict[str, float] = {}

        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(y_true)

        for i in range(n_bins):
            bin_mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])
            if i == n_bins - 1:
                bin_mask = (y_proba >= bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])

            bin_size = bin_mask.sum()
            if bin_size > 0:
                bin_accuracy = y_true[bin_mask].mean()
                bin_confidence = y_proba[bin_mask].mean()
                ece += (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)

        metrics["ece"] = float(ece)

        # Maximum Calibration Error (MCE)
        mce = 0.0
        for i in range(n_bins):
            bin_mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])
            if i == n_bins - 1:
                bin_mask = (y_proba >= bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])

            bin_size = bin_mask.sum()
            if bin_size > 0:
                bin_accuracy = y_true[bin_mask].mean()
                bin_confidence = y_proba[bin_mask].mean()
                mce = max(mce, abs(bin_accuracy - bin_confidence))

        metrics["mce"] = float(mce)

        # Brier Score (also a calibration metric)
        metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))

        return metrics

    def get_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        task_type: str = "classification",
    ) -> MetricsResult:
        """
        Calculate all relevant metrics for a given task type.

        Args:
            y_true: True values/labels
            y_pred: Predicted values/labels
            y_proba: Predicted probabilities (for classification)
            task_type: Type of task ('classification' or 'regression')

        Returns:
            MetricsResult containing all calculated metrics
        """
        if task_type == "classification":
            metrics = self.calculate_classification_metrics(y_true, y_pred, y_proba)
            if y_proba is not None:
                calibration = self.calculate_calibration_metrics(y_true, y_proba)
                metrics.update({f"calibration_{k}": v for k, v in calibration.items()})
        elif task_type == "regression":
            metrics = self.calculate_regression_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return MetricsResult(
            metrics=metrics,
            metric_type=task_type,
            metadata={"n_samples": len(y_true)},
        )
