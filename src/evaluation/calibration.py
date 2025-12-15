"""Calibration analysis utilities and plotting functions."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationCurve:
    """Container for calibration curve data."""
    bin_centers: np.ndarray
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray
    n_bins: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bin_centers": self.bin_centers.tolist(),
            "bin_accuracies": self.bin_accuracies.tolist(),
            "bin_confidences": self.bin_confidences.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "n_bins": self.n_bins,
        }


class CalibrationAnalyzer:
    """Analyze and visualize probability calibration."""

    def __init__(self, n_bins: int = 10):
        """
        Initialize CalibrationAnalyzer.

        Args:
            n_bins: Number of bins for calibration analysis
        """
        self.n_bins = n_bins

    def compute_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> CalibrationCurve:
        """
        Compute calibration curve data.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities

        Returns:
            CalibrationCurve with bin statistics
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        # Handle 2D probability arrays
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        bin_accuracies = np.zeros(self.n_bins)
        bin_confidences = np.zeros(self.n_bins)
        bin_counts = np.zeros(self.n_bins, dtype=int)

        for i in range(self.n_bins):
            if i == self.n_bins - 1:
                bin_mask = (y_proba >= bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])
            else:
                bin_mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])

            bin_counts[i] = bin_mask.sum()
            if bin_counts[i] > 0:
                bin_accuracies[i] = y_true[bin_mask].mean()
                bin_confidences[i] = y_proba[bin_mask].mean()
            else:
                bin_accuracies[i] = np.nan
                bin_confidences[i] = np.nan

        return CalibrationCurve(
            bin_centers=bin_centers,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
            n_bins=self.n_bins,
        )

    def compute_reliability_diagram_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute data for reliability diagram plotting.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary with plotting data
        """
        curve = self.compute_calibration_curve(y_true, y_proba)

        # Filter out empty bins
        valid_mask = curve.bin_counts > 0

        return {
            "mean_predicted_value": curve.bin_confidences[valid_mask],
            "fraction_of_positives": curve.bin_accuracies[valid_mask],
            "bin_counts": curve.bin_counts[valid_mask],
            "bin_centers": curve.bin_centers[valid_mask],
        }

    def compute_ece(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Compute Expected Calibration Error.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities

        Returns:
            ECE value
        """
        curve = self.compute_calibration_curve(y_true, y_proba)
        total_samples = curve.bin_counts.sum()

        if total_samples == 0:
            return 0.0

        ece = 0.0
        for i in range(self.n_bins):
            if curve.bin_counts[i] > 0:
                weight = curve.bin_counts[i] / total_samples
                ece += weight * abs(curve.bin_accuracies[i] - curve.bin_confidences[i])

        return float(ece)

    def compute_mce(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Compute Maximum Calibration Error.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities

        Returns:
            MCE value
        """
        curve = self.compute_calibration_curve(y_true, y_proba)

        mce = 0.0
        for i in range(self.n_bins):
            if curve.bin_counts[i] > 0:
                error = abs(curve.bin_accuracies[i] - curve.bin_confidences[i])
                mce = max(mce, error)

        return float(mce)

    def get_calibration_summary(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Get comprehensive calibration summary.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary with calibration metrics and curve data
        """
        curve = self.compute_calibration_curve(y_true, y_proba)

        return {
            "ece": self.compute_ece(y_true, y_proba),
            "mce": self.compute_mce(y_true, y_proba),
            "curve": curve.to_dict(),
            "n_samples": len(y_true),
            "n_bins": self.n_bins,
        }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    ax=None,
    name: str = "Model",
) -> Any:
    """
    Plot calibration curve (reliability diagram).

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        ax: Matplotlib axes (optional)
        name: Model name for legend

    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    analyzer = CalibrationAnalyzer(n_bins=n_bins)
    data = analyzer.compute_reliability_diagram_data(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    # Plot calibration curve
    ax.plot(
        data["mean_predicted_value"],
        data["fraction_of_positives"],
        "s-",
        label=name,
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return ax


def plot_calibration_histogram(
    y_proba: np.ndarray,
    n_bins: int = 10,
    ax=None,
) -> Any:
    """
    Plot histogram of predicted probabilities.

    Args:
        y_proba: Predicted probabilities
        n_bins: Number of bins
        ax: Matplotlib axes (optional)

    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(y_proba, bins=n_bins, range=(0, 1), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predicted Probabilities")

    return ax
