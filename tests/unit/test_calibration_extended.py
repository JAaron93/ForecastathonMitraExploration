"""Unit tests for CalibrationAnalyzer covering uncovered lines."""
import pytest
import numpy as np
from src.evaluation.calibration import CalibrationAnalyzer, CalibrationCurve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Simple binary classification data."""
    np.random.seed(42)
    n = 500
    y_true = np.random.randint(0, 2, n)
    y_proba = np.clip(np.random.uniform(0, 1, n), 0.01, 0.99)
    return y_true, y_proba


@pytest.fixture
def perfect_data():
    """Perfectly calibrated data."""
    y_true = np.array([1, 1, 0, 0, 1])
    y_proba = np.array([0.95, 0.85, 0.05, 0.15, 0.75])
    return y_true, y_proba


@pytest.fixture
def analyzer():
    return CalibrationAnalyzer(n_bins=10)


# ---------------------------------------------------------------------------
# compute_calibration_curve
# ---------------------------------------------------------------------------

class TestComputeCalibrationCurve:
    def test_basic_output_shape(self, analyzer, binary_data):
        y_true, y_proba = binary_data
        curve = analyzer.compute_calibration_curve(y_true, y_proba)
        assert len(curve.bin_centers) == analyzer.n_bins
        assert len(curve.bin_accuracies) == analyzer.n_bins
        assert len(curve.bin_confidences) == analyzer.n_bins

    def test_2d_proba_second_class_extracted(self, analyzer):
        y_true = np.array([0, 1, 0, 1])
        y_proba_2d = np.array([[0.9, 0.1], [0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
        curve = analyzer.compute_calibration_curve(y_true, y_proba_2d)
        assert curve is not None

    def test_2d_proba_wrong_shape_raises(self, analyzer):
        y_true = np.array([0, 1, 0])
        y_proba = np.random.rand(3, 3)  # 3 classes – invalid
        with pytest.raises(ValueError, match="shape"):
            analyzer.compute_calibration_curve(y_true, y_proba)

    def test_3d_proba_raises(self, analyzer):
        y_true = np.array([0, 1])
        y_proba = np.random.rand(2, 2, 2)  # 3D – invalid
        with pytest.raises(ValueError, match="1D or 2D"):
            analyzer.compute_calibration_curve(y_true, y_proba)

    def test_empty_bins_have_nan(self, analyzer):
        # Concentrate all probability mass in [0.9, 1.0]; lower bins will be empty
        y_true = np.ones(50)
        y_proba = np.full(50, 0.95)
        curve = analyzer.compute_calibration_curve(y_true, y_proba)
        # All bins except the last should have nan
        assert np.isnan(curve.bin_accuracies[0])

    def test_to_dict_roundtrip(self, analyzer, binary_data):
        y_true, y_proba = binary_data
        curve = analyzer.compute_calibration_curve(y_true, y_proba)
        d = curve.to_dict()
        assert isinstance(d["bin_centers"], list)
        assert len(d["bin_accuracies"]) == analyzer.n_bins


# ---------------------------------------------------------------------------
# compute_reliability_diagram_data
# ---------------------------------------------------------------------------

class TestComputeReliabilityDiagramData:
    def test_filters_empty_bins(self, analyzer, binary_data):
        y_true, y_proba = binary_data
        data = analyzer.compute_reliability_diagram_data(y_true, y_proba)
        # All returned values should be from non-empty bins
        assert all(c > 0 for c in data["bin_counts"])

    def test_output_keys(self, analyzer, binary_data):
        y_true, y_proba = binary_data
        data = analyzer.compute_reliability_diagram_data(y_true, y_proba)
        for key in ("mean_predicted_value", "fraction_of_positives", "bin_counts", "bin_centers"):
            assert key in data


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------

class TestComputeECE:
    def test_ece_is_non_negative(self, analyzer, binary_data):
        y_true, y_proba = binary_data
        ece = analyzer.compute_ece(y_true, y_proba)
        assert ece >= 0.0

    def test_ece_perfect_calibration_near_zero(self, analyzer):
        # Synthetic perfectly calibrated data in one bin
        y_true = np.array([1, 1, 0, 0] * 50)  # 50% positive
        y_proba = np.array([0.5] * 200)        # all predict 0.5
        ece = analyzer.compute_ece(y_true, y_proba)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_ece_empty_bins_handled(self, analyzer):
        # Only a few samples — some bins will be empty
        y_true = np.array([1, 0])
        y_proba = np.array([0.95, 0.05])
        ece = analyzer.compute_ece(y_true, y_proba)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# compute_mce
# ---------------------------------------------------------------------------

class TestComputeMCE:
    def test_mce_is_non_negative(self, analyzer, binary_data):
        y_true, y_proba = binary_data
        mce = analyzer.compute_mce(y_true, y_proba)
        assert mce >= 0.0

    def test_mce_gte_ece(self, analyzer, binary_data):
        y_true, y_proba = binary_data
        ece = analyzer.compute_ece(y_true, y_proba)
        mce = analyzer.compute_mce(y_true, y_proba)
        assert mce >= ece - 1e-10


# ---------------------------------------------------------------------------
# get_calibration_summary
# ---------------------------------------------------------------------------

class TestGetCalibrationSummary:
    def test_summary_contains_expected_keys(self, analyzer, binary_data):
        y_true, y_proba = binary_data
        summary = analyzer.get_calibration_summary(y_true, y_proba)
        for key in ("ece", "mce", "curve", "n_samples", "n_bins"):
            assert key in summary

    def test_n_samples_matches_input(self, analyzer, binary_data):
        y_true, y_proba = binary_data
        summary = analyzer.get_calibration_summary(y_true, y_proba)
        assert summary["n_samples"] == len(y_true)

    def test_n_bins_matches_analyzer(self, analyzer, binary_data):
        y_true, y_proba = binary_data
        summary = analyzer.get_calibration_summary(y_true, y_proba)
        assert summary["n_bins"] == analyzer.n_bins


# ---------------------------------------------------------------------------
# plot_calibration_curve & plot_calibration_histogram (smoke tests)
# ---------------------------------------------------------------------------

class TestPlottingFunctions:
    def test_plot_calibration_curve_returns_axes(self, binary_data):
        from src.evaluation.calibration import plot_calibration_curve
        import matplotlib
        matplotlib.use("Agg")
        y_true, y_proba = binary_data
        ax = plot_calibration_curve(y_true, y_proba)
        assert ax is not None

    def test_plot_calibration_histogram_returns_axes(self, binary_data):
        from src.evaluation.calibration import plot_calibration_histogram
        import matplotlib
        matplotlib.use("Agg")
        _, y_proba = binary_data
        ax = plot_calibration_histogram(y_proba)
        assert ax is not None

    def test_plot_histogram_with_2d_proba(self, binary_data):
        from src.evaluation.calibration import plot_calibration_histogram
        import matplotlib
        matplotlib.use("Agg")
        y_true, y_proba = binary_data
        y_proba_2d = np.stack([1 - y_proba, y_proba], axis=1)
        ax = plot_calibration_histogram(y_proba_2d)
        assert ax is not None
