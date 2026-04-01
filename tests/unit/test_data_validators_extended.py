"""Unit tests for DataValidator covering all uncovered lines."""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.validators import DataValidator, QualityMetrics


@pytest.fixture
def baseline_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "price": np.random.normal(100, 10, n),
        "volume": np.random.normal(1000, 100, n),
    })


@pytest.fixture
def shifted_df():
    np.random.seed(99)
    n = 200
    return pd.DataFrame({
        "price": np.random.normal(200, 10, n),   # big shift
        "volume": np.random.normal(1010, 100, n), # small shift
    })


@pytest.fixture
def validator():
    return DataValidator()


@pytest.fixture
def validator_with_log(tmp_path):
    return DataValidator(log_dir=str(tmp_path / "logs"))


# ---------------------------------------------------------------------------
# QualityMetrics.to_dict
# ---------------------------------------------------------------------------

class TestQualityMetricsToDict:
    def test_to_dict_contains_expected_keys(self):
        qm = QualityMetrics(
            null_count={"a": 0},
            unique_count={"a": 5},
            row_count=10,
            column_count=1,
            value_ranges={"a": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.2}},
        )
        d = qm.to_dict()
        assert "null_count" in d
        assert "value_ranges" in d
        assert "psi_scores" in d


# ---------------------------------------------------------------------------
# DataValidator.__init__ with log directory
# ---------------------------------------------------------------------------

class TestInit:
    def test_log_dir_created(self, tmp_path):
        log_dir = tmp_path / "validator_logs"
        validator = DataValidator(log_dir=str(log_dir))
        assert log_dir.exists()


# ---------------------------------------------------------------------------
# calculate_psi edge cases
# ---------------------------------------------------------------------------

class TestCalculatePSI:
    def test_empty_series_returns_zero(self, validator):
        s = pd.Series([], dtype=float)
        assert validator.calculate_psi(s, pd.Series([1.0, 2.0])) == 0.0
        assert validator.calculate_psi(pd.Series([1.0, 2.0]), s) == 0.0

    def test_constant_series_returns_zero(self, validator):
        s = pd.Series([5.0] * 100)
        assert validator.calculate_psi(s, s) == 0.0

    def test_identical_distributions_near_zero(self, validator):
        s = pd.Series(np.random.normal(0, 1, 200))
        psi = validator.calculate_psi(s, s)
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distribution_has_positive_psi(self, validator, baseline_df, shifted_df):
        psi = validator.calculate_psi(baseline_df["price"], shifted_df["price"])
        assert psi > 0.1  # Should detect significant shift


# ---------------------------------------------------------------------------
# get_quality_metrics with baseline (PSI path)
# ---------------------------------------------------------------------------

class TestGetQualityMetrics:
    def test_returns_psi_scores_when_baseline_provided(self, validator, baseline_df, shifted_df):
        metrics = validator.get_quality_metrics(shifted_df, baseline_df=baseline_df)
        assert len(metrics.psi_scores) > 0
        assert "price" in metrics.psi_scores

    def test_no_psi_without_baseline(self, validator, baseline_df):
        metrics = validator.get_quality_metrics(baseline_df)
        assert metrics.psi_scores == {}

    def test_all_na_column_gets_zero_range(self, validator):
        df = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan]})
        metrics = validator.get_quality_metrics(df)
        assert metrics.value_ranges["all_nan"]["min"] == 0.0


# ---------------------------------------------------------------------------
# validate_data_quality
# ---------------------------------------------------------------------------

class TestValidateDataQuality:
    def test_empty_df_is_valid(self, validator):
        df = pd.DataFrame({"a": []})
        is_valid, issues = validator.validate_data_quality(df)
        assert is_valid
        assert issues == []

    def test_high_null_ratio_fails(self, validator):
        df = pd.DataFrame({"a": [np.nan] * 90 + [1.0] * 10})
        is_valid, issues = validator.validate_data_quality(df, max_null_ratio=0.3)
        assert not is_valid
        assert any("nulls" in issue for issue in issues)

    def test_low_variance_column_flagged(self, validator):
        # All the same value → 1 unique / 100 rows = 0.01 unique ratio
        df = pd.DataFrame({"price": [42.0] * 100})
        is_valid, issues = validator.validate_data_quality(df, min_unique_ratio=0.05)
        assert not is_valid
        assert any("low variance" in issue for issue in issues)

    def test_valid_data_passes(self, validator, baseline_df):
        is_valid, issues = validator.validate_data_quality(baseline_df)
        assert is_valid
        assert issues == []


# ---------------------------------------------------------------------------
# log_quality_metrics
# ---------------------------------------------------------------------------

class TestLogQualityMetrics:
    def test_logs_to_json_file(self, validator_with_log, baseline_df):
        metrics = validator_with_log.get_quality_metrics(baseline_df)
        validator_with_log.log_quality_metrics(metrics, run_id="run001", source_name="test")

        log_files = list(Path(validator_with_log.log_dir).glob("*.json"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            data = json.load(f)
        assert data["run_id"] == "run001"
        assert data["source_name"] == "test"

    def test_no_log_dir_is_noop(self, validator, baseline_df):
        # Should log warning but not raise
        metrics = validator.get_quality_metrics(baseline_df)
        validator.log_quality_metrics(metrics, run_id="r1")  # no exception


# ---------------------------------------------------------------------------
# check_distribution_shift
# ---------------------------------------------------------------------------

class TestCheckDistributionShift:
    def test_no_shift_detected(self, validator, baseline_df):
        result = validator.check_distribution_shift(baseline_df, baseline_df.copy())
        assert result["has_significant_shift"] is False
        assert result["shifted_columns"] == []

    def test_significant_shift_detected(self, validator, baseline_df, shifted_df):
        result = validator.check_distribution_shift(baseline_df, shifted_df)
        assert result["has_significant_shift"] is True
        assert "price" in result["shifted_columns"]

    def test_psi_scores_populated(self, validator, baseline_df, shifted_df):
        result = validator.check_distribution_shift(baseline_df, shifted_df)
        assert "price" in result["psi_scores"]
        assert result["psi_scores"]["price"] > 0

    def test_missing_column_in_current_ignored(self, validator, baseline_df):
        current = baseline_df[["price"]].copy()  # drop 'volume'
        result = validator.check_distribution_shift(baseline_df, current)
        assert "volume" not in result["psi_scores"]
