import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.profilers import DataProfiler


@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": ["a", "b", "c", "d", "e"],
        "C": [1.1, 2.2, 3.3, 4.4, 5.5],
    })


def test_profiler_initialization(tmp_path):
    """Test DataProfiler initialization and directory creation."""
    output_dir = tmp_path / "reports"
    profiler = DataProfiler(output_dir=str(output_dir))
    
    assert profiler.output_dir == output_dir
    assert output_dir.exists()
    assert output_dir.is_dir()


@patch("src.data.profilers._get_profile_report_class")
def test_generate_profile_minimal(mock_get_class, sample_df, tmp_path):
    """Test generating a minimal profile report."""
    mock_pr_class = MagicMock()
    mock_report_instance = MagicMock()
    mock_pr_class.return_value = mock_report_instance
    mock_get_class.return_value = mock_pr_class
    
    profiler = DataProfiler(output_dir=str(tmp_path))
    
    report = profiler.generate_profile(
        df=sample_df,
        title="Test Minimal Report",
        minimal=True
    )
    
    assert report == mock_report_instance
    
    # Check that PR class was called with correct minimal config
    mock_pr_class.assert_called_once()
    args, kwargs = mock_pr_class.call_args
    assert args[0] is sample_df
    assert kwargs["title"] == "Test Minimal Report"
    assert kwargs["explorative"] is False
    assert kwargs["interactions"]["continuous"] is False


@patch("src.data.profilers._get_profile_report_class")
def test_generate_profile_full(mock_get_class, sample_df, tmp_path):
    """Test generating a full profile report with time series config."""
    mock_pr_class = MagicMock()
    mock_report_instance = MagicMock()
    mock_pr_class.return_value = mock_report_instance
    mock_get_class.return_value = mock_pr_class
    
    profiler = DataProfiler(output_dir=str(tmp_path))
    
    ts_config = {"tsmode": True, "sortby": "A"}
    report = profiler.generate_profile(
        df=sample_df,
        title="Test Full Report",
        minimal=False,
        time_series_config=ts_config
    )
    
    assert report == mock_report_instance
    
    # Check that PR class was called with correct full config
    mock_pr_class.assert_called_once()
    args, kwargs = mock_pr_class.call_args
    assert args[0] is sample_df
    assert kwargs["title"] == "Test Full Report"
    assert kwargs["explorative"] is True
    assert kwargs["interactions"]["continuous"] is True
    assert kwargs["tsmode"] is True
    assert kwargs["sortby"] == "A"


def test_save_report(tmp_path):
    """Test saving a profile report to file."""
    profiler = DataProfiler(output_dir=str(tmp_path))
    mock_report = MagicMock()
    
    # Test auto-generated path
    path1 = profiler.save_report(report=mock_report, dataset_identifier="test_data")
    assert "test_data_" in path1
    assert path1.endswith("_profile.html")
    mock_report.to_file.assert_called_once_with(path1)
    
    # Test explicit path
    explicit_path = str(tmp_path / "custom_report.html")
    mock_report.reset_mock()
    path2 = profiler.save_report(report=mock_report, output_path=explicit_path)
    assert path2 == explicit_path
    mock_report.to_file.assert_called_once_with(explicit_path)


def test_get_data_quality_summary(tmp_path):
    """Test extracting data quality summary from a mock report."""
    profiler = DataProfiler(output_dir=str(tmp_path))
    
    mock_report = MagicMock()
    
    # Setup mock description dictionary
    mock_description = MagicMock()
    mock_description.table = {
        "n": 100,
        "n_var": 5,
        "n_cells_missing": 10,
        "p_cells_missing": 0.02,
        "memory_size": 1024,
        "types": {"Numeric": 3, "Categorical": 2}
    }
    
    # Setup mock duplicates
    mock_description.duplicates = {"n_duplicates": 2, "p_duplicates": 0.02}
    
    # Setup mock alerts
    alert1 = MagicMock()
    alert1.column_name = "colA"
    alert1.alert_type = "High Correlation"
    alert1.__str__.return_value = "Alert 1"
    
    mock_description.alerts = [
        alert1,
        {"message": "Alert 2"},
        "Alert 3"
    ]
    
    mock_report.get_description.return_value = mock_description
    
    summary = profiler.get_data_quality_summary(mock_report)
    
    assert summary["row_count"] == 100
    assert summary["column_count"] == 5
    assert summary["missing_cells"] == 10
    assert summary["missing_cells_pct"] == 2.0
    assert summary["memory_size"] == 1024
    assert summary["duplicate_rows"] == 2
    assert summary["duplicate_rows_pct"] == 2.0
    assert summary["variable_types"] == {"Numeric": 3, "Categorical": 2}
    assert summary["alerts_count"] == 3
    assert len(summary["alerts"]) == 3
    assert summary["alerts"][0] == {"column": "colA", "type": "High Correlation", "message": "Alert 1"}
    assert summary["alerts"][1] == {"message": "Alert 2"}
    assert summary["alerts"][2] == {"message": "Alert 3"}


def test_compare_profiles(tmp_path):
    """Test comparing two profile reports."""
    profiler = DataProfiler(output_dir=str(tmp_path))
    
    report1 = MagicMock()
    report2 = MagicMock()
    comparison_mock = MagicMock()
    report1.compare.return_value = comparison_mock
    
    comp_report = profiler.compare_profiles(
        report1=report1,
        report2=report2,
        title="Custom Comparison"
    )
    
    report1.compare.assert_called_once_with(report2)
    assert comp_report == comparison_mock
    assert comparison_mock.config.title == "Custom Comparison"


@patch("src.data.profilers.DataProfiler.get_data_quality_summary")
@patch("src.data.profilers.DataProfiler.save_report")
@patch("src.data.profilers.DataProfiler.generate_profile")
def test_generate_and_save_profile(
    mock_generate, mock_save, mock_summary, sample_df, tmp_path
):
    """Test convenience method for generating and saving profile."""
    profiler = DataProfiler(output_dir=str(tmp_path))
    
    mock_report = MagicMock()
    mock_generate.return_value = mock_report
    mock_save.return_value = "/mock/path/report.html"
    mock_summary.return_value = {"row_count": 5}
    
    result = profiler.generate_and_save_profile(
        df=sample_df,
        dataset_identifier="test_data",
        title="My Custom Title",
        minimal=True,
        time_series_config={"tsmode": True}
    )
    
    mock_generate.assert_called_once_with(
        df=sample_df,
        title="My Custom Title",
        minimal=True,
        time_series_config={"tsmode": True}
    )
    
    mock_save.assert_called_once_with(
        report=mock_report,
        dataset_identifier="test_data"
    )
    
    mock_summary.assert_called_once_with(mock_report)
    
    assert result["report_path"] == "/mock/path/report.html"
    assert result["quality_summary"] == {"row_count": 5}

