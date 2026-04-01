"""Extended tests for loaders module."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.data.loaders import DataLoader, ValidationResult

class TestDataLoaderExtended:
    def test_validation_result_to_dict(self):
        vr = ValidationResult(is_valid=False, errors=["e1"], warnings=["w1"], schema_violations={"col": "err"})
        d = vr.to_dict()
        assert d == {
            "is_valid": False, "errors": ["e1"], "warnings": ["w1"], "schema_violations": {"col": "err"}
        }
        
    def test_init_log_dir_creation(self, tmp_path):
        dl = DataLoader(log_dir=str(tmp_path / "new_dir"))
        assert (tmp_path / "new_dir").exists()
        
    @patch("pandas.read_parquet")
    def test_load_parquet_parse_error(self, mock_read, tmp_path):
        import pandas.errors
        mock_read.side_effect = pandas.errors.ParserError("test")
        dl = DataLoader()
        f = tmp_path / "file.parquet"
        f.touch()
        with pytest.raises(pandas.errors.ParserError, match="Error parsing Parquet file"):
            dl.load_parquet(str(f))

    @patch("pandas.read_parquet")
    def test_load_parquet_os_error(self, mock_read, tmp_path):
        mock_read.side_effect = OSError("test")
        dl = DataLoader()
        f = tmp_path / "file.parquet"
        f.touch()
        with pytest.raises(OSError, match="OS error reading Parquet file"):
            dl.load_parquet(str(f))
            
    def test_load_assets_base_dir_not_exists(self):
        dl = DataLoader()
        with pytest.raises(FileNotFoundError, match="Base directory not found"):
            dl.load_assets("/non/existent/dir", [])
            
    def test_load_assets_status_type_checks(self, tmp_path):
        dl = DataLoader()
        # Not a dict
        with pytest.raises(TypeError, match="status must be a dict"):
            dl.load_assets(str(tmp_path), [], status="not_dict")
        # loaded_assets not a list
        with pytest.raises(TypeError, match="status\\['loaded_assets'\\] must be a list"):
            dl.load_assets(str(tmp_path), [], status={"loaded_assets": {}})
        # failed_assets not a dict
        with pytest.raises(TypeError, match="status\\['failed_assets'\\] must be a dict"):
            dl.load_assets(str(tmp_path), [], status={"loaded_assets": [], "failed_assets": []})

    def test_load_from_config_invalid_config(self):
        dl = DataLoader()
        data = dl.load_from_config({"cat1": "not_a_dict"})
        assert data == {}

    def test_load_from_config_missing_keys(self):
        dl = DataLoader()
        data = dl.load_from_config({"cat1": {"path": "/tmp"}}) # Missing assets
        assert data == {}

    def test_dtype_compatible_variations(self):
        dl = DataLoader()
        assert dl._dtype_compatible("float32", "float")
        assert dl._dtype_compatible("int32", "int64")
        assert dl._dtype_compatible("datetime64[ns]", "datetime64")

    @patch("builtins.open")
    def test_log_validation_result(self, mock_open, tmp_path):
        dl = DataLoader(log_dir=str(tmp_path))
        vr = ValidationResult(is_valid=True, errors=[], warnings=[], schema_violations={})
        dl.log_validation_result(vr, "run_1", "/some/path")
        mock_open.assert_called_once()
