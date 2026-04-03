"""Extended tests for loaders module."""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from src.data.loaders import DataLoader, ValidationResult


class TestDataLoaderExtended:
    def test_validation_result_to_dict(self):
        vr = ValidationResult(
            is_valid=False,
            errors=["e1"],
            warnings=["w1"],
            schema_violations={"col": "err"},
        )
        d = vr.to_dict()
        assert d == {
            "is_valid": False,
            "errors": ["e1"],
            "warnings": ["w1"],
            "schema_violations": {"col": "err"},
        }

    def test_init_log_dir_creation(self, tmp_path):
        dl = DataLoader(log_dir=str(tmp_path / "new_dir"))
        assert (tmp_path / "new_dir").exists()

    @patch("src.data.loaders.pd.read_parquet")
    def test_load_parquet_parse_error(self, mock_read, tmp_path):
        mock_read.side_effect = pa.ArrowInvalid("test")
        dl = DataLoader()
        f = tmp_path / "file.parquet"
        f.touch()
        with pytest.raises(pa.ArrowInvalid, match="Error parsing Parquet file"):
            dl.load_parquet(str(f))

    @patch("src.data.loaders.pd.read_parquet")
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
        with pytest.raises(
            TypeError, match="status\\['loaded_assets'\\] must be a list"
        ):
            dl.load_assets(str(tmp_path), [], status={"loaded_assets": {}})
        # failed_assets not a dict
        with pytest.raises(
            TypeError, match="status\\['failed_assets'\\] must be a dict"
        ):
            dl.load_assets(
                str(tmp_path), [], status={"loaded_assets": [], "failed_assets": []}
            )

    def test_load_from_config_invalid_config(self):
        dl = DataLoader()
        data = dl.load_from_config({"cat1": "not_a_dict"})
        assert data == {}

    def test_load_from_config_missing_keys(self):
        dl = DataLoader()
        data = dl.load_from_config({"cat1": {"path": "/tmp"}})  # Missing assets
        assert data == {}

    def test_dtype_compatible_variations(self):
        dl = DataLoader()
        assert dl._dtype_compatible("float32", "float")
        assert dl._dtype_compatible("int32", "int64")
        assert dl._dtype_compatible("datetime64[ns]", "datetime64")

    @patch("src.data.loaders.open", new_callable=mock_open)
    def test_log_validation_result(self, m_open, tmp_path):
        dl = DataLoader(log_dir=str(tmp_path))
        vr = ValidationResult(
            is_valid=False,  # Changed for semantic consistency
            errors=["err1"],
            warnings=["warn1"],
            schema_violations={"col1": "type_mismatch"},
        )
        run_id = "run_1"
        source_path = "/some/path"
        dl.log_validation_result(vr, run_id, source_path)

        # Verify open was called with expected path and mode
        expected_log_path = tmp_path / f"{run_id}_validation_report.json"
        m_open.assert_called_once_with(expected_log_path, "w")

        # Capture and verify written content
        handle = m_open()
        # Collect all chunks passed to write()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        log_content = json.loads(written_data)

        assert log_content["run_id"] == run_id
        assert log_content["source_path"] == source_path
        assert "timestamp" in log_content

        val_res = log_content["validation_result"]
        assert val_res["is_valid"] is False  # Changed to reflect is_valid=False
        assert val_res["errors"] == ["err1"]
        assert val_res["warnings"] == ["warn1"]
        assert val_res["schema_violations"] == {"col1": "type_mismatch"}
