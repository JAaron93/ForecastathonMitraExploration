"""Unit tests for src/utils/logging_config.py."""
import json
import logging
import sys

from src.utils.logging_config import JSONFormatter


class TestJSONFormatter:
    def _make_record(
        self, msg: str, level: int = logging.INFO, **extra
    ) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test.logger",
            level=level,
            pathname="test_logging_init.py",
            lineno=42,
            msg=msg,
            args=(),
            exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_format_returns_valid_json(self):
        formatter = JSONFormatter()
        record = self._make_record("hello world")
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello world"

    def test_format_contains_required_keys(self):
        formatter = JSONFormatter()
        record = self._make_record("test")
        parsed = json.loads(formatter.format(record))
        required_keys = (
            "timestamp", "level", "logger", "message", "module", "funcName", "lineNo"
        )
        for key in required_keys:
            assert key in parsed, f"Missing key: {key}"

    def test_format_level_name(self):
        formatter = JSONFormatter()
        record = self._make_record("warn msg", level=logging.WARNING)
        parsed = json.loads(formatter.format(record))
        assert parsed["level"] == "WARNING"

    def test_format_with_exception_info(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="f.py", lineno=1,
            msg="error occurred", args=(), exc_info=exc_info,
        )
        parsed = json.loads(formatter.format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_format_with_extra_props(self):
        formatter = JSONFormatter()
        record = self._make_record(
            "extra test", props={"request_id": "abc-123"}
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["request_id"] == "abc-123"

    def test_format_without_props_attribute(self):
        """Should not raise even when record has no 'props' attribute."""
        formatter = JSONFormatter()
        record = self._make_record("no props")
        # Ensure the record has no 'props' attr
        assert not hasattr(record, "props")
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "no props"
