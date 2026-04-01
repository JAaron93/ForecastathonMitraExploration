"""Unit tests for ExperimentTracker local file-based tracking."""
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.experiment_tracking import ExperimentRun, ExperimentTracker


# ---------------------------------------------------------------------------
# Helper: build a tracker backed by a temp directory (no MLflow)
# ---------------------------------------------------------------------------

@pytest.fixture
def local_tracker(tmp_path):
    """An ExperimentTracker in local-file mode with monitoring disabled."""
    tracking_uri = str(tmp_path / "runs")
    artifact_location = str(tmp_path / "artifacts")
    with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
        tracker = ExperimentTracker(
            experiment_name="test_experiment",
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
            enable_monitoring=False,
        )
    return tracker


# ---------------------------------------------------------------------------
# ExperimentRun dataclass
# ---------------------------------------------------------------------------

class TestExperimentRun:
    def test_to_dict_roundtrip(self):
        run = ExperimentRun(
            run_id="abc123",
            experiment_name="exp",
            parameters={"lr": 0.01},
            metrics={"acc": 0.95},
            status="completed",
        )
        d = run.to_dict()
        restored = ExperimentRun.from_dict(d)
        assert restored.run_id == run.run_id
        assert restored.parameters == run.parameters
        assert restored.metrics == run.metrics
        assert restored.status == run.status

    def test_to_dict_with_end_time(self):
        run = ExperimentRun(run_id="x", experiment_name="e")
        run.end_time = datetime(2024, 1, 1, 12, 0, 0)
        d = run.to_dict()
        assert d["end_time"] == "2024-01-01T12:00:00"

    def test_from_dict_with_end_time(self):
        d = {
            "run_id": "r1", "experiment_name": "e",
            "parameters": {}, "metrics": {}, "artifacts": [],
            "status": "completed",
            "start_time": "2024-01-01T10:00:00",
            "end_time": "2024-01-01T11:00:00",
            "tags": {},
        }
        run = ExperimentRun.from_dict(d)
        assert run.end_time == datetime(2024, 1, 1, 11, 0, 0)


# ---------------------------------------------------------------------------
# Local directory structure & persistence
# ---------------------------------------------------------------------------

class TestLocalTracker:
    def test_tracking_dir_created_on_init(self, local_tracker):
        assert local_tracker.tracking_dir.exists()

    def test_artifact_location_created_on_init(self, local_tracker):
        artifact_dir = Path(local_tracker.artifact_location)
        assert artifact_dir.exists()

    def test_start_run_sets_current_run(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            run = local_tracker.start_run(run_name="test-run")
        assert local_tracker.current_run is not None
        assert run.status == "running"

    def test_end_run_saves_json(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            run = local_tracker.start_run()
            local_tracker.end_run()

        saved_file = local_tracker.tracking_dir / f"{run.run_id}.json"
        assert saved_file.exists()
        with open(saved_file) as f:
            data = json.load(f)
        assert data["status"] == "completed"

    def test_end_run_clears_current_run(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            local_tracker.start_run()
            local_tracker.end_run()
        assert local_tracker.current_run is None

    def test_end_run_no_active_run_is_noop(self, local_tracker):
        # Should not raise; just log a warning
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            local_tracker.end_run()

    def test_log_metrics_updates_current_run(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            local_tracker.start_run()
            local_tracker.log_metrics({"loss": 0.3, "accuracy": 0.85})
        assert local_tracker.current_run.metrics["loss"] == pytest.approx(0.3)
        assert local_tracker.current_run.metrics["accuracy"] == pytest.approx(0.85)

    def test_log_params_updates_current_run(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            local_tracker.start_run()
            local_tracker.log_params({"lr": 0.001, "batch_size": 32})
        assert local_tracker.current_run.parameters["lr"] == 0.001

    def test_set_tag_updates_current_run(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            local_tracker.start_run()
            local_tracker.set_tag("env", "test")
        assert local_tracker.current_run.tags["env"] == "test"

    def test_log_metrics_raises_without_active_run(self, local_tracker):
        with pytest.raises(RuntimeError, match="No active run"):
            local_tracker.log_metrics({"loss": 0.1})

    def test_log_params_raises_without_active_run(self, local_tracker):
        with pytest.raises(RuntimeError, match="No active run"):
            local_tracker.log_params({"lr": 0.1})

    def test_set_tag_raises_without_active_run(self, local_tracker):
        with pytest.raises(RuntimeError, match="No active run"):
            local_tracker.set_tag("k", "v")


# ---------------------------------------------------------------------------
# get_run / list_runs
# ---------------------------------------------------------------------------

class TestRunRetrieval:
    def test_get_run_returns_correct_run(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            run = local_tracker.start_run()
            local_tracker.log_metrics({"rmse": 0.12})
            local_tracker.end_run()
            restored = local_tracker.get_run(run.run_id)

        assert restored is not None
        assert restored.run_id == run.run_id
        assert restored.metrics["rmse"] == pytest.approx(0.12)

    def test_get_run_returns_none_for_unknown_id(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            result = local_tracker.get_run("does-not-exist")
        assert result is None

    def test_list_runs_returns_completed_runs(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            local_tracker.start_run(run_name="r1")
            local_tracker.end_run()
            local_tracker.start_run(run_name="r2")
            local_tracker.end_run()
            runs = local_tracker.list_runs()

        assert len(runs) >= 2
        assert all(isinstance(r, ExperimentRun) for r in runs)

    def test_list_runs_respects_max_results(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            for _ in range(5):
                local_tracker.start_run()
                local_tracker.end_run()
            runs = local_tracker.list_runs(max_results=3)
        assert len(runs) <= 3


# ---------------------------------------------------------------------------
# cleanup_experiments
# ---------------------------------------------------------------------------

class TestCleanupExperiments:
    def test_cleanup_removes_old_run(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            old_run = local_tracker.start_run()
            local_tracker.end_run()

            # Backdate the run file
            run_file = local_tracker.tracking_dir / f"{old_run.run_id}.json"
            with open(run_file) as f:
                data = json.load(f)
            data["start_time"] = (datetime.now() - timedelta(days=100)).isoformat()
            with open(run_file, "w") as f:
                json.dump(data, f)

            deleted = local_tracker.cleanup_experiments(older_than_days=90)

        assert deleted == 1
        assert not run_file.exists()

    def test_cleanup_keeps_recent_runs(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            local_tracker.start_run()
            local_tracker.end_run()
            deleted = local_tracker.cleanup_experiments(older_than_days=90)
        assert deleted == 0


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_completes_on_success(self, local_tracker):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            with local_tracker:
                local_tracker.log_metrics({"loss": 0.5})

        assert local_tracker.current_run is None

    def test_context_manager_marks_failed_on_exception(self, local_tracker):
        run_id = None
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            try:
                with local_tracker:
                    run_id = local_tracker.current_run.run_id
                    raise RuntimeError("Simulated failure")
            except RuntimeError:
                pass

            if run_id:
                saved = local_tracker.get_run(run_id)
                assert saved is not None
                assert saved.status == "failed"


# ---------------------------------------------------------------------------
# log_artifact (local copy)
# ---------------------------------------------------------------------------

class TestLogArtifact:
    def test_log_artifact_copies_to_local_dir(self, local_tracker, tmp_path):
        src_file = tmp_path / "model.pkl"
        src_file.write_bytes(b"fake_model_bytes")

        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            local_tracker.start_run()
            local_tracker.log_artifact(str(src_file))
            run_id = local_tracker.current_run.run_id
            local_tracker.end_run()

        dest = Path(local_tracker.artifact_location) / run_id / "model.pkl"
        assert dest.exists()

    def test_log_artifact_raises_without_active_run(self, local_tracker, tmp_path):
        src_file = tmp_path / "file.txt"
        src_file.write_text("hello")
        with pytest.raises(RuntimeError, match="No active run"):
            local_tracker.log_artifact(str(src_file))
