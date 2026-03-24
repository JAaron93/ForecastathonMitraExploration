import math
import shutil
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize

from src.utils.experiment_tracking import ExperimentTracker
from src.utils.monitoring import AlertConfig, AlertManager

# Strategy for experiment configurations
experiment_name_strategy = st.text(min_size=1, max_size=20)
metrics_strategy = st.dictionaries(
    st.text(min_size=1), st.floats(min_value=-1000, max_value=1000)
)
params_strategy = st.dictionaries(
    st.text(min_size=1), st.one_of(st.text(), st.integers(), st.floats())
)


class ExperimentTrackingMachine(RuleBasedStateMachine):
    """
    Stateful property test for ExperimentTracker.
    Verifies that tracking state remains consistent through a sequence of operations.
    """

    def __init__(self):
        super().__init__()
        self.tmp_dir = Path(".hypothesis/experiment_tracking")
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        # Use a sub-directory for tracking_uri to avoid MLflow trash issues
        # Explicitly disable MLflow to use local file-based tracking
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            self.tracking_uri = str(self.tmp_dir / "local_runs")
            Path(self.tracking_uri).mkdir(parents=True, exist_ok=True)
            self.artifact_location = str(self.tmp_dir / "artifacts")
            Path(self.artifact_location).mkdir(parents=True, exist_ok=True)
            self.tracker = None
            self.current_run_id = None

    @initialize(experiment_name=experiment_name_strategy)
    def init_tracker(self, experiment_name):
        with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
            self.tracker = ExperimentTracker(
                experiment_name=experiment_name,
                tracking_uri=self.tracking_uri,
                artifact_location=self.artifact_location,
                enable_monitoring=False,
            )
        assert self.tracker.experiment_name == experiment_name

    @rule(run_name=st.one_of(st.none(), st.text(min_size=1)))
    def start_run(self, run_name):
        if self.tracker.current_run is None:
            with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
                run = self.tracker.start_run(run_name=run_name)
            assert run.status == "running"
            self.current_run_id = run.run_id

    @rule(metrics=metrics_strategy)
    def log_metrics(self, metrics):
        if self.tracker.current_run:
            with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
                self.tracker.log_metrics(metrics)
            for k, v in metrics.items():
                assert self.tracker.current_run.metrics[k] == v

    @rule(params=params_strategy)
    def log_params(self, params):
        if self.tracker.current_run:
            with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
                self.tracker.log_params(params)
            for k, v in params.items():
                if isinstance(v, float) and math.isnan(v):
                    assert math.isnan(self.tracker.current_run.parameters[k])
                else:
                    assert self.tracker.current_run.parameters[k] == params[k]

    @rule()
    def end_run(self):
        if self.tracker.current_run:
            run_id = self.tracker.current_run.run_id
            with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
                self.tracker.end_run()
            assert self.tracker.current_run is None

            with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
                saved_run = self.tracker.get_run(run_id)
            assert saved_run is not None
            assert saved_run.status == "completed"

    def teardown(self):
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)


TestExperimentTracker = ExperimentTrackingMachine.TestCase


def test_monitoring_alerts():
    """Test functionality of alerting system."""
    tracker = ExperimentTracker("alert_test", enable_monitoring=True)

    tracker._monitor = MagicMock()
    tracker._alert_manager = MagicMock()

    tracker.add_alert("accuracy", 0.5, operator="<")
    tracker.start_run()
    tracker.log_metrics({"accuracy": 0.4})

    tracker._alert_manager.check_metric.assert_any_call("accuracy", 0.4)

    manager = AlertManager()
    manager.add_alert(
        "accuracy_alert",
        AlertConfig("accuracy", 0.5, "<", "Alert: low accuracy", cooldown_seconds=0),
    )

    msg = manager.check_metric("accuracy", 0.4)
    assert msg == "Alert: low accuracy"

    msg_ok = manager.check_metric("accuracy", 0.6)
    assert msg_ok is None


@patch("psutil.cpu_percent")
@patch("psutil.virtual_memory")
def test_system_monitor_integration(mock_mem, mock_cpu):
    """Test system monitor integration into tracker."""
    mock_cpu.return_value = 50.0
    mock_mem_obj = MagicMock()
    mock_mem_obj.percent = 60.0
    mock_mem_obj.used = 1024**3 * 8  # 8 GB
    mock_mem.return_value = mock_mem_obj

    tracker = ExperimentTracker("monitor_test", enable_monitoring=True)
    tracker._monitor.interval_seconds = 0.1

    run = tracker.start_run()
    time.sleep(0.3)
    tracker.end_run()

    assert "system_cpu_percent" in run.metrics
    assert "system_memory_percent" in run.metrics
    assert run.metrics["system_cpu_percent"] == 50.0


def test_cleanup_policy():
    """Test experiment cleanup policy."""
    tmp_dir = Path(".hypothesis/cleanup_test")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with patch("src.utils.experiment_tracking.MLFLOW_AVAILABLE", False):
        tracker = ExperimentTracker(
            "cleanup_test", tracking_uri=str(tmp_dir), enable_monitoring=False
        )

        old_run = tracker.start_run()
        tracker.end_run()

        run_file = tmp_dir / f"{old_run.run_id}.json"
        with open(run_file, "r") as f:
            data = json.load(f)

        old_date = datetime.now() - timedelta(days=100)
        data["start_time"] = old_date.isoformat()

        with open(run_file, "w") as f:
            json.dump(data, f)

        tracker.start_run()
        tracker.end_run()

        deleted = tracker.cleanup_experiments(older_than_days=90)
    assert deleted == 1

    remaining_files = list(tmp_dir.glob("*.json"))
    assert len(remaining_files) == 1

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
