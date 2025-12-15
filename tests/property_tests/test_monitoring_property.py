"""
Property tests for experiment tracking and monitoring.
"""

from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize
import pytest
from unittest.mock import MagicMock, patch
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.experiment_tracking import ExperimentTracker, ExperimentRun
from src.utils.monitoring import SystemMonitor, AlertConfig

# Strategy for experiment configurations
experiment_name_strategy = st.text(min_size=1, max_size=20)
metrics_strategy = st.dictionaries(st.text(min_size=1), st.floats(min_value=-1000, max_value=1000))
params_strategy = st.dictionaries(st.text(min_size=1), st.one_of(st.text(), st.integers(), st.floats()))

class ExperimentTrackingMachine(RuleBasedStateMachine):
    """
    Stateful property test for ExperimentTracker.
    Verifies that tracking state remains consistent through a sequence of operations.
    """
    
    def __init__(self):
        super().__init__()
        self.tmp_dir = Path(".hypothesis/experiment_tracking")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = str(self.tmp_dir / "runs")
        self.artifact_location = str(self.tmp_dir / "artifacts")
        self.tracker = None
        self.current_run_id = None
        
    @initialize(experiment_name=experiment_name_strategy)
    def init_tracker(self, experiment_name):
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            tracking_uri=self.tracking_uri,
            artifact_location=self.artifact_location,
            enable_monitoring=False 
        )
        assert self.tracker.experiment_name == experiment_name
        
    @rule(run_name=st.one_of(st.none(), st.text(min_size=1)))
    def start_run(self, run_name):
        # Can only start if no run is active (logic in tracker doesn't enforce this strictly, but effectively we want to track one at a time here)
        if self.tracker.current_run is None:
            run = self.tracker.start_run(run_name=run_name)
            assert run.status == "running"
            self.current_run_id = run.run_id
            
    @rule(metrics=metrics_strategy)
    def log_metrics(self, metrics):
        if self.tracker.current_run:
            self.tracker.log_metrics(metrics)
            # Verify metrics are in memory
            for k, v in metrics.items():
                assert self.tracker.current_run.metrics[k] == v
                
    @rule(params=params_strategy)
    def log_params(self, params):
        if self.tracker.current_run:
            self.tracker.log_params(params)
            # Verify params are in memory
            for k, v in params.items():
                # Handle NaN comparison
                import math
                if isinstance(v, float) and math.isnan(v):
                    assert math.isnan(self.tracker.current_run.parameters[k])
                else:
                    assert self.tracker.current_run.parameters[k] == params[k]

    @rule()
    def end_run(self):
        if self.tracker.current_run:
            run_id = self.tracker.current_run.run_id
            self.tracker.end_run()
            assert self.tracker.current_run is None
            
            # Verify persistence
            saved_run = self.tracker.get_run(run_id)
            assert saved_run is not None
            assert saved_run.status == "completed"
            
    def teardown(self):
        # Cleanup temp directory
        import shutil
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

TestExperimentTracker = ExperimentTrackingMachine.TestCase


def test_monitoring_alerts():
    """Test functionality of alerting system."""
    tracker = ExperimentTracker("alert_test", enable_monitoring=True)
    
    # Mock system monitor to avoid threading in simple unit test
    tracker._monitor = MagicMock()
    tracker._alert_manager = MagicMock()
    
    # Configure alert
    tracker.add_alert("accuracy", 0.5, operator="<")
    
    # Start run to enable alerting check in log_metrics
    tracker.start_run()
    
    # Log metric that should trigger
    tracker.log_metrics({"accuracy": 0.4})
    
    # Verify check_metric was called
    tracker._alert_manager.check_metric.assert_any_call("accuracy", 0.4)
    
    # Verify logic of AlertManager itself
    from src.utils.monitoring import AlertManager, AlertConfig
    manager = AlertManager()
    manager.add_alert(
        "accuracy_alert", 
        AlertConfig("accuracy", 0.5, "<", "Alert: low accuracy", cooldown_seconds=0)
    )
    
    msg = manager.check_metric("accuracy", 0.4)
    assert msg == "Alert: low accuracy"
    
    msg_ok = manager.check_metric("accuracy", 0.6)
    assert msg_ok is None

@patch("psutil.cpu_percent")
@patch("psutil.virtual_memory")
def test_system_monitor_integration(mock_mem, mock_cpu):
    """Test system monitor integration into tracker."""
    # Setup mocks
    mock_cpu.return_value = 50.0
    mock_mem_obj = MagicMock()
    mock_mem_obj.percent = 60.0
    mock_mem_obj.used = 1024**3 * 8 # 8 GB
    mock_mem.return_value = mock_mem_obj
    
    tracker = ExperimentTracker("monitor_test", enable_monitoring=True)
    
    # Reduce interval for fast test
    tracker._monitor.interval_seconds = 0.1
    
    run = tracker.start_run()
    time.sleep(0.3) # Let monitor run a few loops
    tracker.end_run()
    
    # Check if system metrics were logged
    assert "system_cpu_percent" in run.metrics
    assert "system_memory_percent" in run.metrics
    assert run.metrics["system_cpu_percent"] == 50.0

def test_cleanup_policy():
    """Test experiment cleanup policy."""
    tmp_dir = Path(".hypothesis/cleanup_test")
    import shutil
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    tracker = ExperimentTracker(
        "cleanup_test", 
        tracking_uri=str(tmp_dir),
        enable_monitoring=False
    )
    
    # Create old run
    old_run = tracker.start_run()
    tracker.end_run()
    
    # Manually backdate the file
    run_file = tmp_dir / f"{old_run.run_id}.json"
    with open(run_file, "r") as f:
        data = json.load(f)
    
    old_date = datetime.now() - timedelta(days=100)
    data["start_time"] = old_date.isoformat()
    
    with open(run_file, "w") as f:
        json.dump(data, f)
        
    # Create new run
    tracker.start_run()
    tracker.end_run()
    
    # Convert 'start_time' string back to datetime for comparison in test
    # The actual cleanup uses file content
    
    deleted = tracker.cleanup_experiments(older_than_days=90)
    assert deleted == 1
    
    remaining_files = list(tmp_dir.glob("*.json"))
    assert len(remaining_files) == 1
    
    import shutil
    shutil.rmtree(tmp_dir)
