"""Experiment tracking integration with MLflow."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging
import os

logger = logging.getLogger(__name__)

# Try to import mlflow, but make it optional
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Experiment tracking will be limited.")


@dataclass
class ExperimentRun:
    """Container for experiment run information."""
    run_id: str
    experiment_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    status: str = "running"
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRun":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            experiment_name=data["experiment_name"],
            parameters=data.get("parameters", {}),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", []),
            status=data.get("status", "unknown"),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            tags=data.get("tags", {}),
        )


class ExperimentTracker:
    """
    Experiment tracking with MLflow integration.
    
    Provides a unified interface for tracking experiments, logging parameters,
    metrics, and artifacts. Falls back to local file-based tracking if MLflow
    is not available.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        enable_monitoring: bool = True,
    ):
        """
        Initialize ExperimentTracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (optional)
            artifact_location: Path for storing artifacts
            enable_monitoring: Whether to enable background system monitoring
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "experiments/mlflow_runs"
        self.artifact_location = artifact_location or "experiments/artifacts"
        self._current_run: Optional[ExperimentRun] = None
        self._mlflow_run = None
        
        # Monitoring
        self.enable_monitoring = enable_monitoring
        if self.enable_monitoring:
            from .monitoring import SystemMonitor, AlertManager, AlertConfig
            self._monitor = SystemMonitor(interval_seconds=10.0)
            self._alert_manager = self._monitor._alert_manager
        else:
            self._monitor = None
            self._alert_manager = None
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.tracking_uri)
            try:
                self._experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=self.artifact_location,
                )
            except mlflow.exceptions.MlflowException:
                # Experiment already exists
                experiment = mlflow.get_experiment_by_name(experiment_name)
                self._experiment_id = experiment.experiment_id if experiment else None
        else:
            self._experiment_id = None
            # Create local tracking directory
            Path(self.tracking_uri).mkdir(parents=True, exist_ok=True)
            Path(self.artifact_location).mkdir(parents=True, exist_ok=True)

    def add_alert(self, metric_name: str, threshold: float, operator: str = '>') -> None:
        """Add a performance alert."""
        if self._alert_manager:
            from .monitoring import AlertConfig
            self._alert_manager.add_alert(
                f"{metric_name}_alert",
                AlertConfig(
                    metric_name=metric_name,
                    threshold=threshold,
                    operator=operator,
                    message_template=f"Performance alert: {{metric}} {{value:.4f}} {operator} {{threshold}}"
                )
            )

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ExperimentRun:
        """
        Start a new experiment run.

        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run

        Returns:
            ExperimentRun instance
        """
        tags = tags or {}
        
        if MLFLOW_AVAILABLE:
            self._mlflow_run = mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=run_name,
                tags=tags,
            )
            run_id = self._mlflow_run.info.run_id
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        self._current_run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.experiment_name,
            tags=tags,
            status="running",
        )
        
        # Start monitoring
        if self._monitor:
            def monitor_callback(usage):
                # Log system metrics
                self.log_metrics({
                    "system_cpu_percent": usage.cpu_percent,
                    "system_memory_percent": usage.memory_percent,
                    "system_memory_gb": usage.memory_used_gb
                })
                
            self._monitor.start(callback=monitor_callback)

        logger.info(f"Started run {run_id} for experiment {self.experiment_name}")
        return self._current_run

    def end_run(self, status: str = "completed") -> None:
        """
        End the current run.

        Args:
            status: Final status of the run
        """
        if self._current_run is None:
            logger.warning("No active run to end")
            return

        # Stop monitoring
        if self._monitor:
            self._monitor.stop()
            # Log max resource usage
            history = self._monitor.get_history()
            if history:
                max_cpu = max(h.cpu_percent for h in history)
                max_mem = max(h.memory_percent for h in history)
                self.log_metrics({
                    "max_cpu_percent": max_cpu,
                    "max_memory_percent": max_mem
                })

        self._current_run.status = status
        self._current_run.end_time = datetime.now()

        if MLFLOW_AVAILABLE and self._mlflow_run:
            mlflow.end_run(status="FINISHED" if status == "completed" else "FAILED")
            self._mlflow_run = None
        else:
            # Save run to local file
            self._save_run_locally()

        logger.info(f"Ended run {self._current_run.run_id} with status {status}")
        self._current_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters for the current run.

        Args:
            params: Dictionary of parameter names to values
        """
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self._current_run.parameters.update(params)

        if MLFLOW_AVAILABLE:
            # MLflow requires string values for params
            mlflow_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(mlflow_params)

        logger.debug(f"Logged parameters: {list(params.keys())}")

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a single parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        self.log_params({key: value})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics for the current run.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for tracking over time
        """
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self._current_run.metrics.update(metrics)
        
        # Check alerts
        if self._alert_manager:
            for k, v in metrics.items():
                self._alert_manager.check_metric(k, v)

        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics, step=step)

        logger.debug(f"Logged metrics: {list(metrics.keys())}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact file.

        Args:
            local_path: Path to the local file
            artifact_path: Optional subdirectory in artifact storage
        """
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self._current_run.artifacts.append(local_path)

        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(local_path, artifact_path)
        else:
            # Copy to local artifact directory
            import shutil
            dest_dir = Path(self.artifact_location) / self._current_run.run_id
            if artifact_path:
                dest_dir = dest_dir / artifact_path
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest_dir)

        logger.debug(f"Logged artifact: {local_path}")

    def log_model_artifact(
        self,
        model_artifact: Any,
        artifact_name: str = "model",
    ) -> None:
        """
        Log a model artifact with metadata.

        Args:
            model_artifact: ModelArtifact instance
            artifact_name: Name for the artifact
        """
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        # Log model metadata as params
        if hasattr(model_artifact, "hyperparameters"):
            self.log_params({
                f"{artifact_name}_hyperparams": json.dumps(model_artifact.hyperparameters)
            })

        # Log training metrics
        if hasattr(model_artifact, "training_metrics"):
            prefixed_metrics = {
                f"{artifact_name}_train_{k}": v 
                for k, v in model_artifact.training_metrics.items()
            }
            self.log_metrics(prefixed_metrics)

        # Log validation metrics
        if hasattr(model_artifact, "validation_metrics"):
            prefixed_metrics = {
                f"{artifact_name}_val_{k}": v 
                for k, v in model_artifact.validation_metrics.items()
            }
            self.log_metrics(prefixed_metrics)

        logger.info(f"Logged model artifact: {artifact_name}")

    def set_tag(self, key: str, value: str) -> None:
        """
        Set a tag for the current run.

        Args:
            key: Tag name
            value: Tag value
        """
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self._current_run.tags[key] = value

        if MLFLOW_AVAILABLE:
            mlflow.set_tag(key, value)

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """
        Get a run by ID.

        Args:
            run_id: Run identifier

        Returns:
            ExperimentRun if found, None otherwise
        """
        if MLFLOW_AVAILABLE:
            try:
                client = MlflowClient()
                run = client.get_run(run_id)
                return ExperimentRun(
                    run_id=run.info.run_id,
                    experiment_name=self.experiment_name,
                    parameters=run.data.params,
                    metrics=run.data.metrics,
                    status=run.info.status,
                    start_time=datetime.fromtimestamp(run.info.start_time / 1000),
                    end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                    tags=run.data.tags,
                )
            except Exception as e:
                logger.error(f"Failed to get run {run_id}: {e}")
                return None
        else:
            # Load from local file
            run_file = Path(self.tracking_uri) / f"{run_id}.json"
            if run_file.exists():
                with open(run_file, "r") as f:
                    return ExperimentRun.from_dict(json.load(f))
            return None

    def list_runs(
        self,
        max_results: int = 100,
        filter_string: Optional[str] = None,
    ) -> List[ExperimentRun]:
        """
        List runs for the experiment.

        Args:
            max_results: Maximum number of runs to return
            filter_string: Optional filter string (MLflow syntax)

        Returns:
            List of ExperimentRun instances
        """
        runs = []

        if MLFLOW_AVAILABLE and self._experiment_id:
            try:
                client = MlflowClient()
                mlflow_runs = client.search_runs(
                    experiment_ids=[self._experiment_id],
                    max_results=max_results,
                    filter_string=filter_string,
                )
                for run in mlflow_runs:
                    runs.append(ExperimentRun(
                        run_id=run.info.run_id,
                        experiment_name=self.experiment_name,
                        parameters=run.data.params,
                        metrics=run.data.metrics,
                        status=run.info.status,
                        start_time=datetime.fromtimestamp(run.info.start_time / 1000),
                        end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                        tags=run.data.tags,
                    ))
            except Exception as e:
                logger.error(f"Failed to list runs: {e}")
        else:
            # Load from local files
            tracking_dir = Path(self.tracking_uri)
            for run_file in tracking_dir.glob("*.json"):
                with open(run_file, "r") as f:
                    runs.append(ExperimentRun.from_dict(json.load(f)))
                if len(runs) >= max_results:
                    break

        return runs

    def cleanup_experiments(self, older_than_days: int = 90) -> int:
        """
        Clean up old experiment runs.
        
        Args:
            older_than_days: Delete runs older than this many days
            
        Returns:
            Number of deleted runs
        """
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        deleted_count = 0
        
        if MLFLOW_AVAILABLE and self._experiment_id:
            client = MlflowClient()
            runs = self.list_runs(max_results=1000)
            for run in runs:
                if run.start_time < cutoff_date:
                    try:
                        client.delete_run(run.run_id)
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete run {run.run_id}: {e}")
        else:
            # Local cleanup
            tracking_dir = Path(self.tracking_uri)
            for run_file in tracking_dir.glob("*.json"):
                try:
                    with open(run_file, "r") as f:
                        run_data = json.load(f)
                    start_time = datetime.fromisoformat(run_data["start_time"])
                    if start_time < cutoff_date:
                        run_file.unlink()
                        # Clean up artifacts if local
                        artifact_dir = Path(self.artifact_location) / run_data["run_id"]
                        if artifact_dir.exists():
                            import shutil
                            shutil.rmtree(artifact_dir)
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to cleanup local run {run_file}: {e}")
                    
        logger.info(f"Cleaned up {deleted_count} experiments older than {older_than_days} days")
        return deleted_count

    def _save_run_locally(self) -> None:
        """Save current run to local file."""
        if self._current_run is None:
            return

        run_file = Path(self.tracking_uri) / f"{self._current_run.run_id}.json"
        with open(run_file, "w") as f:
            json.dump(self._current_run.to_dict(), f, indent=2)

    @property
    def current_run(self) -> Optional[ExperimentRun]:
        """Get the current active run."""
        return self._current_run

    def __enter__(self) -> "ExperimentTracker":
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        status = "failed" if exc_type is not None else "completed"
        self.end_run(status=status)
