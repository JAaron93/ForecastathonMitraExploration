"""
Monitoring utilities for system resources and performance thresholds.
"""

import threading
import time
import psutil
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ResourceUsage:
    """Container for resource usage metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float

@dataclass
class AlertConfig:
    """Configuration for an alert."""
    metric_name: str
    threshold: float
    operator: str  # '>', '<', '>=', '<='
    message_template: str
    cooldown_seconds: int = 60
    
    def check(self, value: float) -> bool:
        """Check if value breaches the threshold."""
        if self.operator == '>':
            return value > self.threshold
        elif self.operator == '<':
            return value < self.threshold
        elif self.operator == '>=':
            return value >= self.threshold
        elif self.operator == '<=':
            return value <= self.threshold
        return False

class AlertManager:
    """
    Manages performance alerts.
    """
    
    def __init__(self):
        self._alerts: Dict[str, AlertConfig] = {}
        self._last_triggered: Dict[str, datetime] = {}
        self._triggered_alerts: List[Dict[str, Any]] = []
        
    def add_alert(self, name: str, config: AlertConfig) -> None:
        """Add an alert configuration."""
        self._alerts[name] = config
        
    def check_metric(self, metric_name: str, value: float) -> Optional[str]:
        """
        Check a metric against relevant alerts.
        
        Returns:
            Alert message if triggered, None otherwise
        """
        triggered_message = None
        now = datetime.now()
        
        for name, config in self._alerts.items():
            if config.metric_name != metric_name:
                continue
                
            if config.check(value):
                # Check cooldown
                last_time = self._last_triggered.get(name)
                if last_time and (now - last_time).total_seconds() < config.cooldown_seconds:
                    continue
                    
                message = config.message_template.format(
                    metric=metric_name,
                    value=value,
                    threshold=config.threshold
                )
                
                self._last_triggered[name] = now
                self._triggered_alerts.append({
                    "name": name,
                    "message": message,
                    "timestamp": now.isoformat(),
                    "value": value
                })
                logger.warning(f"ALERT: {message}")
                triggered_message = message
                
        return triggered_message
        
    def get_triggered_history(self) -> List[Dict[str, Any]]:
        """Get history of triggered alerts."""
        return self._triggered_alerts

class SystemMonitor:
    """
    Monitors system resources in the background.
    """
    
    def __init__(self, interval_seconds: float = 5.0):
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._usage_history: List[ResourceUsage] = []
        self._alert_manager = AlertManager()
        
        # Default alerts
        self._alert_manager.add_alert(
            "high_memory",
            AlertConfig(
                metric_name="memory_percent",
                threshold=90.0,
                operator=">",
                message_template="Memory usage high: {value:.1f}% > {threshold}%"
            )
        )
        
    def start(self, callback: Optional[Callable[[ResourceUsage], None]] = None) -> None:
        """Start monitoring in a background thread."""
        if self._thread is not None:
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            args=(callback,),
            daemon=True
        )
        self._thread.start()
        logger.info("System monitoring started")
        
    def stop(self) -> None:
        """Stop monitoring."""
        if self._thread is None:
            return
            
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        logger.info("System monitoring stopped")
        
    def _monitor_loop(self, callback: Optional[Callable[[ResourceUsage], None]]) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Collect metrics
                cpu = psutil.cpu_percent(interval=1.0)
                memory = psutil.virtual_memory()
                
                usage = ResourceUsage(
                    timestamp=datetime.now(),
                    cpu_percent=cpu,
                    memory_percent=memory.percent,
                    memory_used_gb=memory.used / (1024**3)
                )
                
                self._usage_history.append(usage)
                
                # Check alerts
                self._alert_manager.check_metric("memory_percent", usage.memory_percent)
                self._alert_manager.check_metric("cpu_percent", usage.cpu_percent)
                
                if callback:
                    callback(usage)
                    
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                
            # Wait for next interval (subtracting the 1s sleep from cpu_percent)
            sleep_time = max(0.0, self.interval_seconds - 1.0)
            if self._stop_event.wait(sleep_time):
                break
                
    def get_history(self) -> List[ResourceUsage]:
        """Get recorded usage history."""
        return list(self._usage_history)
        
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get triggered alerts."""
        return self._alert_manager.get_triggered_history()
