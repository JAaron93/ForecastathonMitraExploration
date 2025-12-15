"""Logging configuration for the pipeline."""

import logging
import logging.config
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields safely
        if hasattr(record, "props"):
             log_obj.update(record.props)
             
        return json.dumps(log_obj)

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    config_path: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (INFO, DEBUG, etc.)
        log_dir: Directory to store log files
        config_path: Optional path to YAML config file (not used in this simplified version)
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create distinct loggers for different concerns
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console Handler (Human readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File Handler (JSON Structured)
    # General app logs
    file_handler = logging.FileHandler(f"{log_dir}/app.jsonl")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)
    
    # Separate Error Log
    error_handler = logging.FileHandler(f"{log_dir}/errors.jsonl")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)
    
    logging.info(f"Logging configured with level {log_level}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
