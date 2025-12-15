"""Error handling utilities."""

import functools
import logging
import time
import inspect
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type, Union

logger = logging.getLogger(__name__)


def retry_with_backoff(
    retries: int = 5,
    initial_backoff: float = 2.0,
    backoff_multiplier: float = 2.0,
    max_backoff: float = 60.0,
    retry_exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        retries: Maximum number of retries
        initial_backoff: Initial wait time in seconds
        backoff_multiplier: Multiplier for wait time after each failure
        max_backoff: Maximum wait time in seconds
        retry_exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wait_time = initial_backoff
            attempt = 0
            
            while attempt <= retries:
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    attempt += 1
                    if attempt > retries:
                        logger.error(f"Function {func.__name__} failed after {retries} retries: {e}")
                        raise e
                    
                    logger.warning(
                        f"Retry attempt {attempt}/{retries} for {func.__name__} "
                        f"after error: {e}. Waiting {wait_time}s."
                    )
                    
                    time.sleep(wait_time)
                    wait_time = min(wait_time * backoff_multiplier, max_backoff)
            
        return wrapper
    return decorator


@dataclass
class RecoveryContext:
    """Captures context for error recovery and debugging."""
    run_id: str
    timestamp: float = field(default_factory=time.time)
    exception_type: str = ""
    exception_message: str = ""
    stack_trace: str = ""
    local_variables: Dict[str, str] = field(default_factory=dict)
    system_stats: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_exception(cls, run_id: str, exc: Exception) -> "RecoveryContext":
        """
        Create context from an exception.
        Captures locals from the frame where exception occurred.
        """
        tb = traceback.extract_tb(exc.__traceback__)
        stack_trace = "".join(traceback.format_tb(exc.__traceback__))
        
        # Capture locals from the last frame
        locals_repr = {}
        if exc.__traceback__:
             # Go to the last frame
             ptr = exc.__traceback__
             while ptr.tb_next:
                 ptr = ptr.tb_next
             frame = ptr.tb_frame
             
             # Safely capture locals
             for k, v in frame.f_locals.items():
                 try:
                     val_str = str(v)
                     # Truncate long values
                     if len(val_str) > 500:
                         val_str = val_str[:500] + "..."
                     locals_repr[k] = val_str
                 except Exception:
                     locals_repr[k] = "<unprintable>"
        
        return cls(
            run_id=run_id,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            stack_trace=stack_trace,
            local_variables=locals_repr
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "stack_trace": self.stack_trace,
            "local_variables": self.local_variables,
            "system_stats": self.system_stats
        }
