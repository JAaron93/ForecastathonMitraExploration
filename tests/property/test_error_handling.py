"""Property tests for error handling and system utilities."""

import time
import pytest
from hypothesis import given, settings, strategies as st

from src.utils.error_handling import retry_with_backoff, RecoveryContext
from src.utils.state_persistence import PipelineStateManager


class TransientError(Exception):
    pass

class PersistentError(Exception):
    pass


def test_retry_success_after_failure():
    """Test using mock function."""
    
    attempts = 0
    
    @retry_with_backoff(retries=3, initial_backoff=0.01) # Short backoff for test
    def flaky_func():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientError("Fail")
        return "Success"
        
    result = flaky_func()
    assert result == "Success"
    assert attempts == 3


def test_retry_failure_max_retries():
    """Test max retries reached."""
    attempts = 0
    
    @retry_with_backoff(retries=2, initial_backoff=0.01)
    def failing_func():
        nonlocal attempts
        attempts += 1
        raise PersistentError("Fail")
        
    with pytest.raises(PersistentError):
        failing_func()
        
    # Attempt 1 (initial) + 2 retries = 3 total attempts usually?
    # Logic in retry: attempt starts at 0. Loop: try... except: attempt += 1. if attempt > retries: raise.
    # Initial call (attempt 0 in loop logic? No, loop starts.) 
    # Let's verify logic:
    # attempt = 0
    # while attempt <= retries (2):
    #   try func
    #   except: attempt += 1 (now 1). Check > 2 (No). Sleep. Loop
    #   try func
    #   except: attempt += 1 (now 2). Check > 2 (No). Sleep. Loop
    #   try func
    #   except: attempt += 1 (now 3). Check > 2 (Yes). Raise.
    # So 3 failed calls total.
    
    assert attempts == 3


def test_recovery_context_capture():
    """Verify context capture from exception."""
    try:
        x = 123
        y = "important_context"
        raise ValueError("Something went wrong")
    except ValueError as e:
        ctx = RecoveryContext.from_exception(run_id="test_run", exc=e)
        
    assert ctx.run_id == "test_run"
    assert ctx.exception_type == "ValueError"
    assert "x" in ctx.local_variables
    assert ctx.local_variables["x"] == "123"
    assert "y" in ctx.local_variables
    assert ctx.local_variables["y"] == "important_context"


@given(st.dictionaries(st.text(), st.text()), st.text())
@settings(max_examples=20, deadline=None)
def test_state_persistence_roundtrip(context_data, string_data):
    """
    Property 9: Error handling and state persistence.
    Verify data roundtrip through persistence manager.
    """
    manager = PipelineStateManager(state_dir="tests/temp_state")
    run_id = f"run_{abs(hash(string_data))}"
    
    # Save
    manager.save_state(run_id, "step1", data=string_data, context=context_data)
    
    # Load
    loaded = manager.load_state(run_id, "step1")
    
    assert loaded["data"] == string_data
    assert loaded["context"] == context_data
    
    # Cleanup
    manager.clear_state(run_id)
