"""Pipeline state persistence and recovery."""

import json
import logging
import pickle
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class PipelineStateManager:
    """Manages persistence of pipeline state for recovery."""

    def __init__(self, state_dir: str = "logs/pipeline_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save_state(
        self,
        run_id: str,
        step_name: str,
        data: Any = None,
        context: Dict[str, Any] = None,
        retention_days: int = 30
    ) -> str:
        """
        Save intermediate state.
        
        Args:
            run_id: Unique run identifier
            step_name: Name of the pipeline step
            data: Data objects to save (optional, pickled)
            context: Metadata context (dict, JSON saved)
            retention_days: (Not implemented in MVP, but placeholder for cleanup)
            
        Returns:
            Path to saved state directory
        """
        run_dir = self.state_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        # Save Metadata
        meta = {
            "run_id": run_id,
            "step": step_name,
            "timestamp": timestamp,
            "context": context or {}
        }
        
        meta_path = run_dir / f"{step_name}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        # Save Data if provided
        if data is not None:
             data_path = run_dir / f"{step_name}_data.pkl"
             with open(data_path, "wb") as f:
                 pickle.dump(data, f)
                 
        logger.debug(f"Saved state for {run_id}/{step_name}")
        return str(run_dir)

    def load_state(self, run_id: str, step_name: str) -> Dict[str, Any]:
        """
        Load pipeline state.
        
        Returns:
             Dict with 'data' and 'context'
        """
        run_dir = self.state_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"No state found for run {run_id}")
            
        meta_path = run_dir / f"{step_name}_meta.json"
        if not meta_path.exists():
             raise FileNotFoundError(f"No state found for step {step_name}")
             
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        result = {"context": meta["context"], "meta": meta}
        
        data_path = run_dir / f"{step_name}_data.pkl"
        if data_path.exists():
            with open(data_path, "rb") as f:
                result["data"] = pickle.load(f)
        else:
            result["data"] = None
            
        return result

    def clear_state(self, run_id: str) -> None:
        """Clear state for a run."""
        run_dir = self.state_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
            logger.info(f"Cleared state for {run_id}")
