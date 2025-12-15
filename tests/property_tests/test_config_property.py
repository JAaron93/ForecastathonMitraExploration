"""
Property tests for configuration manager.
"""

from hypothesis import given, strategies as st
import pytest
import json
import yaml
from pathlib import Path
import tempfile
import shutil

from src.utils.config_manager import ConfigManager

# Strategy for generating arbitrary JSON-serializable dictionaries
json_strategy = st.recursive(
    st.dictionaries(st.text(), st.one_of(st.none(), st.booleans(), st.integers(), st.floats(), st.text())),
    lambda children: st.dictionaries(st.text(), children),
    max_leaves=10
)

# Specific schema for testing validation
test_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer", "minimum": 0},
        "nested": {
            "type": "object",
            "properties": {
                "flag": {"type": "boolean"}
            }
        }
    },
    "required": ["name"]
}

@pytest.fixture
def config_env():
    """Setup temporary directory for config tests."""
    temp_dir = Path(tempfile.mkdtemp())
    config_dir = temp_dir / "config"
    schema_dir = temp_dir / "schemas"
    config_dir.mkdir()
    schema_dir.mkdir()
    
    # Write test schema
    with open(schema_dir / "test_schema.json", "w") as f:
        json.dump(test_schema, f)
        
    yield config_dir, schema_dir
    
    shutil.rmtree(temp_dir)

def test_load_and_validate_valid_config(config_env):
    """Test loading and validating a correct configuration."""
    config_dir, schema_dir = config_env
    cm = ConfigManager(str(config_dir), str(schema_dir))
    
    valid_config = {
        "name": "test",
        "count": 10,
        "nested": {"flag": True}
    }
    
    with open(config_dir / "valid.yaml", "w") as f:
        yaml.dump(valid_config, f)
        
    loaded = cm.load_config("valid.yaml", "test_schema.json")
    assert loaded == valid_config

def test_load_and_validate_invalid_config(config_env):
    """Test that invalid configuration raises ValueError."""
    config_dir, schema_dir = config_env
    cm = ConfigManager(str(config_dir), str(schema_dir))
    
    # Missing required 'name'
    invalid_config = {
        "count": 10
    }
    
    with open(config_dir / "invalid.json", "w") as f:
        json.dump(invalid_config, f)
        
    with pytest.raises(ValueError, match="Configuration validation failed"):
        cm.load_config("invalid.json", "test_schema.json")

@given(base=json_strategy, override=json_strategy)
def test_merge_config_properties(base, override):
    """
    Property: Merging should always result in a dictionary containing keys from both,
    with override values taking precedence.
    """
    cm = ConfigManager()
    merged = cm.merge_configs(base, override)
    
    # Check that all override keys are present and equal
    import math
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            # Recursed
            pass
        else:
            if isinstance(v, float) and math.isnan(v):
                assert math.isnan(merged[k])
            else:
                assert merged[k] == v
            
    # Check that base keys not in override are preserved
    for k, v in base.items():
        if k not in override:
            if isinstance(v, float) and math.isnan(v):
                assert math.isnan(merged[k])
            else:
                assert merged[k] == v

def test_config_flexibility(config_env):
    """Test flexibility logic (merging default with experiment)."""
    cm = ConfigManager()
    
    default_config = {
        "model": "xgboost",
        "params": {
            "learning_rate": 0.1,
            "max_depth": 3
        }
    }
    
    experiment_config = {
        "params": {
            "learning_rate": 0.05
        }
    }
    
    merged = cm.merge_configs(default_config, experiment_config)
    
    assert merged["model"] == "xgboost"
    assert merged["params"]["learning_rate"] == 0.05
    assert merged["params"]["max_depth"] == 3
