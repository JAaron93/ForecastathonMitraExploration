
import os
import shutil
import tempfile
import json
import logging
import pytest
from hypothesis import given, settings, strategies as st
from src.utils.config_manager import ConfigManager

# Minimal recursive strategy for generating JSON-compatible dictionaries
json_values = st.recursive(
    st.text(min_size=1) | st.integers() | st.floats(allow_nan=False) | st.booleans(),
    lambda children: st.lists(children) | st.dictionaries(st.text(min_size=1), children),
    max_leaves=10
)

# Strategy for generating dot-separated paths
def paths_from_dict(d, prefix=""):
    paths = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            paths.append(new_prefix)
            paths.extend(paths_from_dict(v, new_prefix))
    return paths

@st.composite
def config_and_path(draw):
    config = draw(st.dictionaries(st.text(min_size=1), json_values, min_size=1, max_size=5))
    paths = paths_from_dict(config)
    if not paths:
        return config, "nonexistent"
    path = draw(st.sampled_from(paths))
    return config, path

class TestConfigManagerProperties:
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.test_dir, "config")
        self.schema_dir = os.path.join(self.config_dir, "schemas")
        os.makedirs(self.schema_dir, exist_ok=True)
        
        self.manager = ConfigManager(config_dir=self.config_dir, schema_dir=self.schema_dir)
        
        yield
        
        shutil.rmtree(self.test_dir)

    @given(config_and_path())
    @settings(max_examples=50)
    def test_get_value_consistency(self, data):
        """
        Property: get_value should correctly retrieve existing values using dot notation.
        """
        config, path = data
        
        # Manually traverse to verify
        keys = path.split('.')
        expected = config
        found = True
        for k in keys:
            if isinstance(expected, dict) and k in expected:
                expected = expected[k]
            else:
                found = False
                break
        
        if found:
            assert self.manager.get_value(config, path) == expected
        else:
            assert self.manager.get_value(config, path, default="DEFAULT") == "DEFAULT"

    @given(st.dictionaries(st.text(min_size=1), json_values), st.text(min_size=1), json_values)
    @settings(max_examples=50)
    def test_set_value_consistency(self, config, path, value):
        """
        Property: set_value should correctly set values, creating intermediate dicts if needed,
        and get_value should retrieve them.
        """
        # Exclude paths containing dots that are actually keys in the initial dictionary
        # (This is a simplified test assumption; in reality keys shouldn't contain dots for this utility)
        if any('.' in k for k in config.keys()):
            return

        # Deep copy to avoid mutating the strategy input
        config_copy = json.loads(json.dumps(config))
        
        self.manager.set_value(config_copy, path, value)
        retrieved = self.manager.get_value(config_copy, path)
        
        # Use json serialization for comparison to handle type differences (e.g. tuples vs lists)
        assert json.dumps(retrieved, sort_keys=True) == json.dumps(value, sort_keys=True)

    @given(st.dictionaries(st.text(min_size=1), json_values), st.dictionaries(st.text(min_size=1), json_values))
    @settings(max_examples=50)
    def test_merge_configs_properties(self, base, override):
        """
        Property: merge_configs should produce a merged result where override values take precedence.
        """
        merged = self.manager.merge_configs(base, override)
        
        # Check that all top-level keys from both are present
        for k in base:
            assert k in merged
        for k in override:
            assert k in merged
            # If not a nested dict merge, override should win
            if not (isinstance(base.get(k), dict) and isinstance(override[k], dict)):
                assert merged[k] == override[k]

    def test_validation_error_message(self):
        """
        Property: Validation errors should contain the path to the error.
        """
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": { "type": "integer" }
                    }
                }
            }
        }
        
        schema_path = os.path.join(self.schema_dir, "test_schema.json")
        with open(schema_path, 'w') as f:
            json.dump(schema, f)
            
        invalid_config = {"level1": {"level2": "not_an_integer"}}
        
        with pytest.raises(ValueError) as excinfo:
            self.manager.validate_config(invalid_config, "test_schema.json")
            
        assert "level1 -> level2" in str(excinfo.value)
