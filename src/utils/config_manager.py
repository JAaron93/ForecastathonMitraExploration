"""
Configuration management utilities.
"""

import os
import yaml
import json
import logging
import jsonschema
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages loading, validation, and merging of configurations.
    """
    
    def __init__(self, config_dir: Optional[str] = None, schema_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.schema_dir = Path(schema_dir) if schema_dir else self.config_dir / "schemas"
        
    def load_config(self, config_name: str, schema_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a configuration file (YAML or JSON).
        Optionally validate against a schema.
        
        Args:
            config_name: Name of config file (e.g. 'data_config.yaml')
            schema_name: Name of schema file (e.g. 'data_config_schema.json')
            
        Returns:
            Loaded configuration dictionary
        """
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, "r") as f:
            if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
                config = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
                
        if schema_name:
            self.validate_config(config, schema_name)
            
        return config

    def validate_config(self, config: Dict[str, Any], schema_name: str) -> None:
        """
        Validate configuration against a schema.
        
        Args:
            config: Configuration dictionary
            schema_name: Name of schema file
        """
        schema_path = self.schema_dir / schema_name
        
        if not schema_path.exists():
             # Fallback: check if schema_name is just the name without path in schema_dir
             if not schema_path.exists():
                 raise FileNotFoundError(f"Schema file not found: {schema_path}")

        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)
            
            jsonschema.validate(instance=config, schema=schema)
            logger.info(f"Configuration successfully validated against {schema_name}")
            
        except jsonschema.exceptions.ValidationError as e:
            path_str = " -> ".join(str(p) for p in e.path) if e.path else "root"
            error_msg = f"Configuration validation failed at '{path_str}': {e.message}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise

    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configurations.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def get_value(self, config: Dict[str, Any], path: str, default: Any = None) -> Any:
        """
        Get a value from configuration using dot notation.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., 'model.params.learning_rate')
            default: Default value if path not found
            
        Returns:
            Value at path or default
        """
        keys = path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
                
        return current

    def set_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """
        Set a value in configuration using dot notation.
        Creates intermediate dictionaries if they don't exist.
        
        Args:
            config: Configuration dictionary (modified in-place)
            path: Dot-separated path
            value: Value to set
        """
        keys = path.split('.')
        current = config
        
        for i, key in enumerate(keys[:-1]):
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
            
        current[keys[-1]] = value

    def update_config(self, config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration with a dictionary of changes.
        This is an alias for merge_configs but implies modifying usage.
        
        Args:
            config: Base configuration
            updates: Dictionary of updates
            
        Returns:
            Updated configuration (new dictionary)
        """
        return self.merge_configs(config, updates)
