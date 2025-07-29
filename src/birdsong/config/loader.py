"""
Configuration loader with hierarchical inheritance support.

This module provides functionality to load YAML configurations with 
template inheritance, environment variable substitution, and validation.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .schema import BirdsongConfig, ConfigValidationError


class ConfigLoader:
    """Configuration loader with inheritance and template support."""
    
    def __init__(self, search_paths: Optional[List[Union[str, Path]]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            search_paths: List of directories to search for configuration files
        """
        self.search_paths = search_paths or [
            Path.cwd() / "configs",
            Path.cwd() / "configs" / "templates", 
            Path(__file__).parent.parent.parent.parent / "configs",
            Path(__file__).parent.parent.parent.parent / "configs" / "templates",
        ]
        
        # Convert to Path objects and ensure they exist
        self.search_paths = [Path(p) for p in self.search_paths]
        self._loaded_configs = {}  # Cache for loaded configurations
        self._inheritance_stack = []  # Track inheritance to prevent cycles
    
    def load_config(
        self, 
        config_path: Union[str, Path], 
        override_values: Optional[Dict[str, Any]] = None
    ) -> BirdsongConfig:
        """
        Load configuration with inheritance and validation.
        
        Args:
            config_path: Path to configuration file or template name
            override_values: Dictionary of values to override in the config
            
        Returns:
            Validated BirdsongConfig object
            
        Raises:
            ConfigValidationError: If configuration is invalid
            FileNotFoundError: If configuration file is not found
        """
        # Resolve the configuration file path
        resolved_path = self._resolve_config_path(config_path)
        
        # Load the raw configuration with inheritance
        raw_config = self._load_with_inheritance(resolved_path)
        
        # Apply environment variable substitution
        raw_config = self._substitute_env_vars(raw_config)
        
        # Apply overrides
        if override_values:
            raw_config = self._apply_overrides(raw_config, override_values)
        
        # Validate and create BirdsongConfig object
        try:
            config = BirdsongConfig.from_dict(raw_config)
            
            # Track inheritance in metadata
            if self._inheritance_stack:
                config.inherits_from = list(self._inheritance_stack)
                
            return config
            
        except Exception as e:
            raise ConfigValidationError(
                f"Failed to load configuration from {resolved_path}: {str(e)}"
            )
        finally:
            # Clear inheritance tracking
            self._inheritance_stack.clear()
            self._loaded_configs.clear()
    
    def load_template(self, template_name: str) -> BirdsongConfig:
        """
        Load a configuration template by name.
        
        Args:
            template_name: Name of the template (without .yaml extension)
            
        Returns:
            Validated BirdsongConfig object
        """
        template_path = f"{template_name}.yaml"
        return self.load_config(template_path)
    
    def list_templates(self) -> List[str]:
        """
        List available configuration templates.
        
        Returns:
            List of template names (without .yaml extension)
        """
        templates = []
        
        for search_path in self.search_paths:
            if search_path.exists() and search_path.is_dir():
                for config_file in search_path.glob("*.yaml"):
                    template_name = config_file.stem
                    if template_name not in templates:
                        templates.append(template_name)
        
        return sorted(templates)
    
    def validate_config_file(self, config_path: Union[str, Path]) -> bool:
        """
        Validate a configuration file without loading it fully.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self.load_config(config_path)
            return True
        except (ConfigValidationError, FileNotFoundError, yaml.YAMLError):
            return False
    
    def _resolve_config_path(self, config_path: Union[str, Path]) -> Path:
        """Resolve configuration file path using search paths."""
        config_path = Path(config_path)
        
        # If absolute path or exists relative to cwd, use as-is
        if config_path.is_absolute():
            if config_path.exists():
                return config_path
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # If relative path exists from cwd, use it
        if config_path.exists():
            return config_path.resolve()
        
        # Search in configured search paths
        for search_path in self.search_paths:
            candidate = search_path / config_path
            if candidate.exists():
                return candidate.resolve()
        
        # Try with .yaml extension if not present
        if not config_path.suffix:
            return self._resolve_config_path(config_path.with_suffix('.yaml'))
        
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found in search paths: "
            f"{[str(p) for p in self.search_paths]}"
        )
    
    def _load_with_inheritance(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration with inheritance resolution."""
        # Prevent circular inheritance
        config_key = str(config_path)
        if config_key in self._inheritance_stack:
            raise ConfigValidationError(
                f"Circular inheritance detected: {' -> '.join(self._inheritance_stack)} -> {config_key}"
            )
        
        # Check cache
        if config_key in self._loaded_configs:
            return self._loaded_configs[config_key].copy()
        
        # Load the YAML file
        try:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Failed to read {config_path}: {e}")
        
        # Track inheritance
        self._inheritance_stack.append(config_key)
        
        # Process inheritance
        if 'inherits_from' in raw_config:
            parent_configs = raw_config.pop('inherits_from')
            if not isinstance(parent_configs, list):
                parent_configs = [parent_configs]
            
            # Load parent configurations
            merged_config = {}
            for parent_path in parent_configs:
                parent_resolved = self._resolve_config_path(parent_path)
                parent_config = self._load_with_inheritance(parent_resolved)
                merged_config = self._deep_merge(merged_config, parent_config)
            
            # Merge current config over parents
            raw_config = self._deep_merge(merged_config, raw_config)
        
        # Cache the result
        self._loaded_configs[config_key] = raw_config.copy()
        
        return raw_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        
        for key, value in override.items():
            if (
                key in result 
                and isinstance(result[key], dict) 
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values."""
        def substitute_value(value):
            if isinstance(value, str):
                # Replace ${VAR_NAME} and ${VAR_NAME:default_value} patterns
                def replace_env_var(match):
                    var_spec = match.group(1)
                    if ':' in var_spec:
                        var_name, default = var_spec.split(':', 1)
                    else:
                        var_name, default = var_spec, ""
                    
                    return os.getenv(var_name, default)
                
                return re.sub(r'\$\{([^}]+)\}', replace_env_var, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value
        
        return substitute_value(config)
    
    def _apply_overrides(
        self, 
        config: Dict[str, Any], 
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply override values to configuration using dot notation."""
        result = config.copy()
        
        for key, value in overrides.items():
            # Support dot notation for nested keys (e.g., "model.encoder_dim")
            keys = key.split('.')
            current = result
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the final value
            current[keys[-1]] = value
        
        return result


# Convenience functions for common usage patterns

def load_config(
    config_path: Union[str, Path],
    search_paths: Optional[List[Union[str, Path]]] = None,
    override_values: Optional[Dict[str, Any]] = None
) -> BirdsongConfig:
    """
    Load configuration with default settings.
    
    Args:
        config_path: Path to configuration file or template name
        search_paths: Additional search paths for configuration files
        override_values: Values to override in the configuration
        
    Returns:
        Validated BirdsongConfig object
    """
    loader = ConfigLoader(search_paths)
    return loader.load_config(config_path, override_values)


def load_template(
    template_name: str,
    search_paths: Optional[List[Union[str, Path]]] = None
) -> BirdsongConfig:
    """
    Load configuration template by name.
    
    Args:
        template_name: Name of template (without .yaml extension)
        search_paths: Additional search paths for templates
        
    Returns:
        Validated BirdsongConfig object
    """
    loader = ConfigLoader(search_paths)
    return loader.load_template(template_name)


def create_config_from_dict(data: Dict[str, Any]) -> BirdsongConfig:
    """
    Create configuration from dictionary with validation.
    
    Args:
        data: Configuration dictionary
        
    Returns:
        Validated BirdsongConfig object
        
    Raises:
        ConfigValidationError: If configuration is invalid
    """
    return BirdsongConfig.from_dict(data) 