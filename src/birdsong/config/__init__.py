"""
Configuration system for birdsong research package.

This module provides hierarchical YAML configuration with schema validation,
inheritance, and template support for reproducible experiments.
"""

from .loader import ConfigLoader, load_config, load_template
from .schema import (
    BirdsongConfig,
    DataConfig, 
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    ExperimentConfig,
    ConfigValidationError
)
from .validation import validate_config, validate_paths, validate_dependencies

__all__ = [
    # Core configuration classes
    "BirdsongConfig",
    "DataConfig", 
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig", 
    "ExperimentConfig",
    
    # Configuration loading
    "ConfigLoader",
    "load_config",
    "load_template",
    
    # Validation
    "validate_config",
    "validate_paths", 
    "validate_dependencies",
    "ConfigValidationError",
] 