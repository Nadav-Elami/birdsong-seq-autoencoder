"""
Runtime validation utilities for birdsong configurations.

This module provides functions for validating configurations at runtime,
checking dependencies, and ensuring system requirements are met.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .schema import BirdsongConfig, ConfigValidationError


def validate_config(config: BirdsongConfig, strict: bool = True) -> List[str]:
    """
    Perform comprehensive runtime validation of configuration.
    
    Args:
        config: Configuration to validate
        strict: If True, raise exception on validation errors
        
    Returns:
        List of validation warning messages
        
    Raises:
        ConfigValidationError: If strict=True and validation fails
    """
    warnings = []
    errors = []
    
    # Validate paths
    try:
        validate_paths(config)
    except ConfigValidationError as e:
        if strict:
            raise
        errors.append(str(e))
    
    # Validate dependencies
    try:
        validate_dependencies(config)
    except ConfigValidationError as e:
        if strict:
            raise  
        errors.append(str(e))
    
    # Validate system resources
    resource_warnings = _validate_system_resources(config)
    warnings.extend(resource_warnings)
    
    # Validate model architecture constraints
    arch_warnings = _validate_model_architecture(config)
    warnings.extend(arch_warnings)
    
    # Validate training configuration
    training_warnings = _validate_training_config(config)
    warnings.extend(training_warnings)
    
    if errors and strict:
        raise ConfigValidationError(
            f"Configuration validation failed with {len(errors)} errors",
            errors=[{"loc": ["validation"], "msg": err} for err in errors]
        )
    
    return warnings


def validate_paths(config: BirdsongConfig) -> None:
    """
    Validate all file and directory paths in configuration.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ConfigValidationError: If required paths are invalid
    """
    errors = []
    
    # Validate data paths
    if config.data.data_path:
        data_path = Path(config.data.data_path)
        if not data_path.exists():
            errors.append(f"Data file not found: {config.data.data_path}")
        elif not data_path.is_file():
            errors.append(f"Data path is not a file: {config.data.data_path}")
        elif not data_path.suffix.lower() in ['.h5', '.hdf5']:
            errors.append(f"Data file must be HDF5 format: {config.data.data_path}")
    
    # Validate output directories and create if needed
    output_dirs = []
    
    if config.data.output_path:
        output_dirs.append(Path(config.data.output_path).parent)
    
    if config.training.checkpoint_path:
        output_dirs.append(Path(config.training.checkpoint_path).parent)
    
    if config.training.plot_dir:
        output_dirs.append(Path(config.training.plot_dir))
    
    if config.evaluation.output_dir:
        output_dirs.append(Path(config.evaluation.output_dir))
    
    for output_dir in output_dirs:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            errors.append(f"Cannot create output directory {output_dir}: {e}")
    
    # Validate evaluation checkpoint
    if config.evaluation.checkpoint_path:
        checkpoint_path = Path(config.evaluation.checkpoint_path)
        if not checkpoint_path.exists():
            errors.append(f"Checkpoint file not found: {config.evaluation.checkpoint_path}")
    
    if errors:
        raise ConfigValidationError(
            "Path validation failed",
            errors=[{"loc": ["paths"], "msg": err} for err in errors]
        )


def validate_dependencies(config: BirdsongConfig) -> None:
    """
    Validate system dependencies and requirements.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ConfigValidationError: If required dependencies are missing
    """
    errors = []
    
    # Check PyTorch availability
    try:
        import torch
    except ImportError:
        errors.append("PyTorch is required but not installed")
    
    # Check CUDA availability if specified
    if hasattr(config.experiment, 'device') and config.experiment.device == "cuda":
        if not torch.cuda.is_available():
            errors.append("CUDA device specified but CUDA is not available")
    
    # Check MPS availability if specified  
    if hasattr(config.experiment, 'device') and config.experiment.device == "mps":
        if not torch.backends.mps.is_available():
            errors.append("MPS device specified but MPS is not available")
    
    # Check required Python packages
    required_packages = [
        ('numpy', 'NumPy'),
        ('h5py', 'HDF5 support'),
        ('matplotlib', 'plotting'),
        ('tqdm', 'progress bars'),
        ('pydantic', 'configuration validation'),
        ('yaml', 'YAML configuration files')
    ]
    
    for package, description in required_packages:
        try:
            __import__(package)
        except ImportError:
            errors.append(f"Required package '{package}' not found (needed for {description})")
    
    # Check system utilities
    if config.data.plot_samples or config.evaluation.plot_samples:
        # Check if display is available for plotting
        if os.name != 'nt' and not os.environ.get('DISPLAY'):
            # Linux/Mac without display - this is a warning, not an error
            pass
    
    if errors:
        raise ConfigValidationError(
            "Dependency validation failed",
            errors=[{"loc": ["dependencies"], "msg": err} for err in errors]
        )


def _validate_system_resources(config: BirdsongConfig) -> List[str]:
    """Validate system resource requirements."""
    warnings = []
    
    # Check available memory
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Estimate memory requirements
        estimated_memory = _estimate_memory_usage(config)
        
        if estimated_memory > available_memory_gb:
            warnings.append(
                f"Estimated memory usage ({estimated_memory:.1f}GB) "
                f"exceeds available memory ({available_memory_gb:.1f}GB)"
            )
        
        # Check experiment memory limit
        if (config.experiment and config.experiment.max_memory_gb and 
            estimated_memory > config.experiment.max_memory_gb):
            warnings.append(
                f"Estimated memory usage ({estimated_memory:.1f}GB) "
                f"exceeds configured limit ({config.experiment.max_memory_gb}GB)"
            )
    except ImportError:
        warnings.append("psutil not available - cannot check memory requirements")
    
    # Check disk space for outputs
    if config.training.checkpoint_path:
        checkpoint_dir = Path(config.training.checkpoint_path).parent
        try:
            disk_usage = shutil.disk_usage(checkpoint_dir)
            free_space_gb = disk_usage.free / (1024**3)
            
            # Estimate checkpoint size (rough estimate)
            estimated_checkpoint_size = _estimate_checkpoint_size(config)
            
            if estimated_checkpoint_size > free_space_gb:
                warnings.append(
                    f"Estimated checkpoint size ({estimated_checkpoint_size:.1f}GB) "
                    f"exceeds available disk space ({free_space_gb:.1f}GB)"
                )
        except (OSError, FileNotFoundError):
            warnings.append(f"Cannot check disk space for checkpoint directory: {checkpoint_dir}")
    
    return warnings


def _validate_model_architecture(config: BirdsongConfig) -> List[str]:
    """Validate model architecture constraints."""
    warnings = []
    
    # Check dimension relationships
    if config.model.latent_dim > config.model.factor_dim:
        warnings.append(
            f"Latent dimension ({config.model.latent_dim}) is larger than "
            f"factor dimension ({config.model.factor_dim}). Consider reducing latent_dim."
        )
    
    if config.model.inferred_input_dim > config.model.latent_dim:
        warnings.append(
            f"Inferred input dimension ({config.model.inferred_input_dim}) is larger than "
            f"latent dimension ({config.model.latent_dim}). Consider reducing inferred_input_dim."
        )
    
    # Check for very large model sizes
    total_params = _estimate_model_parameters(config)
    if total_params > 50_000_000:  # 50M parameters
        warnings.append(
            f"Model has approximately {total_params:,} parameters, which may be "
            f"very large for this type of model. Consider reducing dimensions."
        )
    
    # Check alphabet size consistency
    if config.model.alphabet_size and config.model.alphabet_size < 3:
        warnings.append(
            f"Very small alphabet size ({config.model.alphabet_size}). "
            f"Consider increasing for meaningful sequence modeling."
        )
    
    return warnings


def _validate_training_config(config: BirdsongConfig) -> List[str]:
    """Validate training configuration."""
    warnings = []
    
    # Check batch size
    if config.training.batch_size > 512:
        warnings.append(
            f"Very large batch size ({config.training.batch_size}). "
            f"This may cause memory issues or suboptimal training."
        )
    
    if config.training.batch_size == 1:
        warnings.append(
            "Batch size of 1 may cause training instability. "
            "Consider increasing batch size if possible."
        )
    
    # Check learning rate
    if config.training.learning_rate > 0.01:
        warnings.append(
            f"Very high learning rate ({config.training.learning_rate}). "
            f"This may cause training instability."
        )
    
    if config.training.learning_rate < 1e-6:
        warnings.append(
            f"Very low learning rate ({config.training.learning_rate}). "
            f"Training may be very slow or ineffective."
        )
    
    # Check epoch counts
    if config.training.epochs > 1000:
        warnings.append(
            f"Very high epoch count ({config.training.epochs}). "
            f"Consider early stopping or monitoring for overfitting."
        )
    
    # Check KL annealing schedule
    if config.training.kl_full_epoch <= config.training.kl_start_epoch:
        warnings.append(
            "KL annealing schedule may be incorrect. "
            f"kl_full_epoch ({config.training.kl_full_epoch}) should be > "
            f"kl_start_epoch ({config.training.kl_start_epoch})."
        )
    
    # Check data split ratios
    total_split = config.training.val_split + config.training.test_split
    if total_split > 0.5:
        warnings.append(
            f"Large validation/test split ({total_split:.1%}). "
            f"Consider reducing to leave more data for training."
        )
    
    return warnings


def _estimate_memory_usage(config: BirdsongConfig) -> float:
    """Estimate memory usage in GB."""
    # Rough estimation based on model size and batch size
    model_params = _estimate_model_parameters(config)
    
    # Memory for model parameters (float32 = 4 bytes)
    model_memory = model_params * 4 / (1024**3)
    
    # Memory for gradients (same size as parameters)
    gradient_memory = model_memory
    
    # Memory for optimizer state (Adam needs ~2x parameter memory)
    optimizer_memory = model_memory * 2
    
    # Memory for batch data (rough estimate)
    batch_memory = (
        config.training.batch_size * 
        config.data.seq_max_length * 
        config.model.alphabet_size * 
        4  # float32
    ) / (1024**3)
    
    return model_memory + gradient_memory + optimizer_memory + batch_memory


def _estimate_model_parameters(config: BirdsongConfig) -> int:
    """Estimate total number of model parameters."""
    # Rough estimation based on LFADS architecture
    alphabet_size = config.model.alphabet_size or 8
    
    # Encoder parameters
    encoder_params = (
        config.model.encoder_dim * config.model.encoder_dim * 2 +  # Bidirectional GRU
        config.model.encoder_dim * config.model.latent_dim * 2     # mu and logvar projections
    )
    
    # Controller parameters  
    controller_params = (
        config.model.controller_dim * config.model.controller_dim +  # GRU
        config.model.controller_dim * config.model.inferred_input_dim * 2  # mu and logvar
    )
    
    # Generator parameters
    generator_params = (
        config.model.generator_dim * config.model.generator_dim * 2 +  # Bidirectional GRU
        config.model.generator_dim * config.model.factor_dim              # projection
    )
    
    # Output layer parameters
    output_params = config.model.factor_dim * (alphabet_size ** config.model.order)
    
    return encoder_params + controller_params + generator_params + output_params


def _estimate_checkpoint_size(config: BirdsongConfig) -> float:
    """Estimate checkpoint file size in GB."""
    model_params = _estimate_model_parameters(config)
    
    # Model state dict (float32 = 4 bytes)
    model_size = model_params * 4
    
    # Optimizer state (Adam stores additional state)
    optimizer_size = model_params * 12  # Rough estimate
    
    # Training metadata and history
    metadata_size = 1024 * 1024  # 1MB for metadata
    
    return (model_size + optimizer_size + metadata_size) / (1024**3)


def check_config_compatibility(config1: BirdsongConfig, config2: BirdsongConfig) -> List[str]:
    """
    Check compatibility between two configurations.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        List of compatibility warnings
    """
    warnings = []
    
    # Check model architecture compatibility
    if config1.model.alphabet_size != config2.model.alphabet_size:
        warnings.append(
            f"Alphabet size mismatch: {config1.model.alphabet_size} vs {config2.model.alphabet_size}"
        )
    
    if config1.model.order != config2.model.order:
        warnings.append(
            f"Model order mismatch: {config1.model.order} vs {config2.model.order}"
        )
    
    # Check dimension compatibility
    model_dims = [
        'encoder_dim', 'controller_dim', 'generator_dim', 
        'factor_dim', 'latent_dim', 'inferred_input_dim'
    ]
    
    for dim in model_dims:
        val1 = getattr(config1.model, dim)
        val2 = getattr(config2.model, dim)
        if val1 != val2:
            warnings.append(f"Model {dim} mismatch: {val1} vs {val2}")
    
    return warnings 