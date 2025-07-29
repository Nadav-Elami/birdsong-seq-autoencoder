"""
Configuration schema for birdsong research package.

This module defines Pydantic models for type-safe, validated configuration
with hierarchical inheritance and clear error reporting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic import ValidationError as PydanticValidationError


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    
    def __init__(self, message: str, errors: Optional[List[Dict[str, Any]]] = None):
        self.message = message
        self.errors = errors or []
        super().__init__(message)
    
    def __str__(self) -> str:
        if not self.errors:
            return self.message
        
        error_details = []
        for error in self.errors:
            loc = ".".join(str(x) for x in error.get("loc", []))
            msg = error.get("msg", "Unknown error")
            error_details.append(f"  {loc}: {msg}")
        
        return f"{self.message}\n" + "\n".join(error_details)


class BaseConfig(BaseModel):
    """Base configuration class with common functionality."""
    
    model_config = ConfigDict(
        extra="forbid",  # Prevent unexpected fields
        validate_assignment=True,  # Validate on assignment
        use_enum_values=True,
        arbitrary_types_allowed=True
    )


class DataConfig(BaseConfig):
    """Configuration for data generation and loading."""
    
    # Data generation parameters
    alphabet: List[str] = Field(
        default=["<", "a", "b", "c", "d", "e", "f", ">"],
        description="Alphabet symbols for sequence generation"
    )
    order: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Markov process order (1=bigram, 2=trigram, etc.)"
    )
    
    # Process configuration
    process_configs: List[tuple[str, int]] = Field(
        default=[("1st_order", 8), ("linear", 8)],
        description="List of (process_name, num_processes) tuples"
    )
    
    # Sequence parameters
    num_batches: int = Field(
        default=100,
        ge=1,
        description="Number of time steps per process"
    )
    batch_size: int = Field(
        default=50,
        ge=1,
        description="Number of sequences per batch"
    )
    seq_min_length: int = Field(
        default=10,
        ge=3,
        description="Minimum sequence length"
    )
    seq_max_length: int = Field(
        default=50,
        ge=3,
        description="Maximum sequence length"
    )
    
    # Data paths
    data_path: Optional[str] = Field(
        default=None,
        description="Path to existing data file (H5 format)"
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Output path for generated data"
    )
    
    # Validation options
    validate_data: bool = Field(
        default=True,
        description="Whether to validate generated data"
    )
    plot_samples: bool = Field(
        default=False,
        description="Whether to plot sample sequences"
    )
    
    @field_validator("alphabet")
    @classmethod
    def validate_alphabet(cls, v):
        if "<" not in v or ">" not in v:
            raise ValueError("Alphabet must contain start '<' and end '>' symbols")
        if len(v) < 3:
            raise ValueError("Alphabet must have at least 3 symbols")
        return v
    
    @field_validator("seq_max_length")
    @classmethod
    def validate_seq_lengths(cls, v, info):
        if hasattr(info, 'data') and "seq_min_length" in info.data and v <= info.data["seq_min_length"]:
            raise ValueError("seq_max_length must be greater than seq_min_length")
        return v
    
    @field_validator("data_path")
    @classmethod
    def validate_data_path(cls, v):
        if v is not None and not os.path.exists(v):
            raise ValueError(f"Data file not found: {v}")
        return v


class ModelConfig(BaseConfig):
    """Configuration for LFADS model architecture."""
    
    # Required parameters
    alphabet_size: Optional[int] = Field(
        default=None,
        ge=3,
        description="Size of alphabet (auto-derived from data if None)"
    )
    order: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Markov process order"
    )
    
    # Network dimensions
    encoder_dim: int = Field(
        default=64,
        ge=8,
        le=1024,
        description="Encoder hidden dimension"
    )
    controller_dim: int = Field(
        default=64,
        ge=8, 
        le=1024,
        description="Controller hidden dimension"
    )
    generator_dim: int = Field(
        default=64,
        ge=8,
        le=1024,
        description="Generator hidden dimension"
    )
    factor_dim: int = Field(
        default=32,
        ge=4,
        le=512,
        description="Factor space dimension"
    )
    latent_dim: int = Field(
        default=16,
        ge=2,
        le=256,
        description="Latent space dimension"
    )
    inferred_input_dim: int = Field(
        default=8,
        ge=1,
        le=128,
        description="Inferred input dimension"
    )
    
    # Model hyperparameters
    kappa: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Regularization strength"
    )
    ar_step_size: float = Field(
        default=0.99,
        ge=0.1,
        le=1.0,
        description="Autoregressive step size"
    )
    ar_process_var: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        description="Autoregressive process variance"
    )
    
    # Architecture options
    dropout_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=0.9,
        description="Dropout rate for training"
    )
    use_bias: bool = Field(
        default=True,
        description="Whether to use bias in linear layers"
    )


class TrainingConfig(BaseConfig):
    """Configuration for model training."""
    
    # Training parameters
    epochs: int = Field(
        default=20,
        ge=1,
        le=10000,
        description="Number of training epochs"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Training batch size"
    )
    learning_rate: float = Field(
        default=1e-3,
        ge=1e-6,
        le=1.0,
        description="Learning rate"
    )
    
    # Optimizer configuration
    optimizer: Literal["adam", "sgd", "adamw"] = Field(
        default="adam",
        description="Optimizer type"
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weight decay for optimizer"
    )
    momentum: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Momentum for SGD optimizer"
    )
    
    # Learning rate scheduling
    lr_scheduler: Literal["none", "step", "cosine", "exponential"] = Field(
        default="none",
        description="Learning rate scheduler type"
    )
    lr_step_size: int = Field(
        default=10,
        ge=1,
        description="Step size for step LR scheduler"
    )
    lr_gamma: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Gamma for step LR scheduler"
    )
    lr_min: float = Field(
        default=1e-6,
        ge=1e-8,
        description="Minimum learning rate"
    )
    
    # KL annealing
    kl_start_epoch: int = Field(
        default=2,
        ge=0,
        description="Epoch to start KL annealing"
    )
    kl_full_epoch: int = Field(
        default=10,
        ge=1,
        description="Epoch to reach full KL weight"
    )
    
    # Loss configuration - toggles for each loss component
    enable_kl_loss: bool = Field(
        default=True,
        description="Whether to enable KL divergence loss"
    )
    enable_l1_loss: bool = Field(
        default=True,
        description="Whether to enable L1 regularization"
    )
    enable_l2_loss: bool = Field(
        default=True,
        description="Whether to enable L2 regularization"
    )
    enable_reconstruction_loss: bool = Field(
        default=True,
        description="Whether to enable reconstruction loss"
    )
    
    # Loss weights
    l1_lambda: float = Field(
        default=0.0001,
        ge=0.0,
        le=1.0,
        description="L1 regularization weight"
    )
    l2_lambda: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="L2 regularization weight"
    )
    kl_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="KL divergence weight (final weight after annealing)"
    )
    
    # Training control
    gradient_clip_norm: Optional[float] = Field(
        default=None,
        ge=0.1,
        description="Gradient clipping norm (None to disable)"
    )
    early_stopping_patience: Optional[int] = Field(
        default=None,
        ge=1,
        description="Early stopping patience (None to disable)"
    )
    early_stopping_min_delta: float = Field(
        default=1e-4,
        ge=0.0,
        description="Minimum improvement for early stopping"
    )
    
    # Data loading
    num_workers: int = Field(
        default=0,
        ge=0,
        le=16,
        description="Number of data loading workers"
    )
    pin_memory: bool = Field(
        default=False,
        description="Whether to pin memory for GPU training"
    )
    val_split: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Validation split ratio"
    )
    test_split: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Test split ratio"
    )
    
    # Checkpointing
    checkpoint_path: str = Field(
        default="checkpoints/birdsong_lfads.pt",
        description="Path to save checkpoints"
    )
    save_every: int = Field(
        default=10,
        ge=1,
        description="Save checkpoint every N epochs"
    )
    keep_best: bool = Field(
        default=True,
        description="Whether to keep best checkpoint"
    )
    max_checkpoints: int = Field(
        default=5,
        ge=1,
        description="Maximum number of checkpoints to keep"
    )
    
    # Logging and visualization
    print_every: int = Field(
        default=10,
        ge=1,
        description="Print progress every N steps"
    )
    plot_dir: str = Field(
        default="plots",
        description="Directory for training plots"
    )
    disable_tqdm: bool = Field(
        default=False,
        description="Whether to disable progress bars"
    )
    log_tensorboard: bool = Field(
        default=False,
        description="Whether to log to TensorBoard"
    )
    log_wandb: bool = Field(
        default=False,
        description="Whether to log to Weights & Biases"
    )
    
    # Advanced training options
    mixed_precision: bool = Field(
        default=False,
        description="Whether to use mixed precision training"
    )
    deterministic: bool = Field(
        default=False,
        description="Whether to use deterministic training"
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Random seed for reproducibility"
    )
    
    @field_validator("kl_full_epoch")
    @classmethod
    def validate_kl_epochs(cls, v, info):
        if hasattr(info, 'data') and "kl_start_epoch" in info.data and v <= info.data["kl_start_epoch"]:
            raise ValueError("kl_full_epoch must be greater than kl_start_epoch")
        return v
    
    @model_validator(mode='after')
    def validate_splits(self):
        val_split = self.val_split
        test_split = self.test_split
        if val_split + test_split >= 1.0:
            raise ValueError("val_split + test_split must be less than 1.0")
        return self


class EvaluationConfig(BaseConfig):
    """Configuration for model evaluation."""
    
    # Evaluation parameters
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to model checkpoint for evaluation"
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        le=1024,
        description="Evaluation batch size"
    )
    num_plots: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Number of prediction/evolution plots to generate during evaluation"
    )
    num_samples: int = Field(
        default=10,
        ge=1,
        le=10000,
        description="Number of samples to evaluate (for metrics and summary statistics)"
    )
    
    # Test set evaluation
    use_test_set: bool = Field(
        default=False,
        description="Whether to use test set from checkpoint instead of random samples (recommended for proper evaluation)"
    )
    
    # Analysis options - detailed toggles for each analysis type
    analyze_reconstructions: bool = Field(
        default=True,
        description="Whether to analyze reconstructions"
    )
    analyze_transitions: bool = Field(
        default=True,
        description="Whether to analyze transition matrices"
    )
    analyze_latents: bool = Field(
        default=False,
        description="Whether to perform latent space analysis"
    )
    analyze_factors: bool = Field(
        default=False,
        description="Whether to analyze factor dynamics"
    )
    analyze_trajectories: bool = Field(
        default=False,
        description="Whether to analyze latent trajectories"
    )
    analyze_kl_divergence: bool = Field(
        default=True,
        description="Whether to compute KL divergence metrics"
    )
    analyze_cross_entropy: bool = Field(
        default=True,
        description="Whether to compute cross-entropy metrics"
    )
    analyze_accuracy: bool = Field(
        default=True,
        description="Whether to compute accuracy metrics"
    )
    analyze_js_divergence: bool = Field(
        default=True,
        description="Whether to compute Jensen-Shannon divergence"
    )
    
    # Visualization options - detailed toggles for each plot type
    plot_samples: bool = Field(
        default=True,
        description="Whether to plot sample outputs"
    )
    plot_metrics: bool = Field(
        default=True,
        description="Whether to plot evaluation metrics"
    )
    plot_latents: bool = Field(
        default=False,
        description="Whether to plot latent trajectories"
    )
    plot_transitions: bool = Field(
        default=True,
        description="Whether to plot transition matrices"
    )
    plot_factors: bool = Field(
        default=False,
        description="Whether to plot factor dynamics"
    )
    plot_reconstructions: bool = Field(
        default=True,
        description="Whether to plot reconstruction comparisons"
    )
    plot_summary_metrics: bool = Field(
        default=True,
        description="Whether to plot summary statistics"
    )
    plot_individual_samples: bool = Field(
        default=True,
        description="Whether to plot individual sample analyses"
    )
    
    # Output configuration
    output_dir: str = Field(
        default="evaluation_results",
        description="Directory for evaluation outputs"
    )
    save_reconstructions: bool = Field(
        default=False,
        description="Whether to save reconstruction data"
    )
    export_format: Literal["png", "pdf", "svg"] = Field(
        default="png",
        description="Plot export format"
    )
    export_svg: bool = Field(
        default=True,
        description="Whether to also export plots as SVG"
    )
    export_pdf: bool = Field(
        default=False,
        description="Whether to also export plots as PDF"
    )
    dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="DPI for plot exports"
    )
    
    # Detailed analysis configuration
    smooth_window: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Window size for smoothing estimates"
    )
    compute_per_timestep: bool = Field(
        default=True,
        description="Whether to compute metrics per timestep"
    )
    compute_summary_stats: bool = Field(
        default=True,
        description="Whether to compute summary statistics"
    )
    save_detailed_metrics: bool = Field(
        default=True,
        description="Whether to save detailed per-sample metrics"
    )
    
    # Export options
    export_results: bool = Field(
        default=False,
        description="Export results to JSON and CSV formats"
    )
    export_plots: bool = Field(
        default=True,
        description="Whether to export plots"
    )
    export_metrics: bool = Field(
        default=True,
        description="Whether to export metrics to files"
    )
    export_latents: bool = Field(
        default=False,
        description="Whether to export latent representations"
    )
    
    # Advanced evaluation options
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU for evaluation"
    )
    deterministic_eval: bool = Field(
        default=False,
        description="Whether to use deterministic evaluation"
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Random seed for evaluation reproducibility"
    )
    
    # Latent analysis performance settings
    max_latent_analysis_samples: int = Field(
        default=500,
        ge=100,
        le=10000,
        description="Maximum number of samples to use for latent analysis"
    )
    skip_tsne_for_large_datasets: bool = Field(
        default=True,
        description="Whether to skip t-SNE for datasets larger than 1000 samples"
    )
    latent_analysis_batch_size: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Batch size for latent extraction during analysis"
    )
    
    @field_validator("checkpoint_path")
    @classmethod
    def validate_checkpoint(cls, v):
        import os
        if v is not None and not os.path.exists(v):
            raise ValueError(f"Checkpoint file not found: {v}")
        return v


class ExperimentConfig(BaseConfig):
    """Configuration for complete experiments with metadata."""
    
    # Experiment metadata
    name: str = Field(
        description="Experiment name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Experiment description"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Experiment tags for organization"
    )
    
    # Reproducibility
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Random seed for reproducibility"
    )
    
    # Execution options
    run_data_generation: bool = Field(
        default=True,
        description="Whether to generate data"
    )
    run_training: bool = Field(
        default=True,
        description="Whether to run training"
    )
    run_evaluation: bool = Field(
        default=True,
        description="Whether to run evaluation"
    )
    
    # Output configuration
    output_dir: str = Field(
        default="experiments",
        description="Base directory for experiment outputs"
    )
    save_config: bool = Field(
        default=True,
        description="Whether to save configuration"
    )
    
    # Resource configuration
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Device for computation"
    )
    max_memory_gb: Optional[float] = Field(
        default=None,
        ge=0.1,
        description="Maximum memory usage in GB"
    )


class BirdsongConfig(BaseConfig):
    """Main configuration class combining all components."""
    
    # Configuration metadata
    version: str = Field(
        default="1.0",
        description="Configuration schema version"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Configuration creation timestamp"
    )
    
    # Inheritance metadata
    inherits_from: Optional[List[str]] = Field(
        default=None,
        description="List of configuration files this inherits from"
    )
    
    # Component configurations
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data generation and loading configuration"
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model architecture configuration"
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration"
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Evaluation configuration"
    )
    experiment: Optional[ExperimentConfig] = Field(
        default=None,
        description="Experiment configuration (optional)"
    )
    
    @model_validator(mode='after')
    def validate_config_consistency(self):
        """Validate cross-component consistency."""
        data_config = self.data
        model_config = self.model
        
        if data_config and model_config:
            # If model alphabet_size is manually set, use it to update data alphabet
            if model_config.alphabet_size is not None:
                # Generate appropriate alphabet symbols for the specified size
                from ..data.loader import generate_alphabet_symbols
                data_config.alphabet = generate_alphabet_symbols(model_config.alphabet_size)
            else:
                # Auto-derive alphabet_size from data if not set
                model_config.alphabet_size = len(data_config.alphabet)
            
            # Validate consistency (should always pass now)
            if model_config.alphabet_size != len(data_config.alphabet):
                raise ValueError(
                    f"Model alphabet_size ({model_config.alphabet_size}) "
                    f"doesn't match data alphabet length ({len(data_config.alphabet)})"
                )
            
            if model_config.order != data_config.order:
                raise ValueError(
                    f"Model order ({model_config.order}) "
                    f"doesn't match data order ({data_config.order})"
                )
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict(exclude_none=True)
    
    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        import yaml
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BirdsongConfig":
        """Create configuration from dictionary."""
        try:
            return cls(**data)
        except PydanticValidationError as e:
            raise ConfigValidationError(
                "Configuration validation failed",
                errors=[error for error in e.errors()]
            ) 