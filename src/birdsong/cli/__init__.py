"""
Command-line interfaces for birdsong package.

This package provides comprehensive CLI commands for birdsong research including:
- birdsong-generate: Generate synthetic birdsong data
- birdsong-train: Train LFADS models with hierarchical configuration
- birdsong-eval: Evaluate trained models with analysis
- birdsong-experiment: Orchestrate complete research experiments

All commands support:
- Hierarchical YAML configuration with templates
- Automatic seed tracking and reproducibility
- Dry-run validation mode
- Enhanced error handling and help documentation
"""

from .eval import main as eval_main
from .eval_enhanced import main as eval_enhanced_main
from .experiment import main as experiment_main
from .generate import main as generate_main
from .train import main as train_main
from .train_enhanced import main as train_enhanced_main

__all__ = [
    "eval_main",
    "eval_enhanced_main", 
    "experiment_main",
    "generate_main",
    "train_main",
    "train_enhanced_main",
]
