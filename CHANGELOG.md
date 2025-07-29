# Changelog

All notable changes to the Birdsong Sequential Autoencoder project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Hyperparameter optimization with Optuna integration
- Interactive training dashboards
- Distributed training support
- Performance profiling tools
- Advanced model comparison framework

## [0.1.0] - 2025-01-14

### Added

#### 🔒 Reproducibility Framework
- **Global seed management system** (`src/birdsong/utils/reproducibility.py`)
  - `SeedManager` class for centralized seed management across torch, numpy, and Python's random
  - `set_global_seed()` function for easy reproducible experiment setup
  - Environment fingerprinting with system info, package versions, and hardware details
  - Deterministic mode enabling for PyTorch operations
  - Data integrity verification with SHA256 hashing
  - Comprehensive test suite (21 tests, 100% passing)

#### ⚙️ Configuration System
- **Hierarchical YAML configuration schema** with Pydantic v2 validation
  - 5 main configuration classes: DataConfig, ModelConfig, TrainingConfig, EvaluationConfig, ExperimentConfig
  - Multi-level config inheritance with `inherits_from` support
  - Environment variable substitution (`${VAR:default}` syntax)
  - Cross-component validation and runtime validation
  - Dot notation overrides (`model.encoder_dim: 128`)
  - 6 research scenario templates covering common use cases
  - Error handling with detailed `ConfigValidationError` messages

#### 🎪 Experiment Orchestration
- **Modular experiment components** with standalone capability
  - `ExperimentRunner` for full pipeline orchestration (data→train→eval)
  - Resume functionality with state persistence and recovery
  - Progress tracking with human-readable duration strings
  - Automatic result archiving with comprehensive metadata
  - Stage-wise execution with dependency management

#### 🖥️ Command-Line Interface
- **Comprehensive CLI commands** with enhanced functionality:
  - `birdsong-experiment`: Complete pipeline orchestration (run/resume/list/status)
  - `birdsong-train`: Enhanced training with resumable checkpoints and config integration
  - `birdsong-eval`: Evaluation with analysis options and test set support
  - `birdsong-generate`: Data generation with validation and plotting
  - `birdsong-analyze`: Latent space analysis with multiple visualization methods
  - All commands support dry-run mode, comprehensive help, and error guidance
  - Template support alongside configuration files
  - Automatic seed tracking and metadata logging

#### 🔍 Test Set Evaluation System
- **Proper test set evaluation** with checkpoint consistency
  - Test indices saved in training checkpoints for reproducible evaluation
  - `create_test_loader_from_checkpoint()` function for exact test set reconstruction
  - CLI support via `--use-test-set` flag
  - Configuration file support with `use_test_set: true` setting
  - Graceful fallback and comprehensive error handling

#### 🧠 Latent Space Analysis Suite
- **Comprehensive latent space analysis tools** (`src/birdsong/analysis/latent.py`)
  - Dimensionality reduction: PCA, t-SNE, UMAP with customizable parameters
  - Cluster analysis: K-means, DBSCAN with quality metrics (silhouette, Calinski-Harabasz)
  - Trajectory analysis for sequence data with curvature and length statistics  
  - Interactive exploration widgets using Plotly (optional dependency)
  - `LatentSpaceAnalyzer` class for unified analysis workflow
  - Support for all LFADS latent types (factors, g0, u)
  - Automatic visualization generation and result saving
  - Integration with config toggles and CLI commands

#### 🧪 Enhanced Testing Framework
- **Comprehensive test coverage** across all components
  - Unit tests for all major functionality with >90% coverage target
  - Integration tests for full pipeline workflows
  - Performance tests for memory usage and GPU utilization
  - Scientific validation tests for mathematical correctness
  - Cross-platform compatibility testing

#### 📚 Package Infrastructure
- **Research-grade package structure** following best practices
  - Modular organization: data → models → training → evaluation → analysis
  - Hard 500-line file limit with automatic splitting recommendations
  - Relative imports within package (`from ..models import lfads`)
  - Type hints mandatory on all public APIs
  - Google-style docstrings throughout

### Technical Specifications

- **Python**: 3.10+ with full type hint support
- **Dependencies**: PyTorch 2.0+, NumPy, HDF5, Matplotlib, scikit-learn, Pydantic v2
- **Code Style**: Black formatter (88 char line length), Ruff linter, PEP-8 compliance
- **Testing**: pytest with coverage reporting, CI/CD ready
- **Documentation**: Comprehensive docstrings, README, and example workflows

### Research Features

- **LFADS-inspired Architecture**: Sequential autoencoder for birdsong syntax analysis
- **Birdsong Data Simulation**: Configurable synthetic birdsong generation with realistic syntax patterns
- **Neural Dynamics Modeling**: Latent factor analysis for understanding temporal neural representations
- **Reproducible Experiments**: Full reproducibility across platforms with seed management
- **Scientific Analysis**: Publication-ready visualizations and statistical analysis tools

### Breaking Changes

- Complete rewrite from original example scripts to research-grade package
- New configuration system replaces ad-hoc parameter passing
- Enhanced CLI replaces legacy command interfaces
- Standardized file naming and output organization

### Migration Guide

For users migrating from original example scripts:

1. **Configuration**: Convert script parameters to YAML config files using provided templates
2. **CLI Usage**: Use new enhanced CLI commands (`birdsong-train`, `birdsong-eval`, etc.)
3. **Output Structure**: Outputs now organized in timestamped directories with full metadata
4. **Analysis**: Use new `birdsong-analyze` command for latent space analysis instead of manual scripts

---

**Full Documentation**: See README.md and docs/ directory for complete usage examples and API documentation.

**Research Applications**: This package is designed for computational neuroscience research on birdsong syntax, neural dynamics, and sequence learning.

**Lab Information**: Developed at The Neural Syntax Lab, Weizmann Institute of Science
- **Lab Website**: https://www.weizmann.ac.il/brain-sciences/labs/cohen/
- **Lab GitHub**: [@NeuralSyntaxLab](https://github.com/NeuralSyntaxLab) 