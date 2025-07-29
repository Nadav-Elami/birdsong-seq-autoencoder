# Contributing to Birdsong Sequential Autoencoder

Thank you for your interest in contributing to the Birdsong Sequential Autoencoder project! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)
- [Research Collaboration](#research-collaboration)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- PyTorch 2.0+ compatible system (CPU or CUDA-enabled GPU recommended)

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/Nadav-Elami/birdsong-seq-autoencoder.git
   cd birdsong-seq-autoencoder
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e .[dev]
   ```

4. **Verify Installation**
   ```bash
   pytest tests/
   birdsong-train --help
   ```

## Code Style and Standards

### Python Code Style

We follow **PEP-8** standards with the following tools:

- **Formatter**: [Black](https://black.readthedocs.io/) (line length: 88)
- **Linter**: [Ruff](https://docs.astral.sh/ruff/) 
- **Type Hints**: Required for all public APIs

#### Running Code Formatting

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Fix auto-fixable linting issues
ruff check --fix src/ tests/
```

### File Organization

Follow the established package structure:

```
src/birdsong/
├── data/           # Data loaders and simulation
├── models/         # LFADS model and loss functions  
├── training/       # Training loop and utilities
├── evaluation/     # Evaluation and plotting
├── analysis/       # Latent space analysis tools
├── cli/            # Command-line interfaces
├── config/         # Configuration system
├── experiments/    # Experiment orchestration
└── utils/          # Utilities and reproducibility
```

### Documentation Standards

- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Type Hints**: Mandatory for all public APIs
- **Comments**: Explain *why* not *what* - focus on algorithmic choices and research rationale

Example:
```python
def compute_latent_factors(
    neural_data: torch.Tensor, 
    model: BirdsongLFADSModel2
) -> torch.Tensor:
    """
    Extract latent factors from neural data using trained LFADS model.

    Args:
        neural_data (Tensor): Input neural data, shape (batch, time, neurons)
        model (BirdsongLFADSModel2): Trained LFADS model

    Returns:
        Tensor: Latent factors, shape (batch, time, latent_dim)
        
    Raises:
        ValueError: If neural_data dimensions don't match model expectations
    """
```

## Testing

### Test Requirements

- **Coverage Target**: ≥90% for core functionality
- **Test Types**: Unit tests, integration tests, regression tests
- **Framework**: pytest

### Test Structure

Tests mirror the source structure:
```
tests/
├── test_data.py          # Data loading and simulation tests
├── test_models.py        # Model architecture tests  
├── test_training.py      # Training loop tests
├── test_evaluation.py    # Evaluation and plotting tests
├── test_analysis.py      # Analysis functionality tests
├── integration/          # End-to-end pipeline tests
└── performance/          # Performance and memory tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/birdsong --cov-report=html

# Run specific test categories
pytest tests/integration/
pytest tests/performance/

# Run tests with GPU (if available)
pytest tests/gpu/
```

### Test Writing Guidelines

Each new feature should include:

1. **Happy Path Test**: Normal usage scenario
2. **Edge Case Test**: Boundary conditions and unusual inputs  
3. **Failure Test**: Error handling and validation

## Submitting Changes

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes** following code style and testing requirements

3. **Run Full Test Suite**
   ```bash
   pytest
   black src/ tests/
   ruff check src/ tests/
   ```

4. **Update Documentation** if needed (README, docstrings, etc.)

5. **Commit Changes** with descriptive messages:
   ```bash
   git commit -m "Add latent space clustering analysis

   - Implement K-means and DBSCAN clustering for latent factors
   - Add silhouette score evaluation metrics  
   - Include visualization functions for cluster analysis
   - Add comprehensive test suite for clustering functionality"
   ```

6. **Create Pull Request** with:
   - Clear description of changes
   - References to related issues
   - Test results and coverage information

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Include detailed description for complex changes

## Issue Reporting

### Bug Reports

Please include:

- **Environment Details**: OS, Python version, PyTorch version
- **Reproducible Example**: Minimal code that demonstrates the issue
- **Expected vs Actual Behavior**: Clear description of what went wrong
- **Error Messages**: Full traceback if applicable
- **Data Information**: Dataset characteristics (if relevant)

### Feature Requests

Please include:

- **Research Motivation**: Why this feature would be valuable
- **Use Case**: Specific research scenarios where it would be applied
- **Proposed Implementation**: Ideas for how it could be implemented
- **Related Work**: Any relevant papers or existing implementations

## Research Collaboration

### Academic Use

This package is designed for academic research. If you use it in your research:

1. **Cite the Package**: Include appropriate citations in your papers
2. **Share Results**: We're interested in hearing about your research applications
3. **Contribute Back**: Consider contributing improvements or new features

### Data and Model Sharing

- **Datasets**: Large datasets should not be committed to git - use Git LFS or external storage
- **Trained Models**: Share checkpoints via appropriate platforms (not git)
- **Results**: Share analysis results and plots in appropriate formats

## Development Philosophy

### Research-First Approach

- **Reproducibility**: Every experiment must be fully reproducible
- **Modularity**: Components should work independently and in combination
- **Scientific Rigor**: Maintain high standards for mathematical correctness
- **User Experience**: Balance power with ease of use

### Performance Considerations

- **Memory Efficiency**: Support large datasets and long sequences
- **GPU Utilization**: Efficient use of available hardware
- **Computational Speed**: Optimize bottlenecks without sacrificing clarity

## Questions?

- **Technical Questions**: Open an issue with the "question" label
- **Research Collaboration**: Contact Nadav Elami (nadav.elami@weizmann.ac.il)  
- **Lab Information**: Visit [The Neural Syntax Lab](https://www.weizmann.ac.il/brain-sciences/labs/cohen/)

Thank you for contributing to advancing our understanding of neural syntax and birdsong dynamics! 