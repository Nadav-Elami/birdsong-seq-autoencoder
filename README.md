# Birdsong Sequential Autoencoder

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-000000.svg)](https://github.com/charliermarsh/ruff)

A research-grade Python package for LFADS-style sequential autoencoder analysis of birdsong syntax and neural dynamics. This package provides comprehensive tools for data generation, model training, evaluation, and advanced latent space analysis of birdsong sequences.

## About

Developed at **The Neural Syntax Lab** at the [Weizmann Institute of Science](https://www.weizmann.ac.il/brain-sciences/labs/cohen/), this package implements sequential autoencoders inspired by the LFADS (Latent Factor Analysis via Dynamical Systems) framework for analyzing the temporal structure and syntax of birdsong.

## Installation

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]
```

## Quick Start

### Training a Model

```bash
# Train with default configuration
birdsong-train --config configs/default.yaml

# Train with custom parameters
birdsong-train --config configs/default.yaml --epochs 100 --batch-size 32
```

### Evaluating a Model

```bash
# Evaluate a trained checkpoint
birdsong-eval --checkpoint checkpoints/model_epoch_50.pt

# Generate plots and metrics
birdsong-eval --checkpoint checkpoints/model_epoch_50.pt --output-dir results/
```

### Analyzing Latent Representations

```bash
# Analyze latent space with PCA
birdsong-analyze --checkpoint checkpoints/model_epoch_50.pt --data-path data.h5 --reduction-method pca

# Analyze with t-SNE and clustering
birdsong-analyze --checkpoint checkpoints/model_epoch_50.pt --data-path data.h5 --reduction-method tsne --clustering-method kmeans --n-clusters 5

# Interactive analysis with UMAP
birdsong-analyze --checkpoint checkpoints/model_epoch_50.pt --data-path data.h5 --reduction-method umap --interactive
```

### Python API

```python
from birdsong import BirdsongLFADSModel2, BirdsongDataset, BirdsongTrainer
from birdsong.analysis import LatentSpaceAnalyzer

# Load data
dataset = BirdsongDataset("path/to/data.h5")

# Create model
model = BirdsongLFADSModel2(
    input_size=dataset.input_size,
    hidden_size=128,
    latent_size=32
)

# Train model
trainer = BirdsongTrainer(model, dataset)
trainer.train(epochs=100, batch_size=32)

# Analyze latent representations
analyzer = LatentSpaceAnalyzer(model, device)
analysis_results = analyzer.analyze_latent_space(
    latent_type='factors',
    reduction_method='pca',
    clustering_method='kmeans'
)
```

## Package Structure

```
src/birdsong/
├── data/           # Data loaders and simulation
├── models/         # LFADS model and loss functions
├── training/       # Training loop and utilities
├── evaluation/     # Evaluation and plotting
├── analysis/       # Latent space analysis tools
└── cli/           # Command-line interfaces
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/birdsong

# Run specific test file
pytest tests/test_regression.py
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/
```

## Documentation

- `docs/motivation.md` - Research motivation and background
- `docs/architecture.md` - Package architecture and design decisions

## Acknowledgments

This work was inspired by and builds upon the LFADS framework:

> Pandarinath, C., O'Shea, D.J., Collins, J., Jozefowicz, R., Stavisky, S.D., Kao, J.C., Trautmann, E.M., Kaufman, M.T., Ryu, S.I., Hochberg, L.R. and Henderson, J.M., 2018. [Inferring single-trial neural population dynamics using sequential auto-encoders](https://www.nature.com/articles/s41592-018-0109-9). *Nature methods*, 15(10), pp.805-815.

Special thanks to the broader birdsong research community and the neural computation community for their foundational work in understanding the neural basis of learned vocal behavior.

## Lab & Contact

**The Neural Syntax Lab**  
Weizmann Institute of Science  
Department of Brain Sciences  

**Lab Website**: https://www.weizmann.ac.il/brain-sciences/labs/cohen/  
**Lab GitHub**: [@NeuralSyntaxLab](https://github.com/NeuralSyntaxLab)

**Author**: Nadav Elami (nadav.elami@weizmann.ac.il)  
**GitHub**: [@Nadav-Elami](https://github.com/Nadav-Elami)

For bug reports, feature requests, or research collaborations, please open an issue or contact us directly.

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Nadav Elami, The Neural Syntax Lab, Weizmann Institute of Science 