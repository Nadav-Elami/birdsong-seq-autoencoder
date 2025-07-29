# Birdsong Sequential Autoencoder Package Refactor PRP

## Context

### Feature Summary
Build a research-grade Python package named **birdsongs-seq-autoencoder** that wraps existing LFADS-style birdsong code and exposes clean package layout, CLI utilities, minimal reproducible training, pytest regression tests, and context-engineering rules.

### Functional Requirements
- **Clean package layout** (`src/birdsong/...`) that keeps every line of original scripts but makes them importable
- **CLI utilities**:
  - `birdsong-train --config path/to.yaml`
  - `birdsong-eval --checkpoint ckpt.pt`
- **Minimal, reproducible training run** inside `examples/quickstart.ipynb`
- **Pytest regression tests** that confirm refactor preserves numerical results (loss within 1e-5 of legacy script on 10-step synthetic run)
- **Context-engineering rules for Cursor**:
  - Global rule: `.cursor/rules/project_overview.mdc` (already added)
  - Scoped rules: `data_pipeline.mdc`, `model_conventions.mdc`, `testing.mdc`
- **Docs stub** in `docs/` powered by Markdown; auto-generated API reference later is okay, but at minimum `motivation.md` and `architecture.md`

### Non-functional Requirements / Constraints
- **Research-only**—production hardening, cloud orchestration, and hyper-scale performance are out of scope
- **No numerical drift**: Unit tests must compare tensors from new package against outputs of original scripts
- **≤ 500 LoC per file**; split long legacy scripts if necessary
- **Strict dependency set**: `torch`, `numpy`, `h5py`, `matplotlib`, `tqdm`
- **Everything must install** with `pip install -e .` on Python 3.10+
- **CLI entry-points** must be registered in `pyproject.toml` under `[project.scripts]`
- **Common gotcha**: absolute imports inside old scripts—convert to relative imports to avoid circular dependencies

### Code-base Reconnaissance
- `examples/example code/neural netowrk model/birdsong_lfads_model_2.py` - Main LFADS model with rowwise softmax output (835 lines)
- `examples/example code/neural netowrk model/birdsong_training_2.py` - Training loop with validation, checkpointing, and CLI pattern (710 lines)
- `examples/example code/neural netowrk model/birdsong_data_loader.py` - PyTorch Dataset for HDF5 bigram data (87 lines)
- `examples/example code/neural netowrk model/birdsong_loss.py` - Loss functions with KL divergence and reconstruction loss (205 lines)
- `examples/example code/data generation/birdsong_data_generation.py` - Data simulation with Markov processes and HDF5 aggregation (795 lines)
- `examples/example code/data generation/Birdsong_Data_Aggregation.py` - Multi-process data aggregation with various transition functions (536 lines)
- `examples/example code/evaluation and plotting/birdsong_evaluation_3.py` - Evaluation and visualization scripts

### External Research Hooks
- **PyTorch LFADS reference**: https://github.com/google-research/lfads-torch (architecture inspiration)
- **Cursor rule format**: https://www.cursor.sh/docs/context-engineering (YAML front-matter spec)
- **CCN PDF**: `docs/Tracking_Time_Varying_Syntax_in_Birdsong_with_a_Sequential_Autoencoder_CCN.pdf` (abstract & methods drive doc pages)

## Implementation Blueprint

### 1. Package Structure Setup
- Create `src/birdsong/` directory structure:
  - `src/birdsong/__init__.py` - Package initialization
  - `src/birdsong/data/` - Data loaders and simulation
  - `src/birdsong/models/` - LFADS model and loss functions
  - `src/birdsong/training/` - Training loop and utilities
  - `src/birdsong/evaluation/` - Evaluation and plotting
- Create `pyproject.toml` with dependencies and CLI entry-points
- Create `README.md` with installation and usage instructions

### 2. Model Migration
- **File**: `src/birdsong/models/lfads.py`
- **Algorithm**: Extract `BirdsongLFADSModel2` class from `birdsong_lfads_model_2.py`
- **API Design**: Convert to relative imports, add type hints, Google docstrings
- **Error Handling**: Add input validation for model parameters

### 3. Data Pipeline Migration
- **File**: `src/birdsong/data/loader.py`
- **Algorithm**: Extract `BirdsongDataset` class from `birdsong_data_loader.py`
- **API Design**: Add data validation and error handling for HDF5 files
- **Error Handling**: Handle missing datasets and corrupted files

### 4. Data Generation Pipeline Migration
- **File**: `src/birdsong/data/generation.py`
- **Algorithm**: Extract simulation functions from `birdsong_data_generation.py`:
  - `simulate_birdsong()` - Main simulation with Markov processes
  - `x_init_maker()` - Transition matrix initialization
  - `save_to_hdf5()` - HDF5 data saving
  - `preprocess_simulated_songs()` - Complete simulation pipeline
- **API Design**: Create `BirdsongSimulator` class with configurable parameters
- **Error Handling**: Add validation for alphabet, sequence ranges, and process functions

- **File**: `src/birdsong/data/aggregation.py`
- **Algorithm**: Extract aggregation functions from `Birdsong_Data_Aggregation.py`:
  - Process function definitions (linear, cosine, fourier, etc.)
  - Multi-process data aggregation pipeline
  - Metadata handling and timestamp generation
- **API Design**: Create `BirdsongAggregator` class with process type configuration
- **Error Handling**: Add validation for process functions and aggregation parameters

### 5. Training Pipeline Migration
- **File**: `src/birdsong/training/trainer.py`
- **Algorithm**: Extract training function from `birdsong_training_2.py`
- **API Design**: Create `BirdsongTrainer` class with config-based initialization
- **Error Handling**: Add checkpoint validation and training recovery

### 6. Loss Functions Migration
- **File**: `src/birdsong/models/loss.py`
- **Algorithm**: Extract loss functions from `birdsong_loss.py`
- **API Design**: Create unified loss interface with configurable components
- **Error Handling**: Add numerical stability checks

### 7. CLI Implementation
- **File**: `src/birdsong/cli/train.py`
- **Algorithm**: Create argparse-based CLI using training function
- **API Design**: Support YAML config files and command-line overrides
- **Error Handling**: Validate config files and model checkpoints

- **File**: `src/birdsong/cli/eval.py`
- **Algorithm**: Create evaluation CLI using evaluation functions
- **API Design**: Support multiple checkpoint formats and output options
- **Error Handling**: Validate checkpoint files and output directories

### 8. Evaluation Pipeline Migration
- **File**: `src/birdsong/evaluation/evaluate.py`
- **Algorithm**: Extract evaluation functions from `birdsong_evaluation_3.py`
- **API Design**: Create `BirdsongEvaluator` class with configurable evaluation metrics
- **Error Handling**: Add validation for checkpoint files and output directories

### 9. Quickstart Notebook
- **File**: `examples/quickstart.ipynb`
- **Algorithm**: Import package, generate synthetic data, run one training epoch, plot reconstructions
- **API Design**: Demonstrate complete workflow from data to results
- **Error Handling**: Add cell-level error handling and progress indicators

### 10. Testing Framework
- **File**: `tests/test_regression.py`
- **Algorithm**: Compare outputs between original scripts and new package
- **API Design**: Use pytest fixtures for data and model setup
- **Error Handling**: Assert numerical precision within 1e-5 tolerance

### 11. Documentation
- **File**: `docs/motivation.md`
- **Algorithm**: Extract motivation from CCN PDF abstract
- **API Design**: Clear explanation of research goals and methodology

- **File**: `docs/architecture.md`
- **Algorithm**: Document package structure and design decisions
- **API Design**: Include diagrams and code examples

### 12. Cursor Rules
- **File**: `.cursor/rules/data_pipeline.mdc`
- **Algorithm**: Rules for data loading, simulation, and preprocessing patterns
- **API Design**: Consistent data validation, error handling, and HDF5 operations

- **File**: `.cursor/rules/model_conventions.mdc`
- **Algorithm**: Rules for model architecture and loss function patterns
- **API Design**: Consistent model initialization and forward pass patterns

- **File**: `.cursor/rules/testing.mdc`
- **Algorithm**: Rules for regression testing and numerical validation
- **API Design**: Consistent test structure and assertion patterns

## Validation Gates

Commands that *must* pass:

```bash
ruff check --fix
pytest -q
pip install -e .
birdsong-train --help
birdsong-eval --help
```

## Done Checklist

- [x] Package structure created with `src/birdsong/` layout
- [x] `pyproject.toml` configured with dependencies and CLI entry-points
- [x] All original scripts migrated to package modules with relative imports
- [x] Data generation pipeline migrated (`generation.py`, `aggregation.py`)
- [x] Evaluation pipeline migrated (`evaluate.py`)
- [x] CLI utilities `birdsong-train` and `birdsong-eval` implemented
- [x] `examples/quickstart.ipynb` created and tested
- [x] Pytest regression tests pass with ≤ 1e-5 numerical tolerance
- [x] Documentation files `docs/motivation.md` and `docs/architecture.md` created
- [x] Cursor rules files created and tested
- [x] Package installs successfully with `pip install -e .`
- [x] All tests pass with `pytest -q`
- [x] Code formatting passes with `ruff check --fix`
- [x] README.md badges updated once tests/docs are live

**Confidence to succeed in one pass: 8/10**

*Rationale: The original scripts are well-structured and the migration path is clear. The main challenges are ensuring numerical precision in regression tests and handling the CLI integration, but the modular approach should make this manageable.* 