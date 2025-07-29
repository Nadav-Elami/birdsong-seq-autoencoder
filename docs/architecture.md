# Birdsong Package Architecture

## Package Structure

The birdsong package follows a modular design organized by functionality:

```
src/birdsong/
├── __init__.py          # Package initialization and main imports
├── data/                # Data generation and loading
│   ├── __init__.py
│   ├── generation.py    # Simulation functions and BirdsongSimulator
│   ├── aggregation.py   # Process functions and BirdsongAggregator
│   └── loader.py        # BirdsongDataset for HDF5 data
├── models/              # Neural network models
│   ├── __init__.py
│   ├── lfads.py        # BirdsongLFADSModel2 implementation
│   └── loss.py         # Loss functions and utilities
├── training/            # Training utilities
│   ├── __init__.py
│   └── trainer.py      # BirdsongTrainer class
├── evaluation/          # Evaluation and visualization
│   ├── __init__.py
│   └── evaluate.py     # BirdsongEvaluator and plotting functions
└── cli/                # Command-line interfaces
    ├── __init__.py
    ├── train.py        # Training CLI
    └── eval.py         # Evaluation CLI
```

## Design Principles

### 1. Modularity
Each module has a single responsibility:
- **Data modules**: Handle data generation, loading, and preprocessing
- **Model modules**: Define neural network architectures
- **Training modules**: Manage training loops and optimization
- **Evaluation modules**: Provide metrics and visualization
- **CLI modules**: Expose functionality through command-line interfaces

### 2. Type Safety
All public APIs use type hints to ensure correctness:
```python
def simulate_birdsong(
    num_batches: int,
    batch_size: int,
    seq_range: Tuple[int, int],
    alphabet: List[str],
    num_processes: int,
    order: int = 1,
    process_fn: Optional[Callable] = None
) -> Tuple[np.ndarray, np.ndarray]:
```

### 3. Error Handling
Comprehensive error handling with informative messages:
```python
if '<' not in alphabet or '>' not in alphabet:
    raise ValueError("Alphabet must contain '<' and '>' symbols")
```

### 4. Documentation
Google-style docstrings for all public functions and classes:
```python
def simulate_birdsong(...):
    """
    Simulate song sequences and compute n-gram counts and probability distributions.
    
    Args:
        num_batches: Number of time steps (batches) per process
        batch_size: Number of sequences per batch
        # ... other parameters
        
    Returns:
        Tuple of (ngram_counts, probabilities) tensors
    """
```

## Core Components

### Data Generation (`data/generation.py`)

The data generation module provides:

1. **Simulation Functions**:
   - `simulate_birdsong()`: Main simulation pipeline
   - `x_init_maker()`: Create initial transition matrices
   - `simulate_one_song_order1/2()`: Single sequence simulation

2. **BirdsongSimulator Class**:
   - High-level interface for data generation
   - Configurable parameters and validation
   - Support for custom process functions

3. **HDF5 Integration**:
   - `save_to_hdf5()`: Save simulation results
   - `preprocess_simulated_songs()`: Complete pipeline

### Data Aggregation (`data/aggregation.py`)

The aggregation module provides:

1. **Process Functions**:
   - Linear, nonlinear, and stochastic dynamics
   - Fourier series updates
   - Sparse transition processes

2. **BirdsongAggregator Class**:
   - Multi-process data generation
   - Process type registry
   - Metadata handling

3. **Utility Functions**:
   - `aggregate_data()`: Combine multiple process types
   - `save_aggregated_data()`: Save with metadata

### Model Architecture (`models/lfads.py`)

The LFADS model implements:

1. **Encoder Network**:
   - Bidirectional GRU for sequence encoding
   - Produces initial condition parameters (μ_g0, logvar_g0)

2. **Controller Network**:
   - RNN for inferring latent inputs
   - Uses encoded sequence and approximate factors
   - Produces inferred input parameters (μ_u, logvar_u)

3. **Generator Network**:
   - Bidirectional GRU for factor generation
   - Maps from latent space to factor space
   - Produces transition logits

4. **Output Layer**:
   - Row-wise softmax for transition probabilities
   - Supports first and second-order Markov processes

### Training Pipeline (`training/trainer.py`)

The training module provides:

1. **BirdsongTrainer Class**:
   - Configurable training parameters
   - Checkpointing and validation
   - Progress tracking and visualization

2. **Training Loop**:
   - Mini-batch processing
   - Loss computation and backpropagation
   - Learning rate scheduling

3. **Utilities**:
   - Data loading and preprocessing
   - Model evaluation
   - Logging and metrics

### Evaluation (`evaluation/evaluate.py`)

The evaluation module provides:

1. **BirdsongEvaluator Class**:
   - Model evaluation and metrics
   - Visualization functions
   - Results analysis

2. **Plotting Functions**:
   - `plot_ngram_counts()`: N-gram frequency analysis
   - `plot_transition_plots()`: Transition matrix visualization
   - `plot_summary_metrics()`: Training and evaluation metrics

3. **Metrics**:
   - Reconstruction accuracy
   - KL divergence tracking
   - Temporal consistency measures

## Data Flow

### 1. Data Generation
```
Alphabet + Parameters → Markov Process → N-gram Counts → HDF5 File
```

### 2. Model Training
```
HDF5 File → BirdsongDataset → BirdsongTrainer → Trained Model
```

### 3. Evaluation
```
Trained Model + Test Data → BirdsongEvaluator → Metrics + Plots
```

## Configuration Management

### 1. Model Configuration
Model parameters are specified during initialization:
```python
model = BirdsongLFADSModel2(
    alphabet_size=7,
    order=1,
    encoder_dim=64,
    controller_dim=64,
    generator_dim=64,
    factor_dim=32,
    latent_dim=16,
    inferred_input_dim=8,
    kappa=1.0,
    ar_step_size=0.99,
    ar_process_var=0.1
)
```

### 2. Training Configuration
Training parameters are set in the trainer:
```python
trainer = BirdsongTrainer(
    model=model,
    dataset=dataset,
    batch_size=32,
    learning_rate=1e-3,
    device='cuda'
)
```

### 3. CLI Configuration
Command-line tools support YAML configuration files:
```yaml
model:
  alphabet_size: 7
  order: 1
  encoder_dim: 64
  # ... other parameters

training:
  batch_size: 32
  learning_rate: 1e-3
  epochs: 100
  # ... other parameters
```

## Testing Strategy

### 1. Unit Tests
- Individual function testing
- Edge case handling
- Error condition validation

### 2. Integration Tests
- End-to-end pipeline testing
- Data flow validation
- CLI functionality testing

### 3. Regression Tests
- Numerical precision validation
- Consistency checking
- Performance benchmarking

## Performance Considerations

### 1. Memory Management
- Efficient data loading with HDF5
- Batch processing for large datasets
- Gradient accumulation for large models

### 2. Computational Optimization
- GPU acceleration where available
- Vectorized operations
- Efficient loss computation

### 3. Scalability
- Modular design for easy extension
- Configurable parameters
- Support for distributed training

## Extension Points

### 1. New Process Types
Add new process functions to `data/aggregation.py`:
```python
def custom_process(x, A, t):
    # Custom implementation
    return updated_x

PROCESS_FUNCTIONS["custom"] = custom_process
```

### 2. New Model Architectures
Extend the models module with new architectures:
```python
class CustomModel(nn.Module):
    def __init__(self, ...):
        # Custom implementation
        pass
```

### 3. New Evaluation Metrics
Add metrics to the evaluation module:
```python
def custom_metric(predictions, targets):
    # Custom metric implementation
    return metric_value
```

## Best Practices

### 1. Code Organization
- Keep functions focused and single-purpose
- Use descriptive names and clear documentation
- Follow PEP 8 style guidelines

### 2. Error Handling
- Validate inputs early and often
- Provide informative error messages
- Use appropriate exception types

### 3. Testing
- Write tests for all public APIs
- Test edge cases and error conditions
- Maintain high test coverage

### 4. Documentation
- Document all public functions and classes
- Include usage examples
- Keep documentation up to date

This architecture provides a solid foundation for birdsong research while maintaining flexibility for future extensions and improvements. 