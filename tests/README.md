# Test Organization

This directory contains organized test files for the Birdsong LFADS project.

## Directory Structure

### `/tests/performance/`
- **`simple_performance_test.py`** - Comprehensive performance testing for data loading optimization
- **`test_performance.py`** - Data loader performance testing

### `/tests/gpu/`
- **`simple_gpu_test.py`** - GPU vs CPU speed comparison with performance metrics
- **`test_gpu_usage.py`** - Verifies GPU is actually being used during training

### `/tests/integration/`
- **`simple_test.py`** - Basic functionality test for imports and model creation
- **`test_eval.py`** - Evaluation functionality testing
- **`test_fixes.py`** - Tests PyTorch warnings and test indices fixes
- **`test_resume.py`** - Tests resume functionality with optimized configuration

## Test Categories

### Performance Tests
- Data loading speed optimization
- Batch processing efficiency
- Memory usage optimization

### GPU Tests
- CUDA availability verification
- GPU vs CPU performance comparison
- Training device verification

### Integration Tests
- Basic functionality verification
- Evaluation pipeline testing
- Import and model creation tests
- Fix verification (PyTorch warnings, test indices)
- Resume training functionality

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/performance/
python -m pytest tests/gpu/
python -m pytest tests/integration/

# Run individual tests
python tests/performance/simple_performance_test.py
python tests/gpu/simple_gpu_test.py
python tests/integration/test_fixes.py
python tests/integration/test_resume.py
```

## Test Maintenance

- **Performance tests** should be run when optimizing data loading
- **GPU tests** should be run when setting up new environments
- **Integration tests** should be run after major code changes
- **Fix verification tests** should be run after implementing fixes

## Main Directory Clean

All test files have been moved from the main directory to the organized test structure. The main directory now contains only:
- Essential project files (`run_CLI.py`, `task_board.md`, etc.)
- Project configuration (`pyproject.toml`, `README.md`)
- Source code (`src/`)
- Documentation (`docs/`)
- Configuration (`configs/`) 