"""
Comprehensive reproducibility framework for scientific experiments.

This module provides tools for ensuring reproducible experiments through:
- Global seed management across all RNG sources
- Environment fingerprinting for dependency tracking
- Deterministic operation enforcement
- Verification utilities for reproducibility testing
"""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class SeedManager:
    """
    Manages random seeds across all RNG sources for reproducible experiments.
    
    This class provides a centralized way to set, track, and verify seeds
    across torch, numpy, and Python's random module.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the seed manager.
        
        Args:
            seed (int, optional): Master seed for all RNG sources. If None,
                a random seed will be generated.
        """
        self._master_seed = seed if seed is not None else self._generate_seed()
        self._component_seeds: Dict[str, int] = {}
        self._is_set = False
        
    @staticmethod
    def _generate_seed() -> int:
        """Generate a random seed using OS entropy."""
        return int.from_bytes(os.urandom(4), byteorder='big') % (2**31)
    
    def set_seeds(self) -> Dict[str, int]:
        """
        Set seeds for all RNG sources using the master seed.
        
        Returns:
            Dict[str, int]: Mapping of component names to their assigned seeds.
        """
        # Use master seed to generate component-specific seeds deterministically
        rng = np.random.RandomState(self._master_seed)
        
        # Generate unique seeds for each component
        self._component_seeds = {
            'python_random': rng.randint(0, 2**31),
            'numpy': rng.randint(0, 2**31),
            'torch': rng.randint(0, 2**31),
            'torch_cuda': rng.randint(0, 2**31),
        }
        
        # Set seeds for all RNG sources
        random.seed(self._component_seeds['python_random'])
        np.random.seed(self._component_seeds['numpy'])
        torch.manual_seed(self._component_seeds['torch'])
        
        # Set CUDA seeds if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._component_seeds['torch_cuda'])
            torch.cuda.manual_seed_all(self._component_seeds['torch_cuda'])
        
        self._is_set = True
        return self._component_seeds.copy()
    
    def get_seed_info(self) -> Dict[str, Any]:
        """
        Get comprehensive seed information.
        
        Returns:
            Dict[str, Any]: Seed information including master seed and components.
        """
        return {
            'master_seed': self._master_seed,
            'component_seeds': self._component_seeds.copy(),
            'is_set': self._is_set,
        }
    
    @property
    def master_seed(self) -> int:
        """Get the master seed."""
        return self._master_seed


def set_global_seed(seed: Optional[int] = None) -> SeedManager:
    """
    Set global seed for all RNG sources and enable deterministic operations.
    
    This is the main entry point for setting up reproducible experiments.
    It handles torch, numpy, and Python's random module.
    
    Args:
        seed (int, optional): Seed value. If None, a random seed is generated.
        
    Returns:
        SeedManager: Configured seed manager instance.
        
    Example:
        >>> seed_mgr = set_global_seed(42)
        >>> print(f"Using seed: {seed_mgr.master_seed}")
    """
    seed_manager = SeedManager(seed)
    seed_manager.set_seeds()
    
    # Enable deterministic operations
    enable_deterministic_mode()
    
    return seed_manager


def enable_deterministic_mode() -> None:
    """
    Enable deterministic operations in PyTorch for reproducible results.
    
    This sets various PyTorch flags to ensure deterministic behavior,
    though it may impact performance.
    """
    # Enable deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set additional environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Configure cuDNN for deterministic behavior
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_environment_fingerprint() -> Dict[str, Any]:
    """
    Generate a comprehensive fingerprint of the current environment.
    
    This captures all relevant information needed to reproduce the environment,
    including package versions, hardware info, and system details.
    
    Returns:
        Dict[str, Any]: Environment fingerprint with system and package info.
    """
    fingerprint = {
        'system': {
            'platform': platform.platform(),
            'python_version': sys.version,
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
        'packages': {},
        'torch_info': {},
        'environment_vars': {},
    }
    
    # Capture key package versions
    key_packages = [
        'torch', 'numpy', 'h5py', 'matplotlib', 'tqdm', 'pyyaml'
    ]
    
    for package in key_packages:
        try:
            module = __import__(package)
            fingerprint['packages'][package] = getattr(module, '__version__', 'unknown')
        except ImportError:
            fingerprint['packages'][package] = 'not_installed'
    
    # Capture PyTorch-specific information
    fingerprint['torch_info'] = {
        'version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    # Capture relevant environment variables
    env_vars = ['PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS']
    for var in env_vars:
        fingerprint['environment_vars'][var] = os.environ.get(var, None)
    
    return fingerprint


def verify_reproducibility(
    func: callable,
    args: tuple = (),
    kwargs: dict = None,
    seed: int = 42,
    num_runs: int = 3,
    tolerance: float = 1e-10
) -> Dict[str, Any]:
    """
    Verify that a function produces reproducible results across multiple runs.
    
    Args:
        func (callable): Function to test for reproducibility.
        args (tuple): Positional arguments for the function.
        kwargs (dict): Keyword arguments for the function.
        seed (int): Seed to use for testing.
        num_runs (int): Number of runs to compare.
        tolerance (float): Numerical tolerance for comparing results.
        
    Returns:
        Dict[str, Any]: Verification results including success status and details.
        
    Example:
        >>> def test_func():
        ...     return torch.randn(10)
        >>> result = verify_reproducibility(test_func, seed=42)
        >>> print(f"Reproducible: {result['is_reproducible']}")
    """
    if kwargs is None:
        kwargs = {}
    
    results = []
    hashes = []
    
    for run in range(num_runs):
        # Reset seeds for each run
        set_global_seed(seed)
        
        # Run the function
        try:
            result = func(*args, **kwargs)
            results.append(result)
            
            # Compute hash of the result for comparison
            if isinstance(result, torch.Tensor):
                result_hash = hashlib.md5(result.detach().cpu().numpy().tobytes()).hexdigest()
            elif isinstance(result, np.ndarray):
                result_hash = hashlib.md5(result.tobytes()).hexdigest()
            else:
                result_hash = hashlib.md5(str(result).encode()).hexdigest()
            
            hashes.append(result_hash)
            
        except Exception as e:
            return {
                'is_reproducible': False,
                'error': f"Function failed on run {run}: {str(e)}",
                'num_runs_completed': run,
            }
    
    # Check if all results are identical
    all_hashes_same = len(set(hashes)) == 1
    
    # For numerical results, also check with tolerance
    numerical_same = True
    if results and isinstance(results[0], (torch.Tensor, np.ndarray)):
        for i in range(1, len(results)):
            if isinstance(results[0], torch.Tensor):
                diff = torch.abs(results[0] - results[i]).max().item()
            else:
                diff = np.abs(results[0] - results[i]).max()
            
            if diff > tolerance:
                numerical_same = False
                break
    
    return {
        'is_reproducible': all_hashes_same and numerical_same,
        'hash_identical': all_hashes_same,
        'numerically_identical': numerical_same,
        'num_runs': num_runs,
        'result_hashes': hashes,
        'tolerance_used': tolerance,
    }


def save_reproducibility_info(
    filepath: Union[str, Path],
    seed_manager: SeedManager,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save comprehensive reproducibility information to a file.
    
    Args:
        filepath (Union[str, Path]): Path to save the information.
        seed_manager (SeedManager): Seed manager instance with seed info.
        additional_info (Dict[str, Any], optional): Additional metadata to save.
    """
    info = {
        'seeds': seed_manager.get_seed_info(),
        'environment': get_environment_fingerprint(),
        'additional': additional_info or {},
    }
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2, default=str)


def load_reproducibility_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load reproducibility information from a file.
    
    Args:
        filepath (Union[str, Path]): Path to the saved information.
        
    Returns:
        Dict[str, Any]: Loaded reproducibility information.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_data_hash(data: Union[torch.Tensor, np.ndarray, bytes]) -> str:
    """
    Compute a hash of data for verification purposes.
    
    Args:
        data: Data to hash (tensor, array, or bytes).
        
    Returns:
        str: Hexadecimal hash string.
    """
    if isinstance(data, torch.Tensor):
        data_bytes = data.detach().cpu().numpy().tobytes()
    elif isinstance(data, np.ndarray):
        data_bytes = data.tobytes()
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        data_bytes = str(data).encode()
    
    return hashlib.sha256(data_bytes).hexdigest() 