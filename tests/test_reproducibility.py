"""
Tests for the reproducibility framework.

This module tests all aspects of the reproducibility system including:
- Seed management across all RNG sources
- Environment fingerprinting
- Deterministic operations
- Hash-based verification
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from birdsong.utils.reproducibility import (
    SeedManager,
    compute_data_hash,
    enable_deterministic_mode,
    get_environment_fingerprint,
    load_reproducibility_info,
    save_reproducibility_info,
    set_global_seed,
    verify_reproducibility,
)


class TestSeedManager:
    """Test the SeedManager class functionality."""
    
    def test_seed_manager_initialization_with_seed(self):
        """Test SeedManager initializes correctly with provided seed."""
        seed = 42
        manager = SeedManager(seed)
        assert manager.master_seed == seed
        assert not manager.get_seed_info()['is_set']
    
    def test_seed_manager_initialization_without_seed(self):
        """Test SeedManager generates seed when none provided."""
        manager = SeedManager()
        assert isinstance(manager.master_seed, int)
        assert 0 <= manager.master_seed < 2**31
    
    def test_set_seeds_consistency(self):
        """Test that set_seeds produces consistent component seeds."""
        seed = 123
        manager1 = SeedManager(seed)
        manager2 = SeedManager(seed)
        
        seeds1 = manager1.set_seeds()
        seeds2 = manager2.set_seeds()
        
        assert seeds1 == seeds2
        assert all(key in seeds1 for key in ['python_random', 'numpy', 'torch', 'torch_cuda'])
    
    def test_seed_info_tracking(self):
        """Test that seed information is properly tracked."""
        seed = 456
        manager = SeedManager(seed)
        
        # Before setting seeds
        info_before = manager.get_seed_info()
        assert not info_before['is_set']
        assert info_before['master_seed'] == seed
        
        # After setting seeds
        manager.set_seeds()
        info_after = manager.get_seed_info()
        assert info_after['is_set']
        assert len(info_after['component_seeds']) == 4


class TestGlobalSeedFunction:
    """Test the set_global_seed function."""
    
    def test_global_seed_with_specified_seed(self):
        """Test global seed setting with specified seed."""
        seed = 789
        manager = set_global_seed(seed)
        
        assert manager.master_seed == seed
        assert manager.get_seed_info()['is_set']
    
    def test_global_seed_without_specified_seed(self):
        """Test global seed setting without specified seed."""
        manager = set_global_seed()
        
        assert isinstance(manager.master_seed, int)
        assert manager.get_seed_info()['is_set']
    
    def test_rng_consistency_after_global_seed(self):
        """Test that RNG sources produce consistent results after seeding."""
        seed = 999
        
        # First run
        set_global_seed(seed)
        torch_val1 = torch.randn(5)
        numpy_val1 = np.random.randn(5)
        
        # Second run with same seed
        set_global_seed(seed)
        torch_val2 = torch.randn(5)
        numpy_val2 = np.random.randn(5)
        
        # Results should be identical
        assert torch.equal(torch_val1, torch_val2)
        assert np.array_equal(numpy_val1, numpy_val2)


class TestDeterministicMode:
    """Test deterministic mode functionality."""
    
    def test_enable_deterministic_mode(self):
        """Test that deterministic mode sets correct flags."""
        # Enable deterministic mode
        enable_deterministic_mode()
        
        # Check environment variables are set
        assert os.environ.get('PYTHONHASHSEED') == '0'
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        
        # Check PyTorch settings (these should not raise errors)
        if torch.backends.cudnn.is_available():
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False


class TestEnvironmentFingerprinting:
    """Test environment fingerprinting functionality."""
    
    def test_environment_fingerprint_structure(self):
        """Test that environment fingerprint has correct structure."""
        fingerprint = get_environment_fingerprint()
        
        # Check top-level keys
        required_keys = ['system', 'packages', 'torch_info', 'environment_vars']
        assert all(key in fingerprint for key in required_keys)
        
        # Check system info
        system_keys = ['platform', 'python_version', 'machine', 'processor']
        assert all(key in fingerprint['system'] for key in system_keys)
        
        # Check that key packages are captured
        key_packages = ['torch', 'numpy', 'h5py', 'matplotlib', 'tqdm']
        for package in key_packages:
            assert package in fingerprint['packages']
        
        # Check torch info
        torch_keys = ['version', 'cuda_available', 'device_count']
        assert all(key in fingerprint['torch_info'] for key in torch_keys)
    
    def test_environment_fingerprint_consistency(self):
        """Test that environment fingerprint is consistent across calls."""
        fp1 = get_environment_fingerprint()
        fp2 = get_environment_fingerprint()
        
        # Should be identical (except for any time-dependent info)
        assert fp1['system'] == fp2['system']
        assert fp1['packages'] == fp2['packages']
        assert fp1['torch_info'] == fp2['torch_info']


class TestReproducibilityVerification:
    """Test the reproducibility verification system."""
    
    def test_verify_reproducible_function(self):
        """Test verification of a reproducible function."""
        def deterministic_func():
            torch.manual_seed(42)
            return torch.randn(10)
        
        result = verify_reproducibility(deterministic_func, seed=123, num_runs=3)
        
        assert result['is_reproducible']
        assert result['hash_identical']
        assert result['numerically_identical']
        assert result['num_runs'] == 3
        assert len(result['result_hashes']) == 3
    
    def test_verify_non_reproducible_function(self):
        """Test verification of a non-reproducible function."""
        import time
        
        def truly_non_deterministic():
            # Use system time to ensure different results
            # This will definitely be different across runs
            time_ns = time.time_ns()
            return torch.tensor([float(time_ns % 1000000)]) / 1000000.0
        
        result = verify_reproducibility(truly_non_deterministic, seed=123, num_runs=3)
        
        # This should not be reproducible due to time dependency
        assert not result['is_reproducible']
    
    def test_verify_function_with_args(self):
        """Test verification with function arguments."""
        def func_with_args(size, multiplier=1.0):
            torch.manual_seed(42)
            return torch.randn(size) * multiplier
        
        result = verify_reproducibility(
            func_with_args,
            args=(5,),
            kwargs={'multiplier': 2.0},
            seed=456,
            num_runs=2
        )
        
        assert result['is_reproducible']
        assert result['num_runs'] == 2
    
    def test_verify_function_error_handling(self):
        """Test verification handles function errors gracefully."""
        def failing_func():
            raise ValueError("Test error")
        
        result = verify_reproducibility(failing_func, seed=789)
        
        assert not result['is_reproducible']
        assert 'error' in result
        assert 'Test error' in result['error']


class TestDataHashing:
    """Test data hashing functionality."""
    
    def test_torch_tensor_hashing(self):
        """Test hashing of torch tensors."""
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([1.0, 2.0, 3.0])
        tensor3 = torch.tensor([1.0, 2.0, 4.0])
        
        hash1 = compute_data_hash(tensor1)
        hash2 = compute_data_hash(tensor2)
        hash3 = compute_data_hash(tensor3)
        
        assert hash1 == hash2  # Identical tensors should have same hash
        assert hash1 != hash3  # Different tensors should have different hashes
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex string length
    
    def test_numpy_array_hashing(self):
        """Test hashing of numpy arrays."""
        array1 = np.array([1.0, 2.0, 3.0])
        array2 = np.array([1.0, 2.0, 3.0])
        array3 = np.array([1.0, 2.0, 4.0])
        
        hash1 = compute_data_hash(array1)
        hash2 = compute_data_hash(array2)
        hash3 = compute_data_hash(array3)
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert isinstance(hash1, str)
        assert len(hash1) == 64
    
    def test_bytes_hashing(self):
        """Test hashing of byte data."""
        data1 = b"test data"
        data2 = b"test data"
        data3 = b"different data"
        
        hash1 = compute_data_hash(data1)
        hash2 = compute_data_hash(data2)
        hash3 = compute_data_hash(data3)
        
        assert hash1 == hash2
        assert hash1 != hash3


class TestReproducibilityIO:
    """Test saving and loading reproducibility information."""
    
    def test_save_and_load_reproducibility_info(self):
        """Test saving and loading reproducibility information."""
        seed_manager = SeedManager(12345)
        seed_manager.set_seeds()
        
        additional_info = {'experiment_name': 'test_experiment', 'notes': 'test run'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'repro_info.json'
            
            # Save info
            save_reproducibility_info(filepath, seed_manager, additional_info)
            
            # Check file exists
            assert filepath.exists()
            
            # Load info
            loaded_info = load_reproducibility_info(filepath)
            
            # Verify structure
            assert 'seeds' in loaded_info
            assert 'environment' in loaded_info
            assert 'additional' in loaded_info
            
            # Verify seed info
            assert loaded_info['seeds']['master_seed'] == 12345
            assert loaded_info['seeds']['is_set'] is True
            
            # Verify additional info
            assert loaded_info['additional'] == additional_info
    
    def test_save_creates_directory(self):
        """Test that save function creates necessary directories."""
        seed_manager = SeedManager(999)
        seed_manager.set_seeds()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / 'nested' / 'dir' / 'repro_info.json'
            
            # Directory doesn't exist yet
            assert not nested_path.parent.exists()
            
            # Save should create it
            save_reproducibility_info(nested_path, seed_manager)
            
            # Directory and file should now exist
            assert nested_path.parent.exists()
            assert nested_path.exists()


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_full_reproducibility_workflow(self):
        """Test a complete reproducibility workflow."""
        # Set up reproducible environment
        seed_manager = set_global_seed(42)
        
        # Create some data with known operations
        data = torch.randn(100, 10)
        processed_data = torch.nn.functional.relu(data)
        
        # Compute hash of results
        hash1 = compute_data_hash(processed_data)
        
        # Reset and repeat
        set_global_seed(42)
        data2 = torch.randn(100, 10)
        processed_data2 = torch.nn.functional.relu(data2)
        hash2 = compute_data_hash(processed_data2)
        
        # Should be identical
        assert hash1 == hash2
        assert torch.equal(processed_data, processed_data2)
    
    def test_cross_platform_seed_consistency(self):
        """Test that seeds work consistently (within platform constraints)."""
        # This test verifies that our seeding approach is consistent
        # within a single platform/environment
        
        results = []
        for _ in range(3):
            set_global_seed(1337)
            
            # Mix of operations
            torch_result = torch.randn(5).sum().item()
            numpy_result = np.random.randn(5).sum()
            
            results.append((torch_result, numpy_result))
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert abs(result[0] - first_result[0]) < 1e-10
            assert abs(result[1] - first_result[1]) < 1e-10 