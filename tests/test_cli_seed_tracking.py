"""
Tests for CLI seed tracking and reproducibility functionality.

This module tests the enhanced CLI commands to ensure they properly track
seeds, record metadata, and maintain reproducibility across operations.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from birdsong.cli.base import (
    ReproducibleCLI,
    create_output_filename,
    load_experiment_metadata,
)


class TestReproducibleCLIBase:
    """Test the base ReproducibleCLI functionality."""
    
    def test_create_output_filename_with_seed(self):
        """Test output filename creation includes seed."""
        filename = create_output_filename("test", 42, ".txt")
        assert "seed42" in filename
        assert filename.endswith(".txt")
    
    def test_create_output_filename_with_timestamp(self):
        """Test output filename creation includes timestamp."""
        filename = create_output_filename("test", 123, ".h5", timestamp=True)
        assert "seed123" in filename
        assert filename.endswith(".h5")
        # Should have timestamp format YYYYMMDD_HHMMSS
        parts = filename.split("_")
        assert len(parts) >= 3  # test, seed123, timestamp
    
    def test_create_output_filename_without_timestamp(self):
        """Test output filename creation without timestamp."""
        filename = create_output_filename("test", 456, ".pt", timestamp=False)
        assert filename == "test_seed456.pt"
    
    def test_reproducible_cli_initialization(self):
        """Test ReproducibleCLI initialization."""
        cli = ReproducibleCLI("test", "Test CLI")
        assert cli.command_name == "test"
        assert cli.description == "Test CLI"
        assert cli.seed_manager is None
        assert cli.output_dir is None
        assert cli.config == {}
    
    def test_reproducible_cli_parser_creation(self):
        """Test argument parser creation with common arguments."""
        cli = ReproducibleCLI("test", "Test CLI")
        parser = cli.create_parser()
        
        # Test that common arguments are present
        actions = {action.dest: action for action in parser._actions}
        
        assert 'seed' in actions
        assert 'output_dir' in actions
        assert 'config' in actions
        assert 'verbose' in actions
        assert 'dry_run' in actions
    
    def test_setup_reproducibility(self):
        """Test reproducibility setup."""
        cli = ReproducibleCLI("test", "Test CLI")
        cli.setup_reproducibility(42)
        
        assert cli.seed_manager is not None
        assert cli.seed_manager.master_seed == 42
        assert cli.seed_manager.get_seed_info()['is_set']
    
    def test_setup_output_directory_custom(self):
        """Test output directory setup with custom path."""
        cli = ReproducibleCLI("test", "Test CLI")
        cli.seed_manager = Mock()
        cli.seed_manager.master_seed = 42
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "custom_output"
            result_dir = cli.setup_output_directory(str(output_dir))
            
            assert result_dir == output_dir
            assert output_dir.exists()
            assert cli.output_dir == output_dir
    
    def test_setup_output_directory_auto(self):
        """Test output directory setup with automatic naming."""
        cli = ReproducibleCLI("test", "Test CLI")
        cli.seed_manager = Mock()
        cli.seed_manager.master_seed = 123
        
        result_dir = cli.setup_output_directory()
        
        assert "test_" in str(result_dir)
        assert "seed123" in str(result_dir)
        assert result_dir.exists()
        assert cli.output_dir == result_dir
        
        # Clean up
        import shutil
        shutil.rmtree(result_dir.parent)
    
    def test_save_metadata(self):
        """Test metadata saving functionality."""
        cli = ReproducibleCLI("test", "Test CLI")
        cli.setup_reproducibility(42)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cli.output_dir = Path(temp_dir)
            
            additional_info = {"test_param": "test_value"}
            metadata_path = cli.save_metadata(additional_info)
            
            # Check files were created
            assert metadata_path.exists()
            summary_path = cli.output_dir / "experiment_summary.txt"
            assert summary_path.exists()
            
            # Check metadata content
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert "seeds" in metadata
            assert "environment" in metadata
            assert "additional" in metadata
            assert metadata["additional"]["test_param"] == "test_value"
            assert metadata["seeds"]["master_seed"] == 42
            
            # Check summary content
            summary_content = summary_path.read_text()
            assert "Test Experiment" in summary_content
            assert "Seed: 42" in summary_content
    
    def test_load_experiment_metadata(self):
        """Test loading experiment metadata."""
        cli = ReproducibleCLI("test", "Test CLI")
        cli.setup_reproducibility(999)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cli.output_dir = Path(temp_dir)
            
            # Save metadata first
            cli.save_metadata({"test": "data"})
            
            # Load it back
            loaded = load_experiment_metadata(cli.output_dir)
            
            assert "seeds" in loaded
            assert "environment" in loaded
            assert "additional" in loaded
            assert loaded["seeds"]["master_seed"] == 999
            assert loaded["additional"]["test"] == "data"


class TestCLISeedTracking:
    """Test seed tracking across different CLI commands."""
    
    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset for testing."""
        mock = Mock()
        mock.__len__ = Mock(return_value=100)
        return mock
    
    @pytest.fixture
    def mock_model(self):
        """Mock model for testing."""
        mock = Mock()
        mock.parameters = Mock(return_value=[torch.tensor([1.0])])
        return mock
    
    def test_train_cli_seed_integration(self, mock_dataset, mock_model):
        """Test that training CLI properly integrates seed tracking."""
        from birdsong.cli.train_enhanced import TrainCLI
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock data file
            data_path = Path(temp_dir) / "test_data.h5"
            data_path.touch()
            
            # Mock the CLI validation to skip data path validation
            def mock_validate_setup(self):
                if not self.seed_manager:
                    raise RuntimeError("Reproducibility not set up")
                if not self.output_dir:
                    raise RuntimeError("Output directory not set up")
            
            # Mock the necessary components
            with patch('birdsong.cli.train_enhanced.BirdsongDataset') as mock_dataset_class, \
                 patch('birdsong.cli.train_enhanced.BirdsongLFADSModel2') as mock_model_class, \
                 patch('birdsong.cli.train_enhanced.BirdsongTrainer') as mock_trainer_class, \
                 patch.object(TrainCLI, 'validate_setup', mock_validate_setup), \
                 patch('sys.argv', ['birdsong-train', '--data-path', str(data_path), 
                                   '--seed', '42', '--output-dir', str(temp_dir), 
                                   '--epochs', '1', '--dry-run']):
                
                mock_dataset_class.return_value = mock_dataset
                mock_model_class.return_value = mock_model
                mock_trainer = Mock()
                mock_trainer_class.return_value = mock_trainer
                
                cli = TrainCLI()
                cli.execute()
                
                # Verify seed was set
                assert cli.seed_manager is not None
                assert cli.seed_manager.master_seed == 42
                
                # Verify output directory was created with seed
                assert cli.output_dir is not None
                assert cli.output_dir.exists()
                
                # Verify metadata was saved
                metadata_path = cli.output_dir / 'reproducibility.json'
                assert metadata_path.exists()
                
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                assert metadata['seeds']['master_seed'] == 42
                assert metadata['additional']['command'] == 'train'
    
    def test_eval_cli_seed_integration(self, mock_dataset, mock_model):
        """Test that evaluation CLI properly integrates seed tracking."""
        from birdsong.cli.eval_enhanced import EvalCLI
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock checkpoint file
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
            torch.save({
                'model_state_dict': {'test': torch.tensor([1.0])},
                'epoch': 10,
                'best_loss': 0.5
            }, checkpoint_path)
            
            # Create mock data file
            data_path = Path(temp_dir) / "test_data.h5"
            data_path.touch()
            
            # Mock the CLI validation to skip checkpoint validation
            def mock_validate_setup(self):
                if not self.seed_manager:
                    raise RuntimeError("Reproducibility not set up")
                if not self.output_dir:
                    raise RuntimeError("Output directory not set up")
            
            with patch('birdsong.cli.eval_enhanced.BirdsongDataset') as mock_dataset_class, \
                 patch('birdsong.cli.eval_enhanced.BirdsongLFADSModel2') as mock_model_class, \
                 patch('birdsong.cli.eval_enhanced.BirdsongEvaluator') as mock_evaluator_class, \
                 patch.object(EvalCLI, 'validate_setup', mock_validate_setup), \
                 patch('sys.argv', ['birdsong-eval', '--checkpoint', str(checkpoint_path),
                                   '--data-path', str(data_path), '--seed', '123', 
                                   '--output-dir', str(temp_dir), '--dry-run']):
                
                mock_dataset_class.return_value = mock_dataset
                mock_model_instance = mock_model
                mock_model_instance.load_state_dict = Mock()
                mock_model_instance.eval = Mock()
                mock_model_class.return_value = mock_model_instance
                
                mock_evaluator = Mock()
                mock_evaluator_class.return_value = mock_evaluator
                
                cli = EvalCLI()
                cli.execute()
                
                # Verify seed was set
                assert cli.seed_manager is not None
                assert cli.seed_manager.master_seed == 123
                
                # Verify output directory was created with seed
                assert cli.output_dir is not None
                assert cli.output_dir.exists()
                
                # Verify metadata was saved
                metadata_path = cli.output_dir / 'reproducibility.json'
                assert metadata_path.exists()
                
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                assert metadata['seeds']['master_seed'] == 123
                assert metadata['additional']['command'] == 'eval'
    
    def test_generate_cli_seed_integration(self):
        """Test that generation CLI properly integrates seed tracking."""
        from birdsong.cli.generate import GenerateCLI
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('birdsong.cli.generate.BirdsongSimulator') as mock_simulator_class, \
                 patch('birdsong.cli.generate.BirdsongAggregator') as mock_aggregator_class, \
                 patch('sys.argv', ['birdsong-generate', '--num-songs', '10', 
                                   '--seed', '456', '--output-dir', str(temp_dir), 
                                   '--dry-run']):
                
                # Mock simulator
                mock_simulator = Mock()
                mock_simulator.generate_sequences = Mock(return_value=[[1, 2, 3]] * 10)
                mock_simulator_class.return_value = mock_simulator
                
                # Mock aggregator
                mock_aggregator = Mock()
                mock_aggregator.process_sequences = Mock(return_value=torch.randn(10, 3))
                mock_aggregator.save_data = Mock()
                mock_aggregator_class.return_value = mock_aggregator
                
                cli = GenerateCLI()
                cli.execute()
                
                # Verify seed was set
                assert cli.seed_manager is not None
                assert cli.seed_manager.master_seed == 456
                
                # Verify output directory was created with seed
                assert cli.output_dir is not None
                assert cli.output_dir.exists()
                
                # Verify metadata was saved
                metadata_path = cli.output_dir / 'reproducibility.json'
                assert metadata_path.exists()
                
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                assert metadata['seeds']['master_seed'] == 456
                assert metadata['additional']['command'] == 'generate'


class TestCrossOperationSeedRecovery:
    """Test that seeds can be recovered and experiments can be reproduced."""
    
    def test_seed_recovery_from_metadata(self):
        """Test that seeds can be recovered from saved metadata."""
        cli = ReproducibleCLI("test", "Test CLI")
        cli.setup_reproducibility(777)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cli.output_dir = Path(temp_dir)
            
            # Save metadata
            metadata_path = cli.save_metadata({"experiment_type": "test"})
            
            # Load metadata back
            loaded_metadata = load_experiment_metadata(cli.output_dir)
            
            # Verify seed can be recovered
            recovered_seed = loaded_metadata['seeds']['master_seed']
            assert recovered_seed == 777
            
            # Verify all seed components are present
            assert 'component_seeds' in loaded_metadata['seeds']
            component_seeds = loaded_metadata['seeds']['component_seeds']
            expected_components = ['python_random', 'numpy', 'torch', 'torch_cuda']
            for component in expected_components:
                assert component in component_seeds
    
    def test_reproducibility_across_cli_executions(self):
        """Test that the same seed produces identical results across CLI executions."""
        # This test verifies the conceptual framework - actual reproducibility
        # depends on the specific operations being performed
        
        results = []
        
        for run in range(2):
            cli = ReproducibleCLI("test", "Test CLI") 
            cli.setup_reproducibility(888)  # Same seed for both runs
            
            with tempfile.TemporaryDirectory() as temp_dir:
                cli.output_dir = Path(temp_dir)
                
                # Simulate some reproducible operation
                import torch
                result = torch.randn(10).sum().item()
                results.append(result)
                
                # Save metadata
                cli.save_metadata({"result": result})
                
                # Verify metadata contains correct seed
                metadata_path = cli.output_dir / 'reproducibility.json'
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                assert metadata['seeds']['master_seed'] == 888
        
        # Results should be identical due to same seed
        assert abs(results[0] - results[1]) < 1e-10
    
    def test_filename_consistency_with_seed(self):
        """Test that filenames are consistent when using the same seed."""
        # Test that the same seed produces consistent base filenames
        seed = 999
        base_filename = create_output_filename("experiment", seed, ".h5", timestamp=False)
        
        # Should be exactly "experiment_seed999.h5"
        assert base_filename == "experiment_seed999.h5"
        
        # Test with timestamp should still contain seed
        timestamped_filename = create_output_filename("experiment", seed, ".h5", timestamp=True)
        assert "seed999" in timestamped_filename
        assert timestamped_filename.endswith(".h5")


class TestCLIErrorHandling:
    """Test error handling in CLI commands with seed tracking."""
    
    def test_missing_checkpoint_error(self):
        """Test error handling when checkpoint file is missing."""
        from birdsong.cli.eval_enhanced import EvalCLI
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_checkpoint = Path(temp_dir) / "nonexistent.pt"
            
            with patch('sys.argv', ['birdsong-eval', '--checkpoint', str(nonexistent_checkpoint),
                                   '--seed', '123']):
                
                cli = EvalCLI()
                
                with pytest.raises(SystemExit):  # CLI should exit with error
                    cli.execute()
    
    def test_invalid_config_error(self):
        """Test error handling when config file is invalid."""
        cli = ReproducibleCLI("test", "Test CLI")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            invalid_config_path = f.name
        
        try:
            with pytest.raises(Exception):  # Should raise YAML error
                cli.load_config(invalid_config_path)
        finally:
            os.unlink(invalid_config_path)
    
    def test_setup_validation_error(self):
        """Test validation error when setup is incomplete."""
        cli = ReproducibleCLI("test", "Test CLI")
        
        # Try to validate without setup
        with pytest.raises(RuntimeError, match="Reproducibility not set up"):
            cli.validate_setup()
        
        # Setup reproducibility but not output directory
        cli.setup_reproducibility(42)
        
        with pytest.raises(RuntimeError, match="Output directory not set up"):
            cli.validate_setup()


class TestEnvironmentCapture:
    """Test that environment information is properly captured."""
    
    def test_environment_fingerprint_in_metadata(self):
        """Test that environment fingerprint is included in metadata."""
        cli = ReproducibleCLI("test", "Test CLI")
        cli.setup_reproducibility(42)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cli.output_dir = Path(temp_dir)
            
            # Save metadata
            cli.save_metadata()
            
            # Load and check environment information
            metadata_path = cli.output_dir / 'reproducibility.json'
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert 'environment' in metadata
            env_info = metadata['environment']
            
            # Check that key environment information is present
            assert 'system' in env_info
            assert 'packages' in env_info
            assert 'torch_info' in env_info
            
            # Check specific system information
            assert 'platform' in env_info['system']
            assert 'python_version' in env_info['system']
            
            # Check package information
            assert 'torch' in env_info['packages']
            assert 'numpy' in env_info['packages']
    
    def test_command_line_capture(self):
        """Test that command line arguments are captured."""
        cli = ReproducibleCLI("test", "Test CLI")
        cli.setup_reproducibility(42)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cli.output_dir = Path(temp_dir)
            
            # Save metadata
            cli.save_metadata()
            
            # Check command line was captured
            metadata_path = cli.output_dir / 'reproducibility.json'
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert 'command_line_args' in metadata['additional']
            assert isinstance(metadata['additional']['command_line_args'], list) 