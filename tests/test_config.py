"""
Tests for the birdsong configuration system.

This module tests schema validation, hierarchical inheritance,
environment variable substitution, and error handling.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
import yaml

# Skip tests if pydantic is not available
pytest.importorskip("pydantic")

from birdsong.config import (
    BirdsongConfig,
    ConfigLoader,
    load_config,
    load_template,
    ConfigValidationError,
    validate_config,
    validate_paths,
    validate_dependencies
)
from birdsong.config.schema import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    ExperimentConfig
)


class TestBirdsongConfigSchema:
    """Test the Pydantic configuration schema validation."""
    
    def test_default_config_creation(self):
        """Test creating config with all defaults."""
        config = BirdsongConfig()
        
        assert config.version == "1.0"
        assert len(config.data.alphabet) == 8
        assert config.model.encoder_dim == 64
        assert config.training.epochs == 20
        assert config.evaluation.batch_size == 64
        assert config.experiment is None
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "data": {"alphabet": ["<", "a", "b", ">"], "order": 1},
            "model": {"encoder_dim": 128, "order": 1},
            "training": {"epochs": 50, "batch_size": 64},
            "evaluation": {"batch_size": 32}
        }
        
        config = BirdsongConfig.from_dict(config_dict)
        
        assert len(config.data.alphabet) == 4
        assert config.model.encoder_dim == 128
        assert config.training.epochs == 50
        assert config.evaluation.batch_size == 32
    
    def test_alphabet_validation(self):
        """Test alphabet validation rules."""
        # Valid alphabet
        valid_config = {"data": {"alphabet": ["<", "a", "b", "c", ">"]}}
        config = BirdsongConfig.from_dict(valid_config)
        assert len(config.data.alphabet) == 5
        
        # Missing start symbol
        with pytest.raises(ConfigValidationError):
            invalid_config = {"data": {"alphabet": ["a", "b", "c", ">"]}}
            BirdsongConfig.from_dict(invalid_config)
        
        # Missing end symbol
        with pytest.raises(ConfigValidationError):
            invalid_config = {"data": {"alphabet": ["<", "a", "b", "c"]}}
            BirdsongConfig.from_dict(invalid_config)
        
        # Too short alphabet
        with pytest.raises(ConfigValidationError):
            invalid_config = {"data": {"alphabet": ["<", ">"]}}
            BirdsongConfig.from_dict(invalid_config)
    
    def test_dimension_validation(self):
        """Test model dimension validation."""
        # Valid dimensions
        valid_config = {
            "model": {
                "encoder_dim": 64,
                "factor_dim": 32,
                "latent_dim": 16
            }
        }
        config = BirdsongConfig.from_dict(valid_config)
        assert config.model.encoder_dim == 64
        
        # Invalid dimension (too small)
        with pytest.raises(ConfigValidationError):
            invalid_config = {"model": {"encoder_dim": 4}}  # Below minimum of 8
            BirdsongConfig.from_dict(invalid_config)
        
        # Invalid dimension (too large)
        with pytest.raises(ConfigValidationError):
            invalid_config = {"model": {"encoder_dim": 2048}}  # Above maximum of 1024
            BirdsongConfig.from_dict(invalid_config)
    
    def test_training_validation(self):
        """Test training configuration validation."""
        # Valid KL annealing schedule
        valid_config = {
            "training": {
                "kl_start_epoch": 5,
                "kl_full_epoch": 15,
                "val_split": 0.1,
                "test_split": 0.1
            }
        }
        config = BirdsongConfig.from_dict(valid_config)
        assert config.training.kl_start_epoch == 5
        
        # Invalid KL schedule
        with pytest.raises(ConfigValidationError):
            invalid_config = {
                "training": {
                    "kl_start_epoch": 15,
                    "kl_full_epoch": 10  # Should be > kl_start_epoch
                }
            }
            BirdsongConfig.from_dict(invalid_config)
        
        # Invalid splits (too large)
        with pytest.raises(ConfigValidationError):
            invalid_config = {
                "training": {
                    "val_split": 0.6,
                    "test_split": 0.5  # Total > 1.0
                }
            }
            BirdsongConfig.from_dict(invalid_config)
    
    def test_cross_component_validation(self):
        """Test validation across config components."""
        # Consistent alphabet size and order
        valid_config = {
            "data": {"alphabet": ["<", "a", "b", "c", ">"], "order": 2},
            "model": {"alphabet_size": 5, "order": 2}
        }
        config = BirdsongConfig.from_dict(valid_config)
        assert config.model.alphabet_size == 5
        
        # Auto-derivation of alphabet_size
        auto_config = {
            "data": {"alphabet": ["<", "a", "b", "c", ">"], "order": 1},
            "model": {"alphabet_size": None, "order": 1}
        }
        config = BirdsongConfig.from_dict(auto_config)
        assert config.model.alphabet_size == 5
        
        # Inconsistent order
        with pytest.raises(ConfigValidationError):
            invalid_config = {
                "data": {"order": 1},
                "model": {"order": 2}  # Should match data.order
            }
            BirdsongConfig.from_dict(invalid_config)


class TestConfigLoader:
    """Test the configuration loader with inheritance."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            template_dir = config_dir / "templates"
            template_dir.mkdir()
            yield config_dir
    
    def test_basic_config_loading(self, temp_config_dir):
        """Test loading a basic configuration file."""
        config_file = temp_config_dir / "test.yaml"
        config_data = {
            "data": {"alphabet": ["<", "a", "b", ">"]},
            "model": {"encoder_dim": 128},
            "training": {"epochs": 50}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader([temp_config_dir])
        config = loader.load_config("test.yaml")
        
        assert config.model.encoder_dim == 128
        assert config.training.epochs == 50
    
    def test_inheritance_single_level(self, temp_config_dir):
        """Test single-level configuration inheritance."""
        # Create base config
        base_file = temp_config_dir / "base.yaml"
        base_data = {
            "model": {"encoder_dim": 64, "factor_dim": 32},
            "training": {"epochs": 20, "batch_size": 32}
        }
        with open(base_file, 'w') as f:
            yaml.dump(base_data, f)
        
        # Create child config
        child_file = temp_config_dir / "child.yaml"
        child_data = {
            "inherits_from": ["base.yaml"],
            "model": {"encoder_dim": 128},  # Override
            "training": {"epochs": 50}       # Override
        }
        with open(child_file, 'w') as f:
            yaml.dump(child_data, f)
        
        loader = ConfigLoader([temp_config_dir])
        config = loader.load_config("child.yaml")
        
        # Check overrides
        assert config.model.encoder_dim == 128
        assert config.training.epochs == 50
        
        # Check inherited values
        assert config.model.factor_dim == 32
        assert config.training.batch_size == 32
        
        # Check inheritance tracking
        assert "base.yaml" in str(config.inherits_from[0])
    
    def test_inheritance_multi_level(self, temp_config_dir):
        """Test multi-level configuration inheritance."""
        # Create base config
        base_file = temp_config_dir / "base.yaml"
        base_data = {
            "model": {"encoder_dim": 64, "factor_dim": 32, "latent_dim": 16},
            "training": {"epochs": 20, "batch_size": 32, "learning_rate": 0.001}
        }
        with open(base_file, 'w') as f:
            yaml.dump(base_data, f)
        
        # Create intermediate config
        intermediate_file = temp_config_dir / "intermediate.yaml"
        intermediate_data = {
            "inherits_from": ["base.yaml"],
            "model": {"encoder_dim": 128},      # Override
            "training": {"epochs": 50}          # Override
        }
        with open(intermediate_file, 'w') as f:
            yaml.dump(intermediate_data, f)
        
        # Create final config
        final_file = temp_config_dir / "final.yaml"
        final_data = {
            "inherits_from": ["intermediate.yaml"],
            "model": {"factor_dim": 64},        # Override
            "training": {"learning_rate": 0.01} # Override
        }
        with open(final_file, 'w') as f:
            yaml.dump(final_data, f)
        
        loader = ConfigLoader([temp_config_dir])
        config = loader.load_config("final.yaml")
        
        # Check all levels of inheritance
        assert config.model.encoder_dim == 128     # From intermediate
        assert config.model.factor_dim == 64       # From final
        assert config.model.latent_dim == 16       # From base
        assert config.training.epochs == 50        # From intermediate
        assert config.training.batch_size == 32    # From base
        assert config.training.learning_rate == 0.01  # From final
    
    def test_multiple_inheritance(self, temp_config_dir):
        """Test inheriting from multiple parent configs."""
        # Create first parent
        parent1_file = temp_config_dir / "parent1.yaml"
        parent1_data = {
            "model": {"encoder_dim": 128, "factor_dim": 64},
            "training": {"epochs": 100}
        }
        with open(parent1_file, 'w') as f:
            yaml.dump(parent1_data, f)
        
        # Create second parent
        parent2_file = temp_config_dir / "parent2.yaml"
        parent2_data = {
            "model": {"latent_dim": 32},
            "training": {"batch_size": 64, "learning_rate": 0.01}
        }
        with open(parent2_file, 'w') as f:
            yaml.dump(parent2_data, f)
        
        # Create child inheriting from both
        child_file = temp_config_dir / "child.yaml"
        child_data = {
            "inherits_from": ["parent1.yaml", "parent2.yaml"],
            "training": {"epochs": 200}  # Override
        }
        with open(child_file, 'w') as f:
            yaml.dump(child_data, f)
        
        loader = ConfigLoader([temp_config_dir])
        config = loader.load_config("child.yaml")
        
        # Check inherited values from both parents
        assert config.model.encoder_dim == 128      # From parent1
        assert config.model.factor_dim == 64        # From parent1
        assert config.model.latent_dim == 32        # From parent2
        assert config.training.batch_size == 64     # From parent2
        assert config.training.learning_rate == 0.01 # From parent2
        
        # Check override
        assert config.training.epochs == 200
    
    def test_circular_inheritance_detection(self, temp_config_dir):
        """Test detection of circular inheritance."""
        # Create config A that inherits from B
        config_a = temp_config_dir / "a.yaml"
        a_data = {
            "inherits_from": ["b.yaml"],
            "model": {"encoder_dim": 128}
        }
        with open(config_a, 'w') as f:
            yaml.dump(a_data, f)
        
        # Create config B that inherits from A (circular)
        config_b = temp_config_dir / "b.yaml"
        b_data = {
            "inherits_from": ["a.yaml"],
            "training": {"epochs": 50}
        }
        with open(config_b, 'w') as f:
            yaml.dump(b_data, f)
        
        loader = ConfigLoader([temp_config_dir])
        
        with pytest.raises(ConfigValidationError, match="Circular inheritance"):
            loader.load_config("a.yaml")
    
    def test_environment_variable_substitution(self, temp_config_dir):
        """Test environment variable substitution."""
        config_file = temp_config_dir / "env_test.yaml"
        config_data = {
            "data": {"data_path": "${DATA_PATH:/default/path}"},
            "training": {
                "epochs": "${EPOCHS:50}",
                "checkpoint_path": "${CHECKPOINT_DIR}/model.pt"
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader([temp_config_dir])
        
        # Test with environment variables set
        with patch.dict(os.environ, {
            'DATA_PATH': '/custom/data/path',
            'EPOCHS': '100',
            'CHECKPOINT_DIR': '/custom/checkpoints'
        }):
            config = loader.load_config("env_test.yaml")
            
            assert config.data.data_path == "/custom/data/path"
            assert config.training.epochs == 100  # Should be converted to int
            assert config.training.checkpoint_path == "/custom/checkpoints/model.pt"
        
        # Test with missing environment variables (should use defaults)
        with patch.dict(os.environ, {}, clear=True):
            config = loader.load_config("env_test.yaml")
            
            assert config.data.data_path == "/default/path"
            assert config.training.epochs == 50
            assert config.training.checkpoint_path == "/model.pt"  # Empty default
    
    def test_override_values(self, temp_config_dir):
        """Test applying override values."""
        config_file = temp_config_dir / "test.yaml"
        config_data = {
            "model": {"encoder_dim": 64},
            "training": {"epochs": 20, "batch_size": 32}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader([temp_config_dir])
        
        # Test simple overrides
        overrides = {
            "model.encoder_dim": 128,
            "training.epochs": 100
        }
        
        config = loader.load_config("test.yaml", override_values=overrides)
        
        assert config.model.encoder_dim == 128
        assert config.training.epochs == 100
        assert config.training.batch_size == 32  # Not overridden
    
    def test_template_loading(self, temp_config_dir):
        """Test loading configuration templates."""
        template_dir = temp_config_dir / "templates"
        template_file = template_dir / "test_template.yaml"
        
        template_data = {
            "model": {"encoder_dim": 256},
            "training": {"epochs": 75}
        }
        
        with open(template_file, 'w') as f:
            yaml.dump(template_data, f)
        
        loader = ConfigLoader([temp_config_dir])
        
        # Test template listing
        templates = loader.list_templates()
        assert "test_template" in templates
        
        # Test template loading
        config = loader.load_template("test_template")
        assert config.model.encoder_dim == 256
        assert config.training.epochs == 75


class TestConfigValidation:
    """Test runtime configuration validation."""
    
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        config = BirdsongConfig()
        warnings = validate_config(config, strict=False)
        # Should return warnings list (may be empty)
        assert isinstance(warnings, list)
    
    def test_validate_paths_success(self, tmp_path):
        """Test successful path validation."""
        # Create temporary data file
        data_file = tmp_path / "test_data.h5"
        data_file.touch()
        
        config = BirdsongConfig()
        config.data.data_path = str(data_file)
        
        # Should not raise exception
        validate_paths(config)
    
    def test_validate_paths_missing_file(self):
        """Test path validation with missing file."""
        config = BirdsongConfig()
        config.data.data_path = "/nonexistent/file.h5"
        
        with pytest.raises(ConfigValidationError, match="Data file not found"):
            validate_paths(config)
    
    def test_validate_dependencies_success(self):
        """Test successful dependency validation."""
        config = BirdsongConfig()
        
        # Should not raise exception if all dependencies are available
        validate_dependencies(config)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_validate_dependencies_cuda_unavailable(self, mock_cuda):
        """Test dependency validation when CUDA is unavailable."""
        config = BirdsongConfig()
        config.experiment = ExperimentConfig(
            name="test",
            device="cuda"
        )
        
        with pytest.raises(ConfigValidationError, match="CUDA device specified but CUDA is not available"):
            validate_dependencies(config)


class TestConvenienceFunctions:
    """Test convenience functions for configuration loading."""
    
    def test_load_config_function(self, tmp_path):
        """Test the load_config convenience function."""
        config_file = tmp_path / "test.yaml"
        config_data = {
            "model": {"encoder_dim": 128},
            "training": {"epochs": 50}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(config_file)
        assert config.model.encoder_dim == 128
        assert config.training.epochs == 50
    
    def test_load_template_function(self, tmp_path):
        """Test the load_template convenience function."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        template_file = template_dir / "test.yaml"
        
        template_data = {
            "model": {"encoder_dim": 256},
            "training": {"epochs": 75}
        }
        
        with open(template_file, 'w') as f:
            yaml.dump(template_data, f)
        
        config = load_template("test", search_paths=[tmp_path])
        assert config.model.encoder_dim == 256
        assert config.training.epochs == 75


if __name__ == "__main__":
    pytest.main([__file__]) 