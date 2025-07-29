"""
Tests for config integration with latent space analysis.

This module tests that the latent space analysis properly respects
the configuration toggles for different analysis types.
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os

from birdsong.config.schema import BirdsongConfig, EvaluationConfig
from birdsong.analysis.latent import LatentSpaceAnalyzer


class TestConfigIntegration:
    """Test config integration with latent space analysis."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = BirdsongConfig(
            evaluation=EvaluationConfig(
                analyze_latents=True,
                analyze_factors=False,
                analyze_trajectories=True,
                analyze_reconstructions=True,
                analyze_transitions=True,
                analyze_kl_divergence=True,
                analyze_cross_entropy=True,
                analyze_accuracy=True,
                analyze_js_divergence=True
            )
        )
    
    def test_config_analysis_toggles(self):
        """Test that config analysis toggles are properly set."""
        eval_config = self.config.evaluation
        
        assert eval_config.analyze_latents is True
        assert eval_config.analyze_factors is False
        assert eval_config.analyze_trajectories is True
        assert eval_config.analyze_reconstructions is True
        assert eval_config.analyze_transitions is True
    
    def test_analysis_selection_from_config(self):
        """Test that analysis selection respects config toggles."""
        analyses_to_perform = []
        
        # Check which analyses are enabled
        if self.config.evaluation.analyze_latents:
            analyses_to_perform.append('latents')
        if self.config.evaluation.analyze_factors:
            analyses_to_perform.append('factors')
        if self.config.evaluation.analyze_trajectories:
            analyses_to_perform.append('trajectories')
        
        assert 'latents' in analyses_to_perform
        assert 'factors' not in analyses_to_perform
        assert 'trajectories' in analyses_to_perform
    
    def test_config_override_behavior(self):
        """Test that command line arguments can override config settings."""
        # Simulate command line overrides
        override_latents = True
        override_factors = True
        disable_trajectories = True
        
        analyses_to_perform = []
        
        # Check config settings with overrides
        if self.config.evaluation.analyze_latents and not False:  # No disable
            analyses_to_perform.append('latents')
        if self.config.evaluation.analyze_factors and not False:  # No disable
            analyses_to_perform.append('factors')
        if self.config.evaluation.analyze_trajectories and not disable_trajectories:
            analyses_to_perform.append('trajectories')
        
        # Apply command line overrides
        if override_latents:
            analyses_to_perform.append('latents')
        if override_factors:
            analyses_to_perform.append('factors')
        if override_trajectories:
            analyses_to_perform.append('trajectories')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_analyses = []
        for analysis in analyses_to_perform:
            if analysis not in seen:
                seen.add(analysis)
                unique_analyses.append(analysis)
        
        assert 'latents' in unique_analyses
        assert 'factors' in unique_analyses
        assert 'trajectories' not in unique_analyses  # Disabled by command line
    
    def test_config_validation(self):
        """Test that config validation works with analysis toggles."""
        # Valid config
        valid_config = BirdsongConfig(
            evaluation=EvaluationConfig(
                analyze_latents=True,
                analyze_factors=False,
                analyze_trajectories=True
            )
        )
        
        # Should not raise any exceptions
        assert valid_config.evaluation.analyze_latents is True
        assert valid_config.evaluation.analyze_factors is False
        assert valid_config.evaluation.analyze_trajectories is True
    
    def test_config_serialization(self):
        """Test that config with analysis toggles can be serialized."""
        config_dict = self.config.model_dump()
        
        assert 'evaluation' in config_dict
        assert 'analyze_latents' in config_dict['evaluation']
        assert 'analyze_factors' in config_dict['evaluation']
        assert 'analyze_trajectories' in config_dict['evaluation']
        
        assert config_dict['evaluation']['analyze_latents'] is True
        assert config_dict['evaluation']['analyze_factors'] is False
        assert config_dict['evaluation']['analyze_trajectories'] is True
    
    def test_config_inheritance(self):
        """Test that analysis toggles work with config inheritance."""
        # Create a config that inherits from another
        base_config = BirdsongConfig(
            evaluation=EvaluationConfig(
                analyze_latents=True,
                analyze_factors=False,
                analyze_trajectories=True
            )
        )
        
        # Create derived config
        derived_config = BirdsongConfig(
            inherits_from=["base_config"],
            evaluation=EvaluationConfig(
                analyze_factors=True,  # Override
                analyze_trajectories=False  # Override
            )
        )
        
        # In a real scenario, the inheritance would be handled by the config loader
        # For this test, we just verify the structure
        assert derived_config.evaluation.analyze_factors is True
        assert derived_config.evaluation.analyze_trajectories is False


class TestCLIIntegration:
    """Test CLI integration with config toggles."""
    
    def test_cli_respects_config_toggles(self):
        """Test that CLI respects config analysis toggles."""
        # Mock config with specific toggles
        mock_config = {
            'evaluation': {
                'analyze_latents': True,
                'analyze_factors': False,
                'analyze_trajectories': True,
                'batch_size': 64
            }
        }
        
        # Simulate CLI logic
        analyses_to_perform = []
        
        if mock_config['evaluation']['analyze_latents']:
            analyses_to_perform.append('latents')
        if mock_config['evaluation']['analyze_factors']:
            analyses_to_perform.append('factors')
        if mock_config['evaluation']['analyze_trajectories']:
            analyses_to_perform.append('trajectories')
        
        assert 'latents' in analyses_to_perform
        assert 'factors' not in analyses_to_perform
        assert 'trajectories' in analyses_to_perform
    
    def test_cli_override_behavior(self):
        """Test that CLI can override config settings."""
        mock_config = {
            'evaluation': {
                'analyze_latents': False,
                'analyze_factors': False,
                'analyze_trajectories': False
            }
        }
        
        # Simulate command line arguments
        args = MagicMock()
        args.analyze_latents = True
        args.analyze_factors = False
        args.analyze_trajectories = True
        args.disable_latents = False
        args.disable_factors = False
        args.disable_trajectories = False
        
        # Simulate CLI logic
        analyses_to_perform = []
        
        # Check config settings
        if mock_config['evaluation']['analyze_latents'] and not args.disable_latents:
            analyses_to_perform.append('latents')
        if mock_config['evaluation']['analyze_factors'] and not args.disable_factors:
            analyses_to_perform.append('factors')
        if mock_config['evaluation']['analyze_trajectories'] and not args.disable_trajectories:
            analyses_to_perform.append('trajectories')
        
        # Apply command line overrides
        if args.analyze_latents:
            analyses_to_perform.append('latents')
        if args.analyze_factors:
            analyses_to_perform.append('factors')
        if args.analyze_trajectories:
            analyses_to_perform.append('trajectories')
        
        # Remove duplicates
        seen = set()
        unique_analyses = []
        for analysis in analyses_to_perform:
            if analysis not in seen:
                seen.add(analysis)
                unique_analyses.append(analysis)
        
        assert 'latents' in unique_analyses  # Enabled by command line
        assert 'factors' not in unique_analyses  # Disabled in config and not overridden
        assert 'trajectories' in unique_analyses  # Enabled by command line


if __name__ == "__main__":
    pytest.main([__file__]) 