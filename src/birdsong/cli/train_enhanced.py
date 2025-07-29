"""
Enhanced command-line interface for training birdsong models with reproducibility.

This module provides an enhanced CLI for training Birdsong LFADS models with
automatic seed tracking, reproducibility metadata, and hierarchical configuration support.
"""

import argparse
import os
import sys
from typing import Any, Dict

# Add the src directory to the path so we can import birdsong
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from birdsong.cli.base import ReproducibleCLI, create_output_filename
from birdsong.config import load_config, load_template, validate_config, ConfigValidationError
from birdsong.data.loader import BirdsongDataset
from birdsong.models.lfads import BirdsongLFADSModel2
from birdsong.training.trainer import BirdsongTrainer


class TrainCLI(ReproducibleCLI):
    """Enhanced training CLI with reproducibility and hierarchical configuration features."""
    
    def __init__(self):
        super().__init__(
            command_name="train",
            description="Train a Birdsong LFADS model with automatic seed tracking and hierarchical configuration"
        )
    
    def add_command_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add training-specific command-line arguments."""
        
        # Template argument (config is already provided by base class)
        parser.add_argument(
            "--template", "-t",
            type=str,
            help="Configuration template name (alternative to --config for templates)"
        )
        
        # Data arguments
        parser.add_argument(
            "--data-path",
            type=str,
            help="Override path to training data (HDF5 file)"
        )
        
        # Training arguments
        parser.add_argument(
            "--epochs",
            type=int,
            help="Override number of training epochs"
        )
        
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Override batch size"
        )
        
        parser.add_argument(
            "--learning-rate",
            type=float,
            help="Override learning rate"
        )
        
        # Model arguments
        parser.add_argument(
            "--encoder-dim",
            type=int,
            help="Override encoder dimension"
        )
        
        parser.add_argument(
            "--latent-dim",
            type=int,
            help="Override latent dimension"
        )
        
        # Output arguments
        parser.add_argument(
            "--checkpoint-path",
            type=str,
            help="Override checkpoint save path"
        )
        
        parser.add_argument(
            "--resume",
            type=str,
            help="Path to checkpoint to resume training from"
        )
        
        # Training control
        parser.add_argument(
            "--no-validation",
            action="store_true",
            help="Skip validation during training"
        )
        


    def load_config(self, config_path: str = None, template_name: str = None, 
                   override_values: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Load configuration using the new hierarchical config system.
        
        Args:
            config_path: Path to configuration file
            template_name: Name of configuration template
            override_values: Dictionary of override values
            
        Returns:
            Loaded and validated configuration
        """
        try:
            if template_name:
                print(f"Loading configuration template: {template_name}")
                config = load_template(template_name, override_values=override_values)
            elif config_path:
                print(f"Loading configuration from: {config_path}")
                config = load_config(config_path, override_values=override_values)
            else:
                # Use default training template
                print("No config specified, using 'training_only' template")
                config = load_template('training_only', override_values=override_values)
            
            # Validate configuration
            warnings = validate_config(config, strict=False)
            if warnings:
                print(f"Configuration warnings: {len(warnings)} warnings found")
                for warning in warnings[:5]:  # Show first 5 warnings
                    print(f"  Warning: {warning}")
                if len(warnings) > 5:
                    print(f"  ... and {len(warnings) - 5} more warnings")
            
            return config.model_dump()
            
        except ConfigValidationError as e:
            print(f"Configuration validation error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)

    def validate_setup(self) -> None:
        """Validate that the CLI setup is correct."""
        super().validate_setup()
        
        # Additional training-specific validation
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        
        # Check training configuration exists
        if 'training' not in self.config:
            raise RuntimeError("No training configuration found")
        
        # Check model configuration exists
        if 'model' not in self.config:
            raise RuntimeError("No model configuration found")

    def _get_data_path(self) -> str:
        """Get the data path from configuration or command line."""
        if 'data' in self.config and 'data_path' in self.config['data']:
            return self.config['data']['data_path']
        else:
            raise RuntimeError("No data path specified in configuration")

    def _merge_config_with_args(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Merge configuration with command-line arguments."""
        # Build override values from command line args
        override_values = {}
        
        if args.data_path:
            override_values['data.data_path'] = args.data_path
        if args.epochs:
            override_values['training.epochs'] = args.epochs
        if args.batch_size:
            override_values['training.batch_size'] = args.batch_size
        if args.learning_rate:
            override_values['training.learning_rate'] = args.learning_rate
        if args.encoder_dim:
            override_values['model.encoder_dim'] = args.encoder_dim
        if args.latent_dim:
            override_values['model.latent_dim'] = args.latent_dim
        if args.checkpoint_path:
            override_values['training.checkpoint_path'] = args.checkpoint_path
        if args.no_validation:
            override_values['training.validate'] = False
        
        # Load configuration with overrides
        config_source = args.template or args.config
        return self.load_config(config_source, args.template, override_values)

    def _create_model(self, config: Dict[str, Any]) -> BirdsongLFADSModel2:
        """Create model from configuration."""
        model_config = config['model']
        
        model = BirdsongLFADSModel2(
            alphabet_size=model_config['alphabet_size'],
            order=model_config['order'],
            encoder_dim=model_config['encoder_dim'],
            controller_dim=model_config['controller_dim'],
            generator_dim=model_config['generator_dim'],
            factor_dim=model_config['factor_dim'],
            latent_dim=model_config['latent_dim'],
            inferred_input_dim=model_config['inferred_input_dim'],
            kappa=model_config['kappa'],
            ar_step_size=model_config['ar_step_size'],
            ar_process_var=model_config['ar_process_var']
        )
        
        return model

    def _create_trainer_config(self, config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        """Create trainer configuration from main config."""
        training_config = config['training']
        
        # Build trainer config from our hierarchical config
        trainer_config = {
            "batch_size": training_config['batch_size'],
            "epochs": training_config['epochs'],
            "learning_rate": training_config['learning_rate'],
            "kl_start_epoch": training_config['kl_start_epoch'],
            "kl_full_epoch": training_config['kl_full_epoch'],
            "checkpoint_path": training_config['checkpoint_path'],
            "print_every": training_config.get('print_every', 10),
            "l1_lambda": training_config.get('l1_lambda', 0.0001),
            "enable_kl_loss": training_config.get('enable_kl_loss', True),
            "enable_l2_loss": training_config.get('enable_l2_loss', True),
            "enable_l1_loss": training_config.get('enable_l1_loss', True),
            "disable_tqdm": training_config.get('disable_tqdm', False),
            "num_workers": training_config.get('num_workers', 0),
            "pin_memory": training_config.get('pin_memory', False),
            "val_split": training_config.get('val_split', 0.15),
            "test_split": training_config.get('test_split', 0.1),
            "validate": training_config.get('validate', True)
        }
        
        # Override checkpoint path to be in our output directory
        checkpoint_name = f"checkpoint_{self.seed_manager.master_seed}.pt"
        trainer_config["checkpoint_path"] = str(self.output_dir / checkpoint_name)
        
        return trainer_config

    def run_command(self, args: argparse.Namespace) -> None:
        """Run the training command."""
        try:
            # Setup output directory and reproducibility first
            self.setup_reproducibility(args.seed)
            self.setup_output_directory(args.output_dir)
            
            # Load and merge configuration
            self.config = self._merge_config_with_args(args)
            
            print(f"Training with seed: {self.seed_manager.master_seed}")
            print(f"Output directory: {self.output_dir}")
            
            # Dry run mode
            if args.dry_run:
                print("DRY RUN MODE - Configuration validation only")
                self.validate_setup()
                print("✓ Configuration validated successfully")
                print("✓ Data path exists and is accessible")
                print("✓ Model configuration is valid")
                print("✓ Training configuration is valid")
                print("Dry run completed successfully!")
                return
            
            # Validate setup
            self.validate_setup()
            
            # Load dataset
            data_path = self._get_data_path()
            print(f"Loading dataset from {data_path}")
            dataset = BirdsongDataset(data_path)
            print(f"Dataset loaded: {len(dataset)} samples")
            
            # Create model
            print("Creating model...")
            model = self._create_model(self.config)
            print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
            
            # Create trainer
            trainer_config = self._create_trainer_config(self.config, args)
            trainer = BirdsongTrainer(model, dataset, trainer_config)
            
            # Save metadata before training
            training_metadata = {
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "dataset_size": len(dataset),
                "trainer_config": trainer_config,
                "resume_checkpoint": args.resume
            }
            self.save_metadata(training_metadata)
            
            # Train model
            print("Starting training...")
            trainer.train(resume_from_checkpoint=args.resume)
            
            # Plot training history
            plot_path = self.output_dir / "training_history.png"
            trainer.plot_training_history(str(plot_path))
            print(f"Training history saved to {plot_path}")
            
            # Save final metadata
            final_metadata = {
                "training_completed": True,
                "final_checkpoint": trainer_config["checkpoint_path"]
            }
            self.save_metadata(final_metadata)
            
            print(f"Training completed successfully!")
            print(f"Results saved to: {self.output_dir}")
            
        except Exception as e:
            import traceback
            print(f"Error during training: {e}", file=sys.stderr)
            print("Full traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Save error metadata
            self.save_metadata({"error": str(e), "training_completed": False})
            sys.exit(1)


def main():
    """Main CLI entry point."""
    cli = TrainCLI()
    cli.execute()


if __name__ == "__main__":
    main() 