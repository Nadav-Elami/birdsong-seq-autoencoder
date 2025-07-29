"""
Command-line interface for training birdsong models.

This module provides a CLI for training Birdsong LFADS models with
configurable parameters and YAML configuration files.
"""

import argparse
import os
import sys
from typing import Any

import yaml

# Add the src directory to the path so we can import birdsong
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from birdsong.data.loader import BirdsongDataset
from birdsong.models.lfads import BirdsongLFADSModel2
from birdsong.training.trainer import BirdsongTrainer


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = ['data_path', 'model_params']

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in configuration")

    # Validate model parameters
    model_params = config['model_params']
    required_model_params = ['alphabet_size', 'order']

    for param in required_model_params:
        if param not in model_params:
            raise ValueError(f"Missing required model parameter '{param}'")

    # Validate training parameters
    if 'training_params' in config:
        training_params = config['training_params']

        if 'batch_size' in training_params and training_params['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")

        if 'epochs' in training_params and training_params['epochs'] <= 0:
            raise ValueError("epochs must be positive")

        if 'learning_rate' in training_params and training_params['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")


def create_model(config: dict[str, Any]) -> BirdsongLFADSModel2:
    """
    Create model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized BirdsongLFADSModel2 model
    """
    model_params = config['model_params']

    # Extract required parameters
    alphabet_size = model_params['alphabet_size']
    order = model_params['order']

    # Extract optional parameters with defaults
    encoder_dim = model_params.get('encoder_dim', 64)
    controller_dim = model_params.get('controller_dim', 64)
    generator_dim = model_params.get('generator_dim', 64)
    factor_dim = model_params.get('factor_dim', 32)
    latent_dim = model_params.get('latent_dim', 16)
    inferred_input_dim = model_params.get('inferred_input_dim', 8)
    kappa = model_params.get('kappa', 1.0)
    ar_step_size = model_params.get('ar_step_size', 0.99)
    ar_process_var = model_params.get('ar_process_var', 0.1)

    model = BirdsongLFADSModel2(
        alphabet_size=alphabet_size,
        order=order,
        encoder_dim=encoder_dim,
        controller_dim=controller_dim,
        generator_dim=generator_dim,
        factor_dim=factor_dim,
        latent_dim=latent_dim,
        inferred_input_dim=inferred_input_dim,
        kappa=kappa,
        ar_step_size=ar_step_size,
        ar_process_var=ar_process_var
    )

    return model


def create_trainer_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Create trainer configuration from main config.

    Args:
        config: Main configuration dictionary

    Returns:
        Trainer configuration dictionary
    """
    training_params = config.get('training_params', {})

    # Default trainer config
    trainer_config = {
        "batch_size": 32,
        "epochs": 20,
        "learning_rate": 1e-3,
        "kl_start_epoch": 2,
        "kl_full_epoch": 10,
        "checkpoint_path": "checkpoints/birdsong_lfads.pt",
        "print_every": 10,
        "l1_lambda": 0.0001,
        "plot_dir": "plots",
        "enable_kl_loss": True,
        "enable_l2_loss": True,
        "enable_l1_loss": True,
        "disable_tqdm": False,
        "num_workers": 0,
        "pin_memory": False,
        "val_split": 0.15,
        "test_split": 0.1
    }

    # Override with provided training parameters
    trainer_config.update(training_params)

    return trainer_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train a Birdsong LFADS model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )

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

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Override checkpoint path"
    )

    parser.add_argument(
        "--plot-dir",
        type=str,
        help="Override plot directory"
    )

    args = parser.parse_args()

    try:
        # Load and validate configuration
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        validate_config(config)

        # Override config with command-line arguments
        if args.epochs is not None:
            if 'training_params' not in config:
                config['training_params'] = {}
            config['training_params']['epochs'] = args.epochs

        if args.batch_size is not None:
            if 'training_params' not in config:
                config['training_params'] = {}
            config['training_params']['batch_size'] = args.batch_size

        if args.learning_rate is not None:
            if 'training_params' not in config:
                config['training_params'] = {}
            config['training_params']['learning_rate'] = args.learning_rate

        if args.checkpoint_path is not None:
            if 'training_params' not in config:
                config['training_params'] = {}
            config['training_params']['checkpoint_path'] = args.checkpoint_path

        if args.plot_dir is not None:
            if 'training_params' not in config:
                config['training_params'] = {}
            config['training_params']['plot_dir'] = args.plot_dir

        # Load dataset
        print(f"Loading dataset from {config['data_path']}")
        dataset = BirdsongDataset(config['data_path'])
        print(f"Dataset loaded: {len(dataset)} samples")

        # Create model
        print("Creating model...")
        model = create_model(config)
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

        # Create trainer
        trainer_config = create_trainer_config(config)
        trainer = BirdsongTrainer(model, dataset, trainer_config)

        # Train model
        print("Starting training...")
        trainer.train(resume_from_checkpoint=args.resume)

        # Plot training history
        plot_path = os.path.join(trainer_config['plot_dir'], 'training_history.png')
        trainer.plot_training_history(plot_path)
        print(f"Training history saved to {plot_path}")

        print("Training completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
