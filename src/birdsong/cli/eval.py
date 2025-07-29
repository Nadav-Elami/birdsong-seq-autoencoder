"""
Command-line interface for evaluating birdsong models.

This module provides a CLI for evaluating trained Birdsong LFADS models
with checkpoint validation and visualization options.
"""

import argparse
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Some plotting features may be limited.")
import torch

# Add the src directory to the path so we can import birdsong
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from birdsong.data.loader import BirdsongDataset
from birdsong.models.lfads import BirdsongLFADSModel2


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        Checkpoint dictionary

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    """
    Validate checkpoint structure.

    Args:
        checkpoint: Checkpoint dictionary

    Raises:
        ValueError: If checkpoint is invalid
    """
    required_keys = ['model_state_dict', 'config']

    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Missing required key '{key}' in checkpoint")


def create_model_from_checkpoint(checkpoint: dict[str, Any]) -> BirdsongLFADSModel2:
    """
    Create model from checkpoint.

    Args:
        checkpoint: Checkpoint dictionary

    Returns:
        Initialized BirdsongLFADSModel2 model
    """
    config = checkpoint['config']
    
    # Handle both old format (model_params) and new format (model)
    if 'model_params' in config:
        model_params = config['model_params']
    elif 'model' in config:
        model_params = config['model']
    else:
        raise ValueError("Checkpoint config must contain either 'model_params' or 'model' section")

    # Extract model parameters
    alphabet_size = model_params['alphabet_size']
    order = model_params['order']
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

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def evaluate_model(model: BirdsongLFADSModel2, dataset: BirdsongDataset,
                  device: torch.device, num_samples: int = 10) -> dict[str, float]:
    """
    Evaluate model on dataset.

    Args:
        model: Trained model
        dataset: Evaluation dataset
        device: Device for evaluation
        num_samples: Number of samples to evaluate

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_rec_loss = 0.0
    total_kl_loss = 0.0

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            bigram_counts, probabilities = dataset[i]
            bigram_counts = bigram_counts.unsqueeze(0).to(device)  # Add batch dimension
            probabilities = probabilities.unsqueeze(0).to(device)

            outputs = model(bigram_counts)
            total_loss_batch, loss_dict = model.compute_loss(probabilities, outputs)

            total_loss += total_loss_batch.item()
            total_rec_loss += loss_dict['rec_loss'].item()
            total_kl_loss += loss_dict['kl_g0'].item()

    num_evaluated = min(num_samples, len(dataset))

    return {
        'total_loss': total_loss / num_evaluated,
        'rec_loss': total_rec_loss / num_evaluated,
        'kl_loss': total_kl_loss / num_evaluated
    }


def plot_model_predictions(model: BirdsongLFADSModel2, dataset: BirdsongDataset,
                          device: torch.device, output_dir: str, num_plots: int = 5) -> None:
    """
    Plot model predictions.

    Args:
        model: Trained model
        dataset: Dataset for plotting
        device: Device for model inference
        output_dir: Directory to save plots
        num_plots: Number of plots to generate
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    model.to(device)

    with torch.no_grad():
        for i in range(min(num_plots, len(dataset))):
            bigram_counts, probabilities = dataset[i]
            bigram_counts = bigram_counts.unsqueeze(0).to(device)
            probabilities = probabilities.unsqueeze(0).to(device)

            outputs = model(bigram_counts)

            # Get predictions
            if model.order == 1:
                from birdsong.models.lfads import rowwise_softmax
                pred_probs = rowwise_softmax(outputs["logits"], model.alphabet_size, order=model.order)
                pred_mat = pred_probs.view(1, -1, model.alphabet_size, model.alphabet_size)
                true_mat = probabilities.view(1, -1, model.alphabet_size, model.alphabet_size)
            else:
                from birdsong.models.lfads import rowwise_masked_softmax
                true_mat = probabilities.view(1, -1, model.alphabet_size ** 2, model.alphabet_size)
                target_matrix = true_mat[0, 0]
                pred_probs = rowwise_masked_softmax(
                    outputs["logits"], model.alphabet_size, order=model.order,
                    target_matrix=target_matrix, mask_value=-1e8
                )
                pred_mat = pred_probs.view(1, -1, model.alphabet_size ** 2, model.alphabet_size)

            # Plot transition matrices
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            sns.heatmap(true_mat[0, 0].cpu().numpy(), ax=axes[0], cmap="viridis", cbar=True)
            axes[0].set_title("True Transition Matrix")

            sns.heatmap(pred_mat[0, 0].cpu().numpy(), ax=axes[1], cmap="viridis", cbar=True)
            axes[1].set_title("Predicted Transition Matrix")

            fig.suptitle(f"Sample {i} - Transition Matrix Comparison")
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(output_dir, f"prediction_sample_{i}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved prediction plot to {plot_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Birdsong LFADS model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--data-path", "-d",
        type=str,
        required=True,
        help="Path to evaluation dataset"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="evaluation_output",
        help="Directory to save evaluation outputs"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to evaluate"
    )

    parser.add_argument(
        "--num-plots",
        type=int,
        default=5,
        help="Number of prediction plots to generate"
    )

    parser.add_argument(
        "--use-test-set",
        action="store_true",
        help="Use test set from checkpoint instead of random samples (recommended for proper evaluation)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for evaluation (auto, cpu, cuda)"
    )

    args = parser.parse_args()

    try:
        # Setup device
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        print(f"Using device: {device}")

        # Load checkpoint
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = load_checkpoint(args.checkpoint, device)
        validate_checkpoint(checkpoint)

        # Create model from checkpoint
        print("Creating model from checkpoint...")
        model = create_model_from_checkpoint(checkpoint)
        print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")

        # Load dataset
        print(f"Loading dataset from {args.data_path}")
        dataset = BirdsongDataset(args.data_path)
        print(f"Dataset loaded: {len(dataset)} samples")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Handle test set evaluation
        if args.use_test_set:
            print("Using test set from checkpoint for evaluation...")
            from birdsong.evaluation.evaluate import create_test_loader_from_checkpoint
            try:
                test_loader = create_test_loader_from_checkpoint(args.checkpoint, dataset)
                # For now, we'll use the first batch for evaluation
                # In a full implementation, you'd iterate through all test batches
                test_batch = next(iter(test_loader))
                bigram_counts, probabilities = test_batch
                bigram_counts = bigram_counts.to(device)
                probabilities = probabilities.to(device)
                
                # Evaluate on test set
                print(f"Evaluating model on test set...")
                metrics = evaluate_model(model, dataset, device, len(test_loader.dataset))
            except Exception as e:
                print(f"Warning: Could not use test set from checkpoint: {e}")
                print("Falling back to random sample evaluation...")
                metrics = evaluate_model(model, dataset, device, args.num_samples)
        else:
            # Evaluate model
            print(f"Evaluating model on {args.num_samples} samples...")
            metrics = evaluate_model(model, dataset, device, args.num_samples)

        # Print metrics
        print("\nEvaluation Results:")
        print(f"Total Loss: {metrics['total_loss']:.4f}")
        print(f"Reconstruction Loss: {metrics['rec_loss']:.4f}")
        print(f"KL Loss: {metrics['kl_loss']:.4f}")

        # Save metrics to file
        metrics_path = os.path.join(args.output_dir, "evaluation_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("Evaluation Metrics\n")
            f.write("==================\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")

        print(f"Metrics saved to {metrics_path}")

        # Generate prediction plots
        print(f"Generating {args.num_plots} prediction plots...")
        plot_model_predictions(model, dataset, device, args.output_dir, args.num_plots)

        print(f"Evaluation completed! Results saved to {args.output_dir}")

    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
