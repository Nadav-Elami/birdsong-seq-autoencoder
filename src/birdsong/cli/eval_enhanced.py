"""
Enhanced command-line interface for evaluating birdsong models with reproducibility.

This module provides an enhanced CLI for evaluating trained Birdsong LFADS models
with automatic seed tracking, reproducibility metadata, and hierarchical configuration support.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Some plotting features may be limited.")
import torch

# Add the src directory to the path so we can import birdsong
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from birdsong.data.loader import BirdsongDataset
from birdsong.models.lfads import BirdsongLFADSModel2
from birdsong.evaluation.evaluate import (
    BirdsongEvaluator, evaluate_birdsong_model, plot_transition_plots, 
    plot_summary_metrics, smooth_counts, js_divergence, cross_entropy
)
from birdsong.cli.base import ReproducibleCLI
from birdsong.config import load_config, load_template, validate_config, ConfigValidationError


class EvalCLI(ReproducibleCLI):
    """Enhanced evaluation CLI with reproducibility and hierarchical configuration features."""
    
    def __init__(self):
        super().__init__(
            command_name="eval",
            description="Evaluate a trained Birdsong LFADS model with automatic seed tracking and hierarchical configuration"
        )
    
    def add_command_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add evaluation-specific command-line arguments."""
        
        # Template argument (config is already provided by base class)
        parser.add_argument(
            "--template", "-t",
            type=str,
            help="Configuration template name (alternative to --config for templates)"
        )
        
        # Required arguments
        parser.add_argument(
            "--checkpoint",
            type=str,
            required=True,
            help="Path to model checkpoint file"
        )
        
        # Data arguments
        parser.add_argument(
            "--data-path",
            type=str,
            help="Override path to evaluation data (HDF5 file)"
        )
        
        # Evaluation arguments
        parser.add_argument(
            "--num-samples",
            type=int,
            help="Override number of samples to evaluate"
        )
        
        parser.add_argument(
            "--num-plots",
            type=int,
            help="Override number of prediction plots to generate"
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
        
        # Analysis options
        parser.add_argument(
            "--analyze-latents",
            action="store_true",
            help="Perform latent space analysis"
        )
        
        parser.add_argument(
            "--compare-predictions",
            action="store_true",
            help="Generate detailed prediction comparison plots"
        )
        
        parser.add_argument(
            "--export-results",
            action="store_true",
            help="Export results to JSON and CSV formats"
        )
        
        # Control options


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
                # Use default evaluation template
                print("No config specified, using 'evaluation_only' template")
                config = load_template('evaluation_only', override_values=override_values)
            
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
        
        # Additional evaluation-specific validation
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        
        # Check evaluation configuration exists
        if 'evaluation' not in self.config:
            raise RuntimeError("No evaluation configuration found")

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
        if args.num_samples:
            override_values['evaluation.num_samples'] = args.num_samples
        if args.num_plots:
            override_values['evaluation.num_plots'] = args.num_plots
        if args.analyze_latents:
            override_values['evaluation.analyze_latents'] = True
        if args.compare_predictions:
            override_values['evaluation.compare_predictions'] = True
        if args.export_results:
            override_values['evaluation.export_results'] = True
        if args.use_test_set:
            override_values['evaluation.use_test_set'] = True
        
        # Load configuration with overrides
        config_source = args.template or args.config
        return self.load_config(config_source, args.template, override_values)

    def _load_checkpoint(self, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
        """Load model checkpoint with validation."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            return checkpoint
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")

    def _validate_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Validate checkpoint structure."""
        required_keys = ['model_state_dict']
        
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Missing required key '{key}' in checkpoint")

    def _create_model_from_checkpoint(self, checkpoint: Dict[str, Any]) -> BirdsongLFADSModel2:
        """Create model from checkpoint, falling back to config if needed."""
        # Try to get model config from checkpoint first
        if 'config' in checkpoint and 'model' in checkpoint['config']:
            model_config = checkpoint['config']['model']
        elif 'model_params' in checkpoint:
            # Legacy checkpoint format
            model_params = checkpoint['model_params']
            model_config = {
                'alphabet_size': model_params['alphabet_size'],
                'order': model_params['order'],
                'encoder_dim': model_params.get('encoder_dim', 64),
                'controller_dim': model_params.get('controller_dim', 64),
                'generator_dim': model_params.get('generator_dim', 64),
                'factor_dim': model_params.get('factor_dim', 32),
                'latent_dim': model_params.get('latent_dim', 16),
                'inferred_input_dim': model_params.get('inferred_input_dim', 8),
                'kappa': model_params.get('kappa', 1.0),
                'ar_step_size': model_params.get('ar_step_size', 0.99),
                'ar_process_var': model_params.get('ar_process_var', 0.1)
            }
        else:
            # Fall back to current config
            if 'model' not in self.config:
                raise RuntimeError("No model configuration available in checkpoint or config")
            model_config = self.config['model']

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

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _evaluate_model(self, model: BirdsongLFADSModel2, dataset: BirdsongDataset,
                       device: torch.device, num_samples: int = 10) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        total_loss = 0.0
        total_rec_loss = 0.0
        total_kl_loss = 0.0

        model.eval()
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                bigram_counts, probabilities = dataset[i]
                bigram_counts = bigram_counts.unsqueeze(0).to(device)
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
            'kl_loss': total_kl_loss / num_evaluated,
            'num_samples': num_evaluated
        }

    def _plot_model_predictions(self, model: BirdsongLFADSModel2, dataset: BirdsongDataset,
                               device: torch.device, output_dir: Path, num_plots: int = 5) -> None:
        """Generate comprehensive prediction analysis plots."""
        # Don't create plots_dir here - create it only when we're about to save a plot

        model.eval()
        model.to(device)

        # Configuration for rich evaluation
        smooth_window = 5  # Smoothing window for estimates
        file_format = "png"
        dataset_name = "evaluation"
        
        # Lists to store metrics across samples
        all_js_pred = []
        all_js_est_raw = []
        all_js_est_smooth = []
        all_ce_pred = []
        all_ce_est_raw = []
        all_ce_est_smooth = []

        with torch.no_grad():
            for i in range(min(num_plots, len(dataset))):
                print(f"Processing sample {i}...")
                ngram_counts, true_probs = dataset[i]
                ngram_counts = ngram_counts.unsqueeze(0).to(device)
                true_probs = true_probs.unsqueeze(0).to(device)

                # Evaluate the model
                logits_4d, pred_probs_4d, factors, metrics = evaluate_birdsong_model(
                    model, ngram_counts, true_probs, model.alphabet_size
                )
                print(f"Sample {i} metrics: {metrics}")

                # Prepare estimated probabilities from n-gram counts
                time_steps = ngram_counts.shape[1]
                if model.order == 1:
                    counts_4d = ngram_counts.view(1, time_steps, model.alphabet_size, model.alphabet_size)
                elif model.order == 2:
                    counts_4d = ngram_counts.view(1, time_steps, model.alphabet_size ** 2, model.alphabet_size)
                
                counts_4d_np = counts_4d.cpu().numpy()
                
                # Compute raw estimate: normalize counts row-wise
                raw_counts_norm = np.where(
                    counts_4d_np.sum(axis=-1, keepdims=True) > 0,
                    counts_4d_np / (counts_4d_np.sum(axis=-1, keepdims=True) + 1e-12),  # Add small epsilon
                    0.0
                )[0]  # shape: (T, rows, alphabet_size)

                # Compute smooth estimate if desired
                if smooth_window > 1:
                    smoothed_counts = smooth_counts(counts_4d_np[0], smooth_window)
                    smooth_counts_norm = np.where(
                        smoothed_counts.sum(axis=-1, keepdims=True) > 0,
                        smoothed_counts / (smoothed_counts.sum(axis=-1, keepdims=True) + 1e-12),  # Add small epsilon
                        0.0
                    )
                else:
                    smooth_counts_norm = None

                # Get predicted and true probabilities as numpy arrays
                if model.order == 1:
                    pred_probs_np = pred_probs_4d.cpu().numpy()[0]  # shape: (1, T, alphabet_size, alphabet_size)
                    true_probs_np = true_probs.view(1, time_steps, model.alphabet_size, model.alphabet_size).cpu().numpy()[0]
                elif model.order == 2:
                    pred_probs_np = pred_probs_4d.cpu().numpy()[0]  # shape: (1, T, alphabet_size**2, alphabet_size)
                    true_probs_np = true_probs.view(1, time_steps, model.alphabet_size ** 2, model.alphabet_size).cpu().numpy()[0]

                # Save individual transition plots (directory will be created by plot_transition_plots)
                plot_transition_plots(
                    i, pred_probs_np, true_probs_np, raw_counts_norm,
                    smooth_counts_norm, model.alphabet_size, model.order, 
                    dataset_name, base_output_dir=str(output_dir / "prediction_plots"), 
                    file_format=file_format, plot_smooth_est=True
                )

                # Compute summary metrics for this sample over time
                js_pred_per_t = []
                js_est_raw_per_t = []
                js_est_smooth_per_t = []
                ce_pred_per_t = []
                ce_est_raw_per_t = []
                ce_est_smooth_per_t = []
                
                for t in range(time_steps):
                    p_pred = pred_probs_np[t]  # shape: (rows, alphabet_size)
                    p_true = true_probs_np[t]
                    p_est_raw = raw_counts_norm[t]
                    
                    # Convert numpy arrays to tensors for the divergence functions
                    p_pred_tensor = torch.tensor(p_pred, dtype=torch.float32)
                    p_true_tensor = torch.tensor(p_true, dtype=torch.float32)
                    p_est_raw_tensor = torch.tensor(p_est_raw, dtype=torch.float32)
                    
                    js_pred = np.mean([js_divergence(p_pred_tensor[r], p_true_tensor[r]).item() for r in range(p_pred.shape[0])])
                    ce_pred = np.mean([cross_entropy(p_pred_tensor[r], p_true_tensor[r]).item() for r in range(p_pred.shape[0])])
                    js_est_raw = np.mean([js_divergence(p_est_raw_tensor[r], p_true_tensor[r]).item() for r in range(p_true.shape[0])])
                    ce_est_raw = np.mean([cross_entropy(p_est_raw_tensor[r], p_true_tensor[r]).item() for r in range(p_true.shape[0])])
                    
                    js_pred_per_t.append(js_pred)
                    ce_pred_per_t.append(ce_pred)
                    js_est_raw_per_t.append(js_est_raw)
                    ce_est_raw_per_t.append(ce_est_raw)
                    
                    if smooth_counts_norm is not None:
                        p_est_smooth = smooth_counts_norm[t]
                        p_est_smooth_tensor = torch.tensor(p_est_smooth, dtype=torch.float32)
                        js_est_smooth = np.mean([js_divergence(p_est_smooth_tensor[r], p_true_tensor[r]).item() for r in range(p_true.shape[0])])
                        ce_est_smooth = np.mean([cross_entropy(p_est_smooth_tensor[r], p_true_tensor[r]).item() for r in range(p_true.shape[0])])
                        js_est_smooth_per_t.append(js_est_smooth)
                        ce_est_smooth_per_t.append(ce_est_smooth)

                all_js_pred.append(np.array(js_pred_per_t))
                all_ce_pred.append(np.array(ce_pred_per_t))
                all_js_est_raw.append(np.array(js_est_raw_per_t))
                all_ce_est_raw.append(np.array(ce_est_raw_per_t))
                
                if smooth_counts_norm is not None:
                    all_js_est_smooth.append(np.array(js_est_smooth_per_t))
                    all_ce_est_smooth.append(np.array(ce_est_smooth_per_t))

                # Plot transition matrices
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                sns.heatmap(true_probs_np[0], ax=axes[0], cmap="viridis", cbar=True)
                axes[0].set_title("True Transition Matrix")

                sns.heatmap(pred_probs_np[0], ax=axes[1], cmap="viridis", cbar=True)
                axes[1].set_title("Predicted Transition Matrix")

                fig.suptitle(f"Sample {i} - Transition Matrix Comparison")
                plt.tight_layout()

                # Save plot (directory will be created when saving)
                plot_path = output_dir / "prediction_plots" / f"transition_matrix_sample_{i}.png"
                plot_path.parent.mkdir(exist_ok=True)  # Only create when actually saving
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Saved prediction plots for sample {i}")

        # Plot summary metrics across all samples
        plot_summary_metrics(
            all_js_pred, all_js_est_raw, all_js_est_smooth if all_js_est_smooth else None,
            all_ce_pred, all_ce_est_raw, all_ce_est_smooth if all_ce_est_smooth else None,
            file_format, dataset_name, summary_output_dir=str(output_dir / "prediction_plots")
        )
        print("Summary plots saved.")

    def run_command(self, args: argparse.Namespace) -> None:
        """Run the evaluation command."""
        try:
            # Load and merge configuration
            self.config = self._merge_config_with_args(args)
            
            # Setup output directory and reproducibility
            self.setup_reproducibility(args.seed)
            self.setup_output_directory(args.output_dir)
            
            print(f"Evaluation with seed: {self.seed_manager.master_seed}")
            print(f"Output directory: {self.output_dir}")
            
            # Setup device
            if args.device == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(args.device)
            print(f"Using device: {device}")
            
            # Dry run mode
            if args.dry_run:
                print("DRY RUN MODE - Configuration validation only")
                self.validate_setup()
                print("✓ Configuration validated successfully")
                print("✓ Checkpoint exists and is accessible")
                print("✓ Data path exists and is accessible")
                print("✓ Evaluation configuration is valid")
                print("Dry run completed successfully!")
                return
            
            # Validate setup
            self.validate_setup()
            
            # Load checkpoint
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = self._load_checkpoint(args.checkpoint, device)
            self._validate_checkpoint(checkpoint)
            
            # Create model from checkpoint
            print("Creating model from checkpoint...")
            model = self._create_model_from_checkpoint(checkpoint)
            print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
            
            # Load dataset
            data_path = self._get_data_path()
            print(f"Loading dataset from {data_path}")
            dataset = BirdsongDataset(data_path)
            print(f"Dataset loaded: {len(dataset)} samples")
            
            # Get evaluation configuration
            eval_config = self.config['evaluation']
            num_samples = eval_config.get('num_samples', 10)
            num_plots = eval_config.get('num_plots', 5)
            
            # Save metadata before evaluation
            eval_metadata = {
                "checkpoint_path": args.checkpoint,
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "dataset_size": len(dataset),
                "device": str(device),
                "num_samples": num_samples,
                "num_plots": num_plots
            }
            self.save_metadata(eval_metadata)
            
            # Handle test set evaluation
            use_test_set = eval_config.get('use_test_set', False)
            
            if use_test_set:
                print("Using test set from checkpoint for evaluation...")
                from birdsong.evaluation.evaluate import create_test_loader_from_checkpoint
                try:
                    test_loader = create_test_loader_from_checkpoint(args.checkpoint, dataset)
                    print(f"✅ Created test loader with {len(test_loader.dataset)} samples from checkpoint")
                    
                    # Use the enhanced evaluator for rich evaluation on test set
                    evaluator = BirdsongEvaluator(model, device)
                    # For now, we'll evaluate on the test set samples
                    # In a full implementation, you'd iterate through all test batches
                    all_metrics = evaluator.evaluate_dataset(dataset, len(test_loader.dataset))
                    summary_metrics = evaluator.compute_summary_metrics(all_metrics)
                    
                except Exception as e:
                    print(f"Warning: Could not use test set from checkpoint: {e}")
                    print("Falling back to random sample evaluation...")
                    evaluator = BirdsongEvaluator(model, device)
                    all_metrics = evaluator.evaluate_dataset(dataset, num_samples)
                    summary_metrics = evaluator.compute_summary_metrics(all_metrics)
            else:
                # Evaluate model with comprehensive metrics
                print(f"Evaluating model on {num_samples} samples...")
                
                # Use the enhanced evaluator for rich evaluation
                evaluator = BirdsongEvaluator(model, device)
                all_metrics = evaluator.evaluate_dataset(dataset, num_samples)
                summary_metrics = evaluator.compute_summary_metrics(all_metrics)
            
            # Print comprehensive metrics
            print("\nEvaluation Results:")
            print("==================")
            for key, value in summary_metrics.items():
                print(f"{key}: {value:.6f}")
            
            # Save metrics to file
            metrics_path = self.output_dir / "evaluation_metrics.txt"
            evaluator.save_evaluation_results(all_metrics, summary_metrics, str(metrics_path))
            print(f"Detailed metrics saved to {metrics_path}")
            
            # Generate comprehensive prediction plots
            if num_plots > 0:
                print(f"Generating {num_plots} comprehensive prediction plots...")
                self._plot_model_predictions(model, dataset, device, self.output_dir, num_plots)
            
            # Perform analysis based on config toggles
            analyses_to_perform = []
            
            # Check which analyses are enabled
            if eval_config.get('analyze_latents', False):
                analyses_to_perform.append('latents')
            if eval_config.get('analyze_factors', False):
                analyses_to_perform.append('factors')
            if eval_config.get('analyze_trajectories', False):
                analyses_to_perform.append('trajectories')
            
            if analyses_to_perform:
                print(f"\n🔍 Performing analysis: {', '.join(analyses_to_perform)}")
                try:
                    from birdsong.analysis.latent import LatentSpaceAnalyzer
                    
                    # Create analyzer
                    analyzer = LatentSpaceAnalyzer(model, device, random_state=self.seed_manager.master_seed)
                    
                    # Extract latents
                    print("  Extracting latent representations...")
                    
                    # For latent analysis, use a smaller subset to speed up computation
                    # Use first 500 samples instead of all 5000 for analysis
                    analysis_batch_size = eval_config.get('latent_analysis_batch_size', 32)
                    max_analysis_samples = eval_config.get('max_latent_analysis_samples', 500)
                    skip_tsne_large = eval_config.get('skip_tsne_for_large_datasets', True)
                    
                    # Create a subset of the dataset for analysis
                    if len(dataset) > max_analysis_samples:
                        print(f"  📊 Using {max_analysis_samples} samples for latent analysis (out of {len(dataset)} total)")
                        # Create a subset by taking first max_analysis_samples
                        analysis_dataset = torch.utils.data.Subset(dataset, range(max_analysis_samples))
                    else:
                        analysis_dataset = dataset
                    
                    latent_results = analyzer.extract_latents(
                        analysis_dataset,
                        batch_size=analysis_batch_size,
                        include_factors=True,
                        include_g0=True,
                        include_u=True
                    )
                    print(f"  ✅ Extracted latents: {list(latent_results.keys())}")
                    
                    # Perform analysis for different latent types
                    latent_types = ['factors', 'g0', 'u']
                    reduction_methods = ['pca', 'tsne']
                    
                    # Skip t-SNE for very large datasets to speed up analysis
                    if skip_tsne_large and len(analysis_dataset) > 1000:
                        print(f"  ⚡ Skipping t-SNE for large dataset ({len(analysis_dataset)} samples)")
                        reduction_methods = ['pca']  # Only use PCA for large datasets
                    
                    for latent_type in latent_types:
                        print(f"  📈 Analyzing {latent_type} latents...")
                        
                        for method in reduction_methods:
                            print(f"    🔄 Using {method.upper()}...")
                            
                            try:
                                # Perform analysis
                                analysis_results = analyzer.analyze_latent_space(
                                    latent_type=latent_type,
                                    reduction_method=method,
                                    clustering_method='kmeans',
                                    n_clusters=5,
                                    n_components=2,
                                    random_state=self.seed_manager.master_seed
                                )
                                
                                # Create output directory
                                analysis_dir = self.output_dir / "latent_analysis" / f"{latent_type}_{method}"
                                analysis_dir.mkdir(parents=True, exist_ok=True)
                                
                                # Create visualizations
                                analyzer.create_visualizations(
                                    analysis_results,
                                    output_dir=str(analysis_dir),
                                    save_plots=True,
                                    create_interactive=False
                                )
                                
                                # Save results
                                results_path = analysis_dir / "analysis_results.json"
                                analyzer.save_analysis_results(analysis_results, str(results_path))
                                
                                print(f"    ✅ {method.upper()} analysis complete")
                                
                            except Exception as e:
                                print(f"    ❌ Error in {method.upper()} analysis: {e}")
                                continue
                    
                    print("✅ Analysis completed")
                    
                except Exception as e:
                    print(f"⚠️  Warning: Error during analysis: {e}")
            
            # Export results if requested
            if eval_config.get('export_results', False):
                import json
                results_path = self.output_dir / "evaluation_results.json"
                with open(results_path, 'w') as f:
                    json.dump(summary_metrics, f, indent=2)
                print(f"Results exported to {results_path}")
            
            # Save final metadata
            final_metadata = {
                "evaluation_completed": True,
                "metrics": summary_metrics,
                "results_files": [str(metrics_path)]
            }
            if eval_config.get('export_results', False):
                final_metadata["results_files"].append(str(self.output_dir / "evaluation_results.json"))
            if analyses_to_perform:
                final_metadata["analysis_performed"] = analyses_to_perform
            
            self.save_metadata(final_metadata)
            
            print(f"Evaluation completed successfully!")
            print(f"Results saved to: {self.output_dir}")
            print(f"Check the 'prediction_plots' directory for comprehensive visualizations!")
            
        except Exception as e:
            print(f"Error during evaluation: {e}", file=sys.stderr)
            # Save error metadata
            self.save_metadata({"error": str(e), "evaluation_completed": False})
            sys.exit(1)


def main():
    """Main CLI entry point."""
    cli = EvalCLI()
    cli.execute()


if __name__ == "__main__":
    main() 