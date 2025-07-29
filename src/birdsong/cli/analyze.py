"""
CLI command for latent space analysis.

This module provides a command-line interface for analyzing latent representations
learned by the LFADS model using various dimensionality reduction and clustering techniques.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
import numpy as np

from ..analysis.latent import LatentSpaceAnalyzer
from ..cli.base import ReproducibleCLI
from ..config.loader import load_config
from ..data.loader import BirdsongDataset
from ..models.lfads import BirdsongLFADSModel2
from ..utils.reproducibility import set_global_seed


class AnalyzeCLI(ReproducibleCLI):
    """
    CLI for latent space analysis of LFADS models.
    
    Provides comprehensive analysis of latent representations including:
    - Dimensionality reduction (PCA, t-SNE, UMAP)
    - Clustering analysis (K-means, DBSCAN)
    - Trajectory analysis
    - Interactive visualizations
    """
    
    def __init__(self):
        super().__init__(
            command_name="birdsong-analyze",
            description="Analyze latent representations from trained LFADS models"
        )
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments."""
        # Model and data arguments
        parser.add_argument(
            "--checkpoint",
            type=str,
            required=True,
            help="Path to trained model checkpoint"
        )
        parser.add_argument(
            "--data-path",
            type=str,
            required=True,
            help="Path to HDF5 dataset file"
        )
        parser.add_argument(
            "--config",
            type=str,
            help="Path to configuration file (optional, will use checkpoint config)"
        )
        
        # Analysis parameters
        parser.add_argument(
            "--latent-type",
            type=str,
            choices=["factors", "g0", "u"],
            default="factors",
            help="Type of latent representation to analyze (default: factors)"
        )
        parser.add_argument(
            "--reduction-method",
            type=str,
            choices=["pca", "tsne", "umap"],
            default="pca",
            help="Dimensionality reduction method (default: pca)"
        )
        parser.add_argument(
            "--clustering-method",
            type=str,
            choices=["kmeans", "dbscan"],
            default="kmeans",
            help="Clustering method (default: kmeans)"
        )
        parser.add_argument(
            "--n-clusters",
            type=int,
            default=5,
            help="Number of clusters for K-means (default: 5)"
        )
        parser.add_argument(
            "--n-components",
            type=int,
            default=2,
            help="Number of components for dimensionality reduction (default: 2)"
        )
        
        # Analysis toggles (respect config settings)
        parser.add_argument(
            "--analyze-latents",
            action="store_true",
            help="Enable latent space analysis (overrides config)"
        )
        parser.add_argument(
            "--analyze-factors",
            action="store_true",
            help="Enable factor analysis (overrides config)"
        )
        parser.add_argument(
            "--analyze-trajectories",
            action="store_true",
            help="Enable trajectory analysis (overrides config)"
        )
        parser.add_argument(
            "--disable-latents",
            action="store_true",
            help="Disable latent space analysis (overrides config)"
        )
        parser.add_argument(
            "--disable-factors",
            action="store_true",
            help="Disable factor analysis (overrides config)"
        )
        parser.add_argument(
            "--disable-trajectories",
            action="store_true",
            help="Disable trajectory analysis (overrides config)"
        )
        
        # Output options
        parser.add_argument(
            "--output-dir",
            type=str,
            default="latent_analysis",
            help="Output directory for analysis results (default: latent_analysis)"
        )
        parser.add_argument(
            "--save-plots",
            action="store_true",
            default=True,
            help="Save plots to disk (default: True)"
        )
        parser.add_argument(
            "--interactive",
            action="store_true",
            default=False,
            help="Create interactive visualizations (requires plotly)"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=64,
            help="Batch size for latent extraction (default: 64)"
        )
        
        # Advanced options
        parser.add_argument(
            "--perplexity",
            type=float,
            default=30.0,
            help="Perplexity for t-SNE (default: 30.0)"
        )
        parser.add_argument(
            "--max-iter",
            type=int,
            default=1000,
            help="Maximum iterations for t-SNE (default: 1000, minimum: 250)"
        )
        parser.add_argument(
            "--n-neighbors",
            type=int,
            default=15,
            help="Number of neighbors for UMAP (default: 15)"
        )
        parser.add_argument(
            "--min-dist",
            type=float,
            default=0.1,
            help="Minimum distance for UMAP (default: 0.1)"
        )
    
    def run(self, args: argparse.Namespace) -> None:
        """Execute the latent space analysis."""
        # Set up reproducibility
        seed_manager = set_global_seed(args.seed)
        
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            # Try to load config from checkpoint
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                print("⚠️  No configuration found in checkpoint, using defaults")
                config = None
        
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {device}")
        
        # Load model
        print(f"📦 Loading model from {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model_state = checkpoint['model_state_dict']
            
            # Create model with appropriate parameters
            if config:
                model = BirdsongLFADSModel2(
                    alphabet_size=config.data.alphabet_size,
                    order=config.data.order,
                    encoder_dim=config.model.encoder_dim,
                    controller_dim=config.model.controller_dim,
                    generator_dim=config.model.generator_dim,
                    factor_dim=config.model.factor_dim,
                    latent_dim=config.model.latent_dim,
                    inferred_input_dim=config.model.inferred_input_dim,
                    kappa=config.model.kappa,
                    ar_step_size=config.model.ar_step_size,
                    ar_process_var=config.model.ar_process_var
                )
            else:
                # Try to infer parameters from model state
                print("⚠️  Using default model parameters")
                model = BirdsongLFADSModel2(
                    alphabet_size=7,  # Default
                    order=1,
                    encoder_dim=64,
                    controller_dim=64,
                    generator_dim=64,
                    factor_dim=32,
                    latent_dim=16,
                    inferred_input_dim=8
                )
            
            model.load_state_dict(model_state)
            model.to(device)
            model.eval()
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
        
        # Load dataset
        print(f"📊 Loading dataset from {args.data_path}")
        try:
            dataset = BirdsongDataset(args.data_path)
            print(f"✅ Dataset loaded: {len(dataset)} samples")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            sys.exit(1)
        
        # Create analyzer
        analyzer = LatentSpaceAnalyzer(model, device, random_state=args.seed)
        
        # Extract latents
        print(f"🔍 Extracting {args.latent_type} latent representations...")
        try:
            latent_results = analyzer.extract_latents(
                dataset,
                batch_size=args.batch_size,
                include_factors=True,
                include_g0=True,
                include_u=True
            )
            print(f"✅ Extracted latents: {list(latent_results.keys())}")
        except Exception as e:
            print(f"❌ Error extracting latents: {e}")
            sys.exit(1)
        
        # Determine which analyses to perform based on config and command line args
        analyses_to_perform = []
        
        # Check config settings if available
        if config and hasattr(config, 'evaluation'):
            eval_config = config.evaluation
            
            # Check if latent analysis is enabled in config
            if eval_config.analyze_latents and not args.disable_latents:
                analyses_to_perform.append(('latents', 'factors'))
            
            # Check if factor analysis is enabled in config
            if eval_config.analyze_factors and not args.disable_factors:
                analyses_to_perform.append(('factors', 'factors'))
            
            # Check if trajectory analysis is enabled in config
            if eval_config.analyze_trajectories and not args.disable_trajectories:
                analyses_to_perform.append(('trajectories', 'factors'))
        
        # Override with command line arguments
        if args.analyze_latents:
            analyses_to_perform.append(('latents', 'factors'))
        if args.analyze_factors:
            analyses_to_perform.append(('factors', 'factors'))
        if args.analyze_trajectories:
            analyses_to_perform.append(('trajectories', 'factors'))
        
        # If no specific analyses requested, use the latent type specified
        if not analyses_to_perform:
            analyses_to_perform.append(('latents', args.latent_type))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_analyses = []
        for analysis in analyses_to_perform:
            if analysis not in seen:
                seen.add(analysis)
                unique_analyses.append(analysis)
        
        print(f"📈 Performing {len(unique_analyses)} analysis(es):")
        for analysis_type, latent_type in unique_analyses:
            print(f"  - {analysis_type.upper()} analysis of {latent_type}")
        
        all_results = {}
        
        for analysis_type, latent_type in unique_analyses:
            print(f"\n🔄 Running {analysis_type.upper()} analysis...")
            try:
                analysis_kwargs = {
                    'n_components': args.n_components,
                    'random_state': args.seed
                }
                
                # Add method-specific parameters
                if args.reduction_method == 'tsne':
                    analysis_kwargs.update({
                        'perplexity': args.perplexity,
                        'max_iter': args.max_iter  # Use max_iter for newer scikit-learn
                    })
                elif args.reduction_method == 'umap':
                    analysis_kwargs.update({
                        'n_neighbors': args.n_neighbors,
                        'min_dist': args.min_dist
                    })
                
                analysis_results = analyzer.analyze_latent_space(
                    latent_type=latent_type,
                    reduction_method=args.reduction_method,
                    clustering_method=args.clustering_method,
                    n_clusters=args.n_clusters,
                    **analysis_kwargs
                )
                
                all_results[f"{analysis_type}_{latent_type}"] = analysis_results
                print(f"✅ {analysis_type.upper()} analysis completed successfully")
                
            except Exception as e:
                print(f"❌ Error during {analysis_type} analysis: {e}")
                continue
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations for all results
        print(f"🎨 Creating visualizations in {output_dir}")
        for result_key, analysis_results in all_results.items():
            analysis_type, latent_type = result_key.split('_', 1)
            result_dir = output_dir / f"{analysis_type}_{latent_type}"
            result_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                analyzer.create_visualizations(
                    analysis_results,
                    output_dir=str(result_dir),
                    save_plots=args.save_plots,
                    create_interactive=args.interactive
                )
                print(f"✅ Visualizations created for {analysis_type}")
            except Exception as e:
                print(f"⚠️  Warning: Error creating visualizations for {analysis_type}: {e}")
        
        # Save analysis results
        for result_key, analysis_results in all_results.items():
            analysis_type, latent_type = result_key.split('_', 1)
            result_dir = output_dir / f"{analysis_type}_{latent_type}"
            results_path = result_dir / "analysis_results.json"
            
            try:
                analyzer.save_analysis_results(analysis_results, str(results_path))
                print(f"💾 Analysis results saved to {results_path}")
            except Exception as e:
                print(f"⚠️  Warning: Error saving results for {analysis_type}: {e}")
        
        # Print summary
        print("\n📊 Analysis Summary:")
        for result_key, analysis_results in all_results.items():
            analysis_type, latent_type = result_key.split('_', 1)
            print(f"\n  {analysis_type.upper()} Analysis ({latent_type}):")
            print(f"    Reduction method: {analysis_results['reduction_method']}")
            print(f"    Clustering method: {analysis_results['clustering_method']}")
            print(f"    Number of clusters: {analysis_results['cluster_metrics']['n_clusters']}")
            
            if 'silhouette' in analysis_results['cluster_metrics']:
                silhouette = analysis_results['cluster_metrics']['silhouette']
                if not np.isnan(silhouette):
                    print(f"    Silhouette score: {silhouette:.3f}")
            
            if analysis_results['trajectory_analysis']:
                n_trajectories = len(analysis_results['trajectory_analysis']['trajectories'])
                print(f"    Number of trajectories: {n_trajectories}")
        
        print(f"\n🎯 Analysis complete! Results saved to: {output_dir}")
        print(f"📁 Check {output_dir} for plots and analysis files")


def main():
    """Main entry point for the analyze CLI."""
    cli = AnalyzeCLI()
    cli.run_cli()


if __name__ == "__main__":
    main() 