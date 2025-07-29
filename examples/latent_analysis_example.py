"""
Example script for latent space analysis.

This script demonstrates how to use the latent space analysis tools
to explore and understand the latent representations learned by the LFADS model.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import birdsong
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt

from birdsong.analysis.latent import LatentSpaceAnalyzer
from birdsong.models.lfads import BirdsongLFADSModel2
from birdsong.data.loader import BirdsongDataset


def main():
    """Run the latent space analysis example."""
    print("🔍 Birdsong Latent Space Analysis Example")
    print("=" * 50)
    
    # Check if we have the required files
    checkpoint_path = "outputs/8_syl_100_songs_50_timesteps_linear_1st_order_50_epochs/checkpoint_225432660_best.pt"
    data_path = "examples/example datasets/aggregated_birdsong_data_1st_order_8_syl_linear_100_song_in_batch_50_timesteps_20250612_165300.h5"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please run training first or update the checkpoint path.")
        return
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        print("Please update the data path to point to a valid HDF5 dataset.")
        return
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load model
    print(f"📦 Loading model from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state = checkpoint['model_state_dict']
        
        # Create model (using default parameters for this example)
        model = BirdsongLFADSModel2(
            alphabet_size=8,
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
        return
    
    # Load dataset
    print(f"📊 Loading dataset from {data_path}")
    try:
        dataset = BirdsongDataset(data_path)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create analyzer
    analyzer = LatentSpaceAnalyzer(model, device, random_state=42)
    
    # Extract latents
    print("🔍 Extracting latent representations...")
    try:
        latent_results = analyzer.extract_latents(
            dataset,
            batch_size=32,
            include_factors=True,
            include_g0=True,
            include_u=True
        )
        print(f"✅ Extracted latents: {list(latent_results.keys())}")
    except Exception as e:
        print(f"❌ Error extracting latents: {e}")
        return
    
    # Perform analysis for different latent types
    latent_types = ['factors', 'g0', 'u']
    reduction_methods = ['pca', 'tsne']
    
    for latent_type in latent_types:
        print(f"\n📈 Analyzing {latent_type} latents...")
        
        for method in reduction_methods:
            print(f"  🔄 Using {method.upper()}...")
            
            try:
                # Perform analysis
                analysis_results = analyzer.analyze_latent_space(
                    latent_type=latent_type,
                    reduction_method=method,
                    clustering_method='kmeans',
                    n_clusters=5,
                    n_components=2,
                    random_state=42
                )
                
                # Create output directory
                output_dir = f"latent_analysis_example/{latent_type}_{method}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Create visualizations
                analyzer.create_visualizations(
                    analysis_results,
                    output_dir=output_dir,
                    save_plots=True,
                    create_interactive=False  # Disable interactive for this example
                )
                
                # Save results
                results_path = os.path.join(output_dir, "analysis_results.json")
                analyzer.save_analysis_results(analysis_results, results_path)
                
                # Print summary
                print(f"    ✅ {method.upper()} analysis complete")
                print(f"    📊 Clusters: {analysis_results['cluster_metrics']['n_clusters']}")
                
                if 'silhouette' in analysis_results['cluster_metrics']:
                    silhouette = analysis_results['cluster_metrics']['silhouette']
                    if not np.isnan(silhouette):
                        print(f"    📈 Silhouette score: {silhouette:.3f}")
                
                if analysis_results['trajectory_analysis']:
                    n_trajectories = len(analysis_results['trajectory_analysis']['trajectories'])
                    print(f"    🛤️  Trajectories: {n_trajectories}")
                
            except Exception as e:
                print(f"    ❌ Error in {method.upper()} analysis: {e}")
                continue
    
    print("\n🎯 Analysis complete!")
    print("📁 Check the 'latent_analysis_example' directory for results")
    print("\n📋 Generated files:")
    print("  - factors_pca/: PCA analysis of factors")
    print("  - factors_tsne/: t-SNE analysis of factors")
    print("  - g0_pca/: PCA analysis of g0 latents")
    print("  - g0_tsne/: t-SNE analysis of g0 latents")
    print("  - u_pca/: PCA analysis of u latents")
    print("  - u_tsne/: t-SNE analysis of u latents")
    
    print("\n💡 Tips for interpretation:")
    print("  - Factors: High-level temporal dynamics")
    print("  - g0: Initial conditions for each sequence")
    print("  - u: Inferred inputs driving the dynamics")
    print("  - Clusters: Groups of similar patterns")
    print("  - Trajectories: Temporal evolution of sequences")


if __name__ == "__main__":
    main() 