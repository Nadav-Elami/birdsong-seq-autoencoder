"""
Analysis tools for birdsong research.

This package contains advanced analysis tools for exploring and understanding
the latent representations learned by the LFADS model.
"""

from .latent import (
    LatentSpaceAnalyzer,
    compute_pca,
    compute_tsne,
    compute_umap,
    cluster_latents,
    analyze_trajectories,
    plot_latent_space,
    plot_trajectories,
    plot_cluster_analysis,
    create_interactive_widgets
)

__all__ = [
    'LatentSpaceAnalyzer',
    'compute_pca',
    'compute_tsne', 
    'compute_umap',
    'cluster_latents',
    'analyze_trajectories',
    'plot_latent_space',
    'plot_trajectories',
    'plot_cluster_analysis',
    'create_interactive_widgets'
] 