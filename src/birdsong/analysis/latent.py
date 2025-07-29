"""
Latent space analysis tools for birdsong research.

This module provides comprehensive tools for analyzing the latent representations
learned by the LFADS model, including dimensionality reduction, clustering,
trajectory analysis, and interactive exploration.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

from ..models.lfads import BirdsongLFADSModel2
from ..data.loader import BirdsongDataset


def compute_pca(
    latents: np.ndarray, 
    n_components: int = 2,
    random_state: int = 42
) -> Tuple[np.ndarray, PCA]:
    """
    Compute PCA dimensionality reduction on latent representations.
    
    Args:
        latents: Array of shape (n_samples, n_features) containing latent representations
        n_components: Number of components to keep (default: 2 for visualization)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (transformed_data, pca_model)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    transformed = pca.fit_transform(latents)
    return transformed, pca


def compute_tsne(
    latents: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs
) -> Tuple[np.ndarray, TSNE]:
    """
    Compute t-SNE dimensionality reduction on latent representations.
    
    Args:
        latents: Array of shape (n_samples, n_features) containing latent representations
        n_components: Number of components to keep (default: 2 for visualization)
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for optimization
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments passed to TSNE
        
    Returns:
        Tuple of (transformed_data, tsne_model)
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    transformed = tsne.fit_transform(latents)
    return transformed, tsne


def compute_umap(
    latents: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    **kwargs
) -> Tuple[np.ndarray, Any]:
    """
    Compute UMAP dimensionality reduction on latent representations.
    
    Args:
        latents: Array of shape (n_samples, n_features) containing latent representations
        n_components: Number of components to keep (default: 2 for visualization)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance between points
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments passed to UMAP
        
    Returns:
        Tuple of (transformed_data, umap_model)
        
    Raises:
        ImportError: If UMAP is not available
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        **kwargs
    )
    transformed = reducer.fit_transform(latents)
    return transformed, reducer


def cluster_latents(
    latents: np.ndarray,
    method: str = 'kmeans',
    n_clusters: int = 5,
    random_state: int = 42,
    **kwargs
) -> Tuple[np.ndarray, Any, Dict[str, float]]:
    """
    Perform clustering on latent representations.
    
    Args:
        latents: Array of shape (n_samples, n_features) containing latent representations
        method: Clustering method ('kmeans' or 'dbscan')
        n_clusters: Number of clusters for K-means
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments passed to clustering algorithm
        
    Returns:
        Tuple of (cluster_labels, cluster_model, metrics)
    """
    if method.lower() == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    elif method.lower() == 'dbscan':
        clusterer = DBSCAN(**kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    labels = clusterer.fit_predict(latents)
    
    # Compute clustering metrics
    metrics = {}
    if len(np.unique(labels)) > 1:  # Only compute if we have multiple clusters
        try:
            metrics['silhouette'] = silhouette_score(latents, labels)
        except:
            metrics['silhouette'] = np.nan
            
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(latents, labels)
        except:
            metrics['calinski_harabasz'] = np.nan
    
    metrics['n_clusters'] = len(np.unique(labels[labels >= 0]))  # Exclude noise points for DBSCAN
    
    return labels, clusterer, metrics


def analyze_trajectories(
    latents: np.ndarray,
    sequence_lengths: List[int],
    method: str = 'pca',
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze trajectories in latent space across sequences.
    
    Args:
        latents: Array of shape (n_samples, n_features) containing latent representations
        sequence_lengths: List of sequence lengths for each trajectory
        method: Dimensionality reduction method ('pca', 'tsne', 'umap')
        **kwargs: Additional arguments for dimensionality reduction
        
    Returns:
        Dictionary containing trajectory analysis results
    """
    # Split latents into sequences
    start_idx = 0
    trajectories = []
    
    for length in sequence_lengths:
        end_idx = start_idx + length
        trajectory = latents[start_idx:end_idx]
        trajectories.append(trajectory)
        start_idx = end_idx
    
    # Apply dimensionality reduction to all data
    if method.lower() == 'pca':
        reduced, model = compute_pca(latents, **kwargs)
    elif method.lower() == 'tsne':
        reduced, model = compute_tsne(latents, **kwargs)
    elif method.lower() == 'umap':
        reduced, model = compute_umap(latents, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Split reduced data into trajectories
    start_idx = 0
    reduced_trajectories = []
    
    for length in sequence_lengths:
        end_idx = start_idx + length
        trajectory = reduced[start_idx:end_idx]
        reduced_trajectories.append(trajectory)
        start_idx = end_idx
    
    # Compute trajectory statistics
    trajectory_stats = []
    for i, traj in enumerate(reduced_trajectories):
        if len(traj) > 1:
            # Compute trajectory length
            distances = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
            total_length = np.sum(distances)
            
            # Compute curvature (simplified)
            if len(traj) > 2:
                # Compute angles between consecutive segments
                segments = np.diff(traj, axis=0)
                angles = []
                for j in range(len(segments) - 1):
                    dot_product = np.dot(segments[j], segments[j+1])
                    norms = np.linalg.norm(segments[j]) * np.linalg.norm(segments[j+1])
                    if norms > 0:
                        angle = np.arccos(np.clip(dot_product / norms, -1, 1))
                        angles.append(angle)
                avg_curvature = np.mean(angles) if angles else 0
            else:
                avg_curvature = 0
            
            stats = {
                'trajectory_id': i,
                'length': total_length,
                'avg_curvature': avg_curvature,
                'n_points': len(traj),
                'start_point': traj[0],
                'end_point': traj[-1]
            }
            trajectory_stats.append(stats)
    
    return {
        'trajectories': reduced_trajectories,
        'trajectory_stats': trajectory_stats,
        'reduction_model': model,
        'method': method
    }


def plot_latent_space(
    reduced_data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Latent Space Visualization",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    **kwargs
) -> None:
    """
    Plot latent space visualization.
    
    Args:
        reduced_data: 2D array of reduced latent representations
        labels: Optional cluster labels for coloring
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size
        **kwargs: Additional arguments for scatter plot
    """
    plt.figure(figsize=figsize)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                reduced_data[mask, 0], 
                reduced_data[mask, 1],
                c=[colors[i]], 
                label=f'Cluster {label}',
                alpha=0.7,
                **kwargs
            )
        plt.legend()
    else:
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7, **kwargs)
    
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        # Save in PNG format
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Also save in SVG format if the path doesn't already have .svg extension
        if not save_path.endswith('.svg'):
            svg_path = save_path.replace('.png', '.svg')
            plt.savefig(svg_path, dpi=300, bbox_inches='tight', format='svg')
    
    plt.show()


def plot_trajectories(
    trajectory_analysis: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    max_trajectories: int = 20
) -> None:
    """
    Plot trajectory analysis results.
    
    Args:
        trajectory_analysis: Results from analyze_trajectories function
        save_path: Optional path to save the plot
        figsize: Figure size
        max_trajectories: Maximum number of trajectories to plot
    """
    trajectories = trajectory_analysis['trajectories']
    method = trajectory_analysis['method']
    
    plt.figure(figsize=figsize)
    
    # Plot trajectories with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, min(len(trajectories), max_trajectories)))
    
    for i, traj in enumerate(trajectories[:max_trajectories]):
        plt.plot(traj[:, 0], traj[:, 1], 
                color=colors[i], alpha=0.7, linewidth=2)
        plt.scatter(traj[0, 0], traj[0, 1], 
                   color=colors[i], s=100, marker='o', label=f'Start {i}')
        plt.scatter(traj[-1, 0], traj[-1, 1], 
                   color=colors[i], s=100, marker='s', label=f'End {i}')
    
    plt.title(f'Trajectory Analysis - {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        # Save in PNG format
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Also save in SVG format if the path doesn't already have .svg extension
        if not save_path.endswith('.svg'):
            svg_path = save_path.replace('.png', '.svg')
            plt.savefig(svg_path, dpi=300, bbox_inches='tight', format='svg')
    
    plt.show()


def plot_cluster_analysis(
    latents: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_metrics: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot cluster analysis results.
    
    Args:
        latents: Original latent representations
        cluster_labels: Cluster labels
        cluster_metrics: Clustering metrics
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Silhouette score distribution
    if 'silhouette' in cluster_metrics and not np.isnan(cluster_metrics['silhouette']):
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(latents, cluster_labels)
        
        axes[0].hist(silhouette_vals, bins=20, alpha=0.7)
        axes[0].axvline(cluster_metrics['silhouette'], color='red', linestyle='--', 
                        label=f'Mean: {cluster_metrics["silhouette"]:.3f}')
        axes[0].set_title('Silhouette Score Distribution')
        axes[0].set_xlabel('Silhouette Score')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'Silhouette score not available', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Silhouette Score Distribution')
    
    # Plot 2: Cluster sizes
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    axes[1].bar(range(len(unique_labels)), counts)
    axes[1].set_title('Cluster Sizes')
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Number of Points')
    axes[1].set_xticks(range(len(unique_labels)))
    axes[1].set_xticklabels(unique_labels)
    
    # Plot 3: Metrics summary
    axes[2].text(0.1, 0.8, f"Number of Clusters: {cluster_metrics['n_clusters']}", 
                transform=axes[2].transAxes, fontsize=12)
    if 'silhouette' in cluster_metrics:
        axes[2].text(0.1, 0.6, f"Silhouette Score: {cluster_metrics['silhouette']:.3f}", 
                    transform=axes[2].transAxes, fontsize=12)
    if 'calinski_harabasz' in cluster_metrics:
        axes[2].text(0.1, 0.4, f"Calinski-Harabasz: {cluster_metrics['calinski_harabasz']:.3f}", 
                    transform=axes[2].transAxes, fontsize=12)
    axes[2].set_title('Clustering Metrics')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # Save in PNG format
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Also save in SVG format if the path doesn't already have .svg extension
        if not save_path.endswith('.svg'):
            svg_path = save_path.replace('.png', '.svg')
            plt.savefig(svg_path, dpi=300, bbox_inches='tight', format='svg')
    
    plt.show()


def create_interactive_widgets(
    latents: np.ndarray,
    trajectory_analysis: Optional[Dict[str, Any]] = None,
    cluster_labels: Optional[np.ndarray] = None
) -> None:
    """
    Create interactive exploration widgets for latent space analysis.
    
    Args:
        latents: Array of latent representations
        trajectory_analysis: Optional trajectory analysis results
        cluster_labels: Optional cluster labels
        
    Raises:
        ImportError: If Plotly is not available
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    # Create interactive PCA plot
    pca_data, _ = compute_pca(latents)
    
    if cluster_labels is not None:
        fig = px.scatter(
            x=pca_data[:, 0], 
            y=pca_data[:, 1],
            color=cluster_labels,
            title="Interactive PCA Visualization",
            labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'}
        )
    else:
        fig = px.scatter(
            x=pca_data[:, 0], 
            y=pca_data[:, 1],
            title="Interactive PCA Visualization",
            labels={'x': 'PC1', 'y': 'PC2'}
        )
    
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True
    )
    fig.show()
    
    # Create interactive trajectory plot if available
    if trajectory_analysis is not None:
        trajectories = trajectory_analysis['trajectories']
        
        fig = go.Figure()
        
        for i, traj in enumerate(trajectories[:10]):  # Limit to first 10 for performance
            fig.add_trace(go.Scatter(
                x=traj[:, 0],
                y=traj[:, 1],
                mode='lines+markers',
                name=f'Trajectory {i}',
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Interactive Trajectory Visualization",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            width=800,
            height=600,
            showlegend=True
        )
        fig.show()


class LatentSpaceAnalyzer:
    """
    Comprehensive latent space analysis toolkit for LFADS models.
    
    This class provides a unified interface for analyzing latent representations
    learned by the LFADS model, including dimensionality reduction, clustering,
    trajectory analysis, and interactive visualization.
    """
    
    def __init__(
        self,
        model: BirdsongLFADSModel2,
        device: torch.device,
        random_state: int = 42
    ):
        """
        Initialize the latent space analyzer.
        
        Args:
            model: Trained LFADS model
            device: Device to run computations on
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.device = device
        self.random_state = random_state
        self.model.eval()
        
        # Storage for analysis results
        self.latents = None
        self.factors = None
        self.g0_samples = None
        self.u_samples = None
        self.sequence_lengths = None
        
    def extract_latents(
        self,
        dataset: BirdsongDataset,
        batch_size: int = 64,
        include_factors: bool = True,
        include_g0: bool = True,
        include_u: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract latent representations from the model.
        
        Args:
            dataset: Dataset to extract latents from
            batch_size: Batch size for extraction
            include_factors: Whether to extract factors
            include_g0: Whether to extract g0 samples
            include_u: Whether to extract u samples
            
        Returns:
            Dictionary containing extracted latent representations
        """
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_latents = []
        all_factors = []
        all_g0 = []
        all_u = []
        sequence_lengths = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    input_data = batch[0].to(self.device)
                else:
                    input_data = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(input_data)
                
                # Extract different types of latents
                if include_factors and 'factors' in outputs:
                    factors = outputs['factors'].cpu().numpy()
                    all_factors.append(factors)
                
                if include_g0 and 'g0' in outputs:
                    g0 = outputs['g0'].cpu().numpy()
                    all_g0.append(g0)
                
                if include_u and 'u' in outputs:
                    u = outputs['u'].cpu().numpy()
                    all_u.append(u)
                
                # Store sequence lengths
                batch_size_actual = input_data.shape[0]
                seq_length = input_data.shape[1]
                sequence_lengths.extend([seq_length] * batch_size_actual)
        
        # Concatenate results
        results = {}
        
        if all_factors:
            self.factors = np.concatenate(all_factors, axis=0)
            results['factors'] = self.factors
        
        if all_g0:
            self.g0_samples = np.concatenate(all_g0, axis=0)
            results['g0'] = self.g0_samples
        
        if all_u:
            self.u_samples = np.concatenate(all_u, axis=0)
            results['u'] = self.u_samples
        
        self.sequence_lengths = sequence_lengths
        
        return results
    
    def analyze_latent_space(
        self,
        latent_type: str = 'factors',
        reduction_method: str = 'pca',
        clustering_method: str = 'kmeans',
        n_clusters: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform comprehensive latent space analysis.
        
        Args:
            latent_type: Type of latent to analyze ('factors', 'g0', 'u')
            reduction_method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            clustering_method: Clustering method ('kmeans', 'dbscan')
            n_clusters: Number of clusters for K-means
            **kwargs: Additional arguments for reduction and clustering
            
        Returns:
            Dictionary containing analysis results
        """
        # Get the appropriate latent representations
        if latent_type == 'factors' and self.factors is not None:
            latents = self.factors
        elif latent_type == 'g0' and self.g0_samples is not None:
            latents = self.g0_samples
        elif latent_type == 'u' and self.u_samples is not None:
            latents = self.u_samples
        else:
            raise ValueError(f"Latent type '{latent_type}' not available")
        
        print(f"    📊 Processing {latents.shape[0]} samples with {latents.shape[1]} features...")
        
        # Reshape if needed (flatten time dimension)
        if len(latents.shape) == 3:  # (batch, time, features)
            latents_flat = latents.reshape(-1, latents.shape[-1])
            print(f"    🔄 Reshaped from {latents.shape} to {latents_flat.shape}")
        else:
            latents_flat = latents
        
        # Filter kwargs for dimensionality reduction (remove clustering-specific params)
        reduction_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['n_clusters', 'eps', 'min_samples']}
        
        # Perform dimensionality reduction
        print(f"    🎯 Applying {reduction_method.upper()}...")
        if reduction_method.lower() == 'pca':
            reduced_data, reduction_model = compute_pca(latents_flat, **reduction_kwargs)
        elif reduction_method.lower() == 'tsne':
            # Add progress reporting for t-SNE
            print(f"    ⏳ t-SNE can take a while for large datasets...")
            reduced_data, reduction_model = compute_tsne(latents_flat, **reduction_kwargs)
        elif reduction_method.lower() == 'umap':
            reduced_data, reduction_model = compute_umap(latents_flat, **reduction_kwargs)
        else:
            raise ValueError(f"Unknown reduction method: {reduction_method}")
        
        print(f"    ✅ {reduction_method.upper()} complete")
        
        # Filter kwargs for clustering (remove reduction-specific params)
        clustering_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['n_components', 'perplexity', 'max_iter', 'n_neighbors', 'min_dist']}
        
        # Perform clustering
        print(f"    🎯 Applying {clustering_method} clustering...")
        cluster_labels, cluster_model, cluster_metrics = cluster_latents(
            latents_flat, method=clustering_method, n_clusters=n_clusters, **clustering_kwargs
        )
        print(f"    ✅ Clustering complete")
        
        # Analyze trajectories if sequence information is available
        trajectory_analysis = None
        if self.sequence_lengths is not None:
            print(f"    🎯 Analyzing trajectories...")
            trajectory_analysis = analyze_trajectories(
                latents_flat, self.sequence_lengths, method=reduction_method, **kwargs
            )
            print(f"    ✅ Trajectory analysis complete")
        
        return {
            'latent_type': latent_type,
            'reduction_method': reduction_method,
            'clustering_method': clustering_method,
            'latents': latents_flat,
            'reduced_data': reduced_data,
            'reduction_model': reduction_model,
            'cluster_labels': cluster_labels,
            'cluster_model': cluster_model,
            'cluster_metrics': cluster_metrics,
            'trajectory_analysis': trajectory_analysis
        }
    
    def create_visualizations(
        self,
        analysis_results: Dict[str, Any],
        output_dir: str = "latent_analysis",
        save_plots: bool = True,
        create_interactive: bool = True
    ) -> None:
        """
        Create comprehensive visualizations from analysis results.
        
        Args:
            analysis_results: Results from analyze_latent_space
            output_dir: Directory to save plots
            save_plots: Whether to save plots to disk
            create_interactive: Whether to create interactive widgets
        """
        os.makedirs(output_dir, exist_ok=True)
        
        latent_type = analysis_results['latent_type']
        reduction_method = analysis_results['reduction_method']
        
        # Create latent space plot
        plot_path = os.path.join(output_dir, f"{latent_type}_{reduction_method}_space.png") if save_plots else None
        plot_latent_space(
            analysis_results['reduced_data'],
            labels=analysis_results['cluster_labels'],
            title=f"{latent_type.upper()} - {reduction_method.upper()} Visualization",
            save_path=plot_path
        )
        
        # Create cluster analysis plot
        if analysis_results['cluster_metrics']['n_clusters'] > 1:
            cluster_path = os.path.join(output_dir, f"{latent_type}_cluster_analysis.png") if save_plots else None
            plot_cluster_analysis(
                analysis_results['latents'],
                analysis_results['cluster_labels'],
                analysis_results['cluster_metrics'],
                save_path=cluster_path
            )
        
        # Create trajectory plot if available
        if analysis_results['trajectory_analysis'] is not None:
            traj_path = os.path.join(output_dir, f"{latent_type}_trajectories.png") if save_plots else None
            plot_trajectories(
                analysis_results['trajectory_analysis'],
                save_path=traj_path
            )
        
        # Create interactive widgets
        if create_interactive and PLOTLY_AVAILABLE:
            try:
                create_interactive_widgets(
                    analysis_results['latents'],
                    analysis_results['trajectory_analysis'],
                    analysis_results['cluster_labels']
                )
            except Exception as e:
                print(f"Warning: Could not create interactive widgets: {e}")
    
    def save_analysis_results(
        self,
        analysis_results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Save analysis results to disk.
        
        Args:
            analysis_results: Results from analyze_latent_space
            output_path: Path to save results
        """
        import json
        
        # Prepare results for saving (remove non-serializable objects)
        save_results = {
            'latent_type': analysis_results['latent_type'],
            'reduction_method': analysis_results['reduction_method'],
            'clustering_method': analysis_results['clustering_method'],
            'cluster_metrics': analysis_results['cluster_metrics'],
            'latents_shape': analysis_results['latents'].shape,
            'reduced_data_shape': analysis_results['reduced_data'].shape,
            'n_clusters': analysis_results['cluster_metrics']['n_clusters'],
            'random_state': self.random_state
        }
        
        if analysis_results['trajectory_analysis'] is not None:
            save_results['trajectory_stats'] = analysis_results['trajectory_analysis']['trajectory_stats']
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
        
        # Save numpy arrays separately
        base_path = output_path.replace('.json', '')
        np.save(f"{base_path}_latents.npy", analysis_results['latents'])
        np.save(f"{base_path}_reduced.npy", analysis_results['reduced_data'])
        np.save(f"{base_path}_clusters.npy", analysis_results['cluster_labels'])
        
        print(f"✅ Analysis results saved to {output_path}") 