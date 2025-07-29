"""
Tests for latent space analysis functionality.

This module tests the latent space analysis tools including dimensionality reduction,
clustering, trajectory analysis, and visualization functions.
"""

import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from sklearn.cluster import KMeans

from birdsong.analysis.latent import (
    compute_pca,
    compute_tsne,
    compute_umap,
    cluster_latents,
    analyze_trajectories,
    plot_latent_space,
    plot_trajectories,
    plot_cluster_analysis,
    create_interactive_widgets,
    LatentSpaceAnalyzer
)
from birdsong.models.lfads import BirdsongLFADSModel2
from birdsong.data.loader import BirdsongDataset


class TestDimensionalityReduction:
    """Test dimensionality reduction functions."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.latents = np.random.randn(100, 32)  # 100 samples, 32 features
        self.latents_3d = np.random.randn(50, 10, 16)  # 50 sequences, 10 time steps, 16 features
    
    def test_compute_pca_happy_path(self):
        """Test PCA computation with valid input."""
        reduced, pca_model = compute_pca(self.latents, n_components=2)
        
        assert reduced.shape == (100, 2)
        assert hasattr(pca_model, 'explained_variance_ratio_')
        assert pca_model.n_components == 2
    
    def test_compute_pca_3d_input(self):
        """Test PCA with 3D input (flattened automatically)."""
        reduced, pca_model = compute_pca(self.latents_3d, n_components=3)
        
        assert reduced.shape == (500, 3)  # 50 * 10 = 500 samples
        assert pca_model.n_components == 3
    
    def test_compute_pca_invalid_components(self):
        """Test PCA with invalid number of components."""
        with pytest.raises(ValueError):
            compute_pca(self.latents, n_components=100)  # More components than features
    
    def test_compute_tsne_happy_path(self):
        """Test t-SNE computation with valid input."""
        reduced, tsne_model = compute_tsne(self.latents, n_components=2, n_iter=100)
        
        assert reduced.shape == (100, 2)
        assert tsne_model.n_components == 2
        assert tsne_model.perplexity == 30.0
    
    def test_compute_tsne_custom_params(self):
        """Test t-SNE with custom parameters."""
        reduced, tsne_model = compute_tsne(
            self.latents, 
            n_components=2, 
            perplexity=10.0, 
            n_iter=200
        )
        
        assert reduced.shape == (100, 2)
        assert tsne_model.perplexity == 10.0
        assert tsne_model.n_iter == 200
    
    def test_compute_umap_happy_path(self):
        """Test UMAP computation with valid input."""
        try:
            reduced, umap_model = compute_umap(self.latents, n_components=2)
            
            assert reduced.shape == (100, 2)
            assert umap_model.n_components == 2
            assert umap_model.n_neighbors == 15
        except ImportError:
            pytest.skip("UMAP not available")
    
    def test_compute_umap_custom_params(self):
        """Test UMAP with custom parameters."""
        try:
            reduced, umap_model = compute_umap(
                self.latents,
                n_components=3,
                n_neighbors=10,
                min_dist=0.05
            )
            
            assert reduced.shape == (100, 3)
            assert umap_model.n_neighbors == 10
            assert umap_model.min_dist == 0.05
        except ImportError:
            pytest.skip("UMAP not available")
    
    def test_compute_umap_not_available(self):
        """Test UMAP when not available."""
        with patch('birdsong.analysis.latent.UMAP_AVAILABLE', False):
            with pytest.raises(ImportError):
                compute_umap(self.latents)


class TestClustering:
    """Test clustering functions."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.latents = np.random.randn(100, 16)
    
    def test_cluster_latents_kmeans_happy_path(self):
        """Test K-means clustering with valid input."""
        labels, clusterer, metrics = cluster_latents(
            self.latents, method='kmeans', n_clusters=3
        )
        
        assert len(labels) == 100
        assert len(np.unique(labels)) == 3
        assert isinstance(clusterer, KMeans)
        assert 'n_clusters' in metrics
        assert metrics['n_clusters'] == 3
    
    def test_cluster_latents_dbscan(self):
        """Test DBSCAN clustering."""
        labels, clusterer, metrics = cluster_latents(
            self.latents, method='dbscan', eps=0.5, min_samples=5
        )
        
        assert len(labels) == 100
        assert 'n_clusters' in metrics
        assert metrics['n_clusters'] >= 0  # Can be 0 if no clusters found
    
    def test_cluster_latents_invalid_method(self):
        """Test clustering with invalid method."""
        with pytest.raises(ValueError):
            cluster_latents(self.latents, method='invalid')
    
    def test_cluster_latents_single_cluster(self):
        """Test clustering when only one cluster is found."""
        # Create data that will likely form one cluster
        latents = np.random.randn(100, 16) * 0.1  # Very tight cluster
        
        labels, clusterer, metrics = cluster_latents(
            latents, method='kmeans', n_clusters=3
        )
        
        assert len(labels) == 100
        # Should still work even if only one cluster is found
        assert 'n_clusters' in metrics


class TestTrajectoryAnalysis:
    """Test trajectory analysis functions."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.latents = np.random.randn(200, 16)  # 200 samples
        self.sequence_lengths = [50, 30, 40, 80]  # 4 sequences
    
    def test_analyze_trajectories_pca(self):
        """Test trajectory analysis with PCA."""
        results = analyze_trajectories(
            self.latents, self.sequence_lengths, method='pca'
        )
        
        assert 'trajectories' in results
        assert 'trajectory_stats' in results
        assert 'method' in results
        assert results['method'] == 'pca'
        assert len(results['trajectories']) == 4
    
    def test_analyze_trajectories_tsne(self):
        """Test trajectory analysis with t-SNE."""
        results = analyze_trajectories(
            self.latents, self.sequence_lengths, method='tsne', n_iter=100
        )
        
        assert results['method'] == 'tsne'
        assert len(results['trajectories']) == 4
    
    def test_analyze_trajectories_umap(self):
        """Test trajectory analysis with UMAP."""
        try:
            results = analyze_trajectories(
                self.latents, self.sequence_lengths, method='umap'
            )
            
            assert results['method'] == 'umap'
            assert len(results['trajectories']) == 4
        except ImportError:
            pytest.skip("UMAP not available")
    
    def test_analyze_trajectories_invalid_method(self):
        """Test trajectory analysis with invalid method."""
        with pytest.raises(ValueError):
            analyze_trajectories(self.latents, self.sequence_lengths, method='invalid')
    
    def test_analyze_trajectories_empty_sequences(self):
        """Test trajectory analysis with empty sequence list."""
        results = analyze_trajectories(self.latents, [], method='pca')
        
        assert len(results['trajectories']) == 0
        assert len(results['trajectory_stats']) == 0


class TestVisualization:
    """Test visualization functions."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.reduced_data = np.random.randn(100, 2)
        self.labels = np.random.randint(0, 3, 100)
        self.cluster_metrics = {
            'n_clusters': 3,
            'silhouette': 0.5,
            'calinski_harabasz': 100.0
        }
    
    def test_plot_latent_space_basic(self):
        """Test basic latent space plotting."""
        with patch('matplotlib.pyplot.show'):
            plot_latent_space(self.reduced_data)
    
    def test_plot_latent_space_with_labels(self):
        """Test latent space plotting with cluster labels."""
        with patch('matplotlib.pyplot.show'):
            plot_latent_space(self.reduced_data, labels=self.labels)
    
    def test_plot_latent_space_save(self):
        """Test latent space plotting with save."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plot_latent_space(self.reduced_data, save_path=tmp.name)
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)
    
    def test_plot_cluster_analysis(self):
        """Test cluster analysis plotting."""
        with patch('matplotlib.pyplot.show'):
            plot_cluster_analysis(self.reduced_data, self.labels, self.cluster_metrics)
    
    def test_plot_cluster_analysis_save(self):
        """Test cluster analysis plotting with save."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plot_cluster_analysis(
                self.reduced_data, self.labels, self.cluster_metrics, save_path=tmp.name
            )
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)
    
    def test_plot_trajectories(self):
        """Test trajectory plotting."""
        trajectory_analysis = {
            'trajectories': [np.random.randn(10, 2) for _ in range(3)],
            'method': 'pca'
        }
        
        with patch('matplotlib.pyplot.show'):
            plot_trajectories(trajectory_analysis)
    
    def test_plot_trajectories_save(self):
        """Test trajectory plotting with save."""
        trajectory_analysis = {
            'trajectories': [np.random.randn(10, 2) for _ in range(3)],
            'method': 'pca'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plot_trajectories(trajectory_analysis, save_path=tmp.name)
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)
    
    def test_create_interactive_widgets(self):
        """Test interactive widget creation."""
        try:
            with patch('birdsong.analysis.latent.PLOTLY_AVAILABLE', True):
                with patch('plotly.graph_objects.Figure.show'):
                    create_interactive_widgets(self.reduced_data, cluster_labels=self.labels)
        except ImportError:
            pytest.skip("Plotly not available")
    
    def test_create_interactive_widgets_not_available(self):
        """Test interactive widgets when Plotly not available."""
        with patch('birdsong.analysis.latent.PLOTLY_AVAILABLE', False):
            with pytest.raises(ImportError):
                create_interactive_widgets(self.reduced_data)


class TestLatentSpaceAnalyzer:
    """Test the LatentSpaceAnalyzer class."""
    
    def setup_method(self):
        """Set up test data and model."""
        self.device = torch.device('cpu')
        self.model = BirdsongLFADSModel2(
            alphabet_size=7,
            order=1,
            encoder_dim=32,
            controller_dim=32,
            generator_dim=32,
            factor_dim=16,
            latent_dim=8,
            inferred_input_dim=4
        )
        self.analyzer = LatentSpaceAnalyzer(self.model, self.device)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.model == self.model
        assert self.analyzer.device == self.device
        assert self.analyzer.random_state == 42
        assert self.analyzer.latents is None
        assert self.analyzer.factors is None
    
    def test_extract_latents_mock_dataset(self):
        """Test latent extraction with mock dataset."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        
        # Mock DataLoader
        mock_batch = torch.randn(2, 10, 49)  # 2 samples, 10 time steps, 49 features (7^2)
        
        with patch('torch.utils.data.DataLoader') as mock_loader:
            mock_loader.return_value = [mock_batch]
            
            # Mock model outputs
            mock_outputs = {
                'factors': torch.randn(2, 10, 16),
                'g0': torch.randn(2, 8),
                'u': torch.randn(2, 10, 4)
            }
            self.model.forward = MagicMock(return_value=mock_outputs)
            
            results = self.analyzer.extract_latents(mock_dataset, batch_size=2)
            
            assert 'factors' in results
            assert 'g0' in results
            assert 'u' in results
    
    def test_analyze_latent_space_mock_data(self):
        """Test latent space analysis with mock data."""
        # Set up mock latents
        self.analyzer.factors = np.random.randn(100, 16)
        self.analyzer.sequence_lengths = [50, 50]
        
        results = self.analyzer.analyze_latent_space(
            latent_type='factors',
            reduction_method='pca',
            clustering_method='kmeans',
            n_clusters=3
        )
        
        assert results['latent_type'] == 'factors'
        assert results['reduction_method'] == 'pca'
        assert results['clustering_method'] == 'kmeans'
        assert 'reduced_data' in results
        assert 'cluster_labels' in results
        assert 'cluster_metrics' in results
    
    def test_analyze_latent_space_invalid_latent_type(self):
        """Test analysis with invalid latent type."""
        self.analyzer.factors = np.random.randn(100, 16)
        
        with pytest.raises(ValueError):
            self.analyzer.analyze_latent_space(latent_type='invalid')
    
    def test_analyze_latent_space_no_latents(self):
        """Test analysis when no latents are available."""
        with pytest.raises(ValueError):
            self.analyzer.analyze_latent_space(latent_type='factors')
    
    def test_create_visualizations(self):
        """Test visualization creation."""
        # Set up mock analysis results
        analysis_results = {
            'latent_type': 'factors',
            'reduction_method': 'pca',
            'clustering_method': 'kmeans',
            'latents': np.random.randn(100, 16),
            'reduced_data': np.random.randn(100, 2),
            'cluster_labels': np.random.randint(0, 3, 100),
            'cluster_metrics': {'n_clusters': 3, 'silhouette': 0.5},
            'trajectory_analysis': {
                'trajectories': [np.random.randn(10, 2) for _ in range(3)],
                'method': 'pca'
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('matplotlib.pyplot.show'):
                self.analyzer.create_visualizations(
                    analysis_results, output_dir=tmpdir, save_plots=True
                )
    
    def test_save_analysis_results(self):
        """Test saving analysis results."""
        analysis_results = {
            'latent_type': 'factors',
            'reduction_method': 'pca',
            'clustering_method': 'kmeans',
            'latents': np.random.randn(100, 16),
            'reduced_data': np.random.randn(100, 2),
            'cluster_labels': np.random.randint(0, 3, 100),
            'cluster_metrics': {'n_clusters': 3, 'silhouette': 0.5}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            self.analyzer.save_analysis_results(analysis_results, tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.exists(tmp.name.replace('.json', '_latents.npy'))
            assert os.path.exists(tmp.name.replace('.json', '_reduced.npy'))
            assert os.path.exists(tmp.name.replace('.json', '_clusters.npy'))
            
            # Clean up
            os.unlink(tmp.name)
            os.unlink(tmp.name.replace('.json', '_latents.npy'))
            os.unlink(tmp.name.replace('.json', '_reduced.npy'))
            os.unlink(tmp.name.replace('.json', '_clusters.npy'))


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_input_data(self):
        """Test with empty input data."""
        empty_latents = np.array([])
        
        with pytest.raises(ValueError):
            compute_pca(empty_latents)
    
    def test_single_sample(self):
        """Test with single sample."""
        single_latent = np.random.randn(1, 16)
        
        # Should work for PCA
        reduced, _ = compute_pca(single_latent, n_components=1)
        assert reduced.shape == (1, 1)
        
        # Should work for clustering
        labels, _, metrics = cluster_latents(single_latent, n_clusters=1)
        assert len(labels) == 1
    
    def test_high_dimensional_data(self):
        """Test with very high dimensional data."""
        high_dim_latents = np.random.randn(50, 1000)
        
        # Should work for PCA
        reduced, _ = compute_pca(high_dim_latents, n_components=10)
        assert reduced.shape == (50, 10)
    
    def test_nan_values(self):
        """Test handling of NaN values."""
        latents_with_nan = np.random.randn(100, 16)
        latents_with_nan[0, 0] = np.nan
        
        with pytest.raises(ValueError):
            compute_pca(latents_with_nan)
    
    def test_inf_values(self):
        """Test handling of infinite values."""
        latents_with_inf = np.random.randn(100, 16)
        latents_with_inf[0, 0] = np.inf
        
        with pytest.raises(ValueError):
            compute_pca(latents_with_inf)


if __name__ == "__main__":
    pytest.main([__file__]) 