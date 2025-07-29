#!/usr/bin/env python3
"""
Simple performance test for data loading optimization.
"""

import time
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SimpleBirdsongDataset(Dataset):
    """Simplified version of the dataset for testing."""
    
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = None
        
        # Quick metadata check
        with h5py.File(h5_path, 'r') as hf:
            self.num_samples = hf['bigram_counts'].shape[2]
            self.time_steps = hf['bigram_counts'].shape[1]
            self.feature_dim = hf['bigram_counts'].shape[0]
    
    def _init_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.bigram_counts = self.h5_file['bigram_counts']
            self.probabilities = self.h5_file['probabilities']
    
    def __getitem__(self, idx):
        self._init_h5()
        bigram_counts = self.bigram_counts[:, :, idx].T
        probabilities = self.probabilities[:, :, idx].T
        
        # Use efficient tensor conversion
        bigram_tensor = torch.from_numpy(bigram_counts).float()
        prob_tensor = torch.from_numpy(probabilities).float()
        
        return bigram_tensor, prob_tensor
    
    def __len__(self):
        return self.num_samples
    
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

def test_performance():
    """Test the performance improvements."""
    
    dataset_path = "examples/example datasets/aggregated_birdsong_data_14_syl_linear_50_song_in_batch_50_timesteps_20250608_150659.h5"
    
    print("Testing optimized data loader performance...")
    print(f"Dataset: {dataset_path}")
    
    # Test initialization time
    start_time = time.time()
    dataset = SimpleBirdsongDataset(dataset_path)
    init_time = time.time() - start_time
    print(f"Dataset initialization time: {init_time:.2f} seconds")
    print(f"Dataset size: {len(dataset)} samples")
    
    # Test single sample loading
    start_time = time.time()
    sample = dataset[0]
    single_sample_time = time.time() - start_time
    print(f"Single sample loading time: {single_sample_time:.4f} seconds")
    print(f"Sample shapes: {sample[0].shape}, {sample[1].shape}")
    
    # Test batch loading
    batch_size = 64
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single worker for testing
        pin_memory=False
    )
    
    print(f"\nTesting batch loading with batch_size={batch_size}")
    
    # Warm up
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
    
    # Actual timing
    start_time = time.time()
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        if batch_count >= 10:
            break
    
    total_time = time.time() - start_time
    avg_batch_time = total_time / batch_count
    samples_per_second = (batch_count * batch_size) / total_time
    
    print(f"Total time for {batch_count} batches: {total_time:.2f} seconds")
    print(f"Average batch loading time: {avg_batch_time:.4f} seconds")
    print(f"Throughput: {samples_per_second:.1f} samples/second")
    
    # Estimate full training time
    total_batches = len(dataloader)
    estimated_training_time = total_batches * avg_batch_time
    print(f"\nEstimated training time per epoch: {estimated_training_time/60:.1f} minutes")
    print(f"Estimated time for 20 epochs: {estimated_training_time*20/3600:.1f} hours")
    
    return {
        "init_time": init_time,
        "single_sample_time": single_sample_time,
        "avg_batch_time": avg_batch_time,
        "samples_per_second": samples_per_second,
        "estimated_epoch_time": estimated_training_time
    }

if __name__ == "__main__":
    results = test_performance()
    print(f"\nPerformance Summary:")
    print(f"- Dataset init: {results['init_time']:.2f}s")
    print(f"- Single sample: {results['single_sample_time']:.4f}s")
    print(f"- Batch loading: {results['avg_batch_time']:.4f}s/batch")
    print(f"- Throughput: {results['samples_per_second']:.1f} samples/s")
    print(f"- Estimated epoch time: {results['estimated_epoch_time']/60:.1f} minutes") 