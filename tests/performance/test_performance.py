#!/usr/bin/env python3
"""
Performance test script for the optimized data loader.
"""

import time
import torch
from torch.utils.data import DataLoader
from src.birdsong.data.loader import BirdsongDataset

def test_data_loading_performance():
    """Test the performance of the data loader."""
    
    # Test dataset path
    dataset_path = "examples/example datasets/aggregated_birdsong_data_14_syl_linear_50_song_in_batch_50_timesteps_20250608_150659.h5"
    
    print("Testing data loader performance...")
    print(f"Dataset: {dataset_path}")
    
    # Test initialization time
    start_time = time.time()
    dataset = BirdsongDataset(dataset_path)
    init_time = time.time() - start_time
    print(f"Dataset initialization time: {init_time:.2f} seconds")
    print(f"Dataset size: {len(dataset)} samples")
    
    # Test single sample loading time
    start_time = time.time()
    sample = dataset[0]
    single_sample_time = time.time() - start_time
    print(f"Single sample loading time: {single_sample_time:.4f} seconds")
    print(f"Sample shapes: {sample[0].shape}, {sample[1].shape}")
    
    # Test batch loading time
    batch_size = 64
    num_workers = 0
    pin_memory = False
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\nTesting batch loading with batch_size={batch_size}, num_workers={num_workers}")
    
    # Warm up
    for i, batch in enumerate(dataloader):
        if i >= 2:  # Just warm up with 2 batches
            break
    
    # Actual timing
    start_time = time.time()
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        if batch_count >= 10:  # Test 10 batches
            break
    
    total_time = time.time() - start_time
    avg_batch_time = total_time / batch_count
    samples_per_second = (batch_count * batch_size) / total_time
    
    print(f"Total time for {batch_count} batches: {total_time:.2f} seconds")
    print(f"Average batch loading time: {avg_batch_time:.4f} seconds")
    print(f"Throughput: {samples_per_second:.1f} samples/second")
    
    return {
        "init_time": init_time,
        "single_sample_time": single_sample_time,
        "avg_batch_time": avg_batch_time,
        "samples_per_second": samples_per_second
    }

if __name__ == "__main__":
    results = test_data_loading_performance()
    print(f"\nPerformance Summary:")
    print(f"- Dataset init: {results['init_time']:.2f}s")
    print(f"- Single sample: {results['single_sample_time']:.4f}s")
    print(f"- Batch loading: {results['avg_batch_time']:.4f}s/batch")
    print(f"- Throughput: {results['samples_per_second']:.1f} samples/s") 