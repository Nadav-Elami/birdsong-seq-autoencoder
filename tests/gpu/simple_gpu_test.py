#!/usr/bin/env python3
"""
Simple GPU usage test.
"""

import torch
import time
import h5py
import numpy as np

def test_gpu_speed():
    """Test GPU vs CPU speed with simple operations."""
    
    print("=== GPU Speed Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Test data size similar to your training
    batch_size = 128
    seq_length = 50
    feature_dim = 14  # alphabet size
    
    print(f"\n=== Testing with batch_size={batch_size}, seq_length={seq_length}, features={feature_dim} ===")
    
    # Create test data
    data = torch.randn(batch_size, seq_length, feature_dim, requires_grad=True)
    target = torch.randn(batch_size, seq_length, feature_dim)
    
    # Test CPU
    print("\n--- CPU Test ---")
    device_cpu = torch.device("cpu")
    data_cpu = data.to(device_cpu)
    target_cpu = target.to(device_cpu)
    
    start_time = time.time()
    for i in range(10):  # 10 iterations
        # Simulate forward pass (simpler operation)
        weight = torch.randn(feature_dim, feature_dim, requires_grad=True)
        output = torch.nn.functional.linear(data_cpu, weight)
        loss = torch.nn.functional.mse_loss(output, target_cpu)
        loss.backward()
    cpu_time = time.time() - start_time
    print(f"CPU time for 10 iterations: {cpu_time:.2f} seconds")
    print(f"CPU time per iteration: {cpu_time/10:.2f} seconds")
    
    # Test GPU
    if torch.cuda.is_available():
        print("\n--- GPU Test ---")
        device_gpu = torch.device("cuda")
        data_gpu = data.to(device_gpu)
        target_gpu = target.to(device_gpu)
        
        # Warm up GPU
        for _ in range(3):
            weight = torch.randn(feature_dim, feature_dim, requires_grad=True).to(device_gpu)
            output = torch.nn.functional.linear(data_gpu, weight)
            loss = torch.nn.functional.mse_loss(output, target_gpu)
            loss.backward()
        
        torch.cuda.synchronize()  # Wait for GPU to finish
        
        start_time = time.time()
        for i in range(10):  # 10 iterations
            # Simulate forward pass
            weight = torch.randn(feature_dim, feature_dim, requires_grad=True).to(device_gpu)
            output = torch.nn.functional.linear(data_gpu, weight)
            loss = torch.nn.functional.mse_loss(output, target_gpu)
            loss.backward()
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        print(f"GPU time for 10 iterations: {gpu_time:.2f} seconds")
        print(f"GPU time per iteration: {gpu_time/10:.2f} seconds")
        
        # Speedup
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.1f}x faster")
    
    # Test data loading speed
    print("\n=== Data Loading Test ===")
    
    # Load a small sample from your dataset
    try:
        with h5py.File("examples/example datasets/aggregated_birdsong_data_14_syl_linear_50_song_in_batch_50_timesteps_20250608_150659.h5", 'r') as f:
            # Get first batch
            bigram_counts = torch.tensor(f['bigram_counts'][:batch_size])
            probabilities = torch.tensor(f['probabilities'][:batch_size])
            
            print(f"Loaded data shape: {bigram_counts.shape}")
            print(f"Data type: {bigram_counts.dtype}")
            
            # Test CPU data transfer
            start_time = time.time()
            data_cpu = bigram_counts.to(device_cpu)
            cpu_transfer_time = time.time() - start_time
            print(f"CPU data transfer time: {cpu_transfer_time:.4f} seconds")
            
            # Test GPU data transfer
            if torch.cuda.is_available():
                start_time = time.time()
                data_gpu = bigram_counts.to(device_gpu)
                torch.cuda.synchronize()
                gpu_transfer_time = time.time() - start_time
                print(f"GPU data transfer time: {gpu_transfer_time:.4f} seconds")
                
                if gpu_transfer_time > 0:
                    transfer_speedup = cpu_transfer_time / gpu_transfer_time
                    print(f"GPU transfer speedup: {transfer_speedup:.1f}x")
            
    except Exception as e:
        print(f"Could not load dataset: {e}")

if __name__ == "__main__":
    test_gpu_speed() 