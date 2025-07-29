#!/usr/bin/env python3
"""
Test script to verify GPU usage during training.
"""

import torch
import time
from src.birdsong.data.loader import BirdsongDataset
from src.birdsong.models.lfads import BirdsongLFADSModel2
from src.birdsong.training.trainer import BirdsongTrainer

def test_gpu_usage():
    """Test if training is actually using GPU."""
    
    print("=== GPU Usage Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    dataset = BirdsongDataset("examples/example datasets/aggregated_birdsong_data_14_syl_linear_50_song_in_batch_50_timesteps_20250608_150659.h5")
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create model
    print("\n=== Creating Model ===")
    model = BirdsongLFADSModel2(
        alphabet_size=14,
        order=1,
        encoder_dim=128,
        controller_dim=128,
        generator_dim=128,
        factor_dim=64,
        latent_dim=32,
        inferred_input_dim=16
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    print("\n=== Creating Trainer ===")
    trainer = BirdsongTrainer(model, dataset, config={
        "batch_size": 32,  # Small batch for testing
        "epochs": 1,
        "learning_rate": 0.001,
        "num_workers": 0,  # No workers for testing
        "pin_memory": True,
        "print_every": 1
    })
    
    print(f"Trainer device: {trainer.device}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Test one batch
    print("\n=== Testing One Batch ===")
    start_time = time.time()
    
    # Get one batch
    train_loader, _, _ = trainer.validation_subset.get_loaders(
        batch_size=32,
        num_workers=0,
        pin_memory=True
    )
    
    for batch_idx, (bigram_counts, probabilities) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Input device: {bigram_counts.device}")
        print(f"  Target device: {probabilities.device}")
        
        # Move to device
        bigram_counts = bigram_counts.to(trainer.device)
        probabilities = probabilities.to(trainer.device)
        
        print(f"  After .to(): {bigram_counts.device}")
        
        # Forward pass
        outputs = model(bigram_counts)
        _, loss_dict = model.compute_loss(probabilities, outputs)
        
        print(f"  Loss: {loss_dict['rec_loss'].item():.4f}")
        print(f"  Time: {time.time() - start_time:.2f} seconds")
        break
    
    print(f"\nTotal time for one batch: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    test_gpu_usage() 