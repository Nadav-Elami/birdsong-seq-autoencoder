#!/usr/bin/env python3
"""
Very simple test to check basic functionality.
"""

import sys
sys.path.insert(0, 'src')

print("Testing imports...")

try:
    import torch
    print("✓ torch imported")
    
    from birdsong.models.lfads import BirdsongLFADSModel2
    print("✓ BirdsongLFADSModel2 imported")
    
    from birdsong.data.loader import BirdsongDataset
    print("✓ BirdsongDataset imported")
    
    from birdsong.evaluation.evaluate import evaluate_birdsong_model
    print("✓ evaluate_birdsong_model imported")
    
    print("All imports successful!")
    
    # Test model creation
    print("Testing model creation...")
    model = BirdsongLFADSModel2(
        alphabet_size=8,
        order=1,
        encoder_dim=128,
        controller_dim=128,
        generator_dim=128,
        factor_dim=64,
        latent_dim=32,
        inferred_input_dim=16,
        kappa=1.0,
        ar_step_size=0.99,
        ar_process_var=0.1
    )
    print("✓ Model created successfully")
    
    # Test dataset loading
    print("Testing dataset loading...")
    dataset = BirdsongDataset("examples/example datasets/aggregated_birdsong_data_1st_order_8_syl_linear_100_song_in_batch_50_timesteps_20250612_165300.h5")
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Test checkpoint loading
    print("Testing checkpoint loading...")
    checkpoint = torch.load("outputs/train_20250716_161833_seed547569123/checkpoint_547569123_best.pt", map_location='cpu')
    print("✓ Checkpoint loaded successfully")
    
    print("All basic functionality works!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 