#!/usr/bin/env python3
"""
Simple test script to run evaluation and see what happens.
"""

import sys
import os
sys.path.insert(0, 'src')

from birdsong.cli.eval_enhanced import EvalCLI

def main():
    """Run evaluation test."""
    print("Starting evaluation test...")
    
    # Create CLI instance
    cli = EvalCLI()
    
    # Create mock args
    class MockArgs:
        checkpoint = "outputs/train_20250716_161833_seed547569123/checkpoint_547569123_best.pt"
        config = "configs/example_dataset_large_batch.yaml"
        data_path = "examples/example datasets/aggregated_birdsong_data_1st_order_8_syl_linear_100_song_in_batch_50_timesteps_20250612_165300.h5"
        output_dir = "outputs/eval_test"
        num_plots = 3
        num_samples = 5
        seed = 42
        device = "cpu"
        dry_run = False
        verbose = True
    
    args = MockArgs()
    
    try:
        print("Running evaluation...")
        cli.run_command(args)
        print("Evaluation completed successfully!")
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 