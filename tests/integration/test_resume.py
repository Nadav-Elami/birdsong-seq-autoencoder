#!/usr/bin/env python3
"""
Test script to verify resume functionality with optimized configuration.
"""

import os
import subprocess
import sys

def test_resume_functionality():
    """Test that resume functionality works with the optimized config."""
    
    print("Testing resume functionality with optimized configuration...")
    
    # Configuration
    config_path = "configs/example_dataset_large_batch.yaml"
    data_path = "examples/example datasets/aggregated_birdsong_data_14_syl_linear_50_song_in_batch_50_timesteps_20250608_150659.h5"
    output_dir = "outputs/14_syl_linear_50_song_in_batch_50_timesteps_1st_order_12_epochs"
    
    # Check if checkpoint exists
    checkpoint_pattern = os.path.join(output_dir, "checkpoint_*.pt")
    import glob
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print("❌ No checkpoint files found. Cannot test resume functionality.")
        print("Please run training first to generate checkpoints.")
        return False
    
    # Find the best checkpoint for resuming
    best_checkpoint = None
    for checkpoint in checkpoint_files:
        if "_best.pt" in checkpoint:
            best_checkpoint = checkpoint
            break
    
    if not best_checkpoint:
        # Use the latest checkpoint
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        best_checkpoint = checkpoint_files[0]
    
    print(f"✅ Found checkpoint for resuming: {best_checkpoint}")
    
    # Test resume command
    resume_cmd = [
        "birdsong-train",
        "--config", config_path,
        "--data-path", data_path,
        "--output-dir", output_dir,
        "--resume", best_checkpoint,
        "--epochs", "2"  # Just test 2 more epochs
    ]
    
    print(f"Testing resume command:")
    print(f"  {' '.join(resume_cmd)}")
    
    try:
        # Run the resume command
        result = subprocess.run(resume_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Resume training completed successfully!")
            print("✅ Optimized configuration is working with resume functionality")
            return True
        else:
            print("❌ Resume training failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running resume command: {e}")
        return False

def test_run_cli_resume():
    """Test the run_CLI.py script with resume functionality."""
    
    print("\nTesting run_CLI.py with resume functionality...")
    
    # Check if run_CLI.py exists
    if not os.path.exists("run_CLI.py"):
        print("❌ run_CLI.py not found")
        return False
    
    try:
        # Run the CLI script
        result = subprocess.run(["python", "run_CLI.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ run_CLI.py executed successfully!")
            print("✅ Resume functionality is properly configured")
            return True
        else:
            print("❌ run_CLI.py failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running run_CLI.py: {e}")
        return False

if __name__ == "__main__":
    print("Testing resume functionality with memory optimizations...")
    
    # Test 1: Direct CLI resume
    test1_success = test_resume_functionality()
    
    # Test 2: run_CLI.py resume
    test2_success = test_run_cli_resume()
    
    print(f"\nTest Results:")
    print(f"  Direct CLI resume: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"  run_CLI.py resume: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! Resume functionality is working with optimizations.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.") 