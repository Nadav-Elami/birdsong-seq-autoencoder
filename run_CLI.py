"""
Birdsong LFADS Training CLI Runner

This script provides a convenient way to run the Birdsong LFADS training pipeline
with automatic checkpoint detection and resume functionality.

USAGE EXAMPLES:

1. Start fresh training:
   python run_CLI.py
   (Set RESUME_TRAINING = False)

2. Resume training from latest checkpoint:
   python run_CLI.py
   (Set RESUME_TRAINING = True, RESUME_CHECKPOINT_PATH = None)

3. Resume from specific checkpoint:
   python run_CLI.py
   (Set RESUME_TRAINING = True, RESUME_CHECKPOINT_PATH = "path/to/checkpoint.pt")

4. Run only evaluation:
   Set STAGES_TO_RUN = ['eval']

5. Run full pipeline (data generation + training + evaluation):
   Set STAGES_TO_RUN = ['all']

The script will automatically detect the best checkpoint for resuming training
and the latest checkpoint for evaluation.
"""

import subprocess
import sys
import glob
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION - Edit these variables to set your paths and stages
# =============================================================================

# Paths (edit these to match your setup)
CONFIG_PATH = "configs/example_dataset_large_batch.yaml"
DATA_PATH = "examples/example datasets/aggregated_birdsong_data_14_syl_linear_50_song_in_batch_50_timesteps_20250608_150659.h5"
CHECKPOINT_PATH = None  # Set to checkpoint path if you have one, e.g., "outputs/train_xxx/checkpoint.pt"
OUTPUT_DIR = "outputs/14_syl_linear_50_song_in_batch_50_timesteps_1st_order_15_epochs_with_latents_no_kl"  # Set to output directory if you want specific output, e.g., "my_experiment_output"

# Training resume settings
RESUME_TRAINING = False  # Set to True to resume from latest checkpoint
RESUME_CHECKPOINT_PATH = None  # Set to specific checkpoint path, or leave None for auto-detection

# Which stages to run (edit this list to choose what to run)
# Options: ['data', 'train', 'eval'] or ['all'] for everything
STAGES_TO_RUN = ['train', 'eval']  # Change this to run different stages

# Extra CLI arguments (optional)
EXTRA_ARGS = ""

# Evaluation settings
USE_TEST_SET = True  # Set to True to use test set from checkpoint (recommended)
EVAL_NUM_SAMPLES = 10  # Number of samples to evaluate (ignored if USE_TEST_SET=True)

# =============================================================================
# END CONFIGURATION - Don't edit below this line
# =============================================================================

def find_latest_checkpoint(output_dir="outputs"):
    """Find the latest checkpoint file in the output directory."""
    checkpoint_pattern = os.path.join(output_dir, "**", "checkpoint_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern, recursive=True)
    
    if not checkpoint_files:
        print(f"[run_CLI.py] Warning: No checkpoint files found in {output_dir}")
        return None
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    latest_checkpoint = checkpoint_files[0]
    
    print(f"[run_CLI.py] Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def find_resume_checkpoint(output_dir="outputs"):
    """Find the best checkpoint for resuming training."""
    # First try to find the best checkpoint (lowest validation loss)
    best_checkpoint_pattern = os.path.join(output_dir, "**", "checkpoint_*_best.pt")
    best_checkpoint_files = glob.glob(best_checkpoint_pattern, recursive=True)
    
    if best_checkpoint_files:
        # Sort by modification time (newest first)
        best_checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        best_checkpoint = best_checkpoint_files[0]
        print(f"[run_CLI.py] Found best checkpoint for resuming: {best_checkpoint}")
        return best_checkpoint
    
    # If no best checkpoint, find the latest regular checkpoint
    regular_checkpoint_pattern = os.path.join(output_dir, "**", "checkpoint_*.pt")
    regular_checkpoint_files = glob.glob(regular_checkpoint_pattern, recursive=True)
    
    if regular_checkpoint_files:
        # Sort by modification time (newest first)
        regular_checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        latest_checkpoint = regular_checkpoint_files[0]
        print(f"[run_CLI.py] Found latest checkpoint for resuming: {latest_checkpoint}")
        return latest_checkpoint
    
    print(f"[run_CLI.py] Warning: No checkpoint files found in {output_dir}")
    return None

# Map stages to CLI commands with their specific argument patterns
CLI_COMMANDS = {
    'data': 'birdsong-generate',
    'train': 'birdsong-train',
    'eval': 'birdsong-eval',
}


def run_command(cmd):
    print(f"\n[run_CLI.py] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[run_CLI.py] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def build_command_args(stage, config_path, data_path=None, checkpoint_path=None, output_dir=None, extra_args="", resume_path=None):
    """Build the appropriate arguments for each CLI command."""
    base_args = f'--config "{config_path}"'
    
    if stage == 'data':
        # Data generation CLI: --config + optional overrides
        args = base_args
        if data_path:
            args += f' --data-path "{data_path}"'
        if output_dir:
            args += f' --output-dir "{output_dir}"'
        if extra_args:
            args += f" {extra_args}"
            
    elif stage == 'train':
        # Training CLI: --config + optional overrides + resume
        args = base_args
        if data_path:
            args += f' --data-path "{data_path}"'
        if output_dir:
            args += f' --output-dir "{output_dir}"'
        if resume_path:
            args += f' --resume "{resume_path}"'
        if extra_args:
            args += f" {extra_args}"
            
    elif stage == 'eval':
        # Evaluation CLI: --config + --checkpoint
        args = base_args
        if checkpoint_path:
            args += f' --checkpoint "{checkpoint_path}"'
        else:
            print(f"[run_CLI.py] Warning: No checkpoint specified for evaluation stage")
        if output_dir:
            args += f' --output-dir "{output_dir}"'
        
        # Add test set evaluation if enabled
        if USE_TEST_SET:
            args += ' --use-test-set'
            print(f"[run_CLI.py] Using test set from checkpoint for evaluation")
        else:
            args += f' --num-samples {EVAL_NUM_SAMPLES}'
            print(f"[run_CLI.py] Using {EVAL_NUM_SAMPLES} random samples for evaluation")
            
        if extra_args:
            args += f" {extra_args}"
    
    return args


def main():
    # Resolve stages
    stages = STAGES_TO_RUN
    if 'all' in stages:
        stages = ['data', 'train', 'eval']

    print(f"[run_CLI.py] Configuration:")
    print(f"  Config: {CONFIG_PATH}")
    print(f"  Data: {DATA_PATH}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Stages: {stages}")
    print(f"  Resume Training: {RESUME_TRAINING}")
    print(f"  Extra args: {EXTRA_ARGS}")

    # Run selected stages
    for stage in stages:
        if stage not in CLI_COMMANDS:
            print(f"[run_CLI.py] Unknown stage: {stage}")
            continue
        
        # Handle checkpoint detection for different stages
        checkpoint_path = CHECKPOINT_PATH
        resume_path = RESUME_CHECKPOINT_PATH
        
        if stage == 'train' and RESUME_TRAINING:
            # For training, find resume checkpoint
            if resume_path is None:
                resume_path = find_resume_checkpoint(OUTPUT_DIR)
                if resume_path is None:
                    print(f"[run_CLI.py] Warning: No checkpoint found for resuming training - starting fresh")
                    resume_path = None
                else:
                    print(f"[run_CLI.py] Resuming training from: {resume_path}")
            else:
                print(f"[run_CLI.py] Using specified resume checkpoint: {resume_path}")
                
        elif stage == 'eval' and checkpoint_path is None:
            # For evaluation, find latest checkpoint
            checkpoint_path = find_latest_checkpoint(OUTPUT_DIR)
            if checkpoint_path is None:
                print(f"[run_CLI.py] Error: No checkpoint found for evaluation stage")
                sys.exit(1)
            
        cmd = CLI_COMMANDS[stage]
        args = build_command_args(stage, CONFIG_PATH, DATA_PATH, checkpoint_path, OUTPUT_DIR, EXTRA_ARGS, resume_path)
        full_cmd = f"{cmd} {args}".strip()
        run_command(full_cmd)

    print("\n[run_CLI.py] All selected stages completed successfully.")


if __name__ == "__main__":
    main() 