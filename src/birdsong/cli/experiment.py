"""
Command-line interface for running complete birdsong experiments.

This module provides a CLI for orchestrating complete birdsong research experiments
including data generation, training, and evaluation with automatic dependency management,
resume functionality, and progress tracking.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add the src directory to the path so we can import birdsong
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from birdsong.experiments import ExperimentRunner, run_experiment, resume_experiment


def create_experiment_parser() -> argparse.ArgumentParser:
    """Create the main experiment argument parser."""
    parser = argparse.ArgumentParser(
        description="Run complete birdsong research experiments with automatic pipeline orchestration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Experiment commands')
    
    # Run command
    run_parser = subparsers.add_parser(
        'run',
        help='Run a new experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    run_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file or template name (e.g., 'full_experiment', 'quick_test')"
    )
    
    run_parser.add_argument(
        "--template", "-t",
        type=str,
        help="Configuration template name (alternative to --config for templates)"
    )
    
    run_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for experiment results (auto-generated if not specified)"
    )
    
    run_parser.add_argument(
        "--seed", "-s",
        type=int,
        help="Random seed for reproducibility (auto-generated if not specified)"
    )
    
    run_parser.add_argument(
        "--name",
        type=str,
        help="Experiment name for identification"
    )
    
    run_parser.add_argument(
        "--stages",
        type=str,
        nargs='+',
        choices=['data', 'train', 'eval', 'all'],
        default=['all'],
        help="Stages to run (default: all stages)"
    )
    
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show execution plan without running"
    )
    
    run_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    # Resume command
    resume_parser = subparsers.add_parser(
        'resume',
        help='Resume an interrupted experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    resume_parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory to resume"
    )
    
    resume_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    # List command
    list_parser = subparsers.add_parser(
        'list',
        help='List experiments in the experiments directory',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    list_parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Directory containing experiments"
    )
    
    list_parser.add_argument(
        "--status",
        choices=['all', 'completed', 'failed', 'running'],
        default='all',
        help="Filter experiments by status"
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show status of a specific experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    status_parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory"
    )
    
    return parser


def validate_config_source(args: argparse.Namespace) -> str:
    """Validate and return the configuration source."""
    if args.template:
        return args.template
    elif args.config:
        return args.config
    else:
        # Default to full experiment template
        print("No configuration specified, using 'full_experiment' template")
        return 'full_experiment'


def show_execution_plan(config_source: str, stages: list, output_dir: str, seed: int) -> None:
    """Show the execution plan for a dry run."""
    print("🔍 Experiment Execution Plan")
    print("=" * 40)
    print(f"Configuration: {config_source}")
    print(f"Output Directory: {output_dir}")
    print(f"Seed: {seed}")
    print(f"Stages: {', '.join(stages)}")
    print()
    
    stage_descriptions = {
        'data': 'Generate synthetic birdsong data with specified parameters',
        'train': 'Train LFADS model on generated data',
        'eval': 'Evaluate trained model and generate analysis plots',
        'all': 'Complete pipeline: data generation → training → evaluation'
    }
    
    if 'all' in stages:
        stages = ['data', 'train', 'eval']
    
    print("Execution Steps:")
    for i, stage in enumerate(stages, 1):
        print(f"{i}. {stage.title()}: {stage_descriptions[stage]}")
    
    print("\nDry run completed. Use --no-dry-run to execute.")


def run_command(args: argparse.Namespace) -> int:
    """Execute the run command."""
    try:
        # Validate configuration source
        config_source = validate_config_source(args)
        
        # Setup output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = args.name or "experiment"
            seed_suffix = f"_seed{args.seed}" if args.seed else ""
            output_dir = f"experiments/{exp_name}_{timestamp}{seed_suffix}"
        
        # Dry run mode
        if args.dry_run:
            show_execution_plan(config_source, args.stages, output_dir, args.seed or 12345)
            return 0
        
        # Create experiment runner
        verbose = not args.quiet
        runner = ExperimentRunner(
            config=config_source,
            output_dir=output_dir,
            seed=args.seed,
            verbose=verbose
        )
        
        # Run specified stages
        if 'all' in args.stages:
            success = runner.run_full_experiment()
        else:
            success = True
            if 'data' in args.stages:
                success = success and runner.run_data_generation()
            if 'train' in args.stages and success:
                success = success and runner.run_training()
            if 'eval' in args.stages and success:
                success = success and runner.run_evaluation()
        
        if success:
            print(f"\n✅ Experiment completed successfully!")
            print(f"📂 Results saved to: {runner.output_dir}")
            return 0
        else:
            print(f"\n❌ Experiment failed!")
            print(f"📂 Partial results saved to: {runner.output_dir}")
            return 1
            
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return 1


def resume_command(args: argparse.Namespace) -> int:
    """Execute the resume command."""
    try:
        verbose = not args.quiet
        success = resume_experiment(args.experiment_dir, verbose=verbose)
        
        if success:
            print(f"\n✅ Experiment resumed and completed successfully!")
            return 0
        else:
            print(f"\n❌ Experiment failed during resume!")
            return 1
            
    except Exception as e:
        print(f"❌ Error resuming experiment: {e}")
        return 1


def list_command(args: argparse.Namespace) -> int:
    """Execute the list command."""
    try:
        experiments_dir = Path(args.experiments_dir)
        
        if not experiments_dir.exists():
            print(f"Experiments directory not found: {experiments_dir}")
            return 1
        
        # Find experiment directories
        experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
        
        if not experiment_dirs:
            print(f"No experiments found in {experiments_dir}")
            return 0
        
        print(f"Experiments in {experiments_dir}:")
        print("=" * 50)
        
        for exp_dir in sorted(experiment_dirs):
            # Try to load experiment status
            state_file = exp_dir / "experiment_state.json"
            status = "Unknown"
            
            if state_file.exists():
                try:
                    import json
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    
                    current_stage = state.get('current_stage')
                    if current_stage == 'completed':
                        status = "✅ Completed"
                    elif current_stage:
                        status = f"🔄 {current_stage.replace('_', ' ').title()}"
                    else:
                        status = "📝 Initialized"
                        
                    # Check for errors
                    results = state.get('results', [])
                    if any(not r.get('success', True) for r in results):
                        status = "❌ Failed"
                        
                except Exception:
                    status = "⚠️ Error"
            
            # Filter by status if requested
            if args.status != 'all':
                status_filter_map = {
                    'completed': '✅',
                    'failed': '❌',
                    'running': '🔄'
                }
                if status_filter_map.get(args.status, '') not in status:
                    continue
            
            print(f"{exp_dir.name}: {status}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error listing experiments: {e}")
        return 1


def status_command(args: argparse.Namespace) -> int:
    """Execute the status command."""
    try:
        exp_dir = Path(args.experiment_dir)
        
        if not exp_dir.exists():
            print(f"Experiment directory not found: {exp_dir}")
            return 1
        
        # Load experiment state
        state_file = exp_dir / "experiment_state.json"
        summary_file = exp_dir / "experiment_summary.json"
        
        if summary_file.exists():
            # Show completed experiment summary
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            print(f"Experiment Status: {exp_dir.name}")
            print("=" * 50)
            print(f"Status: ✅ Completed")
            print(f"Seed: {summary.get('seed', 'Unknown')}")
            print(f"Total Duration: {summary.get('total_duration', 'Unknown')}")
            print()
            
            print("Stages:")
            for stage in summary.get('stages', []):
                status = "✅" if stage.get('success', False) else "❌"
                print(f"  {status} {stage['stage'].replace('_', ' ').title()}: {stage.get('duration', 'Unknown')}")
                
                for key, value in stage.get('metrics', {}).items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.6f}")
                    else:
                        print(f"    {key}: {value}")
            
        elif state_file.exists():
            # Show in-progress experiment status
            import json
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            print(f"Experiment Status: {exp_dir.name}")
            print("=" * 50)
            
            current_stage = state.get('current_stage')
            if current_stage == 'completed':
                print(f"Status: ✅ Completed")
            elif current_stage:
                print(f"Status: 🔄 {current_stage.replace('_', ' ').title()}")
            else:
                print(f"Status: 📝 Initialized")
            
            print()
            print("Completed Stages:")
            for result in state.get('results', []):
                status = "✅" if result.get('success', False) else "❌"
                stage_name = result['stage'].replace('_', ' ').title()
                print(f"  {status} {stage_name}")
                
                if result.get('error'):
                    print(f"    Error: {result['error']}")
        else:
            print(f"No experiment state found in {exp_dir}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error getting experiment status: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = create_experiment_parser()
    args = parser.parse_args()
    
    if args.command == 'run':
        return run_command(args)
    elif args.command == 'resume':
        return resume_command(args)
    elif args.command == 'list':
        return list_command(args)
    elif args.command == 'status':
        return status_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 