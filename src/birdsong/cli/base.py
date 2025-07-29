"""
Base CLI functionality for birdsong commands.

This module provides common CLI functionality including:
- Automatic seed management and tracking
- Reproducibility metadata recording
- Enhanced argument parsing with seed support
- Output file management with metadata
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..utils.reproducibility import (
    SeedManager,
    get_environment_fingerprint,
    save_reproducibility_info,
    set_global_seed,
)


class ReproducibleCLI:
    """
    Base class for reproducible CLI commands.
    
    This class provides common functionality for all birdsong CLI commands,
    including automatic seed management and reproducibility tracking.
    """
    
    def __init__(self, command_name: str, description: str):
        """
        Initialize the reproducible CLI.
        
        Args:
            command_name: Name of the CLI command (e.g., 'train', 'eval', 'generate')
            description: Description for the argument parser
        """
        self.command_name = command_name
        self.description = description
        self.seed_manager: Optional[SeedManager] = None
        self.output_dir: Optional[Path] = None
        self.config: Dict[str, Any] = {}
        
    def create_parser(self) -> argparse.ArgumentParser:
        """
        Create argument parser with common CLI arguments.
        
        Returns:
            Configured argument parser with common arguments
        """
        parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Add common arguments for all commands
        parser.add_argument(
            "--seed", "-s",
            type=int,
            help="Random seed for reproducibility (if not provided, one will be generated)"
        )
        
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            help="Output directory for results and metadata"
        )
        
        parser.add_argument(
            "--config", "-c",
            type=str,
            help="Path to YAML configuration file"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output"
        )
        
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate configuration and setup without running"
        )
        
        return parser
    
    def add_command_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add command-specific arguments to the parser.
        
        This method should be overridden by subclasses to add their
        specific command-line arguments.
        
        Args:
            parser: Argument parser to add arguments to
        """
        pass
    
    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from YAML file with validation.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if not config_path:
            return {}
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")
    
    def setup_reproducibility(self, seed: Optional[int] = None) -> None:
        """
        Set up reproducible environment with seed management.
        
        Args:
            seed: Random seed to use (if None, will be generated)
        """
        self.seed_manager = set_global_seed(seed)
        
        if hasattr(self, 'verbose') and self.verbose:
            print(f"🎲 Reproducibility setup complete with seed: {self.seed_manager.master_seed}")
    
    def setup_output_directory(self, output_dir: Optional[str] = None) -> Path:
        """
        Set up output directory with automatic naming if not provided.
        
        Args:
            output_dir: Output directory path (if None, will be auto-generated)
            
        Returns:
            Path to output directory
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Generate automatic output directory name with timestamp and seed
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed_suffix = f"_seed{self.seed_manager.master_seed}" if self.seed_manager else ""
            dir_name = f"{self.command_name}_{timestamp}{seed_suffix}"
            self.output_dir = Path("outputs") / dir_name
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self, 'verbose') and self.verbose:
            print(f"📁 Output directory: {self.output_dir}")
            
        return self.output_dir
    
    def save_metadata(self, additional_info: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save reproducibility metadata and experiment information.
        
        Args:
            additional_info: Additional metadata to save
            
        Returns:
            Path to saved metadata file
        """
        if not self.seed_manager:
            raise RuntimeError("Reproducibility must be set up before saving metadata")
        
        if not self.output_dir:
            raise RuntimeError("Output directory must be set up before saving metadata")
        
        # Prepare metadata
        metadata = {
            'command': self.command_name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'command_line_args': sys.argv,
            'working_directory': os.getcwd(),
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        # Save reproducibility info
        repro_path = self.output_dir / 'reproducibility.json'
        save_reproducibility_info(repro_path, self.seed_manager, metadata)
        
        # Also save a human-readable summary
        summary_path = self.output_dir / 'experiment_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Birdsong {self.command_name.title()} Experiment\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Command: birdsong-{self.command_name}\n")
            f.write(f"Timestamp: {metadata['timestamp']}\n")
            f.write(f"Seed: {self.seed_manager.master_seed}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            if self.config:
                f.write("Configuration:\n")
                f.write("-" * 20 + "\n")
                f.write(yaml.dump(self.config, default_flow_style=False))
                f.write("\n")
            
            f.write("Command Line:\n")
            f.write("-" * 20 + "\n")
            f.write(" ".join(sys.argv) + "\n\n")
        
        if hasattr(self, 'verbose') and self.verbose:
            print(f"💾 Metadata saved to {repro_path}")
            print(f"📄 Summary saved to {summary_path}")
        
        return repro_path
    
    def validate_setup(self) -> None:
        """
        Validate that the CLI setup is correct.
        
        This method should be overridden by subclasses to add their
        specific validation logic.
        """
        if not self.seed_manager:
            raise RuntimeError("Reproducibility not set up")
        
        if not self.output_dir:
            raise RuntimeError("Output directory not set up")
    
    def run_command(self, args: argparse.Namespace) -> None:
        """
        Main command execution logic.
        
        This method should be overridden by subclasses to implement
        their specific command logic.
        
        Args:
            args: Parsed command-line arguments
        """
        raise NotImplementedError("Subclasses must implement run_command")
    
    def execute(self) -> None:
        """
        Main execution entry point that handles common setup and error handling.
        """
        try:
            # Create parser and add command-specific arguments
            parser = self.create_parser()
            self.add_command_arguments(parser)
            
            # Parse arguments
            args = parser.parse_args()
            
            # Store verbose flag for use in other methods
            self.verbose = getattr(args, 'verbose', False)
            
            # Load configuration
            if hasattr(args, 'config') and args.config:
                self.config = self.load_config(args.config)
                if self.verbose:
                    print(f"📋 Configuration loaded from {args.config}")
            
            # Set up reproducibility
            seed = getattr(args, 'seed', None)
            self.setup_reproducibility(seed)
            
            # Set up output directory
            output_dir = getattr(args, 'output_dir', None)
            self.setup_output_directory(output_dir)
            
            # Validate setup
            self.validate_setup()
            
            # Save metadata
            self.save_metadata()
            
            # Check for dry run
            if getattr(args, 'dry_run', False):
                print("🔍 Dry run - validation complete, exiting without execution")
                return
            
            # Run the actual command
            self.run_command(args)
            
            if self.verbose:
                print(f"✅ {self.command_name.title()} completed successfully!")
                
        except KeyboardInterrupt:
            print(f"\n⚠️ {self.command_name.title()} interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error during {self.command_name}: {e}", file=sys.stderr)
            sys.exit(1)


def create_output_filename(base_name: str, seed: int, extension: str = "", 
                         timestamp: bool = True) -> str:
    """
    Create a standardized output filename with seed and optional timestamp.
    
    Args:
        base_name: Base name for the file
        seed: Random seed to include in filename
        extension: File extension (including dot)
        timestamp: Whether to include timestamp
        
    Returns:
        Formatted filename
    """
    parts = [base_name, f"seed{seed}"]
    
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(ts)
    
    filename = "_".join(parts) + extension
    return filename


def load_experiment_metadata(output_dir: Path) -> Dict[str, Any]:
    """
    Load experiment metadata from output directory.
    
    Args:
        output_dir: Path to experiment output directory
        
    Returns:
        Experiment metadata dictionary
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    metadata_path = output_dir / 'reproducibility.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f) 