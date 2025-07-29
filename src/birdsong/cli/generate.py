"""
Enhanced command-line interface for generating synthetic birdsong data with reproducibility.

This module provides a CLI for generating synthetic birdsong datasets with
automatic seed tracking, reproducibility metadata, and hierarchical configuration support.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add the src directory to the path so we can import birdsong
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from birdsong.cli.base import ReproducibleCLI, create_output_filename
from birdsong.config import load_config, load_template, validate_config, ConfigValidationError
from birdsong.data.generation import BirdsongSimulator
from birdsong.data.aggregation import BirdsongAggregator


class GenerateCLI(ReproducibleCLI):
    """Enhanced data generation CLI with reproducibility and hierarchical configuration features."""
    
    def __init__(self):
        super().__init__(
            command_name="generate",
            description="Generate synthetic birdsong data with automatic seed tracking and hierarchical configuration"
        )
    
    def add_command_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add data generation-specific command-line arguments."""
        
        # Template argument (config is already provided by base class)
        parser.add_argument(
            "--template", "-t",
            type=str,
            help="Configuration template name (alternative to --config for templates)"
        )
        
        # Data generation parameters (overrides)
        parser.add_argument(
            "--num-songs",
            type=int,
            help="Override number of songs to generate"
        )
        
        parser.add_argument(
            "--song-length",
            type=int,
            default=50,
            help="Length of each song (number of time steps)"
        )
        
        parser.add_argument(
            "--alphabet-size",
            type=int,
            default=8,
            help="Size of the syllable alphabet"
        )
        
        parser.add_argument(
            "--transition-order",
            type=int,
            default=1,
            choices=[1, 2, 3],
            help="Order of Markov transition model (1=first-order, 2=second-order, etc.)"
        )
        
        # Process function parameters
        parser.add_argument(
            "--process-type",
            type=str,
            default="linear",
            choices=["linear", "cosine", "fourier", "exponential", "polynomial"],
            help="Type of process function to apply to sequences"
        )
        
        parser.add_argument(
            "--noise-level",
            type=float,
            default=0.1,
            help="Noise level to add to generated data (0.0 = no noise, 1.0 = high noise)"
        )
        
        # Output parameters
        parser.add_argument(
            "--output-name",
            type=str,
            help="Base name for output file (if not provided, will be auto-generated)"
        )
        
        parser.add_argument(
            "--format",
            type=str,
            default="hdf5",
            choices=["hdf5", "npz", "csv"],
            help="Output format for generated data"
        )
        
        # Validation parameters
        parser.add_argument(
            "--validate-data",
            action="store_true",
            help="Run validation checks on generated data"
        )
        
        parser.add_argument(
            "--plot-samples",
            action="store_true",
            help="Generate sample plots of the data"
        )
        
        parser.add_argument(
            "--num-plot-samples",
            type=int,
            default=5,
            help="Number of sample plots to generate"
        )
    
    def load_config(self, config_path: str = None, template_name: str = None, 
                   override_values: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Load configuration using the new hierarchical config system.
        
        Args:
            config_path: Path to configuration file
            template_name: Name of configuration template
            override_values: Dictionary of override values
            
        Returns:
            Loaded and validated configuration
        """
        try:
            if template_name:
                print(f"Loading configuration template: {template_name}")
                config = load_template(template_name, override_values=override_values)
            elif config_path:
                print(f"Loading configuration from: {config_path}")
                config = load_config(config_path, override_values=override_values)
            else:
                # Use default data generation template
                print("No config specified, using 'data_generation' template")
                config = load_template('data_generation', override_values=override_values)
            
            # Validate configuration
            warnings = validate_config(config, strict=False)
            if warnings:
                print(f"Configuration warnings: {len(warnings)} warnings found")
                for warning in warnings[:5]:  # Show first 5 warnings
                    print(f"  Warning: {warning}")
                if len(warnings) > 5:
                    print(f"  ... and {len(warnings) - 5} more warnings")
            
            return config.model_dump()
            
        except ConfigValidationError as e:
            print(f"Configuration validation error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)

    def validate_setup(self) -> None:
        """Validate that the CLI setup is correct."""
        super().validate_setup()
        
        # Additional data generation validation
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        
        # Check data configuration exists
        if 'data' not in self.config:
            raise RuntimeError("No data configuration found")
    
    def _merge_config_with_args(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Merge configuration with command-line arguments."""
        # Build override values from command line args
        override_values = {}
        
        if args.num_songs:
            override_values['data.num_batches'] = args.num_songs // 10  # Rough conversion
        if args.song_length:
            override_values['data.sequence_length_range'] = [args.song_length, args.song_length]
        if args.alphabet_size:
            override_values['data.alphabet_size'] = args.alphabet_size
        if args.transition_order:
            override_values['data.order'] = args.transition_order
        if args.process_type:
            # Map to process_names
            if args.process_type == 'linear':
                override_values['data.process_names'] = ['linear']
            elif args.process_type == 'noise':
                override_values['data.process_names'] = ['linear_with_noise']
        if args.output_name:
            override_values['data.output_name'] = args.output_name
        if args.validate_data:
            override_values['data.validate'] = True
        if args.plot_samples:
            override_values['data.plot_samples'] = True
        if args.num_plot_samples:
            override_values['data.num_plot_samples'] = args.num_plot_samples
        
        # Load configuration with overrides
        config_source = args.template or args.config
        return self.load_config(config_source, args.template, override_values)
    
    def _create_simulator(self, config: Dict[str, Any]) -> BirdsongSimulator:
        """Create birdsong simulator from configuration."""
        generation_params = config.get('generation_params', {})
        
        # Default parameters
        default_params = {
            'alphabet_size': 8,
            'transition_order': 1,
            'process_type': 'linear',
            'noise_level': 0.1,
        }
        
        # Merge with provided parameters
        simulator_params = {}
        for key, default_value in default_params.items():
            simulator_params[key] = generation_params.get(key, default_value)
        
        if self.verbose:
            print(f"🎵 Creating simulator with parameters: {simulator_params}")
        
        return BirdsongSimulator(**simulator_params)
    
    def _create_aggregator(self, config: Dict[str, Any]) -> BirdsongAggregator:
        """Create birdsong aggregator from configuration."""
        generation_params = config.get('generation_params', {})
        
        aggregator_params = {
            'process_type': generation_params.get('process_type', 'linear'),
            'output_dir': self.output_dir,
        }
        
        if self.verbose:
            print(f"📊 Creating aggregator with parameters: {aggregator_params}")
        
        return BirdsongAggregator(**aggregator_params)
    
    def _generate_output_filename(self, config: Dict[str, Any]) -> str:
        """Generate output filename with descriptive parameters."""
        generation_params = config.get('generation_params', {})
        output_params = config.get('output_params', {})
        
        # Check if custom name provided
        if output_params.get('output_name'):
            base_name = output_params['output_name']
        else:
            # Generate descriptive name
            num_songs = generation_params.get('num_songs', 100)
            song_length = generation_params.get('song_length', 50)
            alphabet_size = generation_params.get('alphabet_size', 8)
            transition_order = generation_params.get('transition_order', 1)
            process_type = generation_params.get('process_type', 'linear')
            
            base_name = (f"birdsong_data_{num_songs}songs_{song_length}steps_"
                        f"{alphabet_size}syl_{transition_order}order_{process_type}")
        
        # Add seed and timestamp
        seed = self.seed_manager.master_seed
        
        # Determine file extension
        format_type = output_params.get('format', 'hdf5')
        extension_map = {
            'hdf5': '.h5',
            'npz': '.npz',
            'csv': '.csv'
        }
        extension = extension_map.get(format_type, '.h5')
        
        return create_output_filename(base_name, seed, extension)
    
    def _validate_generated_data(self, data_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the generated data and return validation metrics."""
        if self.verbose:
            print("🔍 Validating generated data...")
        
        import h5py
        import numpy as np
        
        validation_results = {}
        
        try:
            with h5py.File(data_path, 'r') as f:
                # Check data structure
                expected_keys = ['sequences', 'processed_data', 'metadata']
                validation_results['structure_valid'] = all(key in f for key in expected_keys)
                
                if 'sequences' in f:
                    sequences = f['sequences'][:]
                    validation_results['num_sequences'] = len(sequences)
                    validation_results['sequence_shape'] = sequences.shape
                    
                    # Check sequence properties
                    validation_results['min_value'] = float(np.min(sequences))
                    validation_results['max_value'] = float(np.max(sequences))
                    validation_results['mean_value'] = float(np.mean(sequences))
                    validation_results['std_value'] = float(np.std(sequences))
                    
                    # Check for NaN or infinite values
                    validation_results['has_nan'] = bool(np.isnan(sequences).any())
                    validation_results['has_inf'] = bool(np.isinf(sequences).any())
                
                if 'processed_data' in f:
                    processed = f['processed_data'][:]
                    validation_results['processed_shape'] = processed.shape
                    validation_results['processed_dtype'] = str(processed.dtype)
                
                if 'metadata' in f:
                    metadata = dict(f['metadata'].attrs)
                    validation_results['metadata_keys'] = list(metadata.keys())
                    validation_results['generation_seed'] = metadata.get('generation_seed', 'unknown')
        
        except Exception as e:
            validation_results['validation_error'] = str(e)
            validation_results['structure_valid'] = False
        
        # Save validation results
        validation_path = self.output_dir / create_output_filename(
            'data_validation', self.seed_manager.master_seed, '.json'
        )
        
        import json
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        if self.verbose:
            print(f"🔍 Validation results saved to {validation_path}")
            if validation_results.get('structure_valid', False):
                print("✅ Data structure validation passed")
                print(f"📊 Generated {validation_results.get('num_sequences', 0)} sequences")
            else:
                print("❌ Data structure validation failed")
        
        return validation_results
    
    def _generate_sample_plots(self, data_path: Path, config: Dict[str, Any]) -> None:
        """Generate sample plots of the generated data."""
        if self.verbose:
            print(f"📊 Generating sample plots...")
        
        import h5py
        import matplotlib.pyplot as plt
        import numpy as np
        
        validation_params = config.get('validation_params', {})
        num_samples = validation_params.get('num_plot_samples', 5)
        
        plot_dir = self.output_dir / 'sample_plots'
        plot_dir.mkdir(exist_ok=True)
        
        try:
            with h5py.File(data_path, 'r') as f:
                if 'sequences' in f and 'processed_data' in f:
                    sequences = f['sequences'][:]
                    processed = f['processed_data'][:]
                    
                    # Generate individual sample plots
                    for i in range(min(num_samples, len(sequences))):
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                        
                        # Plot original sequence
                        ax1.plot(sequences[i])
                        ax1.set_title(f'Original Sequence {i+1}')
                        ax1.set_ylabel('Syllable Index')
                        ax1.grid(True, alpha=0.3)
                        
                        # Plot processed data
                        ax2.plot(processed[i])
                        ax2.set_title(f'Processed Data {i+1}')
                        ax2.set_xlabel('Time Steps')
                        ax2.set_ylabel('Processed Values')
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        
                        plot_path = plot_dir / create_output_filename(
                            f'sample_{i+1}', self.seed_manager.master_seed, '.png'
                        )
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()
                    
                    # Generate summary plot
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    
                    # Sequence length distribution
                    seq_lengths = [len(seq) for seq in sequences]
                    axes[0, 0].hist(seq_lengths, bins=20, alpha=0.7, edgecolor='black')
                    axes[0, 0].set_title('Sequence Length Distribution')
                    axes[0, 0].set_xlabel('Sequence Length')
                    axes[0, 0].set_ylabel('Frequency')
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # Syllable distribution
                    all_syllables = np.concatenate(sequences)
                    unique, counts = np.unique(all_syllables, return_counts=True)
                    axes[0, 1].bar(unique, counts, alpha=0.7, edgecolor='black')
                    axes[0, 1].set_title('Syllable Distribution')
                    axes[0, 1].set_xlabel('Syllable Index')
                    axes[0, 1].set_ylabel('Frequency')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Processed data distribution
                    all_processed = processed.flatten()
                    axes[1, 0].hist(all_processed, bins=50, alpha=0.7, edgecolor='black')
                    axes[1, 0].set_title('Processed Data Distribution')
                    axes[1, 0].set_xlabel('Value')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Example processed sequences
                    for i in range(min(3, len(processed))):
                        axes[1, 1].plot(processed[i], alpha=0.7, label=f'Sequence {i+1}')
                    axes[1, 1].set_title('Sample Processed Sequences')
                    axes[1, 1].set_xlabel('Time Steps')
                    axes[1, 1].set_ylabel('Processed Values')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    summary_path = plot_dir / create_output_filename(
                        'data_summary', self.seed_manager.master_seed, '.png'
                    )
                    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    if self.verbose:
                        print(f"📊 Sample plots saved to {plot_dir}")
                        
        except Exception as e:
            if self.verbose:
                print(f"❌ Error generating sample plots: {e}")
    
    def run_command(self, args: argparse.Namespace) -> None:
        """Run the data generation command."""
        try:
            # Load and merge configuration
            self.config = self._merge_config_with_args(args)
            
            # Setup output directory and reproducibility
            self.setup_reproducibility(args.seed)
            self.setup_output_directory(args.output_dir)
            
            print(f"Data generation with seed: {self.seed_manager.master_seed}")
            print(f"Output directory: {self.output_dir}")
            
            # Dry run mode
            if args.dry_run:
                print("DRY RUN MODE - Configuration validation only")
                self.validate_setup()
                data_config = self.config['data']
                print("✓ Configuration validated successfully")
                print(f"✓ Would generate {data_config.get('num_batches', 10)} batches of {data_config.get('batch_size', 100)} songs")
                print(f"✓ Alphabet size: {data_config.get('alphabet_size', 8)}")
                print(f"✓ Sequence length range: {data_config.get('sequence_length_range', [8, 25])}")
                print(f"✓ Process types: {', '.join(data_config.get('process_names', ['linear']))}")
                print("Dry run completed successfully!")
                return
            
            # Validate setup
            self.validate_setup()
            
            # Get data configuration
            data_config = self.config['data']
            
            # Generate data using the new approach
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"birdsong_data_seed{self.seed_manager.master_seed}_{timestamp}.h5"
            output_path = self.output_dir / output_filename
            
            # Use the BirdsongAggregator to generate data
            aggregator = BirdsongAggregator(
                alphabet_size=data_config['alphabet_size'],
                order=data_config['order']
            )
            
            simulator = BirdsongSimulator(
                alphabet_size=data_config['alphabet_size'],
                order=data_config['order']
            )
            
            print(f"🎵 Generating {data_config['num_batches']} batches of {data_config['batch_size']} songs...")
            
            # Generate and aggregate data
            aggregator.generate_and_aggregate_birdsong_data(
                simulator=simulator,
                num_batches=data_config['num_batches'],
                batch_size=data_config['batch_size'],
                sequence_length_range=data_config['sequence_length_range'],
                process_names=data_config['process_names'],
                save_path=str(output_path)
            )
            
            # Validate generated data
            from birdsong.data.loader import BirdsongDataset
            dataset = BirdsongDataset(str(output_path))
            
            # Save metadata
            generation_metadata = {
                'output_file': str(output_path),
                'dataset_size': len(dataset),
                'file_size_mb': output_path.stat().st_size / (1024 * 1024),
                'data_config': data_config
            }
            self.save_metadata(generation_metadata)
            
            print(f"💾 Data saved to: {output_path}")
            print(f"📊 Dataset size: {len(dataset)} samples")
            print(f"💾 File size: {generation_metadata['file_size_mb']:.2f} MB")
            
            # Validate data if requested
            if data_config.get('validate', False) or args.validate_data:
                print("🔍 Validating generated data...")
                sample_data = dataset[0]
                print(f"✓ Sample data shapes: {[d.shape for d in sample_data]}")
                print(f"✓ Alphabet size: {data_config['alphabet_size']}")
            
            # Generate plots if requested
            if data_config.get('plot_samples', False) or args.plot_samples:
                num_plots = data_config.get('num_plot_samples', args.num_plot_samples or 3)
                print(f"📈 Generating {num_plots} sample plots...")
                
                plot_dir = self.output_dir / "sample_plots"
                plot_dir.mkdir(exist_ok=True)
                
                for i in range(min(num_plots, len(dataset))):
                    bigram_counts, probabilities = dataset[i]
                    
                    import matplotlib.pyplot as plt
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    ax1.imshow(bigram_counts.T, aspect='auto', cmap='viridis')
                    ax1.set_title(f'Sample {i+1}: Bigram Counts')
                    ax1.set_xlabel('Time Steps')
                    ax1.set_ylabel('Bigram Features')
                    
                    ax2.imshow(probabilities.T, aspect='auto', cmap='viridis')
                    ax2.set_title(f'Sample {i+1}: Probabilities')
                    ax2.set_xlabel('Time Steps')
                    ax2.set_ylabel('Probability Features')
                    
                    plt.tight_layout()
                    plot_path = plot_dir / f"sample_{i+1}.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                
                print(f"📈 Sample plots saved to: {plot_dir}")
            
            print("✅ Data generation completed successfully!")
            print(f"📂 Results saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ Error during data generation: {e}", file=sys.stderr)
            # Save error metadata
            self.save_metadata({"error": str(e), "generation_completed": False})
            sys.exit(1)


def main():
    """Main entry point for enhanced data generation CLI."""
    cli = GenerateCLI()
    cli.execute()


if __name__ == "__main__":
    main() 