"""
Experiment orchestration and pipeline management.

This module provides the ExperimentRunner class for orchestrating complete
birdsong research experiments with automatic dependency management, progress
tracking, and resume functionality.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..cli.base import ReproducibleCLI
from ..config import load_config, load_template, validate_config, ConfigValidationError
from ..data.aggregation import BirdsongAggregator
from ..data.generation import BirdsongSimulator  
from ..data.loader import BirdsongDataset
from ..models.lfads import BirdsongLFADSModel2
from ..training.trainer import BirdsongTrainer
from ..utils.reproducibility import SeedManager, set_global_seed


class ExperimentStage(Enum):
    """Enumeration of experiment pipeline stages."""
    
    DATA_GENERATION = "data_generation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    COMPLETED = "completed"
    
    @property
    def display_name(self) -> str:
        """Human-readable stage name."""
        return self.value.replace("_", " ").title()


@dataclass
class ExperimentResult:
    """Results from an experiment stage."""
    
    stage: ExperimentStage
    success: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Duration of the stage execution."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def duration_str(self) -> str:
        """Human-readable duration string."""
        if self.duration:
            total_seconds = int(self.duration.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        return "N/A"


class ExperimentRunner:
    """
    Orchestrates complete birdsong research experiments.
    
    The ExperimentRunner manages the full pipeline from data generation through
    training to evaluation, with automatic dependency management, progress tracking,
    and resume functionality.
    """
    
    def __init__(self, config: Union[str, Dict[str, Any]], output_dir: Optional[str] = None,
                 seed: Optional[int] = None, verbose: bool = True):
        """
        Initialize the experiment runner.
        
        Args:
            config: Configuration file path, template name, or config dictionary
            output_dir: Output directory for all experiment results
            seed: Random seed for reproducibility
            verbose: Whether to print detailed progress information
        """
        self.verbose = verbose
        self.config = self._load_config(config)
        self.seed_manager = set_global_seed(seed)
        
        # Set up output directory with seed-aware naming
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"experiments/experiment_{timestamp}_seed{self.seed_manager.master_seed}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment state
        self.state_file = self.output_dir / "experiment_state.json"
        self.results: List[ExperimentResult] = []
        self.current_stage: Optional[ExperimentStage] = None
        self.experiment_start_time: Optional[datetime] = None
        
        # Load existing state if resuming
        self._load_state()
        
        if self.verbose:
            print(f"🧪 Experiment Runner initialized")
            print(f"📂 Output directory: {self.output_dir}")
            print(f"🎲 Seed: {self.seed_manager.master_seed}")
    
    def _load_config(self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate configuration."""
        try:
            if isinstance(config, dict):
                # Configuration already loaded
                from ..config.schema import BirdsongConfig
                config_obj = BirdsongConfig.model_validate(config)
                return config_obj.model_dump()
            elif isinstance(config, str):
                if config.endswith('.yaml') or config.endswith('.yml') or '/' in config:
                    # File path
                    config_obj = load_config(config)
                else:
                    # Template name
                    config_obj = load_template(config)
                
                # Validate configuration
                warnings = validate_config(config_obj, strict=False)
                if warnings and self.verbose:
                    print(f"⚠️ Configuration warnings: {len(warnings)} warnings found")
                    for warning in warnings[:3]:
                        print(f"  Warning: {warning}")
                    if len(warnings) > 3:
                        print(f"  ... and {len(warnings) - 3} more warnings")
                
                return config_obj.model_dump()
            else:
                raise ValueError(f"Invalid config type: {type(config)}")
                
        except ConfigValidationError as e:
            raise RuntimeError(f"Configuration validation error: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def _load_state(self) -> None:
        """Load experiment state from disk if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Restore results
                self.results = []
                for result_data in state.get('results', []):
                    result = ExperimentResult(
                        stage=ExperimentStage(result_data['stage']),
                        success=result_data['success'],
                        start_time=datetime.fromisoformat(result_data['start_time']),
                        end_time=datetime.fromisoformat(result_data['end_time']) if result_data.get('end_time') else None,
                        outputs=result_data.get('outputs', {}),
                        metrics=result_data.get('metrics', {}),
                        error=result_data.get('error')
                    )
                    self.results.append(result)
                
                # Restore other state
                self.current_stage = ExperimentStage(state['current_stage']) if state.get('current_stage') else None
                self.experiment_start_time = datetime.fromisoformat(state['experiment_start_time']) if state.get('experiment_start_time') else None
                
                if self.verbose:
                    print(f"📂 Resuming experiment from {self.current_stage.display_name if self.current_stage else 'beginning'}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Could not load experiment state: {e}")
                # Continue with fresh state
    
    def _save_state(self) -> None:
        """Save current experiment state to disk."""
        state = {
            'current_stage': self.current_stage.value if self.current_stage else None,
            'experiment_start_time': self.experiment_start_time.isoformat() if self.experiment_start_time else None,
            'results': []
        }
        
        for result in self.results:
            result_data = {
                'stage': result.stage.value,
                'success': result.success,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'outputs': result.outputs,
                'metrics': result.metrics,
                'error': result.error
            }
            state['results'].append(result_data)
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _get_completed_stages(self) -> List[ExperimentStage]:
        """Get list of successfully completed stages."""
        return [result.stage for result in self.results if result.success]
    
    def _stage_completed(self, stage: ExperimentStage) -> bool:
        """Check if a stage has been completed successfully."""
        return stage in self._get_completed_stages()
    
    def _get_stage_output(self, stage: ExperimentStage, key: str) -> Optional[Any]:
        """Get output from a completed stage."""
        for result in self.results:
            if result.stage == stage and result.success:
                return result.outputs.get(key)
        return None
    
    def _start_stage(self, stage: ExperimentStage) -> ExperimentResult:
        """Start a new stage and return its result object."""
        if self.experiment_start_time is None:
            self.experiment_start_time = datetime.now()
        
        self.current_stage = stage
        result = ExperimentResult(
            stage=stage,
            success=False,
            start_time=datetime.now()
        )
        
        if self.verbose:
            print(f"\n🚀 Starting {stage.display_name}...")
            print(f"⏰ Started at: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return result
    
    def _finish_stage(self, result: ExperimentResult, success: bool = True, 
                     error: Optional[str] = None) -> None:
        """Finish a stage and update its result."""
        result.end_time = datetime.now()
        result.success = success
        result.error = error
        
        # Remove any previous result for this stage
        self.results = [r for r in self.results if r.stage != result.stage]
        self.results.append(result)
        
        self._save_state()
        
        if self.verbose:
            status = "✅ Completed" if success else "❌ Failed"
            print(f"{status} {result.stage.display_name} in {result.duration_str}")
            if error:
                print(f"💥 Error: {error}")
    
    def run_data_generation(self) -> bool:
        """Run the data generation stage."""
        if self._stage_completed(ExperimentStage.DATA_GENERATION):
            if self.verbose:
                print(f"⏭️ Data generation already completed, skipping...")
            return True
        
        # Check if data generation should be skipped
        if not self.config.get('experiment', {}).get('run_data_generation', True):
            if self.verbose:
                print(f"⏭️ Data generation disabled in configuration, skipping...")
            return True
        
        result = self._start_stage(ExperimentStage.DATA_GENERATION)
        
        try:
            data_config = self.config['data']
            
            # Check if we have an existing data path
            if data_config.get('data_path'):
                if self.verbose:
                    print(f"📁 Using existing dataset: {data_config['data_path']}")
                
                result.outputs = {
                    'data_path': data_config['data_path'],
                    'dataset_size': len(BirdsongDataset(data_config['data_path'])),
                    'data_filename': os.path.basename(data_config['data_path'])
                }
                
                self._finish_stage(result, success=True)
                return True
            
            # Create simulator with alphabet list
            alphabet = data_config.get('alphabet', ['<', 'a', 'b', 'c', 'd', 'e', '>'])
            simulator = BirdsongSimulator(
                alphabet=alphabet,
                order=data_config['order']
            )
            
            # Create aggregator with alphabet list
            aggregator = BirdsongAggregator(
                alphabet=alphabet,
                order=data_config['order']
            )
            
            # Generate data filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_filename = f"birdsong_data_seed{self.seed_manager.master_seed}_{timestamp}.h5"
            data_path = self.output_dir / data_filename
            
            # Run data generation
            if self.verbose:
                print(f"📊 Generating {data_config['num_batches']} batches of {data_config['batch_size']} songs each...")
            
            aggregator.create_dataset(
                process_configs=data_config['process_configs'],
                num_batches=data_config['num_batches'],
                batch_size=data_config['batch_size'],
                seq_range=(data_config['seq_min_length'], data_config['seq_max_length']),
                output_path=str(data_path)
            )
            
            # Validate generated data
            dataset = BirdsongDataset(str(data_path))
            
            result.outputs = {
                'data_path': str(data_path),
                'dataset_size': len(dataset),
                'data_filename': data_filename
            }
            
            result.metrics = {
                'num_samples': len(dataset),
                'file_size_mb': data_path.stat().st_size / (1024 * 1024)
            }
            
            if self.verbose:
                print(f"📁 Data saved to: {data_path}")
                print(f"📊 Dataset size: {len(dataset)} samples")
                print(f"💾 File size: {result.metrics['file_size_mb']:.2f} MB")
            
            self._finish_stage(result, success=True)
            return True
            
        except Exception as e:
            error_msg = f"Data generation failed: {str(e)}"
            self._finish_stage(result, success=False, error=error_msg)
            return False
    
    def run_training(self) -> bool:
        """Run the training stage."""
        if self._stage_completed(ExperimentStage.TRAINING):
            if self.verbose:
                print(f"⏭️ Training already completed, skipping...")
            return True
        
        # Check dependency
        if not self._stage_completed(ExperimentStage.DATA_GENERATION):
            if self.verbose:
                print(f"❌ Cannot run training: data generation not completed")
            return False
        
        result = self._start_stage(ExperimentStage.TRAINING)
        
        try:
            # Get data path from previous stage
            data_path = self._get_stage_output(ExperimentStage.DATA_GENERATION, 'data_path')
            if not data_path:
                raise RuntimeError("Data path not found from data generation stage")
            
            # Load dataset
            dataset = BirdsongDataset(data_path)
            
            # Create model
            model_config = self.config['model']
            model = BirdsongLFADSModel2(
                alphabet_size=model_config['alphabet_size'],
                order=model_config['order'],
                encoder_dim=model_config['encoder_dim'],
                controller_dim=model_config['controller_dim'],
                generator_dim=model_config['generator_dim'],
                factor_dim=model_config['factor_dim'],
                latent_dim=model_config['latent_dim'],
                inferred_input_dim=model_config['inferred_input_dim'],
                kappa=model_config['kappa'],
                ar_step_size=model_config['ar_step_size'],
                ar_process_var=model_config['ar_process_var']
            )
            
            # Setup training configuration
            training_config = self.config['training'].copy()
            checkpoint_path = self.output_dir / f"checkpoint_seed{self.seed_manager.master_seed}.pt"
            training_config['checkpoint_path'] = str(checkpoint_path)
            training_config['plot_dir'] = str(self.output_dir / "training_plots")
            
            # Create trainer
            trainer = BirdsongTrainer(model, dataset, training_config)
            
            if self.verbose:
                print(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters")
                print(f"🏃 Training for {training_config['epochs']} epochs...")
            
            # Train model
            trainer.train()
            
            # Generate training plots
            plot_path = self.output_dir / "training_history.png"
            trainer.plot_training_history(str(plot_path))
            
            result.outputs = {
                'checkpoint_path': str(checkpoint_path),
                'training_plots_dir': str(self.output_dir / "training_plots"),
                'training_history_plot': str(plot_path)
            }
            
            result.metrics = {
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'training_epochs': training_config['epochs'],
                'final_loss': float(trainer.train_losses[-1]) if trainer.train_losses else 0.0
            }
            
            if self.verbose:
                print(f"💾 Checkpoint saved to: {checkpoint_path}")
                print(f"📈 Training plots saved to: {self.output_dir / 'training_plots'}")
                print(f"📊 Final loss: {result.metrics['final_loss']:.6f}")
            
            self._finish_stage(result, success=True)
            return True
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self._finish_stage(result, success=False, error=error_msg)
            return False
    
    def run_evaluation(self) -> bool:
        """Run the evaluation stage."""
        if self._stage_completed(ExperimentStage.EVALUATION):
            if self.verbose:
                print(f"⏭️ Evaluation already completed, skipping...")
            return True
        
        # Check dependencies
        if not self._stage_completed(ExperimentStage.DATA_GENERATION):
            if self.verbose:
                print(f"❌ Cannot run evaluation: data generation not completed")
            return False
        
        if not self._stage_completed(ExperimentStage.TRAINING):
            if self.verbose:
                print(f"❌ Cannot run evaluation: training not completed")
            return False
        
        result = self._start_stage(ExperimentStage.EVALUATION)
        
        try:
            # Get paths from previous stages
            data_path = self._get_stage_output(ExperimentStage.DATA_GENERATION, 'data_path')
            checkpoint_path = self._get_stage_output(ExperimentStage.TRAINING, 'checkpoint_path')
            
            if not data_path or not checkpoint_path:
                raise RuntimeError("Required paths not found from previous stages")
            
            # Simplified evaluation (would use full evaluation CLI in practice)
            import torch
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Load dataset
            dataset = BirdsongDataset(data_path)
            
            # Create model (simplified - would use proper checkpoint loading)
            model_config = self.config['model']
            model = BirdsongLFADSModel2(
                alphabet_size=model_config['alphabet_size'],
                order=model_config['order'],
                encoder_dim=model_config['encoder_dim'],
                controller_dim=model_config['controller_dim'],
                generator_dim=model_config['generator_dim'],
                factor_dim=model_config['factor_dim'],
                latent_dim=model_config['latent_dim'],
                inferred_input_dim=model_config['inferred_input_dim'],
                kappa=model_config['kappa'],
                ar_step_size=model_config['ar_step_size'],
                ar_process_var=model_config['ar_process_var']
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to(device)
            
            # Evaluate on a few samples
            eval_config = self.config.get('evaluation', {})
            num_samples = eval_config.get('num_samples', 10)
            
            total_loss = 0.0
            with torch.no_grad():
                for i in range(min(num_samples, len(dataset))):
                    bigram_counts, probabilities = dataset[i]
                    bigram_counts = bigram_counts.unsqueeze(0).to(device)
                    probabilities = probabilities.unsqueeze(0).to(device)
                    
                    outputs = model(bigram_counts)
                    loss, _ = model.compute_loss(probabilities, outputs)
                    total_loss += loss.item()
            
            avg_loss = total_loss / min(num_samples, len(dataset))
            
            result.outputs = {
                'evaluation_results': str(self.output_dir / "evaluation_results.json"),
                'num_samples_evaluated': min(num_samples, len(dataset))
            }
            
            result.metrics = {
                'average_loss': avg_loss,
                'num_samples': min(num_samples, len(dataset))
            }
            
            # Save evaluation results
            eval_results = {
                'average_loss': avg_loss,
                'num_samples': min(num_samples, len(dataset)),
                'device': str(device),
                'evaluation_time': datetime.now().isoformat()
            }
            
            with open(self.output_dir / "evaluation_results.json", 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            if self.verbose:
                print(f"📊 Evaluated {num_samples} samples")
                print(f"📈 Average loss: {avg_loss:.6f}")
                print(f"💾 Results saved to: {self.output_dir / 'evaluation_results.json'}")
            
            self._finish_stage(result, success=True)
            return True
            
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            self._finish_stage(result, success=False, error=error_msg)
            return False
    
    def run_full_experiment(self) -> bool:
        """Run the complete experiment pipeline."""
        if self.verbose:
            print(f"🧪 Starting full experiment pipeline")
            print(f"📊 Configuration: {self.config.get('experiment', {}).get('name', 'Unnamed')}")
        
        # Run all stages in order
        stages = [
            (ExperimentStage.DATA_GENERATION, self.run_data_generation),
            (ExperimentStage.TRAINING, self.run_training),
            (ExperimentStage.EVALUATION, self.run_evaluation)
        ]
        
        for stage, runner_func in stages:
            if not runner_func():
                if self.verbose:
                    print(f"❌ Experiment failed at {stage.display_name}")
                return False
        
        # Mark as completed
        self.current_stage = ExperimentStage.COMPLETED
        self._save_state()
        
        # Generate experiment summary
        self._generate_experiment_summary()
        
        if self.verbose:
            print(f"\n🎉 Experiment completed successfully!")
            total_time = datetime.now() - self.experiment_start_time
            print(f"⏰ Total time: {total_time}")
            print(f"📂 Results saved to: {self.output_dir}")
        
        return True
    
    def _generate_experiment_summary(self) -> None:
        """Generate a comprehensive experiment summary."""
        summary = {
            'experiment_id': str(self.output_dir.name),
            'seed': self.seed_manager.master_seed,
            'start_time': self.experiment_start_time.isoformat() if self.experiment_start_time else None,
            'completion_time': datetime.now().isoformat(),
            'total_duration': str(datetime.now() - self.experiment_start_time) if self.experiment_start_time else None,
            'configuration': self.config,
            'stages': []
        }
        
        for result in self.results:
            stage_summary = {
                'stage': result.stage.value,
                'success': result.success,
                'duration': result.duration_str,
                'outputs': result.outputs,
                'metrics': result.metrics
            }
            if result.error:
                stage_summary['error'] = result.error
            
            summary['stages'].append(stage_summary)
        
        # Save detailed summary
        with open(self.output_dir / "experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save human-readable summary
        with open(self.output_dir / "experiment_summary.txt", 'w') as f:
            f.write("Birdsong Experiment Summary\n")
            f.write("===========================\n\n")
            f.write(f"Experiment ID: {summary['experiment_id']}\n")
            f.write(f"Seed: {summary['seed']}\n")
            f.write(f"Completion Time: {summary['completion_time']}\n")
            f.write(f"Total Duration: {summary['total_duration']}\n\n")
            
            f.write("Stages:\n")
            f.write("-------\n")
            for stage in summary['stages']:
                status = "✅" if stage['success'] else "❌"
                f.write(f"{status} {stage['stage'].replace('_', ' ').title()}: {stage['duration']}\n")
                
                if stage.get('metrics'):
                    for key, value in stage['metrics'].items():
                        if isinstance(value, float):
                            f.write(f"   {key}: {value:.6f}\n")
                        else:
                            f.write(f"   {key}: {value}\n")
                
                if stage.get('error'):
                    f.write(f"   Error: {stage['error']}\n")
                f.write("\n")


def run_experiment(config: Union[str, Dict[str, Any]], output_dir: Optional[str] = None,
                  seed: Optional[int] = None, verbose: bool = True) -> bool:
    """
    Convenience function to run a complete experiment.
    
    Args:
        config: Configuration file path, template name, or config dictionary
        output_dir: Output directory for experiment results
        seed: Random seed for reproducibility
        verbose: Whether to print progress information
        
    Returns:
        True if experiment completed successfully, False otherwise
    """
    runner = ExperimentRunner(config, output_dir, seed, verbose)
    return runner.run_full_experiment()


def resume_experiment(experiment_dir: str, verbose: bool = True) -> bool:
    """
    Resume an interrupted experiment.
    
    Args:
        experiment_dir: Directory containing the experiment state
        verbose: Whether to print progress information
        
    Returns:
        True if experiment completed successfully, False otherwise
    """
    exp_path = Path(experiment_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    state_file = exp_path / "experiment_state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"Experiment state file not found: {state_file}")
    
    # Load original config from state or summary
    config_file = None
    if (exp_path / "experiment_summary.json").exists():
        with open(exp_path / "experiment_summary.json", 'r') as f:
            summary = json.load(f)
            config = summary.get('configuration')
            if config:
                runner = ExperimentRunner(config, str(exp_path), verbose=verbose)
                return runner.run_full_experiment()
    
    raise RuntimeError(f"Could not resume experiment: configuration not found in {experiment_dir}") 