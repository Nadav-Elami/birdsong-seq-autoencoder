"""
Experiment orchestration and pipeline management for birdsong research.

This module provides comprehensive experiment orchestration capabilities including:
- End-to-end pipeline execution (data generation → training → evaluation)
- Resume functionality for interrupted experiments
- Progress tracking and time estimation
- Modular component execution with dependency management
- Comprehensive result archiving and metadata tracking
"""

from .runner import (
    ExperimentRunner,
    ExperimentStage,
    ExperimentResult,
    run_experiment,
    resume_experiment,
)

__all__ = [
    "ExperimentRunner",
    "ExperimentStage", 
    "ExperimentResult",
    "run_experiment",
    "resume_experiment",
] 