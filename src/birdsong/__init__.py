"""
Birdsong Sequential Autoencoder Package.

A research-grade Python package for LFADS-style birdsong analysis.
"""

__version__ = "0.1.0"

# Import main components for easy access
from .data.loader import BirdsongDataset
from .evaluation.evaluate import BirdsongEvaluator
from .models.lfads import BirdsongLFADSModel2
from .training.trainer import BirdsongTrainer

__all__ = [
    "BirdsongLFADSModel2",
    "BirdsongDataset",
    "BirdsongTrainer",
    "BirdsongEvaluator",
]
