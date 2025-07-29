"""
Data generation and loading module for birdsong package.
"""

from .aggregation import PROCESS_FUNCTIONS, BirdsongAggregator, aggregate_data
from .generation import BirdsongSimulator, save_to_hdf5, simulate_birdsong
from .loader import BirdsongDataset

__all__ = [
    "BirdsongDataset",
    "BirdsongSimulator",
    "simulate_birdsong",
    "save_to_hdf5",
    "BirdsongAggregator",
    "aggregate_data",
    "PROCESS_FUNCTIONS",
]
