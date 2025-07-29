"""
Data loading utilities for birdsong analysis.

This module contains the BirdsongDataset class for loading HDF5 data
with bigram counts and probabilities.
"""

import os

import h5py
import torch
from torch.utils.data import Dataset


def generate_alphabet_symbols(alphabet_size: int) -> list[str]:
    """
    Generate alphabet symbols for the given size.
    
    Args:
        alphabet_size: Number of symbols needed
        
    Returns:
        List of alphabet symbols including start/end tokens
    """
    if alphabet_size < 3:
        raise ValueError("Alphabet size must be at least 3 (start, end, and at least one symbol)")
    
    # Start and end tokens
    symbols = ["<", ">"]
    
    # Add middle symbols (a, b, c, ...)
    num_middle_symbols = alphabet_size - 2
    for i in range(num_middle_symbols):
        symbols.insert(1 + i, chr(ord('a') + i))
    
    return symbols


class BirdsongDataset(Dataset):
    """
    PyTorch Dataset for Birdsong bigram counts and probabilities.

    Loads data from HDF5 files containing bigram counts and probability
    distributions for birdsong analysis.
    """

    def __init__(self, h5_path: str):
        """
        Initialize the dataset.

        Args:
            h5_path: Path to the HDF5 file containing bigram counts and probabilities.

        Raises:
            FileNotFoundError: If the HDF5 file doesn't exist.
            KeyError: If required datasets are missing from the HDF5 file.
            ValueError: If the data has invalid shape or contains NaN values.
        """
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        self.h5_path = h5_path
        self.h5_file = None  # Will be opened lazily per worker

        # Validate the HDF5 file structure and get metadata only
        with h5py.File(h5_path, 'r') as hf:
            if 'bigram_counts' not in hf:
                raise KeyError(f"Dataset 'bigram_counts' not found in {h5_path}")
            if 'probabilities' not in hf:
                raise KeyError(f"Dataset 'probabilities' not found in {h5_path}")

            # Get dataset info without loading full data
            self.num_samples = hf['bigram_counts'].shape[2]
            self.time_steps = hf['bigram_counts'].shape[1]
            self.feature_dim = hf['bigram_counts'].shape[0]

            # Quick validation of a small sample instead of full dataset
            sample_idx = min(10, self.num_samples - 1)  # Check first 10 samples
            bigram_sample = hf['bigram_counts'][:, :, :sample_idx]
            prob_sample = hf['probabilities'][:, :, :sample_idx]

            if torch.isnan(torch.tensor(bigram_sample)).any():
                raise ValueError("bigram_counts contains NaN values")
            if torch.isnan(torch.tensor(prob_sample)).any():
                raise ValueError("probabilities contains NaN values")

            # Check that shapes match
            if bigram_sample.shape != prob_sample.shape:
                raise ValueError(
                    f"Shape mismatch: bigram_counts {bigram_sample.shape} != "
                    f"probabilities {prob_sample.shape}"
                )

            # Validate probability distributions on sample
            prob_sums = prob_sample.sum(axis=0)  # Sum over feature dimension
            if not (torch.tensor(prob_sums) > 0.99).all():
                raise ValueError("Some probability distributions do not sum to ~1.0")

        self.input_size = self.feature_dim
        self.time_steps = self.time_steps
        self.num_processes = self.num_samples

    def _init_h5(self) -> None:
        """Initialize HDF5 file connection (lazy loading)."""
        if self.h5_file is None:
            try:
                self.h5_file = h5py.File(self.h5_path, 'r')
                self.bigram_counts = self.h5_file['bigram_counts']
                self.probabilities = self.h5_file['probabilities']
            except Exception as e:
                raise RuntimeError(f"Failed to open HDF5 file {self.h5_path}: {e}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the process to fetch.

        Returns:
            Tuple of (bigram_counts, probabilities) for the specified process.
            Shapes: (time_steps, feature_dim) for both tensors.

        Raises:
            IndexError: If idx is out of bounds.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_samples} samples")

        self._init_h5()

        try:
            # Load data directly as numpy arrays and convert to tensors efficiently
            bigram_counts = self.bigram_counts[:, :, idx].T  # (T, F)
            probabilities = self.probabilities[:, :, idx].T

            # Convert to tensors with proper dtype - more efficient conversion
            bigram_counts_tensor = torch.from_numpy(bigram_counts).float()
            probabilities_tensor = torch.from_numpy(probabilities).float()

            # Quick validation
            if torch.isnan(bigram_counts_tensor).any():
                raise ValueError(f"NaN values in bigram_counts for sample {idx}")
            if torch.isnan(probabilities_tensor).any():
                raise ValueError(f"NaN values in probabilities for sample {idx}")

            return bigram_counts_tensor, probabilities_tensor

        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx}: {e}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def get_dataset_info(self) -> dict:
        """
        Get information about the dataset.

        Returns:
            Dictionary containing dataset metadata.
        """
        return {
            "num_samples": self.num_samples,
            "time_steps": self.time_steps,
            "feature_dim": self.feature_dim,
            "file_path": self.h5_path
        }

    def __del__(self):
        """Clean up HDF5 file connection."""
        if self.h5_file is not None:
            self.h5_file.close()
