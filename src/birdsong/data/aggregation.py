"""
Data aggregation module for birdsong simulation.

This module provides process functions and aggregation utilities for multi-process
birdsong data generation with various transition dynamics.
"""

import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

import h5py
import numpy as np

from .generation import simulate_birdsong


# Process Functions
def linear_process(x: np.ndarray, a: np.ndarray, t: int) -> np.ndarray:
    """Linear update to the transition matrix."""
    return x + a


def linear_with_noise(x: np.ndarray, a: np.ndarray, t: int) -> np.ndarray:
    """Linear update with added Gaussian noise."""
    return x + a + np.random.normal(0, 0.1, size=x.shape)


def nonlinear_cosine(x: np.ndarray, a: np.ndarray, t: int) -> np.ndarray:
    """Nonlinear update using a cosine function."""
    return x + a * np.cos(0.1 * t)


def quadratic_decay(x: np.ndarray, a: np.ndarray, t: int) -> np.ndarray:
    """Nonlinear update with quadratic scaling."""
    return x + a / (1 + (t / 10)**2)


def exponential_growth(x: np.ndarray, a: np.ndarray, t: int) -> np.ndarray:
    """Exponential growth: increments by a factor that decays over time."""
    growth_rate = 0.01
    return x + a * np.exp(-growth_rate * t)


def multiplicative_noise(x: np.ndarray, a: np.ndarray, t: int) -> np.ndarray:
    """Noise amplitude depends on the magnitude of x."""
    noise_scale = 0.01 * (np.abs(x) + 1)
    return x + np.random.normal(0, noise_scale, size=x.shape)


def piecewise_linear(x: np.ndarray, a: np.ndarray, t: int) -> np.ndarray:
    """Different linear increments depending on time interval."""
    if t < 10:
        return x + a  # Early stage: simple linear growth
    elif t < 40:
        return x - 0.5 * a  # Mid stage: decreasing trend
    else:
        return x + 0.5 * a  # Late stage: smaller positive increments


def logistic_growth(x: np.ndarray, a: np.ndarray, t: int) -> np.ndarray:
    """Logistic growth toward a carrying capacity."""
    k = 10.0
    return x + a * x * (1 - x / k)


def fourier_update(x: np.ndarray, a: np.ndarray, t: int) -> np.ndarray:
    """
    Nonlinear update using a Fourier series with sine and cosine terms.

    The Fourier coefficients are generated once for each process (detected when t==0)
    and then reused for all subsequent time steps for that process.
    """
    fourier_order = 5
    use_a = False  # Set to True to evaluate at A instead of x.

    # Initialize storage attributes on the function if they don't exist.
    if not hasattr(fourier_update, "coeff_store"):
        fourier_update.coeff_store = {}  # Dictionary to store coefficients per process.
        fourier_update.process_counter = 0  # Counter for the number of processes seen.
        fourier_update.current_process_key = None  # Key for the current process.

    # When t==0, we assume this is the start of a new process.
    if t == 0:
        key = fourier_update.process_counter
        n = x.shape[0]
        cos_coeffs = np.random.uniform(-0.05, 0.05, (n, fourier_order + 1))
        sin_coeffs = np.random.uniform(-0.05, 0.05, (n, fourier_order))
        fourier_update.coeff_store[key] = (cos_coeffs, sin_coeffs)
        fourier_update.current_process_key = key
        fourier_update.process_counter += 1
    else:
        key = fourier_update.current_process_key

    cos_coeffs, sin_coeffs = fourier_update.coeff_store[key]
    new_x = np.empty_like(x)
    for i in range(len(x)):
        theta = a[i] if use_a else x[i]
        update = cos_coeffs[i, 0]  # constant term (order 0)
        for j in range(1, fourier_order + 1):
            update += (cos_coeffs[i, j] * np.cos(j * theta) +
                       sin_coeffs[i, j - 1] * np.sin(j * theta))
        new_x[i] = x[i] + update
    return new_x


def sparse_transition_process(
    x: np.ndarray,
    a: np.ndarray,
    t: int,
    sparsity: float = 0.9,
    noise_scale: float = 0.05
) -> np.ndarray:
    """
    Enforces sparsity in transition matrix logits, automatically supporting order 1 and 2 Markov chains.

    Automatically infers the Markov order and alphabet size from the shape of `x`.

    Args:
        x: Current transition logits (flattened).
        a: Update direction matrix (same shape as x).
        t: Current time step.
        sparsity: Fraction of transitions to zero out.
        noise_scale: Standard deviation of noise to add to kept logits.

    Returns:
        Flattened sparse logits.
    """
    updated = x + a
    large_negative = -1e8
    total_size = len(updated)

    # Try to infer alphabet size and order
    # Case 1: Order 1 → shape is (R, R)
    # Case 2: Order 2 → shape is (R^2, R) → total size = R^3
    cube_root = round(total_size ** (1/3))
    square_root = round(total_size ** 0.5)

    if cube_root ** 3 == total_size:
        alphabet_size = cube_root
        num_contexts = alphabet_size ** 2
        num_choices = alphabet_size
    elif square_root ** 2 == total_size:
        alphabet_size = square_root
        num_contexts = alphabet_size
        num_choices = alphabet_size
    else:
        raise ValueError("Unable to infer Markov order from input size. Expected perfect square or cube.")

    # Reshape logits to (contexts, choices)
    updated_2d = updated.reshape((num_contexts, num_choices))
    sparse_2d = np.full_like(updated_2d, fill_value=large_negative)

    for i in range(num_contexts):
        row = updated_2d[i]
        k = max(1, int((1 - sparsity) * num_choices))
        top_k = np.argpartition(-np.abs(row), k)[:k]
        sparse_2d[i, top_k] = row[top_k] + np.random.normal(0, noise_scale, size=k)

        # Ensure at least one entry remains active
        if np.all(sparse_2d[i] == large_negative):
            best = np.argmax(np.abs(row))
            sparse_2d[i, best] = row[best]

    return sparse_2d.flatten()


# Process function registry
PROCESS_FUNCTIONS = {
    "linear": linear_process,
    "linear_with_noise": linear_with_noise,
    "nonlinear_cosine": nonlinear_cosine,
    "quadratic_decay": quadratic_decay,
    "exponential_growth": exponential_growth,
    "multiplicative_noise": multiplicative_noise,
    "piecewise_linear": piecewise_linear,
    "logistic_growth": logistic_growth,
    "fourier": fourier_update,
    "sparse_90": lambda x, a, t: sparse_transition_process(x, a, t, sparsity=0.9),
    "sparse_99": lambda x, a, t: sparse_transition_process(x, a, t, sparsity=0.99),
}


def aggregate_data(
    process_configs: list[tuple[str, int]],
    num_batches: int,
    batch_size: int,
    seq_range: tuple[int, int],
    alphabet: list[str],
    order: int = 1,
    output_path: str = None
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Aggregate data across multiple process types.

    Args:
        process_configs: List of (process_name, num_processes) tuples
        num_batches: Number of time steps per process
        batch_size: Number of sequences per batch
        seq_range: Minimum and maximum sequence lengths
        alphabet: List of symbols in the alphabet
        order: Markov order (1 for bigram, 2 for trigram)
        output_path: Optional path to save results

    Returns:
        Tuple of (bigram_counts, probabilities, metadata)
    """
    # Initialize storage for aggregated data
    bigram_counts_list = []
    probabilities_list = []
    metadata = []

    # Generate data for each process type
    for process_name, num_processes in process_configs:
        if process_name not in PROCESS_FUNCTIONS:
            raise ValueError(f"Unknown process type: {process_name}")

        print(f"Generating data for process: {process_name} with {num_processes} processes...")

        process_fn = PROCESS_FUNCTIONS[process_name]
        bigram_counts, probabilities = simulate_birdsong(
            num_batches, batch_size, seq_range, alphabet, num_processes, order, process_fn
        )

        bigram_counts_list.append(bigram_counts)
        probabilities_list.append(probabilities)
        metadata.append({
            "process": process_name,
            "num_processes": num_processes,
            "order": order
        })

    # Concatenate all data
    bigram_counts_aggregated = np.concatenate(bigram_counts_list, axis=2)
    probabilities_aggregated = np.concatenate(probabilities_list, axis=2)

    # Save to file if path provided
    if output_path:
        save_aggregated_data(output_path, bigram_counts_aggregated, probabilities_aggregated, metadata)

    return bigram_counts_aggregated, probabilities_aggregated, metadata


def save_aggregated_data(
    output_path: str,
    bigram_counts: np.ndarray,
    probabilities: np.ndarray,
    metadata: list[dict[str, Any]]
) -> None:
    """
    Save aggregated data to HDF5 file.

    Args:
        output_path: Path to output HDF5 file
        bigram_counts: N-gram count tensor
        probabilities: Probability distribution tensor
        metadata: List of metadata dictionaries
    """
    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("bigram_counts", data=bigram_counts)
        hf.create_dataset("probabilities", data=probabilities)
        hf.attrs["metadata"] = str(metadata)

    print(f"Saved aggregated data to {output_path}")
    print(f"Data shape: {bigram_counts.shape}")


class BirdsongAggregator:
    """
    High-level interface for birdsong data aggregation.
    """

    def __init__(self, alphabet: list[str], order: int = 1):
        """
        Initialize aggregator.

        Args:
            alphabet: List of symbols in the alphabet
            order: Markov order (1 for bigram, 2 for trigram)
        """
        self.alphabet = alphabet
        self.order = order
        self.alphabet_size = len(alphabet)

        # Validate alphabet
        if '<' not in alphabet or '>' not in alphabet:
            raise ValueError("Alphabet must contain '<' and '>' symbols")

    def create_dataset(
        self,
        process_configs: list[tuple[str, int]],
        num_batches: int,
        batch_size: int,
        seq_range: tuple[int, int],
        output_path: str = None
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """
        Create a complete aggregated dataset.

        Args:
            process_configs: List of (process_name, num_processes) tuples
            num_batches: Number of time steps per process
            batch_size: Number of sequences per batch
            seq_range: Minimum and maximum sequence lengths
            output_path: Optional path to save results

        Returns:
            Tuple of (bigram_counts, probabilities, metadata)
        """
        if output_path is None:
            # Generate default output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"aggregated_birdsong_data_{timestamp}.h5"

        return aggregate_data(
            process_configs=process_configs,
            num_batches=num_batches,
            batch_size=batch_size,
            seq_range=seq_range,
            alphabet=self.alphabet,
            order=self.order,
            output_path=output_path
        )

    def get_process_function(self, process_name: str) -> Callable:
        """
        Get a process function by name.

        Args:
            process_name: Name of the process function

        Returns:
            Process function
        """
        if process_name not in PROCESS_FUNCTIONS:
            raise ValueError(f"Unknown process type: {process_name}")
        return PROCESS_FUNCTIONS[process_name]

    def list_process_types(self) -> list[str]:
        """
        List available process types.

        Returns:
            List of available process function names
        """
        return list(PROCESS_FUNCTIONS.keys())
