"""
Data generation module for birdsong simulation.

This module provides functions for simulating birdsong sequences using Markov processes
and saving the results to HDF5 format for training.
"""

from collections.abc import Callable

import h5py
import numpy as np


def masked_softmax(x: np.ndarray, mask_value: float = -1e8) -> np.ndarray:
    """
    Compute softmax with masking support.

    Args:
        x: Input array
        mask_value: Value to treat as masked

    Returns:
        Softmax-normalized probabilities, with zeros for fully masked inputs
    """
    if np.all(x == mask_value):
        return np.zeros_like(x)
    else:
        x_stable = x - np.max(x)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x)


def x_init_maker(alphabet: list[str], order: int = 1) -> np.ndarray:
    """
    Create initial transition matrix for Markov process.

    Args:
        alphabet: List of symbols in the alphabet
        order: Markov order (1 for bigram, 2 for trigram)

    Returns:
        Initial transition logits matrix
    """
    alphabet_size = len(alphabet)
    large_negative = -1e8
    large_positive = 1e8

    # Get indices for special symbols
    idx_start = alphabet.index('<')
    idx_end = alphabet.index('>')
    idx_other = [i for i in range(alphabet_size) if i not in [idx_start, idx_end]]

    if order == 1:
        # Create bigram transition matrix
        x = np.zeros((alphabet_size, alphabet_size))

        # Set impossible transitions to large negative values
        x[idx_end, :] = large_negative
        x[idx_end, idx_end] = large_positive  # Allow self-transition for end

        # Start symbol can only transition to other symbols (not start/end)
        x[idx_start, idx_start] = large_negative
        x[idx_start, idx_end] = large_negative

        # Other symbols can transition to anything except start
        for i in idx_other:
            x[i, idx_start] = large_negative

    elif order == 2:
        # Create trigram transition matrix
        x = np.zeros((alphabet_size ** 2, alphabet_size))

        # Define legal state transitions
        def is_legal_pair(x: int, y: int) -> bool:
            """Check if transition from x to y is legal."""
            if x == idx_end:
                return y == idx_end  # End can only self-transition
            elif x == idx_start:
                return y != idx_start and y != idx_end  # Start can't transition to start/end
            else:
                return y != idx_start  # Other symbols can't transition to start

        # Set impossible transitions
        for i in range(alphabet_size ** 2):
            for j in range(alphabet_size):
                if not is_legal_pair(i // alphabet_size, j):
                    x[i, j] = large_negative

    else:
        raise ValueError(f"Unsupported order: {order}. Use order=1 or order=2.")

    return x


def simulate_one_song_order1(
    seq_range: tuple[int, int],
    probs_matrix: np.ndarray,
    alphabet: list[str]
) -> list[str]:
    """
    Simulate a single first-order Markov song sequence.

    Args:
        seq_range: Minimum and maximum sequence lengths
        probs_matrix: Transition probability matrix
        alphabet: List of symbols

    Returns:
        Simulated sequence as list of symbols
    """
    sequence = ['<']
    while len(sequence) < seq_range[1] + 1:
        current_state = alphabet.index(sequence[-1])
        probs = probs_matrix[current_state]
        next_phrase_idx = np.random.choice(len(alphabet), p=probs)
        next_phrase = alphabet[next_phrase_idx]
        sequence.append(next_phrase)

        if next_phrase == '>':
            break

    return sequence


def simulate_one_song_order2(
    seq_range: tuple[int, int],
    probs_matrix: np.ndarray,
    alphabet: list[str],
    idx_start: int,
    idx_end: int
) -> list[str]:
    """
    Simulate a single second-order Markov song sequence.

    Args:
        seq_range: Minimum and maximum sequence lengths
        probs_matrix: Transition probability matrix
        alphabet: List of symbols
        idx_start: Index of start symbol
        idx_end: Index of end symbol

    Returns:
        Simulated sequence as list of symbols
    """
    # Initialize with legal start state
    state = (idx_end, idx_start)  # (>, <) is the only legal restart state
    sequence = [alphabet[idx_start]]

    while len(sequence) < seq_range[1] + 1:
        # Get transition probabilities for current state
        state_idx = state[0] * len(alphabet) + state[1]
        probs = probs_matrix[state_idx]

        # Sample next symbol
        next_phrase_idx = np.random.choice(len(alphabet), p=probs)
        next_phrase = alphabet[next_phrase_idx]
        sequence.append(next_phrase)

        # Update state
        state = (state[1], next_phrase_idx)

        if next_phrase == '>':
            break

    return sequence


def simulate_birdsong(
    num_batches: int,
    batch_size: int,
    seq_range: tuple[int, int],
    alphabet: list[str],
    num_processes: int,
    order: int = 1,
    process_fn: Callable | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate song sequences and compute n-gram counts and probability distributions.

    Args:
        num_batches: Number of time steps (batches) per process
        batch_size: Number of sequences per batch
        seq_range: Minimum and maximum sequence lengths
        alphabet: List of symbols in the alphabet
        num_processes: Number of independent processes
        order: Markov order (1 for bigram, 2 for trigram)
        process_fn: Function to modify the transition matrix at each time step

    Returns:
        Tuple of (ngram_counts, probabilities) tensors
    """
    alphabet_size = len(alphabet)
    ngram_size = alphabet_size ** (order + 1)

    # Initialize tensors
    ngram_counts = np.zeros((ngram_size, num_batches, num_processes), dtype=np.float32)
    probabilities = np.zeros((ngram_size, num_batches, num_processes), dtype=np.float32)

    # Get indices for special symbols
    idx_start = alphabet.index('<')
    idx_end = alphabet.index('>')
    large_negative = -1e8

    for p in range(num_processes):
        # Initialize transition logits
        x_init = x_init_maker(alphabet, order=order)
        x = x_init.copy()

        # Compute final target for linear update
        x_final = np.random.randint(5, size=x.shape).astype(float)
        x_final[x_init == -large_negative] = x_init[x_init == -large_negative]
        x_final[x_init == large_negative] = x_init[x_init == large_negative]

        # Compute change matrix
        with np.errstate(invalid='ignore'):
            a = (x_final - x) / num_batches
        a[np.isnan(a)] = 0

        for t in range(num_batches):
            # Reshape transition logits for n-gram probabilities
            logits = x.reshape((alphabet_size,) * (order + 1))
            probs_matrix = np.apply_along_axis(masked_softmax, -1, logits, mask_value=large_negative)

            # Simulate batch of sequences
            batch_ngram_counts = np.zeros((ngram_size, batch_size), dtype=np.float32)

            for seq_idx in range(batch_size):
                if order == 1:
                    sequence = simulate_one_song_order1(seq_range, probs_matrix, alphabet)
                elif order == 2:
                    sequence = simulate_one_song_order2(seq_range, probs_matrix, alphabet, idx_start, idx_end)
                else:
                    raise ValueError(f"Unsupported order: {order}")

                # Add n-gram counts if sequence is valid
                if len(sequence) > seq_range[0] + 1 and sequence[-1] == '>':
                    seq_embedding = np.zeros(ngram_size)
                    for i in range(len(sequence) - order):
                        indices = tuple(alphabet.index(sequence[j]) for j in range(i, i + order + 1))
                        flat_idx = np.ravel_multi_index(indices, (alphabet_size,) * (order + 1))
                        seq_embedding[flat_idx] += 1
                    batch_ngram_counts[:, seq_idx] = seq_embedding

            # Aggregate n-gram counts
            ngram_counts[:, t, p] = batch_ngram_counts.sum(axis=1)

            # Normalize row-wise
            reshaped_counts = ngram_counts.reshape(
                alphabet_size ** order, alphabet_size, ngram_counts.shape[1], ngram_counts.shape[2]
            )
            row_sums = reshaped_counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            normalized_counts = reshaped_counts / row_sums
            normalized_counts = normalized_counts.reshape(ngram_counts.shape)
            probabilities[:, t, p] = probs_matrix.flatten()

            # Update transition matrix
            if process_fn:
                x = process_fn(x, a, t)
            else:
                x += a

    return normalized_counts, probabilities


def save_to_hdf5(output_path: str, bigram_counts: np.ndarray, probabilities: np.ndarray) -> None:
    """
    Save simulation results to HDF5 file.

    Args:
        output_path: Path to output HDF5 file
        bigram_counts: N-gram count tensor
        probabilities: Probability distribution tensor
    """
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('bigram_counts', data=bigram_counts)
        f.create_dataset('probabilities', data=probabilities)


def preprocess_simulated_songs(
    num_batches: int,
    batch_size: int,
    seq_range: tuple[int, int],
    alphabet: list[str],
    num_processes: int,
    output_path: str,
    process_fn: Callable | None = None
) -> None:
    """
    Complete simulation pipeline: generate data and save to HDF5.

    Args:
        num_batches: Number of time steps per process
        batch_size: Number of sequences per batch
        seq_range: Minimum and maximum sequence lengths
        alphabet: List of symbols in the alphabet
        num_processes: Number of independent processes
        output_path: Path to output HDF5 file
        process_fn: Function to modify transition matrix at each time step
    """
    # Determine order from alphabet size (simplified heuristic)
    order = 1 if len(alphabet) <= 10 else 2

    # Run simulation
    bigram_counts, probabilities = simulate_birdsong(
        num_batches=num_batches,
        batch_size=batch_size,
        seq_range=seq_range,
        alphabet=alphabet,
        num_processes=num_processes,
        order=order,
        process_fn=process_fn
    )

    # Save results
    save_to_hdf5(output_path, bigram_counts, probabilities)

    print(f"Simulation complete. Results saved to {output_path}")
    print(f"Data shape: {bigram_counts.shape}")


class BirdsongSimulator:
    """
    High-level interface for birdsong data simulation.
    """

    def __init__(self, alphabet: list[str], order: int = 1):
        """
        Initialize simulator.

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

    def simulate_dataset(
        self,
        num_batches: int,
        batch_size: int,
        seq_range: tuple[int, int],
        num_processes: int,
        output_path: str,
        process_fn: Callable | None = None
    ) -> None:
        """
        Simulate a complete dataset and save to HDF5.

        Args:
            num_batches: Number of time steps per process
            batch_size: Number of sequences per batch
            seq_range: Minimum and maximum sequence lengths
            num_processes: Number of independent processes
            output_path: Path to output HDF5 file
            process_fn: Function to modify transition matrix at each time step
        """
        preprocess_simulated_songs(
            num_batches=num_batches,
            batch_size=batch_size,
            seq_range=seq_range,
            alphabet=self.alphabet,
            num_processes=num_processes,
            output_path=output_path,
            process_fn=process_fn
        )

    def simulate_single_sequence(
        self,
        seq_range: tuple[int, int],
        transition_matrix: np.ndarray
    ) -> list[str]:
        """
        Simulate a single sequence using given transition matrix.

        Args:
            seq_range: Minimum and maximum sequence lengths
            transition_matrix: Transition probability matrix

        Returns:
            Simulated sequence as list of symbols
        """
        if self.order == 1:
            return simulate_one_song_order1(seq_range, transition_matrix, self.alphabet)
        elif self.order == 2:
            idx_start = self.alphabet.index('<')
            idx_end = self.alphabet.index('>')
            return simulate_one_song_order2(seq_range, transition_matrix, self.alphabet, idx_start, idx_end)
        else:
            raise ValueError(f"Unsupported order: {self.order}")
