"""
Evaluation utilities for birdsong analysis.

This module contains the BirdsongEvaluator class and related utilities
for evaluating LFADS-style birdsong models with various metrics.
"""

import os
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Some plotting features may be limited.")
import torch
import torch.nn.functional as f
from tqdm import tqdm

from ..data.loader import BirdsongDataset
from ..models.lfads import BirdsongLFADSModel2, rowwise_masked_softmax, rowwise_softmax


def create_test_loader_from_checkpoint(checkpoint_path: str, dataset: BirdsongDataset) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the test set using indices saved in the checkpoint.
    
    This ensures that the exact same test samples used during training are used for evaluation,
    preventing data leakage and ensuring consistent evaluation.
    
    Args:
        checkpoint_path: Path to the training checkpoint
        dataset: The full dataset to create test loader from
        
    Returns:
        DataLoader containing only the test set samples
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If test indices are not found in checkpoint
        ValueError: If test indices are invalid
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract test indices
    if 'test_indices' not in checkpoint:
        raise KeyError(f"Test indices not found in checkpoint: {checkpoint_path}")
    
    test_indices = checkpoint['test_indices']
    
    # Validate indices
    if not isinstance(test_indices, (list, torch.Tensor, np.ndarray)):
        raise ValueError("Test indices must be a list, tensor, or numpy array")
    
    if len(test_indices) == 0:
        raise ValueError("Test indices list is empty")
    
    # Convert to list if needed
    if isinstance(test_indices, torch.Tensor):
        test_indices = test_indices.tolist()
    elif isinstance(test_indices, np.ndarray):
        test_indices = test_indices.tolist()
    
    # Validate indices are within dataset bounds
    max_idx = len(dataset) - 1
    for idx in test_indices:
        if not (0 <= idx <= max_idx):
            raise ValueError(f"Test index {idx} is out of bounds for dataset with {len(dataset)} samples")
    
    # Create subset of dataset with test indices
    from torch.utils.data import Subset
    test_subset = Subset(dataset, test_indices)
    
    # Create DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=64,  # Default batch size for evaluation
        shuffle=False,  # Keep order consistent
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False
    )
    
    print(f"✅ Created test loader with {len(test_indices)} samples from checkpoint")
    return test_loader


def rowwise_kl_div(pred_dist: torch.Tensor, true_dist: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute rowwise KL divergence.

    Args:
        pred_dist: Predicted distribution
        true_dist: True distribution
        eps: Small value to avoid log(0)

    Returns:
        KL divergence value
    """
    pred_dist = torch.clamp(pred_dist, eps, 1.0)
    true_dist = torch.clamp(true_dist, eps, 1.0)
    kl = true_dist * torch.log(true_dist / pred_dist)
    return kl.sum(dim=-1).mean()


def rowwise_mse(pred_dist: torch.Tensor, true_dist: torch.Tensor) -> torch.Tensor:
    """
    Compute rowwise Mean Squared Error.

    Args:
        pred_dist: Predicted distribution
        true_dist: True distribution

    Returns:
        MSE value
    """
    return ((pred_dist - true_dist) ** 2).mean()


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence.

    Args:
        p: First distribution
        q: Second distribution
        eps: Small value to avoid log(0)

    Returns:
        JS divergence value
    """
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    js = 0.5 * (torch.sum(p * torch.log(p / m)) + torch.sum(q * torch.log(q / m)))
    return js


def cross_entropy(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute cross-entropy.

    Args:
        p: True distribution
        q: Predicted distribution
        eps: Small value to avoid log(0)

    Returns:
        Cross-entropy value
    """
    q = torch.clamp(q, eps, 1.0)
    return -torch.sum(p * torch.log(q))


def evaluate_birdsong_model(
    model: BirdsongLFADSModel2,
    input_counts: torch.Tensor,
    true_probs: torch.Tensor,
    alphabet_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, float]]:
    """
    Evaluate the model with various metrics.

    Args:
        model: Trained Birdsong LFADS model
        input_counts: Input bigram counts
        true_probs: True probability distributions
        alphabet_size: Size of the alphabet

    Returns:
        Tuple of (logits_4d, pred_probs_4d, factors, metrics)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_counts)
        logits = outputs["logits"]
        factors = outputs.get("factors", None)
        b, t, logit_dim = logits.shape
        order = model.order

        if order == 1:
            # Bigram case: reshape logits to (B, T, α, α)
            logits_4d = logits.view(b, t, alphabet_size, alphabet_size)
            pred_probs_4d = f.softmax(logits_4d, dim=-1)
            # Flatten to (B, T, α²) for loss computation
            pred_probs_flat = pred_probs_4d.view(b, t, alphabet_size ** 2)
            true_probs_flat = true_probs.view(b, t, alphabet_size ** 2)
        elif order == 2:
            # Trigram case: reshape logits to (B, T, α², α)
            logits_4d = logits.view(b, t, alphabet_size ** 2, alphabet_size)
            true_probs_4d = true_probs.view(b, t, alphabet_size ** 2, alphabet_size)
            target_matrix = true_probs_4d[0, 0, :, :]
            # Use masked softmax for trigrams
            pred_probs = rowwise_masked_softmax(
                logits, alphabet_size, order=order,
                target_matrix=target_matrix, mask_value=-1e8
            )
            pred_probs_4d = pred_probs.view(b, t, alphabet_size ** 2, alphabet_size)
            # Flatten to (B, T, α³) for losses
            pred_probs_flat = pred_probs_4d.view(b, t, alphabet_size ** 3)
            true_probs_flat = true_probs_4d.view(b, t, alphabet_size ** 3)
        else:
            raise ValueError("Only order==1 and order==2 are supported.")

        # Cross-Entropy Loss
        if order == 1:
            pred_flat = pred_probs_4d.view(b * t * alphabet_size, alphabet_size)
            true_labels = true_probs.view(b, t, alphabet_size, alphabet_size).argmax(dim=-1)
            true_flat = true_labels.view(b * t * alphabet_size)
        elif order == 2:
            pred_flat = pred_probs_4d.view(b * t * (alphabet_size ** 2), alphabet_size)
            true_labels = true_probs.view(b, t, alphabet_size ** 2, alphabet_size).argmax(dim=-1)
            true_flat = true_labels.view(b * t * (alphabet_size ** 2))

        ce_loss = f.cross_entropy(pred_flat, true_flat)

        # Accuracy
        pred_classes = pred_probs_4d.argmax(dim=-1)
        true_classes = true_probs.view(b, t, -1, alphabet_size).argmax(dim=-1)
        correct = (pred_classes == true_classes).sum().item()
        total = pred_classes.numel()
        accuracy = correct / total if total > 0 else 0.0

        # Rowwise KL and MSE
        kl_val = rowwise_kl_div(pred_probs_flat, true_probs_flat)
        mse_val = rowwise_mse(pred_probs_flat, true_probs_flat)

        # Jensen-Shannon divergence
        js_val = js_divergence(pred_probs_flat, true_probs_flat)

        metrics = {
            "cross_entropy": ce_loss.item(),
            "accuracy": accuracy,
            "rowwise_kl": kl_val.item(),
            "rowwise_mse": mse_val.item(),
            "js_divergence": js_val.item(),
        }

    return logits_4d, pred_probs_4d, factors, metrics


def plot_wrate_factors(model: BirdsongLFADSModel2, factors: torch.Tensor,
                      alphabet_size: int, sequence_idx: int = 0) -> None:
    """
    Plot the time-averaged output of W_rate * factors as a heatmap.

    Args:
        model: Trained model
        factors: Model factors
        alphabet_size: Size of the alphabet
        sequence_idx: Index of sequence to plot
    """
    if factors is None:
        print("No factors returned by model; cannot plot W_rate*factors.")
        return

    try:
        w_factors = model.rate_linear(factors)  # (B, T, out_dim)
    except AttributeError:
        print("Model does not have a rate_linear layer or there's a dimension mismatch.")
        return

    w_factors = w_factors.detach()
    b, t, out_dim = w_factors.shape
    order = model.order

    if order == 1:
        w_factors_4d = w_factors.view(b, t, alphabet_size, alphabet_size)
    elif order == 2:
        w_factors_4d = w_factors.view(b, t, alphabet_size ** 2, alphabet_size)
    else:
        print("Unsupported order for plotting.")
        return

    data_4d = w_factors_4d[sequence_idx].cpu().numpy()

    # (A) Time-averaged heatmap
    avg_data = data_4d.mean(axis=0)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im = ax1.imshow(avg_data, cmap='bwr', aspect='auto')
    ax1.set_title("Mean of [W_rate * factors] across time")
    fig1.colorbar(im, ax=ax1)
    plt.tight_layout()
    plt.show()

    # (B) Time evolution plots
    x = avg_data.shape[0]
    fig2, axs = plt.subplots(1, x, figsize=(4 * x, 4), sharey=True)
    if x == 1:
        axs = [axs]
    for i in range(x):
        ax = axs[i]
        row_data = data_4d[:, i, :]
        for col in range(alphabet_size):
            ax.plot(row_data[:, col], label=f"col {col}")
        ax.set_title(f"Row {i}")
        ax.set_xlabel("Time")
        if i == 0:
            ax.set_ylabel("Value")
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_ngram_counts(pred_probs_4d: torch.Tensor, true_probs: torch.Tensor, 
                     ngram_counts: torch.Tensor, alphabet_size: int, order: int) -> None:
    """
    Plot predicted and true probability distributions alongside estimated n-gram frequency counts.

    Args:
        pred_probs_4d: Predicted probabilities
        true_probs: True probabilities
        ngram_counts: N-gram counts
        alphabet_size: Size of the alphabet
        order: Order of the model (1 for bigrams, 2 for trigrams)
    """
    b, t = pred_probs_4d.shape[:2]

    if order == 1:
        # Reshape counts to (B, T, α, α)
        counts_4d = ngram_counts.view(b, t, alphabet_size, alphabet_size).cpu().numpy()
        # Normalize rowwise
        counts_norm = np.where(
            counts_4d.sum(axis=-1, keepdims=True) > 0,
            counts_4d / counts_4d.sum(axis=-1, keepdims=True),
            0.0
        )
        # For plotting, choose the first batch element
        pred_mat = pred_probs_4d[0].cpu().numpy()  # (T, α, α)
        true_mat = true_probs.view(b, t, alphabet_size, alphabet_size)[0].cpu().numpy()
        counts_mat = counts_norm[0]  # (T, α, α)
        num_rows = alphabet_size
        fig, axs = plt.subplots(1, num_rows, figsize=(5 * num_rows, 5), sharey=True)
        if num_rows == 1:
            axs = [axs]
        cmap = plt.get_cmap('tab10', alphabet_size)
        colors = [cmap(i) for i in range(alphabet_size)]
        for row in range(num_rows):
            ax = axs[row]
            for col in range(alphabet_size):
                ax.plot(pred_mat[:, row, col], '--', color=colors[col], label=f"Pred col={col}")
                ax.plot(true_mat[:, row, col], '-', color=colors[col], label=f"True col={col}")
                ax.plot(counts_mat[:, row, col], ':', color=colors[col], label=f"Count col={col}")
            ax.set_title(f"Row {row}")
            ax.set_xlabel("Time")
            if row == 0:
                ax.set_ylabel("Probability")
            if row == num_rows - 1:
                ax.legend(loc="best")
        plt.tight_layout()
        plt.show()
    elif order == 2:
        # For trigrams, reshape counts to (B, T, α², α)
        counts_4d = ngram_counts.view(b, t, alphabet_size ** 2, alphabet_size).cpu().numpy()
        counts_norm = np.where(
            counts_4d.sum(axis=-1, keepdims=True) > 0,
            counts_4d / counts_4d.sum(axis=-1, keepdims=True),
            0.0
        )
        pred_mat = pred_probs_4d[0].cpu().numpy()  # (T, α², α)
        true_mat = true_probs.view(b, t, alphabet_size ** 2, alphabet_size)[0].cpu().numpy()
        counts_mat = counts_norm[0]  # (T, α², α)
        num_contexts = alphabet_size ** 2
        fig, axs = plt.subplots(1, num_contexts, figsize=(3 * num_contexts, 3), sharey=True)
        if num_contexts == 1:
            axs = [axs]
        cmap = plt.get_cmap('tab10', alphabet_size)
        colors = [cmap(i) for i in range(alphabet_size)]
        for ctx in range(num_contexts):
            ax = axs[ctx]
            for col in range(alphabet_size):
                ax.plot(pred_mat[:, ctx, col], '--', color=colors[col], label=f"Pred col={col}")
                ax.plot(true_mat[:, ctx, col], '-', color=colors[col], label=f"True col={col}")
                ax.plot(counts_mat[:, ctx, col], ':', color=colors[col], label=f"Count col={col}")
            ax.set_title(f"Context {ctx}")
            ax.set_xlabel("Time")
            if ctx == 0:
                ax.set_ylabel("Probability")
            if ctx == num_contexts - 1:
                ax.legend(loc="best")
        plt.tight_layout()
        plt.show()
    else:
        print("Unsupported order for plotting n-gram counts.")


def get_alphabet(alphabet_size: int) -> list[str]:
    """
    Returns an alphabet list. Assumes:
      - The first symbol is '<' (start)
      - The last symbol is '>' (end)
      - The in-between symbols are generated from letters starting at 'a'
    """
    if alphabet_size < 2:
        return ['<']
    num_middle = alphabet_size - 2
    middle = [chr(ord('a') + i) for i in range(num_middle)]
    return ['<'] + middle + ['>']


def get_context_string(row_idx: int, alphabet: list[str], order: int) -> str:
    """
    For order==1, returns the single symbol for this row.
    For order==2, returns a two-character string computed by mapping the row index
    to a pair (first = index // alphabet_size, second = index % alphabet_size).
    """
    if order == 1:
        return alphabet[row_idx]
    elif order == 2:
        a_size = len(alphabet)
        first = alphabet[row_idx // a_size]
        second = alphabet[row_idx % a_size]
        return first + second
    else:
        return str(row_idx)


def safe_context_str(context_str: str) -> str:
    """Replaces characters illegal in filenames."""
    return context_str.replace('<', 'start').replace('>', 'end')


def smooth_counts(counts: np.ndarray, window_size: int) -> np.ndarray:
    """
    Applies a running average over the time dimension.
    counts: numpy array of shape (T, rows, cols)
    Returns a smoothed array of the same shape.
    """
    if window_size <= 1:
        return counts
    
    t, rows, cols = counts.shape
    pad = window_size // 2
    
    # Use edge padding
    padded = np.pad(counts, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    smoothed = np.empty_like(counts)
    
    for t_idx in range(t):
        window_data = padded[t_idx:t_idx + window_size]
        # Handle potential zero sums by using nanmean and then filling with zeros
        window_mean = np.nanmean(window_data, axis=0)
        # Replace any NaN values with 0
        window_mean = np.nan_to_num(window_mean, nan=0.0)
        smoothed[t_idx] = window_mean
    
    return smoothed


def plot_transition_plots(
    process_idx: int, pred_probs_4d: np.ndarray, true_probs: np.ndarray, 
    raw_est: np.ndarray, smooth_est: Optional[np.ndarray], alphabet_size: int, 
    order: int, dataset_name: str, symbols: Optional[List[str]] = None, 
    base_output_dir: str = "process_plots", file_format: str = "png", 
    plot_smooth_est: bool = True
) -> None:
    """
    For each row (context), plots a separate figure showing over time:
      - Predicted probabilities (dashed),
      - True probabilities (solid),
      - Raw estimate (dotted),
      - And if available, a smooth estimate (dash-dot).

    Args:
        process_idx: Index of the process being plotted
        pred_probs_4d: Predicted probabilities as numpy array
        true_probs: True probabilities as numpy array
        raw_est: Raw estimates as numpy array
        smooth_est: Smooth estimates as numpy array (optional)
        alphabet_size: Size of the alphabet
        order: Order of the model
        dataset_name: Name of the dataset
        symbols: Custom symbols for the alphabet
        base_output_dir: Base directory for output
        file_format: File format for saving plots
        plot_smooth_est: Whether to plot smooth estimates
    """
    alphabet = symbols if symbols is not None else get_alphabet(alphabet_size)
    
    # Create directory for this process only when we're about to save files
    process_dir_name = f"{dataset_name}_process_{process_idx}"
    process_dir = Path(base_output_dir) / process_dir_name
    process_dir.mkdir(parents=True, exist_ok=True)

    num_contexts = pred_probs_4d.shape[1]
    cmap = cm.get_cmap('tab10', alphabet_size)

    for row_idx in range(num_contexts):
        context_str = get_context_string(row_idx, alphabet, order)
        safe_ctx = safe_context_str(context_str)
        
        plt.figure(figsize=(10, 6))  # Increased width to accommodate legend
        for col_idx in range(alphabet_size):
            plt.plot(pred_probs_4d[:, row_idx, col_idx], '--', color=cmap(col_idx),
                     label=f"Pred {context_str} → {alphabet[col_idx]}")
            plt.plot(true_probs[:, row_idx, col_idx], '-', color=cmap(col_idx),
                     label=f"True {context_str} → {alphabet[col_idx]}")
            plt.plot(raw_est[:, row_idx, col_idx], ':', color=cmap(col_idx),
                     label=f"Est {context_str} → {alphabet[col_idx]}")
            if plot_smooth_est and smooth_est is not None:
                plt.plot(smooth_est[:, row_idx, col_idx], '-.', color=cmap(col_idx),
                         label=f"Smooth Est {context_str} → {alphabet[col_idx]}")
        
        plt.title(f"Transition from Context: {context_str}")
        plt.xlabel("Time Step")
        plt.ylabel("Probability")
        # Place the legend outside of the plot with more space
        plt.legend(fontsize='small', bbox_to_anchor=(1.15, 1), loc='upper left')
        
        # Use a safer layout approach
        try:
            plt.tight_layout()
        except:
            # If tight_layout fails, use a more conservative approach
            plt.subplots_adjust(right=0.85, bottom=0.15, top=0.9, left=0.1)
        
        filename = f"transition_{safe_ctx}.{file_format}"
        plt.savefig(process_dir / filename, dpi=300, bbox_inches='tight')
        filename_svg = f"transition_{safe_ctx}.svg"
        plt.savefig(process_dir / filename_svg, dpi=300, bbox_inches='tight')
        plt.close()


def plot_summary_metrics(
    all_js_pred: List[np.ndarray], all_js_est_raw: List[np.ndarray], 
    all_js_est_smooth: Optional[List[np.ndarray]],
    all_ce_pred: List[np.ndarray], all_ce_est_raw: List[np.ndarray], 
    all_ce_est_smooth: Optional[List[np.ndarray]],
    file_format: str, dataset_name: str, summary_output_dir: str = "summary_plots"
) -> None:
    """
    Plots two summary figures using the mean and standard deviation across processes:
      - One for mean Jensen–Shannon divergence per time step.
      - One for mean cross entropy per time step.

    Args:
        all_js_pred: List of JS divergence arrays for predicted vs true
        all_js_est_raw: List of JS divergence arrays for raw estimate vs true
        all_js_est_smooth: List of JS divergence arrays for smooth estimate vs true
        all_ce_pred: List of cross entropy arrays for predicted vs true
        all_ce_est_raw: List of cross entropy arrays for raw estimate vs true
        all_ce_est_smooth: List of cross entropy arrays for smooth estimate vs true
        file_format: File format for saving plots
        dataset_name: Name of the dataset
        summary_output_dir: Directory for summary plots
    """
    # Convert lists to numpy arrays
    js_pred_arr = np.array(all_js_pred)  # shape: (n_processes, T)
    ce_pred_arr = np.array(all_ce_pred)

    x = np.arange(js_pred_arr.shape[1])

    # Create output directory only when we're about to save files
    summary_dir = Path(summary_output_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Jensen-Shannon divergence plot
    plt.figure(figsize=(10, 6))

    # Compute mean and std for predicted JS
    mean_js_pred = js_pred_arr.mean(axis=0)
    std_js_pred = js_pred_arr.std(axis=0)
    plt.plot(x, mean_js_pred, color='blue', linewidth=3, label="Mean Pred JS")
    plt.fill_between(x, mean_js_pred - std_js_pred, mean_js_pred + std_js_pred,
                     color='blue', alpha=0.2)

    # Raw estimate JS
    if all_js_est_raw is not None:
        js_est_raw_arr = np.array(all_js_est_raw)
        mean_js_est_raw = js_est_raw_arr.mean(axis=0)
        std_js_est_raw = js_est_raw_arr.std(axis=0)
        plt.plot(x, mean_js_est_raw, color='red', linewidth=3, label="Mean Raw Est JS")
        plt.fill_between(x, mean_js_est_raw - std_js_est_raw, mean_js_est_raw + std_js_est_raw,
                         color='red', alpha=0.2)

    # Smooth estimate JS
    if all_js_est_smooth is not None:
        js_est_smooth_arr = np.array(all_js_est_smooth)
        mean_js_est_smooth = js_est_smooth_arr.mean(axis=0)
        std_js_est_smooth = js_est_smooth_arr.std(axis=0)
        plt.plot(x, mean_js_est_smooth, color='orange', linewidth=3, label="Mean Smooth Est JS")
        plt.fill_between(x, mean_js_est_smooth - std_js_est_smooth, mean_js_est_smooth + std_js_est_smooth,
                         color='orange', alpha=0.2)

    plt.xlabel("Time Step")
    plt.ylabel("JS Divergence")
    plt.title("Mean Jensen–Shannon Divergence per Time Step")
    plt.legend(loc="best")
    
    # Use a safer layout approach
    try:
        plt.tight_layout()
    except:
        # If tight_layout fails, use a more conservative approach
        plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95)
    
    js_filename = summary_dir / f"{dataset_name}_summary_js_divergence.{file_format}"
    plt.savefig(js_filename, dpi=300, bbox_inches='tight')
    js_filename_svg = summary_dir / f"{dataset_name}_summary_js_divergence.svg"
    plt.savefig(js_filename_svg, dpi=300, bbox_inches='tight')
    plt.close()

    # Cross Entropy plot
    plt.figure(figsize=(10, 6))

    # Predicted CE
    mean_ce_pred = ce_pred_arr.mean(axis=0)
    std_ce_pred = ce_pred_arr.std(axis=0)
    plt.plot(x, mean_ce_pred, color='blue', linewidth=3, label="Mean Pred CE")
    plt.fill_between(x, mean_ce_pred - std_ce_pred, mean_ce_pred + std_ce_pred,
                     color='blue', alpha=0.2)

    # Raw estimate CE
    if all_ce_est_raw is not None:
        ce_est_raw_arr = np.array(all_ce_est_raw)
        mean_ce_est_raw = ce_est_raw_arr.mean(axis=0)
        std_ce_est_raw = ce_est_raw_arr.std(axis=0)
        plt.plot(x, mean_ce_est_raw, color='red', linewidth=3, label="Mean Raw Est CE")
        plt.fill_between(x, mean_ce_est_raw - std_ce_est_raw, mean_ce_est_raw + std_ce_est_raw,
                         color='red', alpha=0.2)

    # Smooth estimate CE
    if all_ce_est_smooth is not None:
        ce_est_smooth_arr = np.array(all_ce_est_smooth)
        mean_ce_est_smooth = ce_est_smooth_arr.mean(axis=0)
        std_ce_est_smooth = ce_est_smooth_arr.std(axis=0)
        plt.plot(x, mean_ce_est_smooth, color='orange', linewidth=3, label="Mean Smooth Est CE")
        plt.fill_between(x, mean_ce_est_smooth - std_ce_est_smooth, mean_ce_est_smooth + std_ce_est_smooth,
                         color='orange', alpha=0.2)

    plt.xlabel("Time Step")
    plt.ylabel("Cross Entropy")
    plt.title("Mean Cross Entropy per Time Step")
    plt.legend(loc="best")
    
    # Use a safer layout approach
    try:
        plt.tight_layout()
    except:
        # If tight_layout fails, use a more conservative approach
        plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95)
    
    ce_filename = summary_dir / f"{dataset_name}_summary_cross_entropy.{file_format}"
    plt.savefig(ce_filename, dpi=300, bbox_inches='tight')
    ce_filename_svg = summary_dir / f"{dataset_name}_summary_cross_entropy.svg"
    plt.savefig(ce_filename_svg, dpi=300, bbox_inches='tight')
    plt.close()


class BirdsongEvaluator:
    """
    Comprehensive evaluator for birdsong models with rich visualization capabilities.
    """

    def __init__(self, model: BirdsongLFADSModel2, device: torch.device):
        """
        Initialize the evaluator.

        Args:
            model: Trained Birdsong LFADS model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    def evaluate_dataset(self, dataset: BirdsongDataset, num_samples: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Evaluate the model on a dataset.

        Args:
            dataset: Dataset to evaluate on
            num_samples: Number of samples to evaluate (None for all)

        Returns:
            Dictionary of metrics for each sample
        """
        if num_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(num_samples, len(dataset))

        all_metrics = {
            'cross_entropy': [],
            'accuracy': [],
            'rowwise_kl': [],
            'rowwise_mse': [],
            'js_divergence': []
        }

        for i in tqdm(range(num_samples), desc="Evaluating"):
            ngram_counts, true_probs = dataset[i]
            ngram_counts = ngram_counts.unsqueeze(0).to(self.device)
            true_probs = true_probs.unsqueeze(0).to(self.device)

            _, _, _, metrics = evaluate_birdsong_model(
                self.model, ngram_counts, true_probs, self.model.alphabet_size
            )

            for key, value in metrics.items():
                all_metrics[key].append(value)

        return all_metrics

    def compute_summary_metrics(self, all_metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Compute summary statistics from all metrics.

        Args:
            all_metrics: Dictionary of metrics for each sample

        Returns:
            Dictionary of summary statistics
        """
        summary = {}
        for key, values in all_metrics.items():
            values = np.array(values)
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values))
            summary[f'{key}_min'] = float(np.min(values))
            summary[f'{key}_max'] = float(np.max(values))
        return summary

    def plot_metrics_distribution(self, all_metrics: Dict[str, List[float]], 
                                save_path: Optional[str] = None) -> None:
        """
        Plot distribution of metrics across samples.

        Args:
            all_metrics: Dictionary of metrics for each sample
            save_path: Path to save the plot
        """
        num_metrics = len(all_metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 4))
        if num_metrics == 1:
            axes = [axes]

        for i, (metric_name, values) in enumerate(all_metrics.items()):
            axes[i].hist(values, bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric_name} Distribution')
            axes[i].set_xlabel(metric_name)
            axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        if save_path:
            # Save in PNG format
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Also save in SVG format if the path doesn't already have .svg extension
            if not save_path.endswith('.svg'):
                svg_path = save_path.replace('.png', '.svg')
                plt.savefig(svg_path, dpi=300, bbox_inches='tight', format='svg')
        plt.show()

    def plot_transition_matrices(self, dataset: BirdsongDataset, num_plots: int = 5,
                               save_dir: Optional[str] = None) -> None:
        """
        Plot transition matrices for multiple samples.

        Args:
            dataset: Dataset to plot from
            num_plots: Number of plots to generate
            save_dir: Directory to save plots
        """
        # Only create directory if we're actually going to save files
        if save_dir:
            save_dir = Path(save_dir)
            # Don't create directory here - create it only when we're about to save a plot

        for i in range(min(num_plots, len(dataset))):
            ngram_counts, true_probs = dataset[i]
            ngram_counts = ngram_counts.unsqueeze(0).to(self.device)
            true_probs = true_probs.unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(ngram_counts)
                logits = outputs["logits"]

                if self.model.order == 1:
                    pred_probs = f.softmax(logits.view(1, -1, self.model.alphabet_size, self.model.alphabet_size), dim=-1)
                    true_mat = true_probs.view(1, -1, self.model.alphabet_size, self.model.alphabet_size)
                else:
                    # Handle order 2 case
                    pred_probs = f.softmax(logits.view(1, -1, self.model.alphabet_size ** 2, self.model.alphabet_size), dim=-1)
                    true_mat = true_probs.view(1, -1, self.model.alphabet_size ** 2, self.model.alphabet_size)

            # Plot transition matrices
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Use seaborn if available, otherwise use matplotlib
            if SEABORN_AVAILABLE:
                sns.heatmap(true_mat[0, 0].cpu().numpy(), ax=axes[0], cmap="viridis", cbar=True)
                sns.heatmap(pred_probs[0, 0].cpu().numpy(), ax=axes[1], cmap="viridis", cbar=True)
            else:
                # Fallback to matplotlib
                axes[0].imshow(true_mat[0, 0].cpu().numpy(), cmap="viridis", aspect='auto')
                axes[0].set_title("True Transition Matrix")
                plt.colorbar(axes[0].images[0], ax=axes[0])
                
                axes[1].imshow(pred_probs[0, 0].cpu().numpy(), cmap="viridis", aspect='auto')
                axes[1].set_title("Predicted Transition Matrix")
                plt.colorbar(axes[1].images[0], ax=axes[1])

            fig.suptitle(f"Sample {i} - Transition Matrix Comparison")
            plt.tight_layout()

            if save_dir:
                # Create directory only when we're about to save a plot
                save_dir.mkdir(parents=True, exist_ok=True)
                plot_path = save_dir / f"transition_matrix_sample_{i}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                
                # Also save in SVG format
                svg_path = save_dir / f"transition_matrix_sample_{i}.svg"
                plt.savefig(svg_path, dpi=300, bbox_inches='tight', format='svg')
                print(f"Saved transition matrix plot to {plot_path} and {svg_path}")
            plt.close()

    def save_evaluation_results(self, all_metrics: Dict[str, List[float]], 
                              summary_metrics: Dict[str, float],
                              output_path: str) -> None:
        """
        Save evaluation results to a file.

        Args:
            all_metrics: Dictionary of metrics for each sample
            summary_metrics: Dictionary of summary statistics
            output_path: Path to save results
        """
        with open(output_path, 'w') as f:
            f.write("Evaluation Results\n")
            f.write("==================\n\n")
            
            f.write("Summary Statistics:\n")
            f.write("------------------\n")
            for key, value in summary_metrics.items():
                f.write(f"{key}: {value:.6f}\n")
            
            f.write("\nPer-Sample Metrics:\n")
            f.write("-------------------\n")
            for i in range(len(next(iter(all_metrics.values())))):
                f.write(f"\nSample {i}:\n")
                for metric_name, values in all_metrics.items():
                    f.write(f"  {metric_name}: {values[i]:.6f}\n")
