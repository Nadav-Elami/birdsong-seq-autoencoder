"""
Training utilities for birdsong analysis.

This module contains the BirdsongTrainer class and related utilities
for training LFADS-style birdsong models.
"""

import math
import os

import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Some plotting features may be limited.")
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..data.loader import BirdsongDataset
from ..models.lfads import BirdsongLFADSModel2, rowwise_masked_softmax, rowwise_softmax


def logistic_kl_weight(epoch: int, k: float = 0.005, c: float = 1000) -> float:
    """
    Returns a logistic weighting factor from near 0 to near 1.

    Args:
        epoch: Current epoch
        k: Sharpness parameter
        c: Center parameter

    Returns:
        KL weight between 0 and 1
    """
    return 1.0 / (1.0 + math.exp(-k * (epoch - c)))


class ValidationSubset:
    """
    Utility class for handling dataset subsets and rescaling metrics.
    """

    def __init__(self, dataset: BirdsongDataset, val_split: float = 0.15, test_split: float = 0.1):
        """
        Create train, validation, and test subsets from the dataset.

        Args:
            dataset: The full dataset to split
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
        """
        self.dataset = dataset
        self.val_split = val_split
        self.test_split = test_split
        self.train_indices, self.val_indices, self.test_indices = self._split_dataset()

    def _split_dataset(self) -> tuple[list[int], list[int], list[int]]:
        """Split dataset into train, validation, and test indices."""
        total_len = len(self.dataset)
        test_len = int(total_len * self.test_split)
        val_len = int((total_len - test_len) * self.val_split)
        total_len - test_len - val_len

        indices = torch.randperm(total_len).tolist()
        test_indices = indices[:test_len]
        val_indices = indices[test_len:test_len + val_len]
        train_indices = indices[test_len + val_len:]
        return train_indices, val_indices, test_indices

    def get_loaders(self, batch_size: int, num_workers: int = 0,
                   pin_memory: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Return train, validation, and test DataLoaders.

        Args:
            batch_size: Batch size for all loaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            Subset(self.dataset, self.train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            Subset(self.dataset, self.val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            Subset(self.dataset, self.test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return train_loader, val_loader, test_loader


class MemoryManager:
    """
    Class to manage GPU memory during training.
    """

    def __init__(self, device: torch.device):
        """
        Initialize memory manager.

        Args:
            device: Device to monitor
        """
        self.device = device

    def clear_cache(self) -> None:
        """Clear the CUDA memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def report_memory(self) -> None:
        """Report memory usage on the GPU."""
        if self.device.type == "cuda":
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved(self.device) / 1024**3:.2f} GB")

    def adjust_batch_size(self, current_batch_size: int, factor: float = 0.5) -> int:
        """
        Reduce batch size when memory error occurs.

        Args:
            current_batch_size: Current batch size
            factor: Factor by which to reduce batch size

        Returns:
            Adjusted batch size
        """
        new_batch_size = max(1, int(current_batch_size * factor))
        print(f"Adjusting batch size from {current_batch_size} to {new_batch_size} due to memory constraints.")
        return new_batch_size


def plot_transition_plots(test_loader: DataLoader, model: BirdsongLFADSModel2,
                         processes_to_plot: list[int], plot_dir: str, device: torch.device) -> None:
    """
    Plot transition matrices for specified processes.

    Args:
        test_loader: DataLoader for test data
        model: Trained model
        processes_to_plot: List of test process indices to plot
        plot_dir: Directory to save the plots
        device: Device for model inference
    """
    model.eval()
    # Don't create plot_dir here - create it only when we're about to save a plot

    with torch.no_grad():
        for process_idx in processes_to_plot:
            for _batch_idx, (bigram_counts, probabilities) in enumerate(test_loader):
                bigram_counts = bigram_counts.to(device)
                probabilities = probabilities.to(device)

                if process_idx < bigram_counts.size(0):
                    bigram_counts_process = bigram_counts[process_idx: process_idx + 1]
                    probabilities_process = probabilities[process_idx: process_idx + 1]

                    outputs = model(bigram_counts_process)
                    b, t, out_dim = outputs["logits"].shape
                    alpha = model.alphabet_size

                    if model.order == 1:
                        pred_probs = rowwise_softmax(outputs["logits"], alpha, order=model.order)
                        pred_mat = pred_probs.view(b, t, alpha, alpha)
                        true_mat = probabilities_process.view(b, t, alpha, alpha)
                    elif model.order == 2:
                        true_mat = probabilities_process.view(b, t, alpha ** 2, alpha)
                        target_matrix = true_mat[0, 0]
                        pred_probs = rowwise_masked_softmax(
                            outputs["logits"], alpha, order=model.order,
                            target_matrix=target_matrix, mask_value=-1e8
                        )
                        pred_mat = pred_probs.view(b, t, alpha ** 2, alpha)
                    else:
                        raise ValueError("Unsupported order: use order==1 or order==2.")

                    # Plotting the transition matrices
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Use seaborn if available, otherwise use matplotlib
                    if SEABORN_AVAILABLE:
                        sns.heatmap(true_mat[0, 0].cpu().numpy(), ax=axes[0], cmap="viridis", cbar=True)
                        sns.heatmap(pred_mat[0, 0].cpu().numpy(), ax=axes[1], cmap="viridis", cbar=True)
                    else:
                        # Fallback to matplotlib
                        axes[0].imshow(true_mat[0, 0].cpu().numpy(), cmap="viridis", aspect='auto')
                        axes[0].set_title("True Transition Matrix")
                        plt.colorbar(axes[0].images[0], ax=axes[0])
                        
                        axes[1].imshow(pred_mat[0, 0].cpu().numpy(), cmap="viridis", aspect='auto')
                        axes[1].set_title("Predicted Transition Matrix")
                        plt.colorbar(axes[1].images[0], ax=axes[1])
                    
                    fig.suptitle(f"Process {process_idx}, Time Step 0")
                    plt.tight_layout()

                    # Create plot directory only when we're about to save a plot
                    os.makedirs(plot_dir, exist_ok=True)
                    
                    # Save the figure
                    file_path = os.path.join(plot_dir, f"process_{process_idx}_transition_plot.png")
                    plt.savefig(file_path)
                    plt.close()
                    break  # Stop after processing the requested process


class BirdsongTrainer:
    """
    Trainer class for Birdsong LFADS models.

    This class provides a configurable training interface with checkpointing,
    validation, and visualization capabilities.
    """

    def __init__(self, model: BirdsongLFADSModel2, dataset: BirdsongDataset,
                 config: dict | None = None):
        """
        Initialize the trainer.

        Args:
            model: Birdsong LFADS model to train
            dataset: Dataset for training
            config: Training configuration dictionary
        """
        self.model = model
        self.dataset = dataset
        self.config = config or self._get_default_config()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Log device information
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name(self.device)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f} GB")
        else:
            print("Running on CPU")

        # Setup memory manager
        self.memory_manager = MemoryManager(self.device)

        # Setup data splits
        self.validation_subset = ValidationSubset(
            dataset,
            val_split=self.config.get("val_split", 0.15),
            test_split=self.config.get("test_split", 0.1)
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "rec_loss": [],
            "kl_loss": [],
            "l2_loss": []
        }

    def _get_default_config(self) -> dict:
        """Get default training configuration."""
        return {
            "batch_size": 32,
            "epochs": 20,
            "learning_rate": 1e-3,
            "kl_start_epoch": 2,
            "kl_full_epoch": 10,
            "checkpoint_path": "checkpoints/birdsong_lfads.pt",
            "print_every": 10,
            "l1_lambda": 0.0001,
            "plot_dir": "plots",
            "enable_kl_loss": True,
            "enable_l2_loss": True,
            "enable_l1_loss": True,
            "disable_tqdm": False,
            "num_workers": 0,
            "pin_memory": False
        }

    def train(self, epochs: int | None = None, batch_size: int | None = None,
              resume_from_checkpoint: str | None = None) -> dict[str, list[float]]:
        """
        Train the model.

        Args:
            epochs: Number of training epochs (overrides config)
            batch_size: Batch size (overrides config)
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training history dictionary
        """
        # Override config with provided parameters
        if epochs is not None:
            self.config["epochs"] = epochs
        if batch_size is not None:
            self.config["batch_size"] = batch_size

        # Resume from checkpoint if provided
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            self._load_checkpoint(resume_from_checkpoint)

        # Setup data loaders
        train_loader, val_loader, test_loader = self.validation_subset.get_loaders(
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"]
        )

        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

        # Create checkpoint directory
        os.makedirs(os.path.dirname(self.config["checkpoint_path"]), exist_ok=True)

        # Training loop
        for epoch in range(self.current_epoch, self.config["epochs"]):
            self.current_epoch = epoch

            # Train for one epoch
            train_loss = self._train_epoch(train_loader, optimizer, epoch)

            # Validate
            val_loss = self._validate_epoch(val_loader, epoch)

            # Update best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(optimizer, is_best=True)

            # Save regular checkpoint
            if epoch % 5 == 0:
                self._save_checkpoint(optimizer, is_best=False)

            # Update training history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            
            # Track individual loss components from the last batch of the epoch
            # We'll get these from the validation pass
            val_outputs = None
            val_loss_dict = None
            with torch.no_grad():
                for bigram_counts, probabilities in val_loader:
                    bigram_counts = bigram_counts.to(self.device)
                    probabilities = probabilities.to(self.device)
                    val_outputs = self.model(bigram_counts)
                    _, val_loss_dict = self.model.compute_loss(probabilities, val_outputs)
                    break  # Just get the first batch for loss component tracking
            
            if val_loss_dict:
                self.training_history["rec_loss"].append(val_loss_dict["rec_loss"].item())
                self.training_history["kl_loss"].append(val_loss_dict["kl_g0"].item() + val_loss_dict["kl_u"].item())
                self.training_history["l2_loss"].append(val_loss_dict["l2_loss"].item())

            print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return self.training_history

    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                    epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # KL weight scheduling
        kl_weight = 0.0
        if epoch >= self.config["kl_start_epoch"]:
            if epoch >= self.config["kl_full_epoch"]:
                kl_weight = 1.0
            else:
                kl_weight = logistic_kl_weight(epoch, k=0.005, c=self.config["kl_full_epoch"])

        progress_bar = tqdm(train_loader, disable=self.config["disable_tqdm"])

        for batch_idx, (bigram_counts, probabilities) in enumerate(progress_bar):
            try:
                bigram_counts = bigram_counts.to(self.device)
                probabilities = probabilities.to(self.device)

                # Forward pass
                outputs = self.model(bigram_counts)
                _, loss_dict = self.model.compute_loss(probabilities, outputs)
                
                # Extract individual loss components
                rec_loss = loss_dict["rec_loss"]
                kl_g0 = loss_dict["kl_g0"]
                kl_u = loss_dict["kl_u"]
                l2_reg = loss_dict["l2_loss"]
                
                # Start with reconstruction loss only
                total_loss_batch = rec_loss
                
                # Conditionally add other loss components based on config
                if self.config.get("enable_kl_loss", True):
                    kl_weight = 0.0
                    if epoch >= self.config.get("kl_start_epoch", 2):
                        if epoch >= self.config.get("kl_full_epoch", 10):
                            kl_weight = 1.0
                        else:
                            kl_weight = logistic_kl_weight(epoch, k=0.005, c=self.config.get("kl_full_epoch", 10))
                    total_loss_batch += kl_weight * (kl_g0 + kl_u)
                
                if self.config.get("enable_l2_loss", True):
                    total_loss_batch += l2_reg
                
                if self.config.get("enable_l1_loss", True):
                    l1_lambda = self.config.get("l1_lambda", 0.0001)
                    l1_reg = 0.0  # L1 regularization not implemented yet
                    total_loss_batch += l1_lambda * l1_reg

                # Backward pass
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()

                total_loss += total_loss_batch.item()
                num_batches += 1

                # Update progress bar
                progress_bar.set_description(
                    f"Epoch {epoch+1} - Loss: {total_loss_batch.item():.4f}"
                )

                # Print batch logs
                if batch_idx % self.config["print_every"] == 0:
                    print(f"Batch {batch_idx}: Loss = {total_loss_batch.item():.4f}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.memory_manager.clear_cache()
                    new_batch_size = self.memory_manager.adjust_batch_size(
                        self.config["batch_size"]
                    )
                    print(f"Memory error, reducing batch size to {new_batch_size}")
                    return total_loss / max(num_batches, 1)
                else:
                    raise e

        return total_loss / max(num_batches, 1)

    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for bigram_counts, probabilities in val_loader:
                bigram_counts = bigram_counts.to(self.device)
                probabilities = probabilities.to(self.device)

                outputs = self.model(bigram_counts)
                _, loss_dict = self.model.compute_loss(probabilities, outputs)
                
                # Extract individual loss components
                rec_loss = loss_dict["rec_loss"]
                kl_g0 = loss_dict["kl_g0"]
                kl_u = loss_dict["kl_u"]
                l2_reg = loss_dict["l2_loss"]
                
                # Start with reconstruction loss only
                val_total_loss = rec_loss
                
                # Conditionally add other loss components based on config
                if self.config.get("enable_kl_loss", True):
                    kl_weight = 0.0
                    if epoch >= self.config.get("kl_start_epoch", 2):
                        if epoch >= self.config.get("kl_full_epoch", 10):
                            kl_weight = 1.0
                        else:
                            kl_weight = logistic_kl_weight(epoch, k=0.005, c=self.config.get("kl_full_epoch", 10))
                    val_total_loss += kl_weight * (kl_g0 + kl_u)
                
                if self.config.get("enable_l2_loss", True):
                    val_total_loss += l2_reg
                
                if self.config.get("enable_l1_loss", True):
                    l1_lambda = self.config.get("l1_lambda", 0.0001)
                    l1_reg = 0.0  # L1 regularization not implemented yet
                    val_total_loss += l1_lambda * l1_reg

                total_loss += val_total_loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, optimizer: optim.Optimizer, is_best: bool = False) -> None:
        """Save model checkpoint."""
        # --- PATCH START ---
        # Add model architecture parameters to config['model']
        # Get encoder and controller dimensions from the GRU layers
        encoder_dim = self.model.g0_encoder.hidden_size
        controller_dim = self.model.controller_rnn.hidden_size
        
        self.config['model'] = {
            'alphabet_size': self.model.alphabet_size,
            'order': self.model.order,
            'encoder_dim': encoder_dim,
            'controller_dim': controller_dim,
            'generator_dim': self.model.generator_dim,
            'factor_dim': self.model.factor_dim,
            'latent_dim': self.model.latent_dim,
            'inferred_input_dim': self.model.inferred_input_dim,
            'kappa': self.model.kappa,
            'ar_step_size': self.model.ar_step_size,
            'ar_process_var': self.model.ar_process_var,
        }
        # --- PATCH END ---
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "config": self.config,
            "test_indices": self.validation_subset.test_indices,  # Save test indices for evaluation
            "val_indices": self.validation_subset.val_indices,     # Save val indices for consistency
            "train_indices": self.validation_subset.train_indices   # Save train indices for completeness
        }

        # Save regular checkpoint
        torch.save(checkpoint, self.config["checkpoint_path"])

        # Save best checkpoint
        if is_best:
            best_path = self.config["checkpoint_path"].replace(".pt", "_best.pt")
            torch.save(checkpoint, best_path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]

        # Load indices if available (for compatibility with older checkpoints)
        if "test_indices" in checkpoint:
            self.validation_subset.test_indices = checkpoint["test_indices"]
        if "val_indices" in checkpoint:
            self.validation_subset.val_indices = checkpoint["val_indices"]
        if "train_indices" in checkpoint:
            self.validation_subset.train_indices = checkpoint["train_indices"]

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def plot_training_history(self, save_path: str | None = None) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Training and validation loss
        axes[0, 0].plot(self.training_history["train_loss"], label="Train")
        axes[0, 0].plot(self.training_history["val_loss"], label="Validation")
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend()

        # Reconstruction loss
        if "rec_loss" in self.training_history:
            axes[0, 1].plot(self.training_history["rec_loss"])
            axes[0, 1].set_title("Reconstruction Loss")

        # KL loss
        if "kl_loss" in self.training_history:
            axes[1, 0].plot(self.training_history["kl_loss"])
            axes[1, 0].set_title("KL Loss")

        # L2 loss
        if "l2_loss" in self.training_history:
            axes[1, 1].plot(self.training_history["l2_loss"])
            axes[1, 1].set_title("L2 Loss")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()
