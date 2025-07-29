############################################################
# birdsong_training_2.py
############################################################
import math
import os

import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim

# Import your custom LFADS-style model
from birdsong_lfads_model_2 import (
    BirdsongLFADSModel2,
    rowwise_masked_softmax,
    rowwise_softmax,
)
from torch.cuda import empty_cache, memory_summary
from torch.utils.data import DataLoader, Subset

# matplotlib.use('Agg')  # Uncomment if no display available
from tqdm import tqdm


def logistic_kl_weight(epoch, k=0.005, c=1000):
    """
    Returns a logistic weighting factor from near 0 to near 1,
    where epoch is the current epoch (int),
    k is the 'sharpness' (float),
    c is the 'center' (float).
    """
    return 1.0 / (1.0 + math.exp(-k * (epoch - c)))

class ValidationSubset:
    """
    A utility class for handling dataset subsets and rescaling metrics.
    """

    def __init__(self, dataset, val_split=0.15, test_split=0.1):
        """
        Creates train, validation, and test subsets from the dataset.
        :param dataset: The full dataset to split.
        :param val_split: Proportion of data to be used for validation.
        :param test_split: Proportion of data to be used for testing.
        """
        self.dataset = dataset
        self.val_split = val_split
        self.test_split = test_split
        self.train_indices, self.val_indices, self.test_indices = self._split_dataset()

    def _split_dataset(self):
        total_len = len(self.dataset)
        test_len = int(total_len * self.test_split)
        val_len = int((total_len - test_len) * self.val_split)
        total_len - test_len - val_len

        indices = torch.randperm(total_len).tolist()
        test_indices = indices[:test_len]
        val_indices = indices[test_len:test_len + val_len]
        train_indices = indices[test_len + val_len:]
        return train_indices, val_indices, test_indices

    def get_loaders(self, batch_size, num_workers=0, pin_memory=False):
        """
        Returns train, validation, and test DataLoaders.
        """
        train_loader = DataLoader(Subset(self.dataset, self.train_indices), batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(Subset(self.dataset, self.val_indices), batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(Subset(self.dataset, self.test_indices), batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, val_loader, test_loader


class MemoryManager:
    """
    A class to manage GPU memory during training, including clearing the CUDA cache,
    reporting memory usage, and adjusting batch size dynamically.
    """

    def __init__(self, device):
        self.device = device

    def clear_cache(self):
        """Clear the CUDA memory cache."""
        if self.device.type == "cuda":
            empty_cache()

    def report_memory(self):
        """Report memory usage on the GPU."""
        if self.device.type == "cuda":
            print(memory_summary(device=self.device, abbreviated=False))

    def adjust_batch_size(self, current_batch_size, factor=0.5):
        """
        Halve the batch size when a memory error occurs.

        :param current_batch_size: Current batch size being used.
        :param factor: Factor by which the batch size will be reduced.
        :return: Adjusted batch size.
        """
        new_batch_size = max(1, int(current_batch_size * factor))
        print(f"Adjusting batch size from {current_batch_size} to {new_batch_size} due to memory constraints.")
        return new_batch_size


def plot_transition_plots(test_loader, model, processes_to_plot, plot_dir, device):
    """
    Plots transition matrices for specified processes in the test set and saves the plots.

    :param test_loader: DataLoader for the test dataset.
    :param model: Trained Birdsong LFADS model.
    :param processes_to_plot: List of test process indices to plot.
    :param plot_dir: Directory to save the plots.
    :param device: Device on which to perform model inference.
    """
    model.eval()
    with torch.no_grad():
        for process_idx in processes_to_plot:
            for _batch_idx, (bigram_counts, probabilities) in enumerate(test_loader):
                bigram_counts = bigram_counts.to(device)
                probabilities = probabilities.to(device)

                if process_idx < bigram_counts.size(0):
                    bigram_counts_process = bigram_counts[process_idx: process_idx + 1]
                    probabilities_process = probabilities[process_idx: process_idx + 1]

                    outputs = model(bigram_counts_process)
                    B, T, out_dim = outputs["logits"].shape
                    alpha = model.alphabet_size

                    if model.order == 1:
                        pred_probs = rowwise_softmax(outputs["logits"], alpha, order=model.order)
                        pred_mat = pred_probs.view(B, T, alpha, alpha)
                        true_mat = probabilities_process.view(B, T, alpha, alpha)
                    elif model.order == 2:
                        true_mat = probabilities_process.view(B, T, alpha ** 2, alpha)
                        target_matrix = true_mat[0, 0]
                        pred_probs = rowwise_masked_softmax(outputs["logits"], alpha, order=model.order,
                                                            target_matrix=target_matrix, mask_value=-1e8)
                        pred_mat = pred_probs.view(B, T, alpha ** 2, alpha)
                    else:
                        raise ValueError("Unsupported order: use order==1 or order==2.")

                    # Plotting the transition matrices
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    sns.heatmap(true_mat[0, 0].cpu().numpy(), ax=axes[0], cmap="viridis", cbar=True)
                    axes[0].set_title("True Transition Matrix")
                    sns.heatmap(pred_mat[0, 0].cpu().numpy(), ax=axes[1], cmap="viridis", cbar=True)
                    axes[1].set_title("Predicted Transition Matrix")
                    fig.suptitle(f"Process {process_idx}, Time Step 0")
                    plt.tight_layout()

                    # Save the figure
                    file_path = os.path.join(plot_dir, f"process_{process_idx}_transition_plot.png")
                    plt.savefig(file_path)
                    plt.close()
                    break  # Stop after processing the requested process.


def train_birdsong_lfads(
        model,
        dataset,
        batch_size=32,
        epochs=20,
        kl_start_epoch=2,
        kl_full_epoch=10,
        test_split=0.1,  # Proportion of the dataset to reserve for testing
        lr=1e-3,
        train_split=0.8,
        checkpoint_path="checkpoints/birdsong_lfads_rowwise.pt",
        resume_from_checkpoint=None,
        device=None,
        print_every=10,  # print batch logs every N batches
        l1_lambda=0.0001,  # L1 regularization strength
        plot_dir="Process Plots",  # Directory for saving visualization plots
        enable_kl_loss=True,  # Toggle KL loss
        enable_l2_loss=True,  # Toggle L2 loss
        enable_l1_loss=True,  # Toggle L1 loss
        disable_tqdm=False   # Option to disable tqdm progress bars
):
    os.makedirs(plot_dir, exist_ok=True)  # Ensure plot directory exists
    memory_manager = MemoryManager(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    """
    Train the Birdsong LFADS model using a dataset of bigram counts and probabilities.
    This function performs training and validation, applies a KL divergence weight
    ramping schedule, and computes various losses including reconstruction loss, KL
    loss, L1 regularization, and L2 regularization. The training routine dynamically
    adjusts learning rates and logs performance metrics during training.

    :param model: The instance of the Birdsong LFADS model to train.
    :param dataset: The dataset containing bigram counts and probabilities for training.
    :param batch_size: The number of samples per batch during training and validation.
    :param epochs: The number of training epochs.
    :param kl_start_epoch: The epoch at which the KL divergence weight starts ramping up.
    :param kl_full_epoch: The epoch by which the KL divergence weight reaches its full value.
    :param lr: The initial learning rate for the optimizer.
    :param train_split: The fraction of the dataset to be used for training.
    :param checkpoint_path: The file path for saving the model checkpoints.
    :param device: The device (`'cuda'` or `'cpu'`) on which the model will be trained. If None,
        the device will be automatically assigned based on availability of a GPU.
    :param print_every: The interval of training batches at which logs will be printed.
    :param l1_lambda: The strength of L1 regularization to apply during model training.
    :return: None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ensure checkpoints directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # --- Split dataset into train/val/test ---
    val_manager = ValidationSubset(dataset, val_split=1 - train_split - test_split, test_split=test_split)
    train_loader, val_loader, test_loader = val_manager.get_loaders(batch_size=batch_size, num_workers=4,
                                                                    pin_memory=True)

    print(f"Dataset split: train={len(val_manager.train_indices)}, "
          f"val={len(val_manager.val_indices)}, test={len(val_manager.test_indices)}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # We'll remove the built-in scheduler and use our custom scheduler below.
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Setup live plotting
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Subplots for losses and accuracies

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    patience, patience_counter = 10, 0

    # Separate plots for loss and accuracy
    loss_ax, acc_ax = axes
    loss_ax.set_title("Loss (Train vs Validation)")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_line_train, = loss_ax.plot([], [], label="Train Loss", color="blue")
    loss_line_val, = loss_ax.plot([], [], label="Val Loss", color="orange")
    loss_ax.legend(loc="upper right")

    acc_ax.set_title("Accuracy (Train vs Validation)")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_line_train, = acc_ax.plot([], [], label="Train Acc", color="green")
    acc_line_val, = acc_ax.plot([], [], label="Val Acc", color="red")
    acc_ax.legend(loc="lower right")

    loss_line_train, = loss_ax.plot([], [], label="Train Loss", color="blue")
    loss_line_val, = loss_ax.plot([], [], label="Val Loss", color="orange")
    acc_line_train, = acc_ax.plot([], [], label="Train Acc", color="green")
    acc_line_val, = acc_ax.plot([], [], label="Val Acc", color="red")

    # Initialize a cooldown counter for LR scheduling:
    cooldown_counter = 0

    start_epoch = 0
    best_val_loss = float('inf')

    # Load best model if resuming training
    if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(
            f"  [*] Resumed training from checkpoint at epoch {start_epoch}, with best validation loss {best_val_loss:.4f}")

    for epoch in range(start_epoch, start_epoch + epochs):
        # Print the current epoch at the beginning
        print(f"\nEpoch [{epoch+1}/{start_epoch + epochs}]")

        # --- Decide how much KL weighting to apply this epoch ---
        if epoch < kl_start_epoch:
            kl_weight = 0.0
        elif epoch >= kl_full_epoch:
            kl_weight = 1.0
        else:
            kl_weight = (epoch - kl_start_epoch) / float(kl_full_epoch - kl_start_epoch)
        # Optionally, override with a logistic ramp
        kl_weight = logistic_kl_weight(epoch)

        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train   = 0

        # Loop over training batches
        train_bar = tqdm(enumerate(train_loader, start=1), desc="Training Batches", unit="batch",
                         total=len(train_loader), leave=False, dynamic_ncols=True, disable=disable_tqdm)
        for batch_idx, (bigram_counts, probabilities) in train_bar:
            try:
                bigram_counts = bigram_counts.to(device)  # (B, T, alpha^2)
                probabilities = probabilities.to(device)  # (B, T, alpha^2)

                outputs = model(bigram_counts)
                base_loss, loss_dict = model.compute_loss(probabilities, outputs)
                rec_loss = loss_dict["rec_loss"]
                kl_g0 = loss_dict["kl_g0"]
                kl_u = loss_dict["kl_u"]
                l2_reg = loss_dict["l2_loss"]  # Compute L2 regularization loss

                total_loss = rec_loss

                if enable_kl_loss:
                    total_loss += kl_weight * (kl_g0 + kl_u)

                if enable_l2_loss:
                    total_loss += l2_reg

                if enable_l1_loss:
                    # l1_reg = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                    l1_reg = 0.0
                    total_loss += l1_lambda * l1_reg

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_train_loss += total_loss.item()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    memory_manager.clear_cache()
                    batch_size = memory_manager.adjust_batch_size(batch_size)
                    if batch_size > len(val_manager.train_indices):
                        batch_size = len(val_manager.train_indices)
                        print(f"[INFO] Reducing batch size to {batch_size} due to small dataset size.")
                    train_loader = DataLoader(Subset(dataset, val_manager.train_indices), batch_size=batch_size,
                                              shuffle=True, num_workers=4, pin_memory=True)
                    print(f"Restarting batch {batch_idx} with reduced batch size...")
                else:
                    raise e
            # bigram_counts = bigram_counts.to(device)   # (B, T, alpha^2)
            # probabilities = probabilities.to(device)   # (B, T, alpha^2)
            #
            # outputs = model(bigram_counts)
            # base_loss, loss_dict = model.compute_loss(probabilities, outputs)
            # rec_loss = loss_dict["rec_loss"]
            # kl_g0    = loss_dict["kl_g0"]
            # kl_u     = loss_dict["kl_u"]
            # l2_reg = loss_dict["l2_loss"] # Compute L2 regularization loss
            #
            #
            # total_loss = rec_loss  + kl_weight * (kl_g0 + kl_u)
            # total_loss += l1_lambda * l2_reg * kl_weight
            #
            # # Compute L1 regularization loss
            # # l1_reg = sum(torch.sum(torch.abs(param)) for param in model.parameters())
            # l1_reg = 0.0
            # total_loss += l1_lambda * l1_reg
            #
            # optimizer.zero_grad()
            # total_loss.backward()
            # optimizer.step()
            #
            # total_train_loss += total_loss.item()

            # Compute training accuracy for this batch
            with torch.no_grad():
                B, T, out_dim = outputs["logits"].shape
                alpha = model.alphabet_size
                if model.order == 1:
                    pred_probs = rowwise_softmax(outputs["logits"], alpha, order=model.order)
                    pred_mat = pred_probs.view(B, T, alpha, alpha)
                    true_mat = probabilities.view(B, T, alpha, alpha)
                elif model.order == 2:
                    true_mat = probabilities.view(B, T, alpha ** 2, alpha)
                    target_matrix = true_mat[0, 0, :, :]
                    pred_probs = rowwise_masked_softmax(outputs["logits"], alpha, order=model.order,
                                                        target_matrix=target_matrix, mask_value=-1e8)
                    pred_mat = pred_probs.view(B, T, alpha ** 2, alpha)
                else:
                    raise ValueError("Unsupported order: use order==1 or order==2.")

                pred_classes = pred_mat.argmax(dim=-1)
                true_classes = true_mat.argmax(dim=-1)

                correct_train += (pred_classes == true_classes).sum().item()
                total_train   += pred_classes.numel()

            if (batch_idx % max(1, len(train_loader) // 10)) == 0:
                train_bar.set_postfix({
                    "Rec": rec_loss.item(),
                    "KL_g0": kl_g0.item() if enable_kl_loss else 0.0,
                    "KL_u": kl_u.item() if enable_kl_loss else 0.0,
                    "L2_Reg": l2_reg.item() if enable_l2_loss else 0.0,
                    "Total": total_loss.item(),
                    "BatchAcc": (pred_classes == true_classes).float().mean().item()
                })
        #L1_Reg={l1_reg.item():.4f},
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train if total_train > 0 else 0.0

        # --- Validation pass ---
        model.eval()
        val_loss_sum = 0.0
        correct_val = 0
        total_val   = 0

        with torch.no_grad():
            for bigram_counts, probabilities in tqdm(val_loader, desc="Validation Batches", unit="batch",
                                                     leave=False, dynamic_ncols=True, disable=disable_tqdm):
                bigram_counts = bigram_counts.to(device)
                probabilities = probabilities.to(device)

                outputs = model(bigram_counts)
                base_loss, loss_dict = model.compute_loss(probabilities, outputs)
                rec_loss = loss_dict["rec_loss"]
                kl_g0    = loss_dict["kl_g0"]
                kl_u     = loss_dict["kl_u"]
                l2_reg = loss_dict["l2_loss"]  # Compute L2 regularization loss

                val_total_loss = rec_loss

                if enable_kl_loss:
                    val_total_loss += kl_weight * (kl_g0 + kl_u)

                if enable_l2_loss:
                    val_total_loss += l2_reg

                if enable_l1_loss:
                    # l1_reg = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                    l1_reg = 0.0
                    val_total_loss += l1_lambda * l1_reg

                val_loss_sum += val_total_loss.item()

                B, T, out_dim = outputs["logits"].shape
                alpha = model.alphabet_size
                if model.order == 1:
                    pred_probs = rowwise_softmax(outputs["logits"], alpha, order=model.order)
                    pred_mat = pred_probs.view(B, T, alpha, alpha)
                    true_mat = probabilities.view(B, T, alpha, alpha)
                elif model.order == 2:
                    true_mat = probabilities.view(B, T, alpha ** 2, alpha)
                    target_matrix = true_mat[0, 0, :, :]
                    pred_probs = rowwise_masked_softmax(outputs["logits"], alpha, order=model.order,
                                                        target_matrix=target_matrix, mask_value=-1e8)
                    pred_mat = pred_probs.view(B, T, alpha ** 2, alpha)
                else:
                    raise ValueError("Unsupported order: use order==1 or order==2.")

                pred_classes = pred_mat.argmax(dim=-1)
                true_classes = true_mat.argmax(dim=-1)
                correct_val += (pred_classes == true_classes).sum().item()
                total_val   += pred_classes.numel()

        avg_val_loss = val_loss_sum / len(val_loader) if len(val_loader) else 0.0
        val_acc = correct_val / total_val if total_val > 0 else 0.0

        print(f"Epoch Summary: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}, KL_weight={kl_weight:.4f}")

        # --- Custom Learning Rate Scheduler ---
        # If we have at least 6 previous epochs and no cooldown, check if the current training loss
        # is higher than the maximum loss of the previous 6 epochs.
        if len(train_loss_history) >= 6 and cooldown_counter == 0:
            if avg_train_loss > max(train_loss_history[-6:]):
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.95
                    new_lr = param_group["lr"]
                print(f"Decaying learning rate to {new_lr:.6f} at epoch {epoch+1}")
                cooldown_counter = 6  # Wait 6 epochs before next decay

        # Decrement cooldown counter if active.
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # Check if learning rate has fallen to or below threshold.
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr <= 1e-5:
            print(f"Learning rate reached {current_lr:.6f} at epoch {epoch+1}, stopping training.")
            break

        # Early stopping
        if avg_val_loss > min(val_loss_history[-patience:], default=float('inf')):
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break
        else:
            patience_counter = 0

    # Save best model based on validation loss.
    test_loss = 0.0  # Initialize test loss for conditionally evaluating the test set
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0  # Reset patience counter on improvement
        # Create model_params from the model's attributes for metadata
        model_params = {
            "alphabet_size": model.alphabet_size,
            "order": model.order,
            "encoder_dim": model.encoder_dim if hasattr(model, 'encoder_dim') else None,
            "controller_dim": model.controller_dim if hasattr(model, 'controller_dim') else None,
            "generator_dim": model.generator_dim if hasattr(model, 'generator_dim') else None,
            "factor_dim": model.factor_dim if hasattr(model, 'factor_dim') else None,
            "latent_dim": model.latent_dim if hasattr(model, 'latent_dim') else None,
            "inferred_input_dim": model.inferred_input_dim if hasattr(model, 'inferred_input_dim') else None,
            "kappa": model.kappa if hasattr(model, 'kappa') else None,
            "ar_step_size": model.ar_step_size if hasattr(model, 'ar_step_size') else None,
            "ar_process_var": model.ar_process_var if hasattr(model, 'ar_process_var') else None
        }

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'layer_metadata': model_params,
            'total_epochs': start_epoch + epochs
        }, checkpoint_path)
        print(f"  [*] Best model saved to {checkpoint_path}, val_loss={best_val_loss:.4f}")

        # # Evaluate the model on the test set after saving the best model
        # model.eval()
        # correct_test, total_test = 0, 0
        # test_loss = 0.0
        #
        # with torch.no_grad():
        #     for bigram_counts, probabilities in test_loader:
        #         bigram_counts = bigram_counts.to(device)
        #         probabilities = probabilities.to(device)
        #
        #         # Compute predictions and losses
        #         outputs = model(bigram_counts)
        #         _, loss_dict = model.compute_loss(probabilities, outputs)
        #         test_loss += loss_dict["rec_loss"].item()
        #
        #         # Compute test accuracy
        #         B, T, out_dim = outputs["logits"].shape
        #         alpha = model.alphabet_size
        #         if model.order == 1:
        #             pred_probs = rowwise_softmax(outputs["logits"], alpha, order=model.order)
        #             pred_mat = pred_probs.view(B, T, alpha, alpha)
        #             true_mat = probabilities.view(B, T, alpha, alpha)
        #         elif model.order == 2:
        #             true_mat = probabilities.view(B, T, alpha ** 2, alpha)
        #             target_matrix = true_mat[0, 0, :, :]
        #             pred_probs = rowwise_masked_softmax(outputs["logits"], alpha, order=model.order,
        #                                                 target_matrix=target_matrix, mask_value=-1e8)
        #             pred_mat = pred_probs.view(B, T, alpha ** 2, alpha)
        #         else:
        #             raise ValueError("Unsupported order: use order==1 or order==2.")
        #
        #         pred_classes = pred_mat.argmax(dim=-1)
        #         true_classes = true_mat.argmax(dim=-1)
        #         correct_test += (pred_classes == true_classes).sum().item()
        #         total_test += pred_classes.numel()
        #
        # # Compute average test loss and accuracy
        # avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
        # test_acc = correct_test / total_test if total_test > 0 else float('nan')
        # print(f"Test set evaluation: Loss={avg_test_loss:.4f}, Accuracy={test_acc:.3f}")

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        acc_line_train.set_data(range(len(train_acc_history)), train_acc_history)
        acc_line_val.set_data(range(len(val_acc_history)), val_acc_history)
        acc_ax.relim()
        acc_ax.autoscale_view()
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    # === Final evaluation on Test Set ===
    # (make sure test_loader was created alongside train_loader & val_loader)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_loss = 0.0
    correct   = 0
    total     = 0

    with torch.no_grad():
        for bigram_counts, probabilities in test_loader:
            bigram_counts = bigram_counts.to(device)    # (B, T, αⁿ)
            probabilities = probabilities.to(device)    # (B, T, αⁿ)

            # forward + loss
            outputs = model(bigram_counts)
            _, loss_dict = model.compute_loss(probabilities, outputs)
            rec_loss = loss_dict["rec_loss"]
            kl_g0    = loss_dict["kl_g0"]
            kl_u     = loss_dict["kl_u"]
            # assume full KL weight at end of training
            loss = rec_loss # + (kl_g0 + kl_u)
            test_loss += loss.item()

            # --- accuracy ---
            B, T, _ = outputs["logits"].shape
            alpha   = model.alphabet_size

            if model.order == 1:
                # bigram case
                pred_probs = rowwise_softmax(
                    outputs["logits"],
                    alpha,
                    order=model.order
                )
                pred_mat = pred_probs.view(B, T, alpha, alpha)
                true_mat = probabilities.view(B, T, alpha, alpha)

            elif model.order == 2:
                # trigram case
                # probabilities shaped (B, T, α³) ⇒ view as (B, T, α², α)
                true_mat = probabilities.view(B, T, alpha**2, alpha)

                # we need a “target matrix” of valid α×α² transitions
                # here we just take it from the first sample & time‐step
                target_matrix = true_mat[0, 0, :, :]

                pred_probs = rowwise_masked_softmax(
                    outputs["logits"],
                    alpha,
                    order=model.order,
                    target_matrix=target_matrix,
                    mask_value=-1e8
                )
                pred_mat = pred_probs.view(B, T, alpha**2, alpha)

            else:
                raise ValueError("Unsupported order: use order==1 or order==2.")

            preds = pred_mat.argmax(dim=-1)
            trues = true_mat.argmax(dim=-1)
            correct += (preds == trues).sum().item()
            total   += preds.numel()

    avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
    test_acc      = correct / total
    if len(test_loader) == 0:
        print("[WARNING] Skipping test evaluation as test set is empty.")
    else:
        print(f"*** Test Loss = {avg_test_loss:.4f},  Test Acc = {test_acc:.3f} ***")

    # Visualize transition plots for 2 processes in the test set
    processes_to_plot = [0, 1]  # Plot first two processes
    plot_dir = "transition_plots"  # Directory to save plots
    os.makedirs(plot_dir, exist_ok=True)
    plot_transition_plots(test_loader, model, processes_to_plot, plot_dir, device)


if __name__ == "__main__":
    order = 1

    import h5py
    h5_path = "aggregated_datasets/aggregated_birdsong_data_10_syl_linear_100_song_in_batch_25_timesteps_20250505_122822.h5"

    # Get shape info from HDF5
    with h5py.File(h5_path, "r") as hf:
        alphabet_size = math.ceil(hf["bigram_counts"].shape[0] ** (1 / (order + 1)))
        time_steps = hf["bigram_counts"].shape[1]
        total_samples = hf["bigram_counts"].shape[2]


    from birdsong_data_loader import BirdsongDataset
    dataset = BirdsongDataset(h5_path)
    print(f"Loaded dataset with alphabet_size={alphabet_size}, time_steps={time_steps}, total_samples={total_samples}")

    model_params = {
        "alphabet_size": alphabet_size,
        "order": order,
        "encoder_dim": 256,
        "controller_dim": 4,
        "generator_dim": 256,
        "factor_dim": 338,
        "latent_dim": 128,  # MUST BE EVEN
        "inferred_input_dim": 4,
        "kappa": 1.0,
        "ar_step_size": 0.99,
        "ar_process_var": 0.1
    }

    model = BirdsongLFADSModel2(**model_params)

    # Option to toggle resume training on or off
    resume_training = True
    resume_from_checkpoint = (
        f"checkpoints/{os.path.splitext(os.path.basename(h5_path))[0]}.pt"
        if resume_training else None
    )

    train_birdsong_lfads(
        model=model,
        dataset=dataset,
        batch_size=64,
        epochs=1,
        kl_start_epoch=5,
        kl_full_epoch=1000,
        lr=1e-3,
        train_split=0.8,
        test_split=0.1,
        checkpoint_path=f"checkpoints/{os.path.splitext(os.path.basename(h5_path))[0]}.pt",
        resume_from_checkpoint=resume_from_checkpoint,
        print_every=10,
        l1_lambda=1e-1,
        enable_kl_loss=False,  # Toggle KL loss
        enable_l2_loss=False,  # Toggle L2 loss
        enable_l1_loss=False,  # Toggle L1 loss
        disable_tqdm=False     # Set to True if tqdm progress bars still cause issues
    )
