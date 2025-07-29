# import os
# import torch
# import math
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# import torch.nn.functional as F
# import matplotlib.cm as cm
#
# from birdsong_lfads_model_2 import BirdsongLFADSModel2, rowwise_softmax, rowwise_masked_softmax
# from BirdsongAEModel import BirdsongAEModel
# from birdsong_data_loader import BirdsongDataset
# from birdsong_data_generation import simulate_birdsong, x_init_maker
#
# ###############################################################################
# # Configuration variables
# ###############################################################################
# # PROCESS_TO_RUN: Use 'all' to run over the entire dataset or a comma-separated string
# # (e.g. "0,1") to run only those processes.
# PROCESS_TO_RUN =  "10,11,12"
#
# FILE_FORMAT = 'png'  # or  'png'
#
# # Smoothing configuration: SMOOTH_WINDOW = 1 means raw estimate; >1 applies a running average.
# SMOOTH_WINDOW = 5  # change as desired; 1 means no smoothing
# PLOT_SMOOTH_EST = True  # if True and SMOOTH_WINDOW > 1, then plot smooth estimates
#
# SAVE_INDIVIDUAL_PLOTS = True # Save each transition plot rather than displaying it.
# BASE_OUTPUT_DIR = 'Process Plots'
# SUMMARY_OUTPUT_DIR = 'summary plots'
#
# ###############################################################################
# # Helper Functions for Evaluation Metrics
# ###############################################################################
#
# def rowwise_kl_div(pred_dist, true_dist, eps=1e-9):
#     """
#     Computes rowwise KL divergence:
#       KL(true || pred) = sum(true * log(true/pred))
#     Both pred_dist and true_dist should be tensors of shape (..., K),
#     where for order==1 K = α² and for order==2 K = α³.
#     """
#     pred_dist = torch.clamp(pred_dist, eps, 1.0)
#     true_dist = torch.clamp(true_dist, eps, 1.0)
#     kl = true_dist * torch.log(true_dist / pred_dist)
#     return kl.sum(dim=-1).mean()
#
#
# def rowwise_mse(pred_dist, true_dist):
#     """
#     Computes rowwise Mean Squared Error.
#     """
#     return ((pred_dist - true_dist) ** 2).mean()
#
#
# ###############################################################################
# # Evaluation Function
# ###############################################################################
#
# def evaluate_birdsong_model(model, input_counts, true_probs, alphabet_size):
#     """
#     Evaluates the model by:
#       1. Forward passing the input n‑gram counts.
#       2. Reshaping the model’s logits according to the order:
#          - For order==1, logits (B, T, α²) → reshape to (B, T, α, α).
#          - For order==2, logits (B, T, α³) → reshape to (B, T, α², α).
#       3. Computing predicted probabilities via softmax (or masked softmax for trigrams).
#       4. Flattening the predictions (and true probabilities) to compute:
#          - Cross‑entropy loss (treating each row as a classification problem).
#          - Accuracy.
#          - Row‑wise KL divergence and MSE.
#
#     Returns:
#       - logits_4d: The raw logits reshaped to 4D.
#       - pred_probs_4d: The predicted probabilities as a 4D tensor.
#       - factors: Any extra factors output by the model.
#       - metrics: A dictionary with computed metrics.
#     """
#     model.eval()
#     with torch.no_grad():
#         outputs = model(input_counts)
#         logits = outputs["logits"]  # For order==1: (B, T, α²); for order==2: (B, T, α³)
#         factors = outputs.get("factors", None)
#         B, T, logit_dim = logits.shape
#         order = model.order  # either 1 or 2
#
#         if order == 1:
#             # Bigram case: reshape logits to (B, T, α, α)
#             logits_4d = logits.view(B, T, alphabet_size, alphabet_size)
#             pred_probs_4d = F.softmax(logits_4d, dim=-1)
#             # Flatten to (B, T, α²) for loss computation.
#             pred_probs_flat = pred_probs_4d.view(B, T, alphabet_size ** 2)
#             true_probs_flat = true_probs.view(B, T, alphabet_size ** 2)
#         elif order == 2:
#             # Trigram case: reshape logits to (B, T, α², α)
#             logits_4d = logits.view(B, T, alphabet_size ** 2, alphabet_size)
#             true_probs_4d = true_probs.view(B, T, alphabet_size ** 2, alphabet_size)
#             target_matrix = true_probs_4d[0,0,:,:]
#             # Use the masked softmax so that rows with all illegal entries remain zeros.
#             pred_probs = rowwise_masked_softmax(logits, alphabet_size, order=order,target_matrix=target_matrix, mask_value=-1e8)
#             pred_probs_4d = pred_probs.view(B, T, alphabet_size ** 2, alphabet_size)
#
#             # Flatten to (B, T, α³) for losses.
#             pred_probs_flat = pred_probs_4d.view(B, T, alphabet_size ** 3)
#             true_probs_flat = true_probs_4d.view(B, T, alphabet_size ** 3)
#         else:
#             raise ValueError("Only order==1 and order==2 are supported.")
#
#         # --- Cross-Entropy Loss ---
#         if order == 1:
#             # There are α rows per time step.
#             pred_flat = pred_probs_4d.view(B * T * alphabet_size, alphabet_size)
#             true_labels = true_probs.view(B, T, alphabet_size, alphabet_size).argmax(dim=-1)
#             true_flat = true_labels.view(B * T * alphabet_size)
#         elif order == 2:
#             # There are α² rows per time step.
#             pred_flat = pred_probs_4d.view(B * T * (alphabet_size ** 2), alphabet_size)
#             true_labels = true_probs.view(B, T, alphabet_size ** 2, alphabet_size).argmax(dim=-1)
#             true_flat = true_labels.view(B * T * (alphabet_size ** 2))
#         ce_loss = F.cross_entropy(pred_flat, true_flat)
#
#         # --- Accuracy ---
#         pred_classes = pred_probs_4d.argmax(dim=-1)  # For order==1: (B, T, α); for order==2: (B, T, α²)
#         true_classes = true_probs.view(B, T, -1, alphabet_size).argmax(dim=-1)
#         correct = (pred_classes == true_classes).sum().item()
#         total = pred_classes.numel()
#         accuracy = correct / total if total > 0 else 0.0
#
#         # --- Rowwise KL and MSE ---
#         kl_val = rowwise_kl_div(pred_probs_flat, true_probs_flat)
#         mse_val = rowwise_mse(pred_probs_flat, true_probs_flat)
#
#         metrics = {
#             "cross_entropy": ce_loss.item(),
#             "accuracy": accuracy,
#             "rowwise_kl": kl_val.item(),
#             "rowwise_mse": mse_val.item(),
#         }
#
#     return logits_4d, pred_probs_4d, factors, metrics
#
#
# ###############################################################################
# # Plotting Functions
# ###############################################################################
#
# def plot_wrate_factors(model, factors, alphabet_size, sequence_idx=0):
#     """
#     Plots the time-averaged (row-wise) output of W_rate * factors as a heatmap,
#     as well as the time evolution of each row.
#
#     For order==1, the output of rate_linear is reshaped to (B, T, α, α).
#     For order==2, it is reshaped to (B, T, α², α).
#     """
#     if factors is None:
#         print("No factors returned by model; cannot plot W_rate*factors.")
#         return
#     try:
#         w_factors = model.rate_linear(factors)  # (B, T, out_dim)
#     except AttributeError:
#         print("Model does not have a rate_linear layer or there's a dimension mismatch.")
#         return
#     w_factors = w_factors.detach()
#     B, T, out_dim = w_factors.shape
#     order = model.order
#     if order == 1:
#         w_factors_4d = w_factors.view(B, T, alphabet_size, alphabet_size)
#     elif order == 2:
#         w_factors_4d = w_factors.view(B, T, alphabet_size ** 2, alphabet_size)
#     else:
#         print("Unsupported order for plotting.")
#         return
#     data_4d = w_factors_4d[sequence_idx].cpu().numpy()  # (T, X, alphabet_size)
#
#     # (A) Time-averaged heatmap.
#     avg_data = data_4d.mean(axis=0)  # (X, alphabet_size)
#     fig1, ax1 = plt.subplots(figsize=(6, 5))
#     im = ax1.imshow(avg_data, cmap='bwr', aspect='auto')
#     ax1.set_title("Mean of [W_rate * factors] across time")
#     fig1.colorbar(im, ax=ax1)
#     plt.tight_layout()
#     plt.show()
#
#     # (B) Time evolution plots.
#     X = avg_data.shape[0]
#     fig2, axs = plt.subplots(1, X, figsize=(4 * X, 4), sharey=True)
#     if X == 1:
#         axs = [axs]
#     for i in range(X):
#         ax = axs[i]
#         row_data = data_4d[:, i, :]  # (T, alphabet_size)
#         for col in range(alphabet_size):
#             ax.plot(row_data[:, col], label=f"col {col}")
#         ax.set_title(f"Row {i}")
#         ax.set_xlabel("Time")
#         if i == 0:
#             ax.set_ylabel("Value")
#         ax.legend()
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_ngram_counts(pred_probs_4d, true_probs, ngram_counts, alphabet_size, order):
#     """
#     Plots the predicted and true rowwise probability distributions alongside the estimated
#     n-gram frequency counts (normalized rowwise) for each row.
#
#     For order==1 (bigrams):
#       - ngram_counts is reshaped to (B, T, α, α).
#     For order==2 (trigrams):
#       - ngram_counts is reshaped to (B, T, α², α).
#
#     This function plots one subplot per row.
#     """
#     B, T = pred_probs_4d.shape[:2]
#     if order == 1:
#         # Reshape counts to (B, T, α, α)
#         counts_4d = ngram_counts.view(B, T, alphabet_size, alphabet_size).cpu().numpy()
#         # Normalize rowwise.
#         counts_norm = np.where(
#             counts_4d.sum(axis=-1, keepdims=True) > 0,
#             counts_4d / counts_4d.sum(axis=-1, keepdims=True),
#             0.0
#         )
#         # For plotting, choose the first batch element.
#         pred_mat = pred_probs_4d[0].cpu().numpy()  # (T, α, α)
#         true_mat = true_probs.view(B, T, alphabet_size, alphabet_size)[0].cpu().numpy()
#         counts_mat = counts_norm[0]  # (T, α, α)
#         num_rows = alphabet_size
#         fig, axs = plt.subplots(1, num_rows, figsize=(5 * num_rows, 5), sharey=True)
#         if num_rows == 1:
#             axs = [axs]
#         cmap = plt.get_cmap('tab10', alphabet_size)
#         colors = [cmap(i) for i in range(alphabet_size)]
#         for row in range(num_rows):
#             ax = axs[row]
#             for col in range(alphabet_size):
#                 ax.plot(pred_mat[:, row, col], '--', color=colors[col], label=f"Pred col={col}")
#                 ax.plot(true_mat[:, row, col], '-', color=colors[col], label=f"True col={col}")
#                 ax.plot(counts_mat[:, row, col], ':', color=colors[col], label=f"Count col={col}")
#             ax.set_title(f"Row {row}")
#             ax.set_xlabel("Time")
#             if row == 0:
#                 ax.set_ylabel("Probability")
#             if row == num_rows - 1:
#                 ax.legend(loc="best")
#         plt.tight_layout()
#         plt.show()
#     elif order == 2:
#         # For trigrams, reshape counts to (B, T, α², α)
#         counts_4d = ngram_counts.view(B, T, alphabet_size ** 2, alphabet_size).cpu().numpy()
#         counts_norm = np.where(
#             counts_4d.sum(axis=-1, keepdims=True) > 0,
#             counts_4d / counts_4d.sum(axis=-1, keepdims=True),
#             0.0
#         )
#         pred_mat = pred_probs_4d[0].cpu().numpy()  # (T, α², α)
#         true_mat = true_probs.view(B, T, alphabet_size ** 2, alphabet_size)[0].cpu().numpy()
#         counts_mat = counts_norm[0]  # (T, α², α)
#         num_contexts = alphabet_size ** 2
#         fig, axs = plt.subplots(1, num_contexts, figsize=(3 * num_contexts, 3), sharey=True)
#         if num_contexts == 1:
#             axs = [axs]
#         cmap = plt.get_cmap('tab10', alphabet_size)
#         colors = [cmap(i) for i in range(alphabet_size)]
#         for ctx in range(num_contexts):
#             ax = axs[ctx]
#             for col in range(alphabet_size):
#                 ax.plot(pred_mat[:, ctx, col], '--', color=colors[col], label=f"Pred col={col}")
#                 ax.plot(true_mat[:, ctx, col], '-', color=colors[col], label=f"True col={col}")
#                 ax.plot(counts_mat[:, ctx, col], ':', color=colors[col], label=f"Count col={col}")
#             ax.set_title(f"Context {ctx}")
#             ax.set_xlabel("Time")
#             if ctx == 0:
#                 ax.set_ylabel("Probability")
#             if ctx == num_contexts - 1:
#                 ax.legend(loc="best")
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("Unsupported order for plotting n-gram counts.")
#
#
# ###############################################################################
# # Utility Functions for Divergence & Cross Entropy
# ###############################################################################
# def js_divergence(p, q, eps=1e-9):
#     """Computes the Jensen–Shannon divergence between two distributions."""
#     p = np.clip(p, eps, 1)
#     q = np.clip(q, eps, 1)
#     m = (p + q) / 2
#     kl_pm = np.sum(p * np.log(p / m), axis=-1)
#     kl_qm = np.sum(q * np.log(q / m), axis=-1)
#     return 0.5 * (kl_pm + kl_qm)
#
#
# def cross_entropy(p, q, eps=1e-9):
#     """Computes the cross entropy between two distributions."""
#     p = np.clip(p, eps, 1)
#     return -np.sum(q * np.log(p), axis=-1)
#
#
# ###############################################################################
# # Helper Functions for Context Strings and Alphabet
# ###############################################################################
# def get_alphabet(alphabet_size):
#     """
#     Returns an alphabet list. Assumes:
#       - The first symbol is '<' (start)
#       - The last symbol is '>' (end)
#       - The in-between symbols are generated from letters starting at 'a'
#     """
#     if alphabet_size < 2:
#         return ['<']
#     num_middle = alphabet_size - 2
#     middle = [chr(ord('a') + i) for i in range(num_middle)]
#     return ['<'] + middle + ['>']
#
#
# def get_context_string(row_idx, alphabet, order):
#     """
#     For order==1, returns the single symbol for this row.
#     For order==2, returns a two-character string computed by mapping the row index
#     to a pair (first = index // alphabet_size, second = index % alphabet_size).
#     """
#     if order == 1:
#         return alphabet[row_idx]
#     elif order == 2:
#         a_size = len(alphabet)
#         first = alphabet[row_idx // a_size]
#         second = alphabet[row_idx % a_size]
#         return first + second
#     else:
#         return str(row_idx)
#
#
# def safe_context_str(context_str):
#     """Replaces characters illegal in filenames."""
#     return context_str.replace('<', 'start').replace('>', 'end')
#
#
# ###############################################################################
# # Helper function to compute a running average (smoothing) over time.
# ###############################################################################
# def smooth_counts(counts, window_size):
#     """
#     Applies a running average over the time dimension.
#     counts: numpy array of shape (T, rows, cols)
#     Returns a smoothed array of the same shape.
#     """
#     if window_size <= 1:
#         return counts
#     T, rows, cols = counts.shape
#     pad = window_size // 2
#     # Use edge padding
#     padded = np.pad(counts, ((pad, pad), (0, 0), (0, 0)), mode='edge')
#     smoothed = np.empty_like(counts)
#     for t in range(T):
#         smoothed[t] = np.mean(padded[t:t + window_size], axis=0)
#     return smoothed
#
#
# ###############################################################################
# # Function: Plot Individual Transition Plots for a Process
# ###############################################################################
# def plot_transition_plots(process_idx, pred_probs_4d, true_probs, raw_est, smooth_est, alphabet_size, order,
#                           dataset_name):
#     """
#     For each row (context), plots a separate figure showing over time:
#       - Predicted probabilities (dashed),
#       - True probabilities (solid),
#       - Raw estimate (dotted),
#       - And if available, a smooth estimate (dash-dot).
#
#     The plot title includes the context and the legend entries are labeled as:
#       "True {context} → {symbol}", "Pred {context} → {symbol}", "Est {context} → {symbol}"
#       and, if applicable, "Smooth Est {context} → {symbol}".
#
#     The legend is placed outside the main plot.
#     Figures are saved into a process-specific directory.
#     """
#     T = pred_probs_4d.shape[0]
#     alphabet = get_alphabet(alphabet_size)
#
#     # Create directory for this process.
#     process_dir_name = f"{dataset_name}_process_{process_idx}"
#     process_dir = os.path.join(BASE_OUTPUT_DIR, process_dir_name)
#     os.makedirs(process_dir, exist_ok=True)
#
#     num_contexts = pred_probs_4d.shape[1]
#     cmap = cm.get_cmap('tab10', alphabet_size)
#
#     for row_idx in range(num_contexts):
#         context_str = get_context_string(row_idx, alphabet, order)
#         safe_ctx = safe_context_str(context_str)
#         plt.figure(figsize=(8, 6))
#         for col_idx in range(alphabet_size):
#             plt.plot(pred_probs_4d[:, row_idx, col_idx], '--', color=cmap(col_idx),
#                      label=f"Pred {context_str} → {alphabet[col_idx]}")
#             plt.plot(true_probs[:, row_idx, col_idx], '-', color=cmap(col_idx),
#                      label=f"True {context_str} → {alphabet[col_idx]}")
#             # plt.plot(raw_est[:, row_idx, col_idx], ':', color=cmap(col_idx),
#             #          label=f"Est {context_str} → {alphabet[col_idx]}")
#             if PLOT_SMOOTH_EST and smooth_est is not None:
#                 plt.plot(smooth_est[:, row_idx, col_idx], '-.', color=cmap(col_idx),
#                          label=f"Smooth Est {context_str} → {alphabet[col_idx]}")
#         plt.title(f"Transition from Context: {context_str}")
#         plt.xlabel("Time Step")
#         plt.ylabel("Probability")
#         # Place the legend outside of the plot.
#         plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.tight_layout()
#         filename = f"transition_{safe_ctx}.{FILE_FORMAT}"
#         plt.savefig(os.path.join(process_dir, filename))
#         plt.close()
#
#
# ###############################################################################
# # Function: Plot Summary Metrics Across Processes
# ###############################################################################
# def plot_summary_metrics(all_js_pred, all_js_est_raw, all_js_est_smooth,
#                          all_ce_pred, all_ce_est_raw, all_ce_est_smooth, file_format, dataset_name):
#     """
#     Plots two summary figures using the mean and standard deviation across processes:
#       - One for mean Jensen–Shannon divergence per time step.
#       - One for mean cross entropy per time step.
#
#     For each metric:
#       - The mean is plotted as a solid line.
#       - The area between (mean - std) and (mean + std) is filled with low opacity.
#
#     The saved filenames include the dataset name.
#
#     Colors used:
#       - Predicted vs. True: Blue.
#       - Raw Estimate vs. True: Red.
#       - Smooth Estimate vs. True: Orange.
#     """
#     # Convert lists to numpy arrays.
#     js_pred_arr = np.array(all_js_pred)  # shape: (n_processes, T)
#     ce_pred_arr = np.array(all_ce_pred)
#
#     x = np.arange(js_pred_arr.shape[1])
#
#     # Jensen-Shannon divergence plot.
#     plt.figure(figsize=(10, 6))
#
#     # Compute mean and std for predicted JS.
#     mean_js_pred = js_pred_arr.mean(axis=0)
#     std_js_pred = js_pred_arr.std(axis=0)
#     plt.plot(x, mean_js_pred, color='blue', linewidth=3, label="Mean Pred JS")
#     plt.fill_between(x, mean_js_pred - std_js_pred, mean_js_pred + std_js_pred,
#                      color='blue', alpha=0.2)
#
#     # Raw estimate JS.
#     if all_js_est_raw is not None:
#         js_est_raw_arr = np.array(all_js_est_raw)
#         mean_js_est_raw = js_est_raw_arr.mean(axis=0)
#         std_js_est_raw = js_est_raw_arr.std(axis=0)
#         plt.plot(x, mean_js_est_raw, color='red', linewidth=3, label="Mean Raw Est JS")
#         plt.fill_between(x, mean_js_est_raw - std_js_est_raw, mean_js_est_raw + std_js_est_raw,
#                          color='red', alpha=0.2)
#
#     # Smooth estimate JS.
#     if all_js_est_smooth is not None:
#         js_est_smooth_arr = np.array(all_js_est_smooth)
#         mean_js_est_smooth = js_est_smooth_arr.mean(axis=0)
#         std_js_est_smooth = js_est_smooth_arr.std(axis=0)
#         plt.plot(x, mean_js_est_smooth, color='orange', linewidth=3, label="Mean Smooth Est JS")
#         plt.fill_between(x, mean_js_est_smooth - std_js_est_smooth, mean_js_est_smooth + std_js_est_smooth,
#                          color='orange', alpha=0.2)
#
#     plt.xlabel("Time Step")
#     plt.ylabel("JS Divergence")
#     plt.title("Mean Jensen–Shannon Divergence per Time Step")
#     plt.legend(loc="best")
#     plt.tight_layout()
#     js_filename = os.path.join(SUMMARY_OUTPUT_DIR, f"{dataset_name}_summary_js_divergence.{file_format}")
#     plt.savefig(js_filename)
#     plt.close()
#
#     # Cross Entropy plot.
#     plt.figure(figsize=(10, 6))
#
#     # Predicted CE.
#     mean_ce_pred = ce_pred_arr.mean(axis=0)
#     std_ce_pred = ce_pred_arr.std(axis=0)
#     plt.plot(x, mean_ce_pred, color='blue', linewidth=3, label="Mean Pred CE")
#     plt.fill_between(x, mean_ce_pred - std_ce_pred, mean_ce_pred + std_ce_pred,
#                      color='blue', alpha=0.2)
#
#     # Raw estimate CE.
#     if all_ce_est_raw is not None:
#         ce_est_raw_arr = np.array(all_ce_est_raw)
#         mean_ce_est_raw = ce_est_raw_arr.mean(axis=0)
#         std_ce_est_raw = ce_est_raw_arr.std(axis=0)
#         plt.plot(x, mean_ce_est_raw, color='red', linewidth=3, label="Mean Raw Est CE")
#         plt.fill_between(x, mean_ce_est_raw - std_ce_est_raw, mean_ce_est_raw + std_ce_est_raw,
#                          color='red', alpha=0.2)
#
#     # Smooth estimate CE.
#     if all_ce_est_smooth is not None:
#         ce_est_smooth_arr = np.array(all_ce_est_smooth)
#         mean_ce_est_smooth = ce_est_smooth_arr.mean(axis=0)
#         std_ce_est_smooth = ce_est_smooth_arr.std(axis=0)
#         plt.plot(x, mean_ce_est_smooth, color='orange', linewidth=3, label="Mean Smooth Est CE")
#         plt.fill_between(x, mean_ce_est_smooth - std_ce_est_smooth, mean_ce_est_smooth + std_ce_est_smooth,
#                          color='orange', alpha=0.2)
#
#     plt.xlabel("Time Step")
#     plt.ylabel("Cross Entropy")
#     plt.title("Mean Cross Entropy per Time Step")
#     plt.legend(loc="best")
#     plt.tight_layout()
#     ce_filename = os.path.join(SUMMARY_OUTPUT_DIR, f"{dataset_name}_summary_cross_entropy.{file_format}")
#     plt.savefig(ce_filename)
#     plt.close()
#
#
# ###############################################################################
# # Main Evaluation Loop
# ###############################################################################
# if __name__ == "__main__":
#     order = 1  # set order to 1 or 2
#
#     # 1) Load dataset.
#     h5_path = "aggregated_datasets/aggregated_birdsong_data_1st_order_6_syl_fourier_20_song_in_batch_50_timesteps_20250312_170533.h5"
#     dataset_name = os.path.splitext(os.path.basename(h5_path))[0]
#     with h5py.File(h5_path, "r") as hf:
#         # For order==2: assume shape is (alphabet_size^(order+1), T)
#         alphabet_size = math.ceil(hf["bigram_counts"].shape[0] ** (1 / (order + 1)))
#         time_steps = hf["bigram_counts"].shape[1]
#         print(f"Loaded dataset with alphabet_size={alphabet_size}, time_steps={time_steps}")
#
#     # 2) Load model checkpoint.
#     checkpoint_path = "checkpoints/birdsong_lfads_2_1st_order_6_syl_fourier_processes_with_kl.pt"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = BirdsongLFADSModel2(
#         alphabet_size=alphabet_size,
#         order=order,
#         encoder_dim=128,
#         controller_dim=128,
#         generator_dim=128,
#         factor_dim=18,
#         latent_dim=18,  # MUST BE EVEN
#         inferred_input_dim=9,
#         kappa=5.0,
#         ar_step_size=0.90,
#         ar_process_var=0.1
#     )
#     ckpt = torch.load(checkpoint_path, map_location=device)  # Consider setting weights_only=True if needed.
#     model.load_state_dict(ckpt["model_state_dict"])
#     model.to(device)
#     model.eval()
#
#     # Create output directories.
#     os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
#     os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)
#
#     # Prepare lists for summary metrics.
#     all_js_pred = []  # for predicted vs. true (JS)
#     all_js_est_raw = []  # for raw estimate vs. true (JS)
#     all_js_est_smooth = []  # for smooth estimate vs. true (JS), if available
#     all_ce_pred = []  # for predicted vs. true (CE)
#     all_ce_est_raw = []  # for raw estimate vs. true (CE)
#     all_ce_est_smooth = []  # for smooth estimate vs. true (CE), if available
#
#     # Process selection: allow a comma-separated list or 'all'.
#     dataset_obj = BirdsongDataset(h5_path)
#     num_processes = len(dataset_obj)
#     if isinstance(PROCESS_TO_RUN, str) and PROCESS_TO_RUN.strip().lower() != 'all':
#         if ',' in PROCESS_TO_RUN:
#             indices = [int(x.strip()) for x in PROCESS_TO_RUN.split(',')]
#         else:
#             indices = [int(PROCESS_TO_RUN)]
#     else:
#         indices = range(num_processes)
#
#     for idx in indices:
#         print(f"Processing process {idx}...")
#         ngram_counts, true_probs = dataset_obj[idx]
#         # Add batch dimension and move to device.
#         ngram_counts_t = ngram_counts.unsqueeze(0).to(device)
#         true_probs_t = true_probs.unsqueeze(0).to(device)
#
#         # Evaluate the model.
#         logits_4d, pred_probs_4d, factors, metrics = evaluate_birdsong_model(model, ngram_counts_t, true_probs_t,
#                                                                              alphabet_size)
#         print(f"Metrics for process {idx}:")
#         for k, v in metrics.items():
#             print(f"  {k}: {v:.4f}")
#
#         # Prepare estimated probabilities from n-gram counts.
#         if order == 1:
#             counts_4d = ngram_counts_t.view(1, time_steps, alphabet_size, alphabet_size)
#         elif order == 2:
#             counts_4d = ngram_counts_t.view(1, time_steps, alphabet_size ** 2, alphabet_size)
#         counts_4d = counts_4d.cpu().numpy()
#
#         # Compute raw estimate: normalize counts row-wise.
#         raw_counts_norm = np.where(
#             counts_4d.sum(axis=-1, keepdims=True) > 0,
#             counts_4d / counts_4d.sum(axis=-1, keepdims=True),
#             0.0
#         )[0]  # shape: (T, rows, alphabet_size)
#
#         # Compute smooth estimate if desired.
#         if SMOOTH_WINDOW > 1:
#             smoothed_counts = smooth_counts(counts_4d[0], SMOOTH_WINDOW)
#             smooth_counts_norm = np.where(
#                 smoothed_counts.sum(axis=-1, keepdims=True) > 0,
#                 smoothed_counts / smoothed_counts.sum(axis=-1, keepdims=True),
#                 0.0
#             )
#         else:
#             smooth_counts_norm = None
#
#         # Get predicted and true probabilities as numpy arrays.
#         if order == 1:
#             pred_probs_np = pred_probs_4d.cpu().numpy()  # shape: (1, T, alphabet_size, alphabet_size)
#             true_probs_np = true_probs_t.view(1, time_steps, alphabet_size, alphabet_size).cpu().numpy()
#         elif order == 2:
#             pred_probs_np = pred_probs_4d.cpu().numpy()  # shape: (1, T, alphabet_size**2, alphabet_size)
#             true_probs_np = true_probs_t.view(1, time_steps, alphabet_size ** 2, alphabet_size).cpu().numpy()
#         pred_probs_np = pred_probs_np[0]  # (T, rows, alphabet_size)
#         true_probs_np = true_probs_np[0]
#
#         # Save individual transition plots.
#         if SAVE_INDIVIDUAL_PLOTS:
#             plot_transition_plots(idx, pred_probs_np, true_probs_np, raw_counts_norm,
#                                   smooth_counts_norm, alphabet_size, order, dataset_name)
#
#         # Compute summary metrics for this process over time.
#         js_pred_per_t = []
#         js_est_raw_per_t = []
#         js_est_smooth_per_t = []  # if available
#         ce_pred_per_t = []
#         ce_est_raw_per_t = []
#         ce_est_smooth_per_t = []
#         for t in range(time_steps):
#             p_pred = pred_probs_np[t]  # shape: (rows, alphabet_size)
#             p_true = true_probs_np[t]
#             p_est_raw = raw_counts_norm[t]
#             js_pred = np.mean([js_divergence(p_pred[r], p_true[r]) for r in range(p_pred.shape[0])])
#             ce_pred = np.mean([cross_entropy(p_pred[r], p_true[r]) for r in range(p_pred.shape[0])])
#             js_est_raw = np.mean([js_divergence(p_est_raw[r], p_true[r]) for r in range(p_true.shape[0])])
#             ce_est_raw = np.mean([cross_entropy(p_est_raw[r], p_true[r]) for r in range(p_true.shape[0])])
#             js_pred_per_t.append(js_pred)
#             ce_pred_per_t.append(ce_pred)
#             js_est_raw_per_t.append(js_est_raw)
#             ce_est_raw_per_t.append(ce_est_raw)
#             if smooth_counts_norm is not None:
#                 p_est_smooth = smooth_counts_norm[t]
#                 js_est_smooth = np.mean([js_divergence(p_est_smooth[r], p_true[r]) for r in range(p_true.shape[0])])
#                 ce_est_smooth = np.mean([cross_entropy(p_est_smooth[r], p_true[r]) for r in range(p_true.shape[0])])
#                 js_est_smooth_per_t.append(js_est_smooth)
#                 ce_est_smooth_per_t.append(ce_est_smooth)
#         all_js_pred.append(np.array(js_pred_per_t))
#         all_ce_pred.append(np.array(ce_pred_per_t))
#         all_js_est_raw.append(np.array(js_est_raw_per_t))
#         all_ce_est_raw.append(np.array(ce_est_raw_per_t))
#         if smooth_counts_norm is not None:
#             all_js_est_smooth.append(np.array(js_est_smooth_per_t))
#             all_ce_est_smooth.append(np.array(ce_est_smooth_per_t))
#
#     # Plot summary figures.
#     plot_summary_metrics(all_js_pred, all_js_est_raw, all_js_est_smooth if all_js_est_smooth else None,
#                          all_ce_pred, all_ce_est_raw, all_ce_est_smooth if all_ce_est_smooth else None,
#                          FILE_FORMAT, dataset_name)
#     print("Summary plots saved.")
#
#
#
#
# # checkpoint_path = "checkpoints/birdsong_lfads_2_1st_order_6_syl_fourier_processes_with_kl.pt"
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model = BirdsongLFADSModel2(
# #         alphabet_size=alphabet_size,
# #         order=order,
# #         encoder_dim=128,
# #         controller_dim=128,
# #         generator_dim=128,
# #         factor_dim=18,
# #         latent_dim=18,  # MUST BE EVEN
# #         inferred_input_dim=9,
# #         kappa=5.0,
# #         ar_step_size=0.90,
# #         ar_process_var=0.1
# #     )

import math
import os

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.io  # for loading .mat files
import torch
import torch.nn.functional as F
from birdsong_data_loader import BirdsongDataset
from birdsong_lfads_model_2 import (
    BirdsongLFADSModel2,
    rowwise_masked_softmax,
)

###############################################################################
# Configuration variables
###############################################################################
# Set this file path to either your simulated HDF5 file or your real .mat file.
DATA_FILE = "aggregated_datasets/aggregated_birdsong_data_10_syl_linear_100_song_in_batch_25_timesteps_20250505_122822.h5"
# Example for real data:
# DATA_FILE = "real_data/bengalese_finch.mat"

FILE_FORMAT = 'png'
SMOOTH_WINDOW = 5  # 1 means no smoothing
PLOT_SMOOTH_EST = True
SAVE_INDIVIDUAL_PLOTS = True  # Save each transition plot rather than displaying it.
BASE_OUTPUT_DIR = 'Process Plots'
SUMMARY_OUTPUT_DIR = 'summary plots'
PROCESS_TO_RUN = "10,11,12"  # Used in the simulated data branch

order = 1  # Set to 1 (bigrams) or 2 (trigrams)
checkpoint_path = "checkpoints/aggregated_birdsong_data_10_syl_linear_100_song_in_batch_25_timesteps_20250505_122822.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# Helper Functions for Evaluation Metrics
###############################################################################
def rowwise_kl_div(pred_dist, true_dist, eps=1e-9):
    pred_dist = torch.clamp(pred_dist, eps, 1.0)
    true_dist = torch.clamp(true_dist, eps, 1.0)
    kl = true_dist * torch.log(true_dist / pred_dist)
    return kl.sum(dim=-1).mean()


def rowwise_mse(pred_dist, true_dist):
    return ((pred_dist - true_dist) ** 2).mean()


###############################################################################
# Evaluation Function
###############################################################################
def evaluate_birdsong_model(model, input_counts, true_probs, alphabet_size):
    model.eval()
    with torch.no_grad():
        outputs = model(input_counts)
        logits = outputs["logits"]  # For order==1: (B, T, α²)
        factors = outputs.get("factors", None)
        B, T, logit_dim = logits.shape
        if order == 1:
            logits_4d = logits.view(B, T, alphabet_size, alphabet_size)
            pred_probs_4d = F.softmax(logits_4d, dim=-1)
            pred_probs_flat = pred_probs_4d.view(B, T, alphabet_size ** 2)
            true_probs_flat = true_probs.view(B, T, alphabet_size ** 2)
        elif order == 2:
            logits_4d = logits.view(B, T, alphabet_size ** 2, alphabet_size)
            true_probs_4d = true_probs.view(B, T, alphabet_size ** 2, alphabet_size)
            target_matrix = true_probs_4d[0, 0, :, :]
            pred_probs = rowwise_masked_softmax(logits, alphabet_size, order=order,
                                                target_matrix=target_matrix, mask_value=-1e8)
            pred_probs_4d = pred_probs.view(B, T, alphabet_size ** 2, alphabet_size)
            pred_probs_flat = pred_probs_4d.view(B, T, alphabet_size ** 3)
            true_probs_flat = true_probs_4d.view(B, T, alphabet_size ** 3)
        else:
            raise ValueError("Only order==1 and order==2 are supported.")

        if order == 1:
            pred_flat = pred_probs_4d.view(B * T * alphabet_size, alphabet_size)
            true_labels = true_probs.view(B, T, alphabet_size, alphabet_size).argmax(dim=-1)
            true_flat = true_labels.view(B * T * alphabet_size)
        elif order == 2:
            pred_flat = pred_probs_4d.view(B * T * (alphabet_size ** 2), alphabet_size)
            true_labels = true_probs.view(B, T, alphabet_size ** 2, alphabet_size).argmax(dim=-1)
            true_flat = true_labels.view(B * T * (alphabet_size ** 2))
        ce_loss = F.cross_entropy(pred_flat, true_flat)

        pred_classes = pred_probs_4d.argmax(dim=-1)
        true_classes = true_probs.view(B, T, -1, alphabet_size).argmax(dim=-1)
        correct = (pred_classes == true_classes).sum().item()
        total = pred_classes.numel()
        accuracy = correct / total if total > 0 else 0.0

        kl_val = rowwise_kl_div(pred_probs_flat, true_probs_flat)
        mse_val = rowwise_mse(pred_probs_flat, true_probs_flat)

        metrics = {
            "cross_entropy": ce_loss.item(),
            "accuracy": accuracy,
            "rowwise_kl": kl_val.item(),
            "rowwise_mse": mse_val.item(),
        }

    return logits_4d, pred_probs_4d, factors, metrics


###############################################################################
# Plotting Functions (unchanged)
###############################################################################
def plot_wrate_factors(model, factors, alphabet_size, sequence_idx=0):
    if factors is None:
        print("No factors returned by model; cannot plot W_rate*factors.")
        return
    try:
        w_factors = model.rate_linear(factors)
    except AttributeError:
        print("Model does not have a rate_linear layer or there's a dimension mismatch.")
        return
    w_factors = w_factors.detach()
    B, T, _ = w_factors.shape
    if order == 1:
        w_factors_4d = w_factors.view(B, T, alphabet_size, alphabet_size)
    elif order == 2:
        w_factors_4d = w_factors.view(B, T, alphabet_size ** 2, alphabet_size)
    else:
        print("Unsupported order for plotting.")
        return
    data_4d = w_factors_4d[sequence_idx].cpu().numpy()
    avg_data = data_4d.mean(axis=0)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im = ax1.imshow(avg_data, cmap='bwr', aspect='auto')
    ax1.set_title("Mean of [W_rate * factors] across time")
    fig1.colorbar(im, ax=ax1)
    plt.tight_layout()
    plt.show()
    X = avg_data.shape[0]
    fig2, axs = plt.subplots(1, X, figsize=(4 * X, 4), sharey=True)
    if X == 1:
        axs = [axs]
    for i in range(X):
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


def plot_ngram_counts(pred_probs_4d, true_probs, ngram_counts, alphabet_size, order):
    B, T = pred_probs_4d.shape[:2]
    if order == 1:
        counts_4d = ngram_counts.view(B, T, alphabet_size, alphabet_size).cpu().numpy()
        counts_norm = np.where(
            counts_4d.sum(axis=-1, keepdims=True) > 0,
            counts_4d / counts_4d.sum(axis=-1, keepdims=True),
            0.0
        )
        pred_mat = pred_probs_4d[0].cpu().numpy()
        true_mat = true_probs.view(B, T, alphabet_size, alphabet_size)[0].cpu().numpy()
        counts_mat = counts_norm[0]
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
        counts_4d = ngram_counts.view(B, T, alphabet_size ** 2, alphabet_size).cpu().numpy()
        counts_norm = np.where(
            counts_4d.sum(axis=-1, keepdims=True) > 0,
            counts_4d / counts_4d.sum(axis=-1, keepdims=True),
            0.0
        )
        pred_mat = pred_probs_4d[0].cpu().numpy()
        true_mat = true_probs.view(B, T, alphabet_size ** 2, alphabet_size)[0].cpu().numpy()
        counts_mat = counts_norm[0]
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


def js_divergence(p, q, eps=1e-9):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    m = (p + q) / 2
    kl_pm = np.sum(p * np.log(p / m), axis=-1)
    kl_qm = np.sum(q * np.log(q / m), axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def cross_entropy(p, q, eps=1e-9):
    p = np.clip(p, eps, 1)
    return -np.sum(q * np.log(p), axis=-1)


def get_alphabet(alphabet_size):
    if alphabet_size < 2:
        return ['<']
    num_middle = alphabet_size - 2
    middle = [chr(ord('a') + i) for i in range(num_middle)]
    return ['<'] + middle + ['>']


def get_context_string(row_idx, alphabet, order):
    if order == 1:
        return alphabet[row_idx]
    elif order == 2:
        a_size = len(alphabet)
        first = alphabet[row_idx // a_size]
        second = alphabet[row_idx % a_size]
        return first + second
    else:
        return str(row_idx)


def safe_context_str(context_str):
    return context_str.replace('<', 'start').replace('>', 'end')


def smooth_counts(counts, window_size):
    if window_size <= 1:
        return counts
    T, rows, cols = counts.shape
    pad = window_size // 2
    padded = np.pad(counts, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    smoothed = np.empty_like(counts)
    for t in range(T):
        smoothed[t] = np.mean(padded[t:t + window_size], axis=0)
    return smoothed


def plot_transition_plots(process_idx, pred_probs_4d, true_probs, raw_est, smooth_est, alphabet_size, order,
                          dataset_name, symbols=None):
    alphabet = symbols if symbols is not None else get_alphabet(alphabet_size)
    process_dir_name = f"{dataset_name}_process_{process_idx}"
    process_dir = os.path.join(BASE_OUTPUT_DIR, process_dir_name)
    os.makedirs(process_dir, exist_ok=True)
    num_contexts = pred_probs_4d.shape[1]
    cmap = cm.get_cmap('tab10', alphabet_size)
    for row_idx in range(num_contexts):
        context_str = get_context_string(row_idx, alphabet, order)
        safe_ctx = safe_context_str(context_str)
        plt.figure(figsize=(8, 6))
        for col_idx in range(alphabet_size):
            plt.plot(pred_probs_4d[:, row_idx, col_idx], '--', color=cmap(col_idx),
                     label=f"Pred {context_str} → {alphabet[col_idx]}")
            plt.plot(true_probs[:, row_idx, col_idx], '-', color=cmap(col_idx),
                     label=f"True {context_str} → {alphabet[col_idx]}")
            if PLOT_SMOOTH_EST and smooth_est is not None:
                plt.plot(smooth_est[:, row_idx, col_idx], '-.', color=cmap(col_idx),
                         label=f"Smooth Est {context_str} → {alphabet[col_idx]}")
        plt.title(f"Transition from Context: {context_str}")
        plt.xlabel("Time Step")
        plt.ylabel("Probability")
        plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        filename = f"transition_{safe_ctx}.{FILE_FORMAT}"
        plt.savefig(os.path.join(process_dir, filename))
        plt.close()


def plot_summary_metrics(all_js_pred, all_js_est_raw, all_js_est_smooth,
                         all_ce_pred, all_ce_est_raw, all_ce_est_smooth, file_format, dataset_name):
    js_pred_arr = np.array(all_js_pred)
    ce_pred_arr = np.array(all_ce_pred)
    x = np.arange(js_pred_arr.shape[1])
    plt.figure(figsize=(10, 6))
    mean_js_pred = js_pred_arr.mean(axis=0)
    std_js_pred = js_pred_arr.std(axis=0)
    plt.plot(x, mean_js_pred, color='blue', linewidth=3, label="Mean Pred JS")
    plt.fill_between(x, mean_js_pred - std_js_pred, mean_js_pred + std_js_pred,
                     color='blue', alpha=0.2)
    if all_js_est_raw is not None:
        js_est_raw_arr = np.array(all_js_est_raw)
        mean_js_est_raw = js_est_raw_arr.mean(axis=0)
        std_js_est_raw = js_est_raw_arr.std(axis=0)
        plt.plot(x, mean_js_est_raw, color='red', linewidth=3, label="Mean Raw Est JS")
        plt.fill_between(x, mean_js_est_raw - std_js_est_raw, mean_js_est_raw + std_js_est_raw,
                         color='red', alpha=0.2)
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
    plt.tight_layout()
    js_filename = os.path.join(SUMMARY_OUTPUT_DIR, f"{dataset_name}_summary_js_divergence.{FILE_FORMAT}")
    plt.savefig(js_filename)
    plt.close()
    plt.figure(figsize=(10, 6))
    mean_ce_pred = ce_pred_arr.mean(axis=0)
    std_ce_pred = ce_pred_arr.std(axis=0)
    plt.plot(x, mean_ce_pred, color='blue', linewidth=3, label="Mean Pred CE")
    plt.fill_between(x, mean_ce_pred - std_ce_pred, mean_ce_pred + std_ce_pred,
                     color='blue', alpha=0.2)
    if all_ce_est_raw is not None:
        ce_est_raw_arr = np.array(all_ce_est_raw)
        mean_ce_est_raw = ce_est_raw_arr.mean(axis=0)
        std_ce_est_raw = ce_est_raw_arr.std(axis=0)
        plt.plot(x, mean_ce_est_raw, color='red', linewidth=3, label="Mean Raw Est CE")
        plt.fill_between(x, mean_ce_est_raw - std_ce_est_raw, mean_ce_est_raw + std_ce_est_raw,
                         color='red', alpha=0.2)
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
    plt.tight_layout()
    ce_filename = os.path.join(SUMMARY_OUTPUT_DIR, f"{dataset_name}_summary_cross_entropy.{FILE_FORMAT}")
    plt.savefig(ce_filename)
    plt.close()


###############################################################################
# Main Evaluation Loop
###############################################################################
if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)

    # Branch based on file extension.
    if DATA_FILE.endswith('.h5'):
        # ---------------- Simulated Data Branch ----------------
        with h5py.File(DATA_FILE, "r") as hf:
            alphabet_size = math.ceil(hf["bigram_counts"].shape[0] ** (1 / (order + 1)))
            time_steps = hf["bigram_counts"].shape[1]
            print(f"Loaded simulated dataset with alphabet_size={alphabet_size}, time_steps={time_steps}")
        dataset_name = os.path.splitext(os.path.basename(DATA_FILE))[0]

        # Instantiate the model with the correct alphabet_size.
        model = BirdsongLFADSModel2(
            alphabet_size=alphabet_size,
            order=order,
            encoder_dim=256,
            controller_dim=4,
            generator_dim=256,
            factor_dim=338,
            latent_dim=128,  # MUST BE EVEN
            inferred_input_dim=4,
            kappa=1.0,
            ar_step_size=0.99,
            ar_process_var=0.1,
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        from birdsong_data_loader import BirdsongDataset

        dataset_obj = BirdsongDataset(DATA_FILE)
        num_processes = len(dataset_obj)
        if isinstance(PROCESS_TO_RUN, str) and PROCESS_TO_RUN.strip().lower() != 'all':
            if ',' in PROCESS_TO_RUN:
                indices = [int(x.strip()) for x in PROCESS_TO_RUN.split(',')]
            else:
                indices = [int(PROCESS_TO_RUN)]
        else:
            indices = range(num_processes)

        all_js_pred, all_js_est_raw, all_js_est_smooth = [], [], []
        all_ce_pred, all_ce_est_raw, all_ce_est_smooth = [], [], []

        for idx in indices:
            print(f"Processing process {idx}...")
            ngram_counts, true_probs = dataset_obj[idx]
            ngram_counts_t = ngram_counts.unsqueeze(0).to(device)
            true_probs_t = true_probs.unsqueeze(0).to(device)

            logits_4d, pred_probs_4d, factors, metrics = evaluate_birdsong_model(
                model, ngram_counts_t, true_probs_t, alphabet_size)
            print(f"Metrics for process {idx}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            if order == 1:
                counts_4d = ngram_counts_t.view(1, time_steps, alphabet_size, alphabet_size)
            elif order == 2:
                counts_4d = ngram_counts_t.view(1, time_steps, alphabet_size ** 2, alphabet_size)
            counts_4d_np = counts_4d.cpu().numpy()
            raw_counts_norm = np.where(
                counts_4d_np.sum(axis=-1, keepdims=True) > 0,
                counts_4d_np / counts_4d_np.sum(axis=-1, keepdims=True),
                0.0
            )[0]

            if SMOOTH_WINDOW > 1:
                smoothed_counts = smooth_counts(counts_4d_np[0], SMOOTH_WINDOW)
                smooth_counts_norm = np.where(
                    smoothed_counts.sum(axis=-1, keepdims=True) > 0,
                    smoothed_counts / smoothed_counts.sum(axis=-1, keepdims=True),
                    0.0
                )
            else:
                smooth_counts_norm = None

            if order == 1:
                pred_probs_np = pred_probs_4d.cpu().numpy()[0]
                true_probs_np = true_probs_t.view(1, time_steps, alphabet_size, alphabet_size).cpu().numpy()[0]
            elif order == 2:
                pred_probs_np = pred_probs_4d.cpu().numpy()[0]
                true_probs_np = true_probs_t.view(1, time_steps, alphabet_size ** 2, alphabet_size).cpu().numpy()[0]

            if SAVE_INDIVIDUAL_PLOTS:
                plot_transition_plots(idx, pred_probs_np, true_probs_np, raw_counts_norm,
                                      smooth_counts_norm, alphabet_size, order, dataset_name)

            js_pred_per_t, js_est_raw_per_t, js_est_smooth_per_t = [], [], []
            ce_pred_per_t, ce_est_raw_per_t, ce_est_smooth_per_t = [], [], []
            for t in range(time_steps):
                p_pred = pred_probs_np[t]
                p_true = true_probs_np[t]
                p_est_raw = raw_counts_norm[t]
                js_pred = np.mean([js_divergence(p_pred[r], p_true[r]) for r in range(p_pred.shape[0])])
                ce_pred = np.mean([cross_entropy(p_pred[r], p_true[r]) for r in range(p_pred.shape[0])])
                js_est_raw = np.mean([js_divergence(p_est_raw[r], p_true[r]) for r in range(p_true.shape[0])])
                ce_est_raw = np.mean([cross_entropy(p_est_raw[r], p_true[r]) for r in range(p_true.shape[0])])
                js_pred_per_t.append(js_pred)
                ce_pred_per_t.append(ce_pred)
                js_est_raw_per_t.append(js_est_raw)
                ce_est_raw_per_t.append(ce_est_raw)
                if smooth_counts_norm is not None:
                    p_est_smooth = smooth_counts_norm[t]
                    js_est_smooth = np.mean([js_divergence(p_est_smooth[r], p_true[r]) for r in range(p_true.shape[0])])
                    ce_est_smooth = np.mean([cross_entropy(p_est_smooth[r], p_true[r]) for r in range(p_true.shape[0])])
                    js_est_smooth_per_t.append(js_est_smooth)
                    ce_est_smooth_per_t.append(ce_est_smooth)
            all_js_pred.append(np.array(js_pred_per_t))
            all_ce_pred.append(np.array(ce_pred_per_t))
            all_js_est_raw.append(np.array(js_est_raw_per_t))
            all_ce_est_raw.append(np.array(ce_est_raw_per_t))
            if smooth_counts_norm is not None:
                all_js_est_smooth.append(np.array(js_est_smooth_per_t))
                all_ce_est_smooth.append(np.array(ce_est_smooth_per_t))

    elif DATA_FILE.endswith('.mat'):
        # ---------------- Real Data Branch ----------------
        print("Loading real data from .mat file...")
        mat_data = scipy.io.loadmat(DATA_FILE)

        # Process cleanSymbols: 1x13 cell array with each cell a character.
        # We assume each cell is directly convertible to a string.
        alphabet = [str(s) for s in mat_data['cleanSymbols'].flatten()]
        alphabet_size = len(alphabet)  # Should be 13.

        # Process transitionMatrix and rawCounts.
        # They are 1x82 cell arrays where each cell is a 13x13 matrix.
        transition_cells = mat_data['transitionMatrices'].flatten()
        rawCounts_cells = mat_data['rawCounts'].flatten()
        # Stack along the time axis: shape becomes (82, 13, 13)
        transitionMatrix = np.stack(list(transition_cells), axis=0)
        rawCounts = np.stack(list(rawCounts_cells), axis=0)
        time_steps = transitionMatrix.shape[0]  # Should be 82.

        # Flatten each 13x13 matrix row-wise: each becomes a vector of length 169.
        # Then transpose to get shape (169, time_steps).
        transitionMatrix_flat = transitionMatrix.reshape(time_steps, -1).T
        rawCounts_flat = rawCounts.reshape(time_steps, -1).T

        dataset_name = os.path.splitext(os.path.basename(DATA_FILE))[0]

        # Instantiate the model with the correct alphabet_size.
        model = BirdsongLFADSModel2(
            alphabet_size=alphabet_size,
            order=order,
            encoder_dim=64,
            controller_dim=1,
            generator_dim=128,
            factor_dim=338,
            latent_dim=84,  # MUST BE EVEN
            inferred_input_dim=4,
            kappa=5.0,
            ar_step_size=0.90,
            ar_process_var=0.1
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        # For real data, we have one process. Convert the flat arrays to tensors.
        # rawCounts_flat and transitionMatrix_flat are of shape (169, time_steps).
        # We need to create tensors of shape (1, time_steps, 169) and then view as (1, time_steps, 13, 13).
        ngram_counts_t = torch.tensor(rawCounts_flat, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
        true_probs_t = torch.tensor(transitionMatrix_flat, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)

        logits_4d, pred_probs_4d, factors, metrics = evaluate_birdsong_model(
            model, ngram_counts_t, true_probs_t, alphabet_size)
        print("Metrics for real data:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        if order == 1:
            counts_4d = ngram_counts_t.view(1, time_steps, alphabet_size, alphabet_size)
        elif order == 2:
            counts_4d = ngram_counts_t.view(1, time_steps, alphabet_size ** 2, alphabet_size)
        counts_4d_np = counts_4d.cpu().numpy()
        raw_counts_norm = np.where(
            counts_4d_np.sum(axis=-1, keepdims=True) > 0,
            counts_4d_np / counts_4d_np.sum(axis=-1, keepdims=True),
            0.0
        )[0]

        if SMOOTH_WINDOW > 1:
            smoothed_counts = smooth_counts(counts_4d_np[0], SMOOTH_WINDOW)
            smooth_counts_norm = np.where(
                smoothed_counts.sum(axis=-1, keepdims=True) > 0,
                smoothed_counts / smoothed_counts.sum(axis=-1, keepdims=True),
                0.0
            )
        else:
            smooth_counts_norm = None

        if order == 1:
            pred_probs_np = pred_probs_4d.cpu().numpy()[0]
            true_probs_np = true_probs_t.view(1, time_steps, alphabet_size, alphabet_size).cpu().numpy()[0]
        elif order == 2:
            pred_probs_np = pred_probs_4d.cpu().numpy()[0]
            true_probs_np = true_probs_t.view(1, time_steps, alphabet_size ** 2, alphabet_size).cpu().numpy()[0]

        if SAVE_INDIVIDUAL_PLOTS:
            plot_transition_plots(0, pred_probs_np, true_probs_np, raw_counts_norm,
                                  smooth_counts_norm, alphabet_size, order, dataset_name, symbols=alphabet)

        js_pred_per_t, js_est_raw_per_t, js_est_smooth_per_t = [], [], []
        ce_pred_per_t, ce_est_raw_per_t, ce_est_smooth_per_t = [], [], []
        for t in range(time_steps):
            p_pred = pred_probs_np[t]
            p_true = true_probs_np[t]
            p_est_raw = raw_counts_norm[t]
            js_pred = np.mean([js_divergence(p_pred[r], p_true[r]) for r in range(p_pred.shape[0])])
            ce_pred = np.mean([cross_entropy(p_pred[r], p_true[r]) for r in range(p_pred.shape[0])])
            js_est_raw = np.mean([js_divergence(p_est_raw[r], p_true[r]) for r in range(p_true.shape[0])])
            ce_est_raw = np.mean([cross_entropy(p_est_raw[r], p_true[r]) for r in range(p_true.shape[0])])
            js_pred_per_t.append(js_pred)
            ce_pred_per_t.append(ce_pred)
            js_est_raw_per_t.append(js_est_raw)
            ce_est_raw_per_t.append(ce_est_raw)
            if smooth_counts_norm is not None:
                p_est_smooth = smooth_counts_norm[t]
                js_est_smooth = np.mean([js_divergence(p_est_smooth[r], p_true[r]) for r in range(p_true.shape[0])])
                ce_est_smooth = np.mean([cross_entropy(p_est_smooth[r], p_true[r]) for r in range(p_true.shape[0])])
                js_est_smooth_per_t.append(js_est_smooth)
                ce_est_smooth_per_t.append(ce_est_smooth)
        all_js_pred.append(np.array(js_pred_per_t))
        all_ce_pred.append(np.array(ce_pred_per_t))
        all_js_est_raw.append(np.array(js_est_raw_per_t))
        all_ce_est_raw.append(np.array(ce_est_raw_per_t))
        if smooth_counts_norm is not None:
            all_js_est_smooth.append(np.array(js_est_smooth_per_t))
            all_ce_est_smooth.append(np.array(ce_est_smooth_per_t))
    else:
        raise ValueError("Unsupported file type. DATA_FILE must be .h5 or .mat.")


    # Plot summary figures.
    plot_summary_metrics(all_js_pred, all_js_est_raw, all_js_est_smooth if all_js_est_smooth else None,
                         all_ce_pred, all_ce_est_raw, all_ce_est_smooth if all_ce_est_smooth else None,
                         FILE_FORMAT, dataset_name)
    print("Summary plots saved.")

# checkpoint_path="checkpoints/gy60r6_BengFinch_1st_order_13_syl_mixed_processes_with_L2.pt"
# model = BirdsongLFADSModel2(
#         alphabet_size=alphabet_size,
#         order=order,
#         encoder_dim=64,
#         controller_dim=64,
#         generator_dim=64,
#         factor_dim=338,
#         latent_dim=26,  # MUST BE EVEN
#         inferred_input_dim=26,
#         kappa=3.0,
#         ar_step_size=0.99,
#         ar_process_var=0.1
#     )
