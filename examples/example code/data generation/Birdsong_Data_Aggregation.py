# # Birdsong Data Aggregation
#
# import numpy as np
# import h5py
# import os
# from birdsong_data_generation import simulate_birdsong, x_init_maker, save_to_hdf5, simulate_variable_batches_and_time_steps
# from datetime import datetime
#
# # Define global parameters
# alphabet = ['<', 'a', 'b', 'c','d','e', '>']  # Example alphabet
# num_batches = 400  # Number of time steps per process
# batch_size = 30  # Number of sequences per batch
# seq_range = (2, 12)  # Min and max sequence length
# order = 1
#
# # Define process functions
# def linear_process(x, A, t):
#     """Linear update to the transition matrix."""
#     return x + A
#
# def linear_with_noise(x, A, t):
#     """Linear update with added Gaussian noise."""
#     return x + A + np.random.normal(0, 0.1, size=x.shape)
#
# def nonlinear_cosine(x, A, t):
#     """Nonlinear update using a cosine function."""
#     return x + A * np.cos(0.1 * t)
#
# def quadratic_decay(x, A, t):
#     """Nonlinear update with quadratic scaling."""
#     return x + A / (1 + (t / 10)**2)
#
# def exponential_growth(x, A, t):
#     """Exponential growth: increments by a factor that decays over time."""
#     growth_rate = 0.01
#     return x + A * np.exp(-growth_rate * t)
#
# def multiplicative_noise(x, A, t):
#     """Noise amplitude depends on the magnitude of x."""
#     noise_scale = 0.01 * (np.abs(x) + 1)
#     return x + np.random.normal(0, noise_scale, size=x.shape)
#
# def piecewise_linear(x, A, t):
#     """Different linear increments depending on time interval."""
#     if t < 10:
#         return x + A  # Early stage: simple linear growth
#     elif t < 40:
#         return x - 0.5 * A  # Mid stage: decreasing trend
#     else:
#         return x + 0.5 * A  # Late stage: smaller positive increments
#
# def logistic_growth(x, A, t):
#     """Logistic growth toward a carrying capacity."""
#     K = 10.0
#     return x + A * x * (1 - x / K)
#
# # Aggregate data across processes
# process_types = [
#     ("linear", linear_process, 8000),  # 200 processes with linear updates
#     ("linear_with_noise", linear_with_noise, 4000),  # 200 processes with noise
#     ("nonlinear_cosine", nonlinear_cosine, 6000),  # 300 processes with cosine updates
#     ("quadratic_decay", quadratic_decay, 6000),  # 300 processes with quadratic decay
#     ("exponential_growth", exponential_growth, 4000),
#     ("multiplicative_noise", multiplicative_noise, 6000),
#     ("piecewise_linear", piecewise_linear, 6000),
#     ("logistic_growth", logistic_growth, 6000)
# ]
#
#
# # Initialize storage for aggregated data
# bigram_counts_list = []
# probabilities_list = []
# metadata = []
#
# # Generate data for each process type
# for process_name, process_fn, num_processes in process_types:
#     print(f"Generating data for process: {process_name} with {num_processes} processes...")
#     bigram_counts, probabilities = simulate_birdsong(
#         num_batches, batch_size, seq_range, alphabet, num_processes, order, process_fn
#     )
#     bigram_counts_list.append(bigram_counts)
#     probabilities_list.append(probabilities)
#     metadata.append({"process": process_name, "num_processes": num_processes})
#
# # Concatenate all data
# bigram_counts_aggregated = np.concatenate(bigram_counts_list, axis=2)
# probabilities_aggregated = np.concatenate(probabilities_list, axis=2)
#
# # Save aggregated data to HDF5
# output_dir = "./aggregated_datasets"
# os.makedirs(output_dir, exist_ok=True)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_path = os.path.join(output_dir, f"aggregated_birdsong_data_{timestamp}.h5")
# print(f"Saving aggregated data to {output_path}...")
#
# with h5py.File(output_path, "w") as hf:
#     hf.create_dataset("bigram_counts", data=bigram_counts_aggregated)
#     hf.create_dataset("probabilities", data=probabilities_aggregated)
#     hf.attrs["metadata"] = str(metadata)  # Save metadata as an attribute
#
# print("Data aggregation complete.")

# # Birdsong Data Aggregation
#
# import numpy as np
# import h5py
# import os
# from birdsong_data_generation import simulate_birdsong, simulate_variable_batches_and_time_steps
# from datetime import datetime
#
# # Define global parameters
# alphabet = ['<', '0', '1', '2', '3', '>']  # Example alphabet '<', 'a', 'b', 'c', 'd','e', 'f', 'g', 'h', 'i' ,'j','k','l','m','n','o','p','q','r','>'
# # ['<', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', '>'] # lb4444 N_phrase Types
# time_step_range = (50, 50)  # Range for variable time steps
# batch_size_range = (10, 10)  # Range for variable batch sizes
# seq_range = (3, 12)  # Min and max sequence length
# order = 1
#
# # Define process functions
# def linear_process(x, A, t):
#     """Linear update to the transition matrix."""
#     return x + A
#
# def linear_with_noise(x, A, t):
#     """Linear update with added Gaussian noise."""
#     return x + A + np.random.normal(0, 0.1, size=x.shape)
#
# def nonlinear_cosine(x, A, t):
#     """Nonlinear update using a cosine function."""
#     return x + A * 2 * np.cos(0.1 * t)
#
# def quadratic_decay(x, A, t):
#     """Nonlinear update with quadratic scaling."""
#     return x + A / (1 + (t / 10)**2)
#
# def exponential_growth(x, A, t):
#     """Exponential growth: increments by a factor that decays over time."""
#     growth_rate = 0.001
#     return x + A * np.exp(-growth_rate * t)
#
# def multiplicative_noise(x, A, t):
#     """Noise amplitude depends on the magnitude of x."""
#     noise_scale = 0.01 * (np.abs(x) + 1)
#     return x + np.random.normal(0, noise_scale, size=x.shape)
#
# def piecewise_linear(x, A, t):
#     """Different linear increments depending on time interval."""
#     if t < 10:
#         return x + A  # Early stage: simple linear growth
#     elif t < 40:
#         return x - 0.5 * A  # Mid stage: decreasing trend
#     else:
#         return x + 0.5 * A  # Late stage: smaller positive increments
#
# def logistic_growth(x, A, t):
#     """Logistic growth toward a carrying capacity."""
#     K = 500.0
#     return x + A * x * (1 - x / K)
#
#
# def fourier_update(x, A, t):
#     """Nonlinear update using a Fourier series with sine and cosine terms.
#
#     The Fourier coefficients are generated once for each process (detected when t==0)
#     and then reused for all subsequent time steps for that process.
#
#     Parameters:
#     -----------
#     x : np.ndarray
#         Current state vector.
#     A : np.ndarray
#         Constant step vector computed as (x_final - x_init)/num_batches.
#     t : int or float
#         Current time step.
#
#     Returns:
#     --------
#     np.ndarray
#         Updated state vector.
#     """
#     fourier_order = 5
#     use_A = False  # Set to True to evaluate at A instead of x.
#
#     # Initialize storage attributes on the function if they don't exist.
#     if not hasattr(fourier_update, "coeff_store"):
#         fourier_update.coeff_store = {}  # Dictionary to store coefficients per process.
#         fourier_update.process_counter = 0  # Counter for the number of processes seen.
#         fourier_update.current_process_key = None  # Key for the current process.
#
#     # When t==0, we assume this is the start of a new process.
#     if t == 0:
#         key = fourier_update.process_counter
#         n = x.shape[0]
#         cos_coeffs = np.random.uniform(-0.05, 0.05, (n, fourier_order + 1))
#         sin_coeffs = np.random.uniform(-0.05, 0.05, (n, fourier_order))
#         fourier_update.coeff_store[key] = (cos_coeffs, sin_coeffs)
#         fourier_update.current_process_key = key
#         fourier_update.process_counter += 1
#     else:
#         key = fourier_update.current_process_key
#
#     cos_coeffs, sin_coeffs = fourier_update.coeff_store[key]
#     new_x = np.empty_like(x)
#     for i in range(len(x)):
#         # Choose evaluation point based on use_A flag.
#         theta = A[i] if use_A else x[i]
#         update = cos_coeffs[i, 0]  # constant term (order 0)
#         for j in range(1, fourier_order + 1):
#             update += (cos_coeffs[i, j] * np.cos(j * theta) +
#                        sin_coeffs[i, j - 1] * np.sin(j * theta))
#         new_x[i] = x[i] + update
#     return new_x
#
#
# # Aggregate data across processes
# process_types = [
#     #("linear", linear_process, 1500000),
#     ("fourier", fourier_update, 50000),  # Add Fourier update processes
#     # ("linear_with_noise", linear_with_noise, 200000),
#     # ("nonlinear_cosine", nonlinear_cosine, 50000),
#     # ("quadratic_decay", quadratic_decay, 20000),
#     # ("exponential_growth", exponential_growth, 40000),
#     # ("multiplicative_noise", multiplicative_noise, 60000),
#     # ("piecewise_linear", piecewise_linear, 60000),
#     # ("logistic_growth", logistic_growth, 200000)
# ]
#
# # Initialize storage for aggregated data
# bigram_counts_list = []
# probabilities_list = []
# metadata = []
#
# # Generate data for each process type
# for process_name, process_fn, num_processes in process_types:
#     print(f"Generating data for process: {process_name} with {num_processes} processes...")
#
#     # New simulation with variable time steps and batch sizes
#     bigram_counts, probabilities = simulate_variable_batches_and_time_steps(
#         num_processes=num_processes,
#         time_step_range=time_step_range,
#         batch_size_range=batch_size_range,
#         seq_range=seq_range,
#         alphabet=alphabet,
#         order=order,
#         process_fn=process_fn
#     )
#
#     # Commenting out the old simulation code
#     # bigram_counts, probabilities = simulate_birdsong(
#     #     num_batches, batch_size, seq_range, alphabet, num_processes, order, process_fn
#     # )
#
#     bigram_counts_list.append(bigram_counts)
#     probabilities_list.append(probabilities)
#     metadata.append({"process": process_name, "num_processes": num_processes, "order": order})
#
# # Concatenate all data
# bigram_counts_aggregated = np.concatenate(bigram_counts_list, axis=2)
# probabilities_aggregated = np.concatenate(probabilities_list, axis=2)
#
# # Save aggregated data to HDF5
# output_dir = "./aggregated_datasets"
# os.makedirs(output_dir, exist_ok=True)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_path = os.path.join(output_dir, f"aggregated_birdsong_data_1st_order_6_syl_fourier_10_song_in_batch_50_timesteps_{timestamp}.h5")
# print(f"Saving aggregated data to {output_path}...")
#
# with h5py.File(output_path, "w") as hf:
#     hf.create_dataset("bigram_counts", data=bigram_counts_aggregated)
#     hf.create_dataset("probabilities", data=probabilities_aggregated)
#     hf.attrs["metadata"] = str(metadata)  # Save metadata as an attribute
#
# print("Data aggregation complete.")

# Birdsong Data Aggregation

import os
from datetime import datetime

import h5py
import numpy as np
from birdsong_data_generation import (
    simulate_variable_batches_and_time_steps,
)

# Define global parameters
alphabet = ['<', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','D', 'E', 'F', 'G', 'H', 'I', 'J', '>']  # Example alphabet
# ['<', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', '>'] # lb4444 N_phrase Types
time_step_range = (75, 75)  # Range for variable time steps
batch_size_range = (50, 50)  # Range for variable batch sizes
seq_range = (3, 12)  # Min and max sequence length
order = 1


# Define process functions
def linear_process(x, a, t):
    """Linear update to the transition matrix."""
    return x + a


def linear_with_noise(x, a, t):
    """Linear update with added Gaussian noise."""
    return x + a + np.random.normal(0, 0.1, size=x.shape)


def nonlinear_cosine(x, a, t):
    """Nonlinear update using a cosine function."""
    return x + a * 2 * np.cos(0.1 * t)


def quadratic_decay(x, a, t):
    """Nonlinear update with quadratic scaling."""
    return x + a / (1 + (t / 10) ** 2)


def exponential_growth(x, a, t):
    """Exponential growth: increments by a factor that decays over time."""
    growth_rate = 0.001
    return x + a * np.exp(-growth_rate * t)


def multiplicative_noise(x, a, t):
    """Noise amplitude depends on the magnitude of x."""
    noise_scale = 0.01 * (np.abs(x) + 1)
    return x + np.random.normal(0, noise_scale, size=x.shape)


def piecewise_linear(x, a, t):
    """Different linear increments depending on time interval."""
    if t < 10:
        return x + a  # Early stage: simple linear growth
    elif t < 40:
        return x - 0.5 * a  # Mid stage: decreasing trend
    else:
        return x + 0.5 * a  # Late stage: smaller positive increments


def logistic_growth(x, a, t):
    """Logistic growth toward a carrying capacity."""
    k = 500.0
    return x + a * x * (1 - x / k)


def fourier_update(x, a, t):
    """Nonlinear update using a Fourier series with sine and cosine terms.

    The Fourier coefficients are generated once for each process (detected when t==0)
    and then reused for all subsequent time steps for that process.
    """
    fourier_order = 5
    use_a = False  # Set to True to evaluate at A instead of x.

    if not hasattr(fourier_update, "coeff_store"):
        fourier_update.coeff_store = {}  # Dictionary to store coefficients per process.
        fourier_update.process_counter = 0  # Counter for the number of processes seen.
        fourier_update.current_process_key = None  # Key for the current process.

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

def sparse_transition_process(x, a, t, sparsity=0.9, noise_scale=0.05):
    """
    Enforces sparsity in transition matrix logits, automatically supporting order 1 and 2 Markov chains.

    Automatically infers the Markov order and alphabet size from the shape of `x`.

    Args:
        x (np.ndarray): Current transition logits (flattened).
        a (np.ndarray): Update direction matrix (same shape as x).
        t (int): Current time step.
        sparsity (float): Fraction of transitions to zero out.
        noise_scale (float): Standard deviation of noise to add to kept logits.

    Returns:
        np.ndarray: Flattened sparse logits.
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



# Define process types (you can add more types if needed)
process_types = [
    # ("fourier", fourier_update, 200000),  # In this example, only the fourier update is active
     ("linear", linear_process, 50000),
    # ("linear_with_noise", linear_with_noise, 200000),
    # ("nonlinear_cosine", nonlinear_cosine, 200000),
    # ("quadratic_decay", quadratic_decay, 20000),
    # ("exponential_growth", exponential_growth, 40000),
    # ("multiplicative_noise", multiplicative_noise, 60000),
    # ("piecewise_linear", piecewise_linear, 60000),
    # ("logistic_growth", logistic_growth, 200000)
    #("sparse_90", lambda x, A, t: sparse_transition_process(x, A, t, sparsity=0.9), 50000),
    # ("sparse_99", lambda x, A, t: sparse_transition_process(x, A, t, sparsity=0.99), 50),
]

# Set up HDF5 output file and parameters for incremental saving
output_dir = "./aggregated_datasets"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir,
                           f"aggregated_birdsong_data_20_syl_linear_50_song_in_batch_75_timesteps_{timestamp}.h5")
print(f"Saving aggregated data to {output_path}...")

# Batch size for simulation to avoid OOM errors
chunk_size = 5000

# Initialize a counter to track the total number of processes written so far
total_processes_written = 0

# This list will accumulate metadata for each process type (including the process index range)
metadata_all = []

# Open the HDF5 file for writing
with h5py.File(output_path, "w") as hf:
    # Loop over each process type
    for process_name, process_fn, num_processes in process_types:
        print(f"Generating data for process: {process_name} with {num_processes} processes...")
        process_start_index = total_processes_written
        processes_remaining = num_processes

        # Process the simulations in chunks to avoid memory issues
        while processes_remaining > 0:
            current_chunk = min(chunk_size, processes_remaining)

            # Generate simulation data for the current chunk
            bigram_counts_chunk, probabilities_chunk = simulate_variable_batches_and_time_steps(
                num_processes=current_chunk,
                time_step_range=time_step_range,
                batch_size_range=batch_size_range,
                seq_range=seq_range,
                alphabet=alphabet,
                order=order,
                process_fn=process_fn
            )

            # On the very first chunk, create the datasets with expandable third dimensions.
            if "bigram_counts" not in hf:
                shape_bc = bigram_counts_chunk.shape  # e.g., (time_steps, other_dim, chunk_size)
                shape_prob = probabilities_chunk.shape
                hf.create_dataset(
                    "bigram_counts", data=bigram_counts_chunk,
                    maxshape=(shape_bc[0], shape_bc[1], None),
                    chunks=True
                )
                hf.create_dataset(
                    "probabilities", data=probabilities_chunk,
                    maxshape=(shape_prob[0], shape_prob[1], None),
                    chunks=True
                )
                total_processes_written += current_chunk
            else:
                # Append new chunk data by resizing the datasets along the third dimension
                current_shape_bc = hf["bigram_counts"].shape
                new_shape_bc = (current_shape_bc[0], current_shape_bc[1], current_shape_bc[2] + current_chunk)
                hf["bigram_counts"].resize(new_shape_bc)
                hf["bigram_counts"][:, :, current_shape_bc[2]:] = bigram_counts_chunk

                current_shape_prob = hf["probabilities"].shape
                new_shape_prob = (current_shape_prob[0], current_shape_prob[1], current_shape_prob[2] + current_chunk)
                hf["probabilities"].resize(new_shape_prob)
                hf["probabilities"][:, :, current_shape_prob[2]:] = probabilities_chunk

                total_processes_written += current_chunk

            processes_remaining -= current_chunk
            hf.flush()  # Flush data to disk after each chunk
            print(f"Processes remaining: {processes_remaining}")

        # Record metadata for this process type along with its index range
        metadata_all.append({
            "process": process_name,
            "num_processes": num_processes,
            "order": order,
            "start_index": process_start_index,
            "end_index": total_processes_written
        })

    # Save metadata as an attribute (you could also use a separate dataset if needed)
    hf.attrs["metadata"] = str(metadata_all)

print("Data aggregation complete.")
