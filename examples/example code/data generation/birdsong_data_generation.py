
import h5py
import numpy as np

# === Helper Functions === #

# def softmax(z):
#     """
#     Compute the softmax of an array.
#
#     Args:
#         z (np.ndarray): Input array.
#
#     Returns:
#         np.ndarray: Softmax-normalized probabilities.
#     """
#     exp_z = np.exp(z - np.max(z))  # Numerical stability with max subtraction
#     return exp_z / exp_z.sum()

# Define a custom softmax that returns zeros if all values are the mask_value.
def masked_softmax(x, mask_value=-1e8):
    # Check if all entries in x are equal to mask_value.
    if np.all(x == mask_value):
        return np.zeros_like(x)
    else:
        # Compute standard softmax in a numerically stable way.
        x_stable = x - np.max(x)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x)


# def simulate_birdsong(num_batches, batch_size, seq_range, alphabet, num_processes, order=1, process_fn=None):
#     """
#     Simulate song sequences and compute n-gram counts and probability distributions for multiple processes.
#
#     Args:
#         num_batches (int): Number of time steps (batches) per process.
#         batch_size (int): Number of sequences per batch.
#         seq_range (tuple): Minimum and maximum sequence lengths.
#         alphabet (list): List of symbols in the alphabet.
#         num_processes (int): Number of independent processes.
#         order (int): Markov order (1 for bigram, 2 for trigram).
#         process_fn (callable): Function to modify the transition matrix at each time step.
#
#     Returns:
#         np.ndarray: Tensor of n-gram counts of shape (alphabet_size**(order+1), num_batches, num_processes).
#         np.ndarray: Tensor of probability distributions of shape (alphabet_size**(order+1), num_batches, num_processes).
#     """
#     alphabet_size = len(alphabet)
#     ngram_size = alphabet_size ** (order + 1)
#
#     # Initialize tensors for storing n-gram counts and probabilities
#     ngram_counts = np.zeros((ngram_size, num_batches, num_processes), dtype=np.float32)
#     probabilities = np.zeros((ngram_size, num_batches, num_processes), dtype=np.float32)
#
#     for p in range(num_processes):
#         x_init = x_init_maker(alphabet, order=order)
#         x = x_init.copy()  # Initialize the transition matrix logits
#
#         # Compute final state for the linear transition
#         x_final = np.random.randint(5, size=x.shape).astype(float)
#         x_final[x_init == 1e8] = x_init[x_init == 1e8]
#         x_final[x_init == -1e8] = x_init[x_init == -1e8]
#
#         # Compute the change matrix A (default linear update)
#         with np.errstate(invalid='ignore'):
#             A = (x_final - x) / num_batches
#         A[np.isnan(A)] = 0
#
#         for t in range(num_batches):
#             # Reshape transition logits for n-gram probabilities
#             logits = x.reshape((alphabet_size,) * (order + 1))
#             probs_matrix = np.apply_along_axis(softmax, -1, logits)
#
#             # Simulate batch of sequences
#             batch_ngram_counts = np.zeros((ngram_size, batch_size), dtype=np.float32)
#
#             for seq_idx in range(batch_size):
#                 sequence = ['<']  # Start every sequence with '<'
#                 while len(sequence) < seq_range[1] + 1:
#                     if len(sequence) < order:
#                         # Handle missing context at the start (assume '<' fills the gap)
#                         prev_indices = tuple(alphabet.index('<') for _ in range(order - len(sequence)))
#                         prev_indices += tuple(alphabet.index(c) for c in sequence)
#                     else:
#                         prev_indices = tuple(alphabet.index(c) for c in sequence[-order:])
#
#                     # Get transition probabilities
#                     probs = probs_matrix[prev_indices]
#                     next_phrase_idx = np.random.choice(alphabet_size, p=probs)
#                     next_phrase = alphabet[next_phrase_idx]
#                     sequence.append(next_phrase)
#
#                     if next_phrase == '>':
#                         break  # End sequence at '>'
#
#                 # Add n-gram counts
#                 if len(sequence) > seq_range[0] + 1 and sequence[-1] == '>':
#                     seq_embedding = np.zeros(ngram_size)
#                     for i in range(len(sequence) - order):
#                         indices = tuple(alphabet.index(sequence[j]) for j in range(i, i + order + 1))
#                         flat_idx = np.ravel_multi_index(indices, (alphabet_size,) * (order + 1))
#                         seq_embedding[flat_idx] += 1
#                     batch_ngram_counts[:, seq_idx] = seq_embedding
#
#             # Aggregate n-gram counts and probabilities
#             ngram_counts[:, t, p] = batch_ngram_counts.sum(axis=1)
#
#             # Reshape to treat rows as groups
#             reshaped_counts = ngram_counts.reshape(alphabet_size ** order, alphabet_size, ngram_counts.shape[1],
#                                                    ngram_counts.shape[2])
#
#             # Sum each row across alphabet_size dimension
#             row_sums = reshaped_counts.sum(axis=1, keepdims=True)
#
#             # Avoid division by zero
#             row_sums[row_sums == 0] = 1
#
#             # Normalize the counts row-wise
#             normalized_counts = reshaped_counts / row_sums
#
#             # Reshape back to original dimensions if needed
#             normalized_counts = normalized_counts.reshape(ngram_counts.shape)
#             probabilities[:, t, p] = probs_matrix.flatten()
#
#             # Update transition matrix using custom or default process function
#             if process_fn:
#                 x = process_fn(x, A, t)
#             else:
#                 x += A
#
#     return normalized_counts, probabilities

# def simulate_birdsong(num_batches, batch_size, seq_range, alphabet, num_processes, order=1, process_fn=None):
#     """
#     Simulate song sequences and compute n-gram counts and probability distributions for multiple processes.
#
#     Args:
#         num_batches (int): Number of time steps (batches) per process.
#         batch_size (int): Number of sequences per batch.
#         seq_range (tuple): Minimum and maximum sequence lengths.
#         alphabet (list): List of symbols in the alphabet.
#         num_processes (int): Number of independent processes.
#         order (int): Markov order (1 for bigram, 2 for trigram).
#         process_fn (callable): Function to modify the transition matrix at each time step.
#
#     Returns:
#         np.ndarray: Tensor of n-gram counts of shape (alphabet_size**(order+1), num_batches, num_processes).
#         np.ndarray: Tensor of probability distributions of shape (alphabet_size**(order+1), num_batches, num_processes).
#     """
#     alphabet_size = len(alphabet)
#     ngram_size = alphabet_size ** (order + 1)
#
#     # Initialize tensors for storing n-gram counts and (for record-keeping) probabilities.
#     ngram_counts = np.zeros((ngram_size, num_batches, num_processes), dtype=np.float32)
#     probabilities = np.zeros((ngram_size, num_batches, num_processes), dtype=np.float32)
#
#     # Get indices for special symbols.
#     idx_start = alphabet.index('<')
#     idx_end   = alphabet.index('>')
#     idx_other = [i for i in range(alphabet_size) if i not in [idx_start, idx_end]]
#     large_negative = -1e8
#
#     for p in range(num_processes):
#         # Initialize transition logits using your x_init_maker.
#         x_init = x_init_maker(alphabet, order=order)
#         x = x_init.copy()  # current transition logits
#
#         # Compute a final target transition matrix (logits) for the linear update.
#         x_final = np.random.randint(5, size=x.shape).astype(float)
#         x_final[x_init == -large_negative] = x_init[x_init == -large_negative]
#         x_final[x_init == large_negative] = x_init[x_init == large_negative]
#
#         with np.errstate(invalid='ignore'):
#             A = (x_final - x) / num_batches
#         A[np.isnan(A)] = 0
#
#         for t in range(num_batches):
#             # Reshape the transition logits into a tensor of shape:
#             #   (alphabet_size,)^(order+1)
#             logits = x.reshape((alphabet_size,) * (order + 1))
#             # Compute the transition probabilities along the last axis.
#             probs_matrix = np.apply_along_axis(masked_softmax, -1, logits, mask_value=large_negative)
#
#             # We'll collect n-gram counts for this batch (across batch_size sequences)
#             batch_ngram_counts = np.zeros((ngram_size, batch_size), dtype=np.float32)
#
#             for seq_idx in range(batch_size):
#                 # --- Initialize the sequence ---
#                 if order == 1:
#                     # For a first-order chain, simply start with '<'.
#                     sequence = ['<']
#                     current_state = alphabet.index('<')
#                 elif order == 2:
#                     # For a second-order chain, initialize the state to the forced restart (>,<).
#                     # This is the only legal state starting with '>' and it forces a chain restart.
#                     state = (idx_end, idx_start)
#                     # We output only the second symbol as the beginning of the chain.
#                     sequence = [alphabet[idx_start]]
#                 else:
#                     raise ValueError("Only order==1 or order==2 are supported.")
#
#                 # --- Simulate the sequence ---
#                 while len(sequence) < seq_range[1]:
#                     if order == 1:
#                         # For order 1, the context is the last symbol.
#                         context = alphabet.index(sequence[-1])
#                         probs = probs_matrix[context]
#                         next_phrase_idx = np.random.choice(alphabet_size, p=probs)
#                         next_phrase = alphabet[next_phrase_idx]
#                         sequence.append(next_phrase)
#                     elif order == 2:
#                         # For order 2, use the current state (an ordered pair) as context.
#                         # (The probability vector is indexed by a tuple.)
#                         probs = probs_matrix[state]
#                         next_phrase_idx = np.random.choice(alphabet_size, p=probs)
#                         next_phrase = alphabet[next_phrase_idx]
#                         sequence.append(next_phrase)
#                         # Update the state by shifting: (current second symbol, new symbol)
#                         state = (state[1], next_phrase_idx)
#                     if next_phrase == '>':
#                         break  # End sequence when the end symbol is reached.
#
#                 # --- Collect n-gram counts if the sequence meets the minimum length and ends with '>' ---
#                 if len(sequence) > seq_range[0] + 1 and sequence[-1] == '>':
#                     seq_embedding = np.zeros(ngram_size)
#                     # For each n-gram (of length order+1) in the sequence, update counts.
#                     for i in range(len(sequence) - order):
#                         indices = tuple(alphabet.index(sequence[j]) for j in range(i, i + order + 1))
#                         flat_idx = np.ravel_multi_index(indices, (alphabet_size,) * (order + 1))
#                         seq_embedding[flat_idx] += 1
#                     batch_ngram_counts[:, seq_idx] = seq_embedding
#
#             # Aggregate n-gram counts for this batch into process p at time t.
#             ngram_counts[:, t, p] = batch_ngram_counts.sum(axis=1)
#
#             # (For record-keeping, here we simply store the flattened probabilities tensor.)
#             probabilities[:, t, p] = probs_matrix.flatten()
#
#             # Update the transition logits using the process function (if provided) or a default linear update.
#             if process_fn:
#                 x = process_fn(x, A, t)
#             else:
#                 x += A
#
#     # Normalize the n-gram counts within groups corresponding to the first order n-grams.
#     reshaped_counts = ngram_counts.reshape(alphabet_size ** order, alphabet_size, ngram_counts.shape[1],
#                                            ngram_counts.shape[2])
#     row_sums = reshaped_counts.sum(axis=1, keepdims=True)
#     row_sums[row_sums == 0] = 1  # avoid division by zero
#     normalized_counts = reshaped_counts / row_sums
#     normalized_counts = normalized_counts.reshape(ngram_counts.shape)
#
#     return normalized_counts, probabilities

def simulate_one_song_order1(seq_range, probs_matrix, alphabet):
    """
    Simulate one song for a first-order chain.
    The song must:
      - start with '<'
      - end with '>'
      - have length between seq_range[0] and seq_range[1]
    If the simulated song reaches the maximum length without ending with '>',
    the simulation is restarted.

    Args:
        seq_range (tuple): (min_length, max_length)
        probs_matrix: A probability matrix of shape (alphabet_size, alphabet_size) where
                      row i gives the distribution over the next symbol when in state i.
        alphabet (list): List of symbols.

    Returns:
        song (list of str): A valid song.
    """
    min_len, max_len = seq_range
    alphabet_size = len(alphabet)

    while True:
        song = ['<']  # always start with '<'
        # Simulate until we either hit an end token or reach max length.
        while len(song) < max_len:
            context = alphabet.index(song[-1])
            probs = probs_matrix[context]
            next_idx = np.random.choice(alphabet_size, p=probs)
            song.append(alphabet[next_idx])
            if song[-1] == '>':
                break
        # Check if song is valid: must be at least min_len and end with '>'.
        if len(song) >= min_len and song[-1] == '>':
            return song
        # Otherwise, discard and resimulate.


def simulate_one_song_order2(seq_range, probs_matrix, alphabet, idx_start, idx_end):
    """
    Simulate one song for a second-order chain.
    The song must:
      - start with '<'
      - end with '>'
      - have length between seq_range[0] and seq_range[1]
    For order==2 we maintain a state (a tuple of two indices) that is used to index into
    the probability tensor. The forced restart state is (>,<), so we initialize accordingly.

    If the simulated song reaches the maximum length without ending with '>',
    the simulation is restarted.

    Args:
        seq_range (tuple): (min_length, max_length)
        probs_matrix: A probability tensor of shape (alphabet_size, alphabet_size, alphabet_size)
                      (after reshaping the logits) so that for a state (i, j) the vector probs_matrix[(i,j)]
                      gives the distribution over the next symbol.
        alphabet (list): List of symbols.
        idx_start (int): Index of the start symbol '<'.
        idx_end (int): Index of the end symbol '>'.

    Returns:
        song (list of str): A valid song.
    """
    min_len, max_len = seq_range
    alphabet_size = len(alphabet)

    while True:
        song = [alphabet[idx_start]]  # song starts with '<'
        state = (idx_end, idx_start)  # forced restart state (>,<)
        # Simulate until reaching an end or maximum length.
        while len(song) < max_len:
            probs = probs_matrix[state]
            next_idx = np.random.choice(alphabet_size, p=probs)
            song.append(alphabet[next_idx])
            state = (state[1], next_idx)
            if song[-1] == '>':
                break
        if len(song) >= min_len and song[-1] == '>':
            return song
        # Otherwise, resimulate by starting over.


# --- Modified simulate_birdsong function ---
def simulate_birdsong(num_batches, batch_size, seq_range, alphabet, num_processes, order=1, process_fn=None):
    """
    Simulate song sequences and compute n-gram counts and probability distributions.

    For each process and each batch (time step), we simulate exactly batch_size songs.
    Each song is simulated independently so that it starts with '<' and ends with '>',
    and its length lies between seq_range[0] and seq_range[1].
    Then we concatenate these songs (in order) to form one long sequence.
    n-gram counts (bigram if order==1, trigram if order==2) are then computed over that concatenated sequence.

    Args:
        num_batches (int): Number of batches (time steps) per process.
        batch_size (int): Number of songs per batch.
        seq_range (tuple): (min_song_length, max_song_length) for each song.
        alphabet (list): List of symbols (including '<' and '>').
        num_processes (int): Number of independent processes.
        order (int): Markov order (1 for bigram, 2 for trigram).
        process_fn (callable): Function to update the transition matrix at each time step.

    Returns:
        ngram_counts: np.ndarray of shape (alphabet_size^(order+1), num_batches, num_processes)
        probabilities: np.ndarray of shape (alphabet_size^(order+1), num_batches, num_processes)
    """
    alphabet_size = len(alphabet)
    ngram_size = alphabet_size ** (order + 1)
    ngram_counts = np.zeros((ngram_size, num_batches, num_processes), dtype=np.float32)
    probabilities = np.zeros((ngram_size, num_batches, num_processes), dtype=np.float32)

    idx_start = alphabet.index('<')
    idx_end = alphabet.index('>')
    large_negative = -1e8

    for p in range(num_processes):
        # Initialize the transition matrix logits.
        x_init = x_init_maker(alphabet, order=order)
        x = x_init.copy()

        # Compute a target transition matrix for the update.
        x_final = np.random.randint(5, size=x.shape).astype(float)
        x_final[x_init == -large_negative] = x_init[x_init == -large_negative]
        x_final[x_init == large_negative] = x_init[x_init == large_negative]
        with np.errstate(invalid='ignore'):
            A = (x_final - x) / num_batches
        A[np.isnan(A)] = 0

        for t in range(num_batches):
            # Reshape logits to shape (alphabet_size)^(order+1) and compute probabilities.
            logits = x.reshape((alphabet_size,) * (order + 1))
            probs_matrix = np.apply_along_axis(masked_softmax, -1, logits, mask_value=large_negative)
            # Note: In the order==1 case, probs_matrix is indexed by an integer (current symbol);
            # in order==2, it is indexed by a tuple (current state).

            # For this batch, simulate exactly batch_size songs and concatenate them.
            songs = []
            for _s in range(batch_size):
                if order == 1:
                    song = simulate_one_song_order1(seq_range, probs_matrix, alphabet)
                elif order == 2:
                    song = simulate_one_song_order2(seq_range, probs_matrix, alphabet, idx_start, idx_end)
                else:
                    raise ValueError("Only order==1 or order==2 supported.")
                songs.append(song)
            # Concatenate the songs (without adding any extra tokens; each song already begins with '<' and ends with '>')
            concatenated = []
            for song in songs:
                concatenated.extend(song)

            # Compute n-gram counts over the concatenated sequence.
            seq_embedding = np.zeros(ngram_size)
            if len(concatenated) >= (seq_range[0] + 1):
                for i in range(len(concatenated) - order):
                    indices = tuple(alphabet.index(concatenated[j]) for j in range(i, i + order + 1))
                    flat_idx = np.ravel_multi_index(indices, (alphabet_size,) * (order + 1))
                    seq_embedding[flat_idx] += 1

            # Here, we store the counts for this batch (for this process).
            # (If you wish to have one column per song, you can store them separately.
            # Here we sum over the batch.)
            ngram_counts[:, t, p] = seq_embedding
            probabilities[:, t, p] = probs_matrix.flatten()

            # Update the transition matrix.
            if process_fn:
                x = process_fn(x, A, t)
            else:
                x += A

    # Normalize the n-gram counts rowwise.
    # For order==1, reshape to (alphabet_size, alphabet_size, num_batches, num_processes)
    # For order==2, reshape to (alphabet_size^2, alphabet_size, num_batches, num_processes)
    reshaped_counts = ngram_counts.reshape(alphabet_size ** order, alphabet_size, ngram_counts.shape[1],
                                           ngram_counts.shape[2])
    row_sums = reshaped_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized_counts = reshaped_counts / row_sums
    normalized_counts = normalized_counts.reshape(ngram_counts.shape)

    return normalized_counts, probabilities



def save_to_hdf5(output_path, bigram_counts, probabilities):
    """
    Save the processed data into an HDF5 file format.

    Args:
        output_path (str): Path to save the HDF5 file.
        bigram_counts (np.ndarray): Tensor of bigram counts.
        probabilities (np.ndarray): Tensor of probability distributions.
    """
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('bigram_counts', data=bigram_counts)
        hf.create_dataset('probabilities', data=probabilities)

def preprocess_simulated_songs(num_batches, batch_size, seq_range, alphabet, num_processes, output_path, process_fn=None):
    """
    Simulate and preprocess song data for LFADS.

    Args:
        num_batches (int): Number of time steps (batches) per process.
        batch_size (int): Number of sequences per batch.
        seq_range (tuple): Minimum and maximum sequence lengths.
        alphabet (list): List of symbols in the alphabet.
        num_processes (int): Number of independent processes.
        output_path (str): Path to save the processed HDF5 file.
        process_fn (callable): Function to modify the transition matrix at each time step.
    """

    print("Simulating song data...")
    bigram_counts, probabilities = simulate_birdsong(num_batches, batch_size, seq_range, alphabet, num_processes, process_fn)

    print(f"Saving simulated data to {output_path}...")
    save_to_hdf5(output_path, bigram_counts, probabilities)
    print("Preprocessing complete.")

# def x_init_maker(alphabet, order=1):
#     """
#     Create the initial transition logits (x_init) for the simulation.
#     Supports bigram (order=1) and trigram (order=2) transitions.
#
#     Args:
#         alphabet (list): List of symbols in the alphabet.
#         order (int): Markov order (1 for bigram, 2 for trigram).
#
#     Returns:
#         np.ndarray: Initial logits for transition probabilities.
#     """
#     alphabet_size = len(alphabet)
#     transition_shape = (alphabet_size,) * (order + 1)
#     x_init = np.random.randint(5, size=transition_shape).astype(float)
#
#     # Special values for large negative and positive logits
#     large_negative = -1e8
#     large_positive = 1e8
#
#     # Indices for '<', 'end', and any other letter
#     idx_start = alphabet.index('<')  # Start symbol
#     idx_end = alphabet.index('>')    # End symbol
#     idx_other = [i for i in range(alphabet_size) if i not in [idx_start, idx_end]]
#
#     # Iterate through all possible transitions
#     for idx in np.ndindex(transition_shape):
#         prev_states = idx[:-1]  # Previous 'order' states
#         current_state = idx[-1]  # Current state
#
#         # Rules for 1st-Order Transitions (Bigram)
#         if order == 1:
#             # From '<' to '<' or '>' (blocked)
#             if prev_states[0] == idx_start and current_state in [idx_start, idx_end]:
#                 x_init[idx] = large_negative
#
#             # From '>' to anything but '<' (blocked), except allow '>' to '<'
#             if prev_states[0] == idx_end and current_state != idx_start:
#                 x_init[idx] = large_negative
#             elif prev_states[0] == idx_end and current_state == idx_start:
#                 x_init[idx] = large_positive
#
#             # To '<' from anywhere but '>' (blocked)
#             if current_state == idx_start and prev_states[0] != idx_end:
#                 x_init[idx] = large_negative
#
#         # Rules for 2nd-Order Transitions (Trigram)
#         if order == 2:
#             # Condition 1: (1, 1, 1) or (1, 1, end) -> Two consecutive '<' cannot be followed by '<' or 'end'
#             if prev_states[0] == idx_start and prev_states[1] == idx_start:
#                 if current_state in [idx_start, idx_end]:
#                     x_init[idx] = large_negative
#
#             # Condition 2: (1, #, 1) -> '<' cannot follow any other letter except itself
#             if prev_states[0] == idx_start and prev_states[1] in idx_other and current_state == idx_start:
#                 x_init[idx] = large_negative
#
#             # Condition 3: (1, end, #) or (1, end, end) -> No valid transitions after '<' and 'end'
#             if prev_states[0] == idx_start and prev_states[1] == idx_end:
#                 x_init[idx] = large_negative
#
#             # Condition 4: (#, end, #) or (#, end, end) -> No valid transitions after any letter and 'end'
#             if prev_states[1] == idx_end:
#                 x_init[idx] = large_negative
#
#             # Condition 5: (#, #, 1) -> '<' cannot follow two other letters
#             if prev_states[0] in idx_other and prev_states[1] in idx_other and current_state == idx_start:
#                 x_init[idx] = large_negative
#
#             # Condition 6: (end, end, #) or (end, end, end) -> No transitions after two consecutive 'end's
#             if prev_states[0] == idx_end and prev_states[1] == idx_end:
#                 x_init[idx] = large_negative
#
#             # Condition 7: (end, 1, #) or (end, 1, end) -> Only '<' can follow 'end, <'
#             if prev_states[0] == idx_end and prev_states[1] == idx_start and current_state != idx_start:
#                 x_init[idx] = large_negative
#
#             # Guaranteed Transitions:
#             # Condition 8: (#, end, 1) -> '<' must follow ('#', 'end')
#             if prev_states[0] in idx_other and prev_states[1] == idx_end and current_state == idx_start:
#                 x_init[idx] = large_positive
#
#             # Condition 9: (end, 1, 1) -> '<' must follow ('end', '<')
#             if prev_states[0] == idx_end and prev_states[1] == idx_start and current_state == idx_start:
#                 x_init[idx] = large_positive
#
#     return x_init.flatten()

def x_init_maker(alphabet, order=1):
    """
    Create the initial transition logits for a Markov chain of a given order.
    The chain’s state is either a single symbol (order==1) or an ordered pair (order==2).
    The transition matrix is constructed so that each row corresponds to one state and
    each column to a candidate next symbol. (The returned array is flattened row-wise.)

    Allowed transitions obey the following rules:
      - The start symbol "<" may appear only as the first symbol.
      - The end symbol ">" may appear only in the final (or second) position.
      - In a transition:
          * For order==1, if the current symbol is a middle symbol then allowed next symbols
            are the middle symbols plus the end symbol.
          * If the current symbol is "<" (which, for order 1, represents the beginning of a chain),
            then allowed next symbols are the middle symbols (i.e. any symbol except "<" and ">").
          * If the current symbol is ">", then the only allowed next symbol is "<"
            (to force a chain restart).
      - For order==2 the state is an ordered pair (x,y). The allowed next symbols are determined solely
        by the second symbol y:
          * If y is a middle symbol then allowed next symbols are the middle symbols plus the end symbol.
          * If y is "<" (which can only occur in the forced chaining pair (>,<)) then allowed next symbols
            are the middle symbols.
          * If y is ">" then the only allowed next symbol is "<".

    In the output logits:
      - Disallowed transitions are set to a very large negative value (to force near-zero probability).
      - When exactly one transition is allowed, its logit is set to a very large positive value.
      - Otherwise, the base logits are drawn randomly (and thus need not be uniform).

    Args:
        alphabet (list): List of symbols. Must include "<" and ">".
        order (int): Markov order (1 or 2).

    Returns:
        np.ndarray: Flattened 1D array of transition logits.
                    If order==1, shape = (alphabet_size, alphabet_size).
                    If order==2, shape = (alphabet_size**2, alphabet_size).
    """
    alphabet_size = len(alphabet)
    large_negative = -1e8
    large_positive = 1e8

    # Identify the special indices.
    idx_start = alphabet.index('<')  # start symbol
    idx_end = alphabet.index('>')  # end symbol
    # "Middle" symbols are all the ones that are neither start nor end.
    idx_other = [i for i in range(alphabet_size) if i not in [idx_start, idx_end]]

    # --- CASE: First-order Markov chain ---
    if order == 1:
        # State space: each state is a single symbol, so we have a (alphabet_size, alphabet_size) matrix.
        # Row i corresponds to the current symbol; column j corresponds to the candidate next symbol.
        #x_init = np.random.randint(5, size=(alphabet_size, alphabet_size)).astype(float)
        x_init = np.random.uniform(0, 5, size=(alphabet_size, alphabet_size)).astype(float)

        for x in range(alphabet_size):
            # Determine allowed next symbols based solely on the current state x.
            if x == idx_start:
                # For the start symbol "<", allowed next symbols are the middle symbols.
                allowed_z = idx_other.copy()
            elif x in idx_other:
                # For a middle symbol, allowed next symbols are the middle symbols (to continue)
                # or the end symbol ">" (to terminate the chain).
                allowed_z = idx_other.copy() + [idx_end]
            elif x == idx_end:
                # For the end symbol, the only allowed next symbol is the start symbol "<"
                # to force a chain restart.
                allowed_z = [idx_start]
            else:
                allowed_z = []

            # For each candidate next symbol z, if z is not allowed, block it.
            for z in range(alphabet_size):
                if z not in allowed_z:
                    x_init[x, z] = large_negative
            # If exactly one transition is allowed, force its logit high.
            if len(allowed_z) == 1:
                x_init[x, allowed_z[0]] = large_positive

        return x_init.flatten()

    # --- CASE: Second-order Markov chain ---
    elif order == 2:
        # State space: ordered pairs. There are alphabet_size**2 rows.
        # Each row corresponds to a pair (x,y) (with row index = x * alphabet_size + y).
        # The candidate next symbol z (column index) will update the pair to (y,z).
        # x_init = np.random.randint(5, size=(alphabet_size ** 2, alphabet_size)).astype(float) # for integer logits.
        x_init = np.random.uniform(0, 5, size=(alphabet_size ** 2, alphabet_size)).astype(float)

        # Define a helper to decide if an ordered pair (x,y) is legal.
        def is_legal_pair(x, y):
            if x == idx_start:
                # A pair starting with "<" is legal only if y is a middle symbol.
                return y in idx_other
            elif x in idx_other:
                # For a middle symbol x, the pair is legal if y is either a middle symbol or the end symbol.
                return (y in idx_other) or (y == idx_end)
            elif x == idx_end:
                # If the first symbol is the end symbol, then the only legal pair is (>,<) (the forced chaining pair).
                return y == idx_start
            else:
                return False

        for x in range(alphabet_size):
            for y in range(alphabet_size):
                row_index = x * alphabet_size + y
                if not is_legal_pair(x, y):
                    # If the pair (x,y) is illegal, block all transitions.
                    x_init[row_index, :] = large_negative
                    continue
                # Otherwise, determine allowed next symbols z based on the active symbol y.
                # (We are going to form the new pair (y, z).)
                if y == idx_start:
                    # The only legal occurrence of "<" is in the forced chaining pair (>,<).
                    # In that case, when forming (<,z) (i.e. starting a new chain),
                    # allowed z are the middle symbols.
                    allowed_z = idx_other.copy()
                elif y in idx_other:
                    # For a middle symbol y, allowed new symbols are any middle symbol (to continue)
                    # or the end symbol (to terminate).
                    allowed_z = idx_other.copy() + [idx_end]
                elif y == idx_end:
                    # If y is the end symbol, then the only allowed next symbol is "<" (to force a new chain).
                    allowed_z = [idx_start]
                else:
                    allowed_z = []

                # For each candidate new symbol z, if z is not allowed, block it.
                for z in range(alphabet_size):
                    if z not in allowed_z:
                        x_init[row_index, z] = large_negative
                # If exactly one transition is allowed, force its logit high.
                if len(allowed_z) == 1:
                    x_init[row_index, allowed_z[0]] = large_positive

        return x_init.flatten()

    else:
        raise ValueError("Order must be 1 or 2.")


def simulate_variable_batches_and_time_steps(
    num_processes, time_step_range, batch_size_range, seq_range, alphabet, order=1, process_fn=None
):
    """
    Simulate processes with varying time steps and batch sizes, and compute n-gram counts and probabilities.

    Args:
        num_processes (int): Number of independent processes.
        time_step_range (tuple): (min_steps, max_steps) for the number of time steps (which will correspond to num_batches).
        batch_size_range (tuple): (min_batch, max_batch) for the batch size.
        seq_range (tuple): Min and max sequence lengths.
        alphabet (list): List of symbols in the alphabet.
        order (int): Markov order (1 for bigram, 2 for trigram).
        process_fn (callable): Function to modify the transition matrix at each time step.

    Returns:
        np.ndarray: bigram_counts (ngram_size, max_time_steps, num_processes)
        np.ndarray: probabilities (ngram_size, max_time_steps, num_processes)
        np.ndarray: true_time_steps for each process
        np.ndarray: batch_sizes for each process (just a single batch size per process in this example)
    """
    alphabet_size = len(alphabet)
    ngram_size = alphabet_size ** (order + 1)

    # We need the maximum time steps to allocate arrays
    max_time_steps = time_step_range[1]
    bigram_counts = np.zeros((ngram_size, max_time_steps, num_processes), dtype=np.float32)
    probabilities = np.zeros((ngram_size, max_time_steps, num_processes), dtype=np.float32)
    true_time_steps = np.zeros(num_processes, dtype=np.int32)
    batch_sizes = np.zeros(num_processes, dtype=np.int32)

    for p in range(num_processes):
        # Decide on the number of time steps (which equals num_batches)
        num_time_steps = np.random.randint(time_step_range[0], time_step_range[1] + 1)
        true_time_steps[p] = num_time_steps

        # Decide on the batch size for this entire run
        batch_size = np.random.randint(batch_size_range[0], batch_size_range[1] + 1)
        batch_sizes[p] = batch_size

        # Now simulate once using num_batches = num_time_steps and batch_size = chosen batch_size
        bigram_counts_t, probabilities_t = simulate_birdsong(
            num_batches=num_time_steps,
            batch_size=batch_size,
            seq_range=seq_range,
            alphabet=alphabet,
            num_processes=1,  # one process at a time
            order=order,
            process_fn=process_fn,
        )
        # bigram_counts_t, probabilities_t have shape (ngram_size, num_time_steps, 1)
        # We can directly place them into the output arrays.

        # Normalize the bigram counts by the batch size if necessary
        bigram_counts[:, :num_time_steps, p] = bigram_counts_t[:, :, 0]
        probabilities[:, :num_time_steps, p] = probabilities_t[:, :, 0]

        print(f"Generated data for process: {p} / {num_processes}")

    return bigram_counts, probabilities



# # Example usage
# if __name__ == "__main__":
#     # Simulation parameters
#     alphabet = ['<', 'a', 'b', 'c', 'd', 'e', '>']  # Example alphabet
#     num_batches = 400  # Number of time steps
#     batch_size = 50  # Number of sequences per batch
#     seq_range = (5, 15)  # Min and max sequence length
#     num_processes = 500  # Number of processes
#     output_path = "./simulated_birdsong_data.h5"
#
#     # Example process function
#     def custom_process(x, A, t):
#         """
#         Custom process function to update transition matrix.
#
#         Args:
#             x (np.ndarray): Current logits.
#             A (np.ndarray): Update matrix.
#             t (int): Current time step.
#
#         Returns:
#             np.ndarray: Updated logits.
#         """
#         return x + A + np.random.normal(0, 0.1, size=x.shape)  # Add noise to the linear process
#
#     # Run preprocessing
#     preprocess_simulated_songs(num_batches, batch_size, seq_range, alphabet, num_processes, output_path, process_fn=custom_process)

