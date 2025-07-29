import h5py
import torch
from torch.utils.data import Dataset

# class BirdsongDataset(Dataset):
#     """
#     PyTorch Dataset for Birdsong bigram counts and probabilities.
#     """
#     def __init__(self, h5_path):
#         """
#         Args:
#             h5_path (str): Path to the HDF5 file containing bigram counts and probabilities.
#         """
#         self.h5_path = h5_path
#         with h5py.File(h5_path, 'r') as hf:
#             self.bigram_counts = hf['bigram_counts'][:]
#             self.probabilities = hf['probabilities'][:]
#
#     def __len__(self):
#         """
#         Returns:
#             int: Number of processes (sequences) in the dataset.
#         """
#         return self.bigram_counts.shape[2]
#
#     def __getitem__(self, idx):
#         """
#         Args:
#             idx (int): Index of the process to fetch.
#
#         Returns:
#             tuple: (bigram_counts, probabilities) for a specific process.
#         """
#         bigram_counts = self.bigram_counts[:, :, idx]  # Shape: (alphabet_size**2, time_steps)
#         probabilities = self.probabilities[:, :, idx]  # Shape: (alphabet_size**2, time_steps)
#
#         # Transpose to match PyTorch conventions: (time_steps, alphabet_size**2)
#         bigram_counts = bigram_counts.T  # Shape: (time_steps, alphabet_size**2)
#         probabilities = probabilities.T  # Shape: (time_steps, alphabet_size**2)
#
#         return torch.tensor(bigram_counts, dtype=torch.float32), torch.tensor(probabilities, dtype=torch.float32)


class BirdsongDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = None  # Will be opened lazily per worker
        with h5py.File(h5_path, 'r') as hf:
            self.num_samples = hf['bigram_counts'].shape[2]
            self.time_steps = hf['bigram_counts'].shape[1]
            self.feature_dim = hf['bigram_counts'].shape[0]

    def _init_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.bigram_counts = self.h5_file['bigram_counts']
            self.probabilities = self.h5_file['probabilities']

    def __getitem__(self, idx):
        self._init_h5()
        bigram_counts = self.bigram_counts[:, :, idx].T  # (T, F)
        probabilities = self.probabilities[:, :, idx].T
        return torch.tensor(bigram_counts, dtype=torch.float32), torch.tensor(probabilities, dtype=torch.float32)

    def __len__(self):
        return self.num_samples


# Example usage
# if __name__ == "__main__":
#     h5_path = "./aggregated_birdsong_data.h5"  # Path to your dataset
#     dataset = BirdsongDataset(h5_path)
#
#     # Create DataLoader
#     batch_size = 32
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     # Iterate through DataLoader
#     for batch_idx, (bigram_counts, probabilities) in enumerate(dataloader):
#         print(f"Batch {batch_idx}:")
#         print(f"Bigram counts shape: {bigram_counts.shape}")  # (batch_size, time_steps, alphabet_size**2)
#         print(f"Probabilities shape: {probabilities.shape}")  # (batch_size, time_steps, alphabet_size**2)
#         break
