import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableGaussianPrior(nn.Module):
    def __init__(self, latent_dim, var_min=1e-6, var_max=1e3, mean_init=0.0, var_init=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.var_min = var_min
        self.var_max = var_max

        # Mean parameter
        self.mean = nn.Parameter(torch.full((latent_dim,), mean_init))

        # If var_min < var_max, variance is trainable and constrained
        if var_min < var_max:
            # Convert var_normalized to a torch tensor
            var_normalized = (var_init - var_min) / (var_max - var_min)
            var_normalized = torch.tensor(var_normalized, dtype=torch.float32)

            var_logit_init = torch.log(var_normalized / (1 - var_normalized + 1e-12))
            self.var_logit = nn.Parameter(var_logit_init * torch.ones(latent_dim))
            self.trainable_var = True
        else:
            # Fixed variance
            self.register_buffer('fixed_var', torch.full((latent_dim,), var_init))
            self.trainable_var = False

    def forward(self, batch_size, time_steps):
        # Broadcast mean and logvar to match (batch_size, time_steps, latent_dim)
        mean = self.mean.unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, -1)
        if self.trainable_var:
            # Constrain variance between var_min and var_max
            var = self.var_min + (self.var_max - self.var_min) * torch.sigmoid(self.var_logit)
            logvar = var.log().unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, -1)
        else:
            logvar = self.fixed_var.log().unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, -1)

        return mean, logvar



# def kl_divergence(z_mean, z_logvar):
#     """
#     Compute KL divergence between the learned posterior and a unit Gaussian prior.
#
#     Args:
#         z_mean (torch.Tensor): Mean of the posterior (batch_size, time_steps, latent_dim)
#         z_logvar (torch.Tensor): Log variance of the posterior (batch_size, time_steps, latent_dim)
#
#     Returns:
#         torch.Tensor: KL divergence for the entire batch (scalar).
#     """
#     # Apply the formula along the latent_dim dimension
#     # Resulting shape after sum along dim=-1: (batch_size, time_steps)
#     kl = -0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - torch.exp(z_logvar), dim=-1)
#
#     # Now we have a KL divergence per time step. We have two main options:
#     # Option 1: sum over time steps and then average over batch
#     kl = kl.sum(dim=-1)  # now shape is (batch_size,)
#     return kl.mean()
#
#     # If you prefer to average over time steps as well, you could do:
#     # kl = kl.mean(dim=-1)  # average over time steps, shape: (batch_size,)
#     # return kl.mean()

def kl_divergence_with_learnable_prior(z_mean, z_logvar, p_mean, p_logvar):
    # z_mean, z_logvar: (batch_size, time_steps, latent_dim)
    # p_mean, p_logvar: (batch_size, time_steps, latent_dim)
    kl = 0.5 * torch.sum(
        p_logvar - z_logvar +
        (torch.exp(z_logvar) + (z_mean - p_mean)**2) / torch.exp(p_logvar) - 1,
        dim=-1
    )
    # kl shape: (batch_size, time_steps)
    kl = kl.sum(dim=-1)  # sum over time steps
    return kl.mean()     # mean over batch


def reconstruction_loss(predicted_logits, true_probabilities, alphabet_size):
    """
    Compute reconstruction loss using cross-entropy between predicted probabilities and true probabilities.

    Args:
        predicted_logits (torch.Tensor): Predicted logits (batch_size, time_steps, alphabet_size**2).
        true_probabilities (torch.Tensor): True probabilities (batch_size, time_steps, alphabet_size**2).
        alphabet_size (int): Size of the alphabet.

    Returns:
        torch.Tensor: Mean reconstruction loss across the batch.
    """
    # Reshape to separate rows and columns of the transition matrix
    batch_size, time_steps, _ = predicted_logits.shape
    predicted_logits = predicted_logits.view(batch_size, time_steps, alphabet_size, alphabet_size)
    true_probabilities = true_probabilities.view(batch_size, time_steps, alphabet_size, alphabet_size)

    # Compute probabilities from logits using softmax row-wise
    predicted_probabilities = F.softmax(predicted_logits, dim=-1)  # Shape: [batch_size, time_steps, alphabet_size, alphabet_size]

    # Add a small value to avoid numerical issues with log(0)
    epsilon = 1e-8
    predicted_probabilities = predicted_probabilities.clamp(min=epsilon)
    true_probabilities = true_probabilities.clamp(min=epsilon)

    # Compute cross-entropy (row-wise comparison for each transition matrix row)
    cross_entropy_loss = -true_probabilities * torch.log(predicted_probabilities)  # Element-wise
    loss_per_row = cross_entropy_loss.sum(dim=-1)  # Sum over columns (row-wise comparison)

    # Average over rows, time steps, and batch
    mean_loss = loss_per_row.mean()  # Scalar loss
    return mean_loss


# def birdsong_loss(predicted_logits, true_probabilities, z_mean, z_logvar, alphabet_size, beta=1.0):
#     """
#     Combined loss for BirdsongLFADS.
#
#     Args:
#         predicted_logits (torch.Tensor): Predicted logits (batch_size, time_steps, alphabet_size, alphabet_size).
#         true_probabilities (torch.Tensor): True probabilities (batch_size, time_steps, alphabet_size, alphabet_size).
#         z_mean (torch.Tensor): Mean of the posterior latent distribution (batch_size, latent_dim).
#         z_logvar (torch.Tensor): Log variance of the posterior latent distribution (batch_size, latent_dim).
#         alphabet_size (int): Size of the alphabet.
#         beta (float): Weighting factor for KL divergence.
#
#     Returns:
#         torch.Tensor: Total loss (reconstruction + beta * KL divergence).
#     """
#     # Flatten logits and probabilities back to original shape
#     predicted_logits = predicted_logits.view(
#         predicted_logits.shape[0],  # Batch size
#         predicted_logits.shape[1],  # Time steps
#         alphabet_size**2            # Flattened vocab size
#     )
#     true_probabilities = true_probabilities.view(
#         true_probabilities.shape[0],  # Batch size
#         true_probabilities.shape[1],  # Time steps
#         alphabet_size**2              # Flattened vocab size
#     )
#
#     # Compute reconstruction loss
#     rec_loss = reconstruction_loss(predicted_logits, true_probabilities, alphabet_size)
#
#     # Compute KL divergence
#     kl_loss = kl_divergence(z_mean, z_logvar)
#
#     # Combine losses
#     total_loss = rec_loss + beta * kl_loss
#
#     return total_loss, rec_loss, kl_loss

def birdsong_loss(predicted_logits, true_probabilities, z_mean, z_logvar, p_mean, p_logvar, alphabet_size, beta=1.0):
    """
    Combined loss for BirdsongLFADS with a learnable Gaussian prior p(z).

    Args:
        predicted_logits (torch.Tensor): (batch_size, time_steps, alphabet_size, alphabet_size)
        true_probabilities (torch.Tensor): (batch_size, time_steps, alphabet_size, alphabet_size)
        z_mean (torch.Tensor): posterior mean (batch_size, time_steps, latent_dim)
        z_logvar (torch.Tensor): posterior log variance (batch_size, time_steps, latent_dim)
        p_mean (torch.Tensor): prior mean (batch_size, time_steps, latent_dim)
        p_logvar (torch.Tensor): prior log variance (batch_size, time_steps, latent_dim)
        alphabet_size (int): size of the alphabet
        beta (float): weighting factor for KL divergence

    Returns:
        torch.Tensor: total_loss
        torch.Tensor: rec_loss
        torch.Tensor: kl_loss
    """
    # Flatten logits and probabilities
    batch_size, time_steps, _, _ = predicted_logits.shape
    predicted_logits_flat = predicted_logits.view(batch_size, time_steps, alphabet_size**2)
    true_probabilities_flat = true_probabilities.view(batch_size, time_steps, alphabet_size**2)

    rec_loss = reconstruction_loss(predicted_logits_flat, true_probabilities_flat, alphabet_size)

    # Compute KL divergence with the learnable prior
    kl_loss = kl_divergence_with_learnable_prior(z_mean, z_logvar, p_mean, p_logvar)

    total_loss = rec_loss + beta * kl_loss
    return total_loss, rec_loss, kl_loss


# # Example usage
# if __name__ == "__main__":
#     # Example shapes
#     batch_size = 32
#     time_steps = 400
#     alphabet_size = 7
#     vocab_size = alphabet_size ** 2
#     latent_dim = 16
#
#     # Example tensors
#     predicted_logits = torch.rand(batch_size, time_steps, vocab_size)
#     true_probabilities = F.softmax(torch.rand(batch_size, time_steps, vocab_size), dim=-1)
#     z_mean = torch.rand(batch_size, latent_dim)
#     z_logvar = torch.rand(batch_size, latent_dim)
#
#     # Compute loss
#     total_loss, rec_loss, kl_loss = birdsong_loss(predicted_logits, true_probabilities, z_mean, z_logvar, beta=1.0)
#     print(f"Total Loss: {total_loss.item():.4f}")
#     print(f"Reconstruction Loss: {rec_loss.item():.4f}")
#     print(f"KL Divergence: {kl_loss.item():.4f}")
