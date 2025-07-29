"""
Loss functions for birdsong analysis.

This module contains loss functions for LFADS-style birdsong analysis,
including reconstruction loss, KL divergence, and learnable priors.
"""


import torch
import torch.nn as nn
import torch.nn.functional as f


class LearnableGaussianPrior(nn.Module):
    """
    Learnable Gaussian prior for variational inference.

    This module implements a learnable Gaussian prior that can be used
    in variational autoencoders to learn the prior distribution.
    """

    def __init__(self, latent_dim: int, var_min: float = 1e-6, var_max: float = 1e3,
                 mean_init: float = 0.0, var_init: float = 1.0):
        """
        Initialize the learnable Gaussian prior.

        Args:
            latent_dim: Dimension of the latent space
            var_min: Minimum variance constraint
            var_max: Maximum variance constraint
            mean_init: Initial mean value
            var_init: Initial variance value
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.var_min = var_min
        self.var_max = var_max

        # Mean parameter
        self.mean = nn.Parameter(torch.full((latent_dim,), mean_init))

        # Variance parameter (trainable or fixed)
        if var_min < var_max:
            # Trainable variance with constraints
            var_normalized = (var_init - var_min) / (var_max - var_min)
            var_normalized = torch.tensor(var_normalized, dtype=torch.float32)
            var_logit_init = torch.log(var_normalized / (1 - var_normalized + 1e-12))
            self.var_logit = nn.Parameter(var_logit_init * torch.ones(latent_dim))
            self.trainable_var = True
        else:
            # Fixed variance
            self.register_buffer('fixed_var', torch.full((latent_dim,), var_init))
            self.trainable_var = False

    def forward(self, batch_size: int, time_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prior mean and log variance.

        Args:
            batch_size: Batch size
            time_steps: Number of time steps

        Returns:
            Tuple of (mean, logvar) tensors of shape (batch_size, time_steps, latent_dim)
        """
        # Broadcast mean and logvar to match (batch_size, time_steps, latent_dim)
        mean = self.mean.unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, -1)

        if self.trainable_var:
            # Constrain variance between var_min and var_max
            var = self.var_min + (self.var_max - self.var_min) * torch.sigmoid(self.var_logit)
            logvar = var.log().unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, -1)
        else:
            logvar = self.fixed_var.log().unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, -1)

        return mean, logvar


def kl_divergence_with_learnable_prior(z_mean: torch.Tensor, z_logvar: torch.Tensor,
                                      p_mean: torch.Tensor, p_logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between posterior and learnable prior.

    Args:
        z_mean: Posterior mean (batch_size, time_steps, latent_dim)
        z_logvar: Posterior log variance (batch_size, time_steps, latent_dim)
        p_mean: Prior mean (batch_size, time_steps, latent_dim)
        p_logvar: Prior log variance (batch_size, time_steps, latent_dim)

    Returns:
        KL divergence averaged over batch
    """
    # Numerical stability check
    if torch.isnan(z_mean).any() or torch.isnan(z_logvar).any():
        raise ValueError("NaN values in posterior parameters")
    if torch.isnan(p_mean).any() or torch.isnan(p_logvar).any():
        raise ValueError("NaN values in prior parameters")

    # Compute KL divergence
    kl = 0.5 * torch.sum(
        p_logvar - z_logvar +
        (torch.exp(z_logvar) + (z_mean - p_mean)**2) / torch.exp(p_logvar) - 1,
        dim=-1
    )

    # Sum over time steps, mean over batch
    kl = kl.sum(dim=-1)
    return kl.mean()


def reconstruction_loss(predicted_logits: torch.Tensor, true_probabilities: torch.Tensor,
                      alphabet_size: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute reconstruction loss using cross-entropy.

    Args:
        predicted_logits: Predicted logits (batch_size, time_steps, alphabet_size**2)
        true_probabilities: True probabilities (batch_size, time_steps, alphabet_size**2)
        alphabet_size: Size of the alphabet
        eps: Small value to avoid numerical issues

    Returns:
        Mean reconstruction loss across the batch
    """
    # Numerical stability check
    if torch.isnan(predicted_logits).any():
        raise ValueError("NaN values in predicted logits")
    if torch.isnan(true_probabilities).any():
        raise ValueError("NaN values in true probabilities")

    # Reshape to separate rows and columns of the transition matrix
    batch_size, time_steps, _ = predicted_logits.shape
    predicted_logits = predicted_logits.view(batch_size, time_steps, alphabet_size, alphabet_size)
    true_probabilities = true_probabilities.view(batch_size, time_steps, alphabet_size, alphabet_size)

    # Compute probabilities from logits using softmax row-wise
    predicted_probabilities = f.softmax(predicted_logits, dim=-1)

    # Add small value to avoid numerical issues with log(0)
    predicted_probabilities = predicted_probabilities.clamp(min=eps)
    true_probabilities = true_probabilities.clamp(min=eps)

    # Compute cross-entropy (row-wise comparison for each transition matrix row)
    cross_entropy_loss = -true_probabilities * torch.log(predicted_probabilities)
    loss_per_row = cross_entropy_loss.sum(dim=-1)  # Sum over columns

    # Average over rows, time steps, and batch
    return loss_per_row.mean()


def birdsong_loss(predicted_logits: torch.Tensor, true_probabilities: torch.Tensor,
                  z_mean: torch.Tensor, z_logvar: torch.Tensor,
                  p_mean: torch.Tensor, p_logvar: torch.Tensor,
                  alphabet_size: int, beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss for BirdsongLFADS with learnable Gaussian prior.

    Args:
        predicted_logits: Predicted logits (batch_size, time_steps, alphabet_size, alphabet_size)
        true_probabilities: True probabilities (batch_size, time_steps, alphabet_size, alphabet_size)
        z_mean: Posterior mean (batch_size, time_steps, latent_dim)
        z_logvar: Posterior log variance (batch_size, time_steps, latent_dim)
        p_mean: Prior mean (batch_size, time_steps, latent_dim)
        p_logvar: Prior log variance (batch_size, time_steps, latent_dim)
        alphabet_size: Size of the alphabet
        beta: Weighting factor for KL divergence

    Returns:
        Tuple of (total_loss, rec_loss, kl_loss)
    """
    # Flatten logits and probabilities
    batch_size, time_steps, _, _ = predicted_logits.shape
    predicted_logits_flat = predicted_logits.view(batch_size, time_steps, alphabet_size**2)
    true_probabilities_flat = true_probabilities.view(batch_size, time_steps, alphabet_size**2)

    # Compute reconstruction loss
    rec_loss = reconstruction_loss(predicted_logits_flat, true_probabilities_flat, alphabet_size)

    # Compute KL divergence with the learnable prior
    kl_loss = kl_divergence_with_learnable_prior(z_mean, z_logvar, p_mean, p_logvar)

    # Combine losses
    total_loss = rec_loss + beta * kl_loss

    return total_loss, rec_loss, kl_loss


class BirdsongLoss(nn.Module):
    """
    Unified loss interface for birdsong analysis.

    This class provides a configurable loss function that can handle
    different loss components and weighting schemes.
    """

    def __init__(self, alphabet_size: int, beta: float = 1.0,
                 use_learnable_prior: bool = False, eps: float = 1e-8):
        """
        Initialize the loss function.

        Args:
            alphabet_size: Size of the alphabet
            beta: Weighting factor for KL divergence
            use_learnable_prior: Whether to use learnable prior
            eps: Small value for numerical stability
        """
        super().__init__()
        self.alphabet_size = alphabet_size
        self.beta = beta
        self.use_learnable_prior = use_learnable_prior
        self.eps = eps

        if use_learnable_prior:
            self.prior = LearnableGaussianPrior(latent_dim=16)  # Default latent dim
        else:
            self.prior = None

    def forward(self, predicted_logits: torch.Tensor, true_probabilities: torch.Tensor,
                z_mean: torch.Tensor, z_logvar: torch.Tensor,
                batch_size: int, time_steps: int) -> dict[str, torch.Tensor]:
        """
        Compute the total loss.

        Args:
            predicted_logits: Predicted logits
            true_probabilities: True probabilities
            z_mean: Posterior mean
            z_logvar: Posterior log variance
            batch_size: Batch size
            time_steps: Number of time steps

        Returns:
            Dictionary containing loss components
        """
        # Compute reconstruction loss
        rec_loss = reconstruction_loss(predicted_logits, true_probabilities, self.alphabet_size, self.eps)

        # Compute KL divergence
        if self.use_learnable_prior and self.prior is not None:
            p_mean, p_logvar = self.prior(batch_size, time_steps)
            kl_loss = kl_divergence_with_learnable_prior(z_mean, z_logvar, p_mean, p_logvar)
        else:
            # Use standard unit Gaussian prior
            p_mean = torch.zeros_like(z_mean)
            p_logvar = torch.zeros_like(z_logvar)
            kl_loss = kl_divergence_with_learnable_prior(z_mean, z_logvar, p_mean, p_logvar)

        # Compute total loss
        total_loss = rec_loss + self.beta * kl_loss

        return {
            "total_loss": total_loss,
            "rec_loss": rec_loss,
            "kl_loss": kl_loss,
            "beta": self.beta
        }
