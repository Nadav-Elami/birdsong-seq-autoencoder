"""
LFADS model for birdsong analysis.

This module contains the BirdsongLFADSModel2 class and related utility functions
for LFADS-style birdsong analysis with rowwise softmax output.
"""


import torch
import torch.nn as nn
import torch.nn.functional as f


def rowwise_softmax(flat_factors: torch.Tensor, alphabet_size: int, order: int = 1) -> torch.Tensor:
    """
    Standard rowwise softmax:
    Reshape a (B, T, alphabet_size^(order+1)) tensor to
      (B, T, alphabet_size^order, alphabet_size),
    apply softmax over the last dimension, then flatten back.
    """
    B, T, _ = flat_factors.shape
    new_shape = (B, T, alphabet_size ** order, alphabet_size)
    mat = flat_factors.view(*new_shape)
    mat_softmax = f.softmax(mat, dim=-1)
    return mat_softmax.view(B, T, alphabet_size ** (order + 1))


def kl_div_gaussian(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                   mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """
    Compute KL(N(q|mu_q, var_q) || N(p|mu_p, var_p)) for diagonal Gaussians.

    KL = 0.5 * sum( log(var_p/var_q) + (var_q + (mu_q - mu_p)^2) / var_p - 1 )

    Args:
        mu_q: Mean of distribution q
        logvar_q: Log variance of distribution q
        mu_p: Mean of distribution p
        logvar_p: Log variance of distribution p

    Returns:
        KL divergence summed over latent dimension
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        logvar_p - logvar_q
        + (var_q + (mu_q - mu_p)**2) / var_p
        - 1.0
    )
    return kl.sum(dim=-1)


def rowwise_softmax(flat_factors: torch.Tensor, alphabet_size: int, order: int = 1) -> torch.Tensor:
    """
    Standard rowwise softmax.

    Reshape a (b, t, alphabet_size^(order+1)) tensor to
      (b, t, alphabet_size^order, alphabet_size),
    apply softmax over the last dimension, then flatten back.

    Args:
        flat_factors: Input tensor of shape (b, t, alphabet_size^(order+1))
        alphabet_size: Size of the alphabet
        order: Markov order (1 for bigram, 2 for trigram)

    Returns:
        Softmax-normalized tensor of same shape as input
    """
    b, t, _ = flat_factors.shape
    new_shape = (b, t, alphabet_size ** order, alphabet_size)
    mat = flat_factors.view(*new_shape)
    mat = f.softmax(mat, dim=-1)
    return mat.view(b, t, alphabet_size ** (order + 1))


def rowwise_masked_softmax(
    flat_factors: torch.Tensor,
    alphabet_size: int,
    order: int = 1,
    target_matrix: torch.Tensor | None = None,
    mask_value: float = -1e8,
    eps: float = 1e-12,
    tol: float = 1e-3
) -> torch.Tensor:
    """
    Applies a masked rowwise softmax to the input tensor.

    Args:
        flat_factors: Input tensor of shape (b, t, alphabet_size^(order+1))
        alphabet_size: Size of the alphabet
        order: Markov order (1 for bigram, 2 for trigram)
        target_matrix: Optional tensor of shape (alphabet_size^order, alphabet_size)
                      to derive mask from
        mask_value: Value to assign to disallowed positions
        eps: Small value to avoid log(0)
        tol: Tolerance for checking row-sum

    Returns:
        Masked softmax-normalized tensor of same shape as input
    """
    b, t, _ = flat_factors.shape
    new_shape = (b, t, alphabet_size ** order, alphabet_size)
    mat = flat_factors.view(*new_shape)

    # If a target_matrix is provided, derive the mask from it
    if target_matrix is not None:
        row_sums = target_matrix.sum(dim=1)
        allowed_rows = row_sums > (1 - tol)
        derived_mask = allowed_rows.unsqueeze(1).expand_as(target_matrix)
        mask = derived_mask
    else:
        mask = None

    # Apply mask if provided
    if mask is not None:
        mask = mask.unsqueeze(0).unsqueeze(0)
        mat = mat.masked_fill(~mask, mask_value)

    # Flatten and compute softmax
    mat_flat = mat.view(-1, alphabet_size)
    fully_masked = (mat_flat == mask_value).all(dim=1)
    soft = f.softmax(mat_flat, dim=1)
    soft = torch.where(fully_masked.unsqueeze(1), torch.zeros_like(soft), soft)

    # Reshape back
    soft = soft.view(b, t, alphabet_size ** order, alphabet_size)
    return soft.view(b, t, alphabet_size ** (order + 1))


def reconstruction_loss_rowwise(pred_dist: torch.Tensor, target: torch.Tensor,
                              alphabet_size: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Rowwise reconstruction loss.

    Args:
        pred_dist: Row-wise normalized predictions (batch, time, alpha^2)
        target: Target distributions (batch, time, alpha^2)
        alphabet_size: Size of the alphabet
        eps: Small value to avoid log(0)

    Returns:
        Cross-entropy loss averaged over batch/time
    """
    pred_dist = torch.clamp(pred_dist, eps, 1.0)
    ce = -target * torch.log(pred_dist)
    # Return scalar by averaging over all dimensions
    return ce.sum(dim=-1).mean()


class BirdsongLFADSModel2(nn.Module):
    """
    LFADS-style model for birdsong data with rowwise softmax output.

    The final "rates" become distributions over each row of an
    alphabet_size x alphabet_size matrix.
    """

    def __init__(
        self,
        alphabet_size: int,       # e.g. 7
        order: int = 1,
        encoder_dim: int = 64,    # per-direction for the g0 encoder
        controller_dim: int = 64, # per-direction for the controller encoder
        generator_dim: int = 64,  # generator RNN hidden size
        factor_dim: int = 32,     # dimension of factors f_t
        latent_dim: int = 16,     # dimension of g0
        inferred_input_dim: int = 8, # dimension of u_t
        kappa: float = 1.0,       # prior variance scaling for g0
        ar_step_size: float = 0.99, # alpha in AR(1)
        ar_process_var: float = 0.1 # sigma^2_e in AR(1)
    ):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.order = order
        self.out_dim = alphabet_size ** (order + 1)
        self.bigram_dim = self.out_dim
        self.latent_dim = latent_dim
        self.inferred_input_dim = inferred_input_dim
        self.factor_dim = factor_dim
        self.generator_dim = generator_dim
        self.kappa = kappa
        self.ar_step_size = ar_step_size
        self.ar_process_var = ar_process_var

        # g0 Encoder (bidirectional)
        self.g0_encoder = nn.GRU(
            input_size=self.bigram_dim,
            hidden_size=encoder_dim,
            batch_first=True,
            bidirectional=True
        )
        self.g0_linear = nn.Linear(2 * encoder_dim, 2 * latent_dim)

        # Controller (bidirectional) - FIXED: Added missing controller components
        self.controller_encoder = nn.GRU(
            input_size=self.bigram_dim,
            hidden_size=controller_dim,
            batch_first=True,
            bidirectional=True
        )
        self.controller_rnn = nn.GRU(
            input_size=2 * controller_dim + factor_dim,
            hidden_size=controller_dim,
            batch_first=True
        )
        self.u_linear = nn.Linear(controller_dim, 2 * inferred_input_dim)

        # Generator (bidirectional) - FIXED: Changed to bidirectional like example
        half_latent = latent_dim // 2  # must be integer!
        self.latent_to_gen_fwd = nn.Linear(half_latent, generator_dim)
        self.latent_to_gen_bwd = nn.Linear(half_latent, generator_dim)

        self.generator_rnn = nn.GRU(
            input_size=inferred_input_dim,
            hidden_size=generator_dim,
            batch_first=True,
            bidirectional=True
        )
        self.factors_linear = nn.Linear(2 * generator_dim, factor_dim)
        self.rate_linear = nn.Linear(factor_dim, self.bigram_dim)  # Now "logits"

        # AR(1) prior
        self.register_buffer("alpha_ar", torch.tensor(ar_step_size, dtype=torch.float))
        self.register_buffer("sigma2_e", torch.tensor(ar_process_var, dtype=torch.float))

    def encode_g0(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode g0 from input sequence."""
        _, h = self.g0_encoder(x)
        h = h.view(2, -1, self.g0_encoder.hidden_size)
        h = h.transpose(0, 1)
        h = h.reshape(h.size(0), -1)
        mu_logvar = self.g0_linear(h)
        mu_g0 = mu_logvar[:, :self.latent_dim]
        logvar_g0 = mu_logvar[:, self.latent_dim:]
        return mu_g0, logvar_g0

    def sample_g0(self, mu_g0: torch.Tensor, logvar_g0: torch.Tensor) -> torch.Tensor:
        """Sample g0 using reparameterization trick."""
        std_g0 = torch.exp(0.5 * logvar_g0)
        eps = torch.randn_like(std_g0)
        return mu_g0 + eps * std_g0

    def encode_u(self, x: torch.Tensor, factors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produce mu_u, logvar_u for each time step via:
          1) Bidirectional encoder -> e^con_t
          2) Forward controller RNN over t=1..T with input [e^con_t, f_{t-1}].
        """
        enc_out, _ = self.controller_encoder(x)  # (batch, time, 2*ctrl_dim)
        B, T, _ = enc_out.shape

        # Shift factors by one step so that f_{t-1} lines up with x_t
        f_shift = torch.zeros_like(factors)
        f_shift[:, 1:] = factors[:, :-1]

        # inputs to controller = cat(enc_out_t, f_shift_t)
        ctrl_in = torch.cat([enc_out, f_shift], dim=-1)  # (B, T, 2*ctrl_dim + factor_dim)
        ctrl_out, _ = self.controller_rnn(ctrl_in)       # (B, T, controller_dim)
        u_params = self.u_linear(ctrl_out)               # (B, T, 2*inferred_input_dim)
        mu_u, logvar_u = torch.chunk(u_params, 2, dim=-1)
        return mu_u, logvar_u

    def sample_u(self, mu_u: torch.Tensor, logvar_u: torch.Tensor) -> torch.Tensor:
        """Sample u using reparameterization trick."""
        std_u = torch.exp(0.5 * logvar_u)
        eps = torch.randn_like(std_u)
        return mu_u + eps * std_u

    def forward_generator(self, g0: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional generator RNN:
          1) Split g0 => (forward half, backward half)
          2) Transform each half => forward init, backward init
          3) Concatenate => shape (2, B, generator_dim)
          4) Run bidirectional generator => (B, T, 2*generator_dim)
          5) factors => W_factors(gen_out), shape (B, T, factor_dim)
          6) logits => W_rate(factors)
        """
        B, LD = g0.shape
        LD_half = LD // 2
        # e.g. if latent_dim=16 => LD_half=8 => g0_fwd=..., g0_bwd=...

        g0_fwd = g0[:, :LD_half]  # (B, LD_half)
        g0_bwd = g0[:, LD_half:]  # (B, LD_half)

        # Transform each half
        init_fwd = self.latent_to_gen_fwd(g0_fwd).unsqueeze(0)  # (1, B, generator_dim)
        init_bwd = self.latent_to_gen_bwd(g0_bwd).unsqueeze(0)  # (1, B, generator_dim)

        # shape => (2, B, generator_dim)
        g0_init = torch.cat([init_fwd, init_bwd], dim=0)

        # generator_rnn => bidirectional => output shape => (B, T, 2*generator_dim)
        gen_out, _ = self.generator_rnn(u, g0_init)

        # map 2*gdim => factor_dim
        f_t = self.factors_linear(gen_out)  # (B, T, factor_dim)

        logits = self.rate_linear(f_t)  # (B, T, bigram_dim)

        return f_t, logits

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Full forward pass: returns a dict with 'logits' as the un-normalized
        rowwise outputs, which you'll then softmax rowwise.
        """
        # 1) Encode g0
        mu_g0, logvar_g0 = self.encode_g0(x)
        g0_post = self.sample_g0(mu_g0, logvar_g0)

        # 2) Quick first pass with u=0 -> approximate factors
        B, T, _ = x.shape
        u_init = torch.zeros(B, T, self.inferred_input_dim, device=x.device)
        f_init, _ = self.forward_generator(g0_post, u_init)

        # 3) Encode u
        mu_u, logvar_u = self.encode_u(x, f_init)
        u_post = self.sample_u(mu_u, logvar_u)

        # 4) Final generator pass
        f_t, logits = self.forward_generator(g0_post, u_post)

        return {
            "logits": logits,  # (B, T, alpha^2) un-normalized
            "factors": f_t,
            "mu_g0": mu_g0,
            "logvar_g0": logvar_g0,
            "mu_u": mu_u,
            "logvar_u": logvar_u,
            "u": u_post,  # Added so compute_loss can access it
        }

    def compute_recurrent_l2_reg(self) -> torch.Tensor:
        """Compute L2 regularization for recurrent weights."""
        l2_reg = 0.0
        # Regularize controller RNN recurrent weights
        for name, param in self.controller_rnn.named_parameters():
            if "weight_hh" in name:
                l2_reg += torch.sum(param ** 2)
        # Regularize generator RNN recurrent weights
        for name, param in self.generator_rnn.named_parameters():
            if "weight_hh" in name:
                l2_reg += torch.sum(param ** 2)
        return l2_reg

    def compute_loss(self, x: torch.Tensor, outputs: dict[str, torch.Tensor] | None = None,
                    num_samples: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the total loss including reconstruction, KL divergence, and regularization.

        Args:
            x: Target data
            outputs: Model outputs (if None, will run forward pass)
            num_samples: Number of samples for Monte Carlo estimation

        Returns:
            Total loss and loss components dictionary
        """
        if outputs is None:
            outputs = self.forward(x)

        # Extract outputs
        u = outputs["u"]
        logits = outputs["logits"]
        mu_g0 = outputs["mu_g0"]
        logvar_g0 = outputs["logvar_g0"]
        mu_u = outputs["mu_u"]
        logvar_u = outputs["logvar_u"]

        # Compute reconstruction loss - match the example code exactly
        # Use appropriate softmax based on order
        if self.order == 1:
            predicted_probabilities = rowwise_softmax(logits, self.alphabet_size, order=self.order)
        elif self.order == 2:
            # For order 2, we need to derive a mask from the target data
            batch_size, time_steps, _ = x.shape
            new_shape = (batch_size, time_steps, self.alphabet_size ** self.order, self.alphabet_size)
            target_matrix = x.view(*new_shape)
            tol = 1e-3
            row_sums = target_matrix.sum(dim=-1)
            mask = row_sums > (1 - tol)
            derived_mask = mask[0, 0]  # Use first sample as representative
            final_mask = derived_mask.unsqueeze(1).expand(self.alphabet_size ** self.order, self.alphabet_size)
            
            predicted_probabilities = rowwise_masked_softmax(
                logits, 
                self.alphabet_size, 
                order=self.order,
                target_matrix=final_mask,
                mask_value=-1e8
            )
        else:
            raise ValueError(f"Unsupported order: {self.order}. Use order==1 or order==2.")
        
        # Add a small value to avoid numerical issues with log(0)
        epsilon = 1e-8
        predicted_probabilities = predicted_probabilities.clamp(min=epsilon)
        true_probabilities = x.clamp(min=epsilon)

        # Compute cross-entropy (row-wise comparison for each transition matrix row)
        cross_entropy_loss = -true_probabilities * torch.log(predicted_probabilities)  # Element-wise
        rec_loss = cross_entropy_loss.sum(dim=-1).mean()  # Sum over rows, average over batch/time

        # Compute KL divergence for g0
        kl_g0 = kl_div_gaussian(mu_g0, logvar_g0,
                               torch.zeros_like(mu_g0),
                               torch.log(torch.full_like(mu_g0, self.kappa)))

        # Compute KL divergence for u
        alpha = self.alpha_ar
        sigma2_e = self.sigma2_e
        sigma2_p = sigma2_e / (1.0 - alpha ** 2)  # stationary var

        # KL for first time step
        kl_u1 = kl_div_gaussian(mu_u[:, 0], logvar_u[:, 0],
                               torch.zeros_like(mu_u[:, 0]),
                               torch.log(torch.full_like(mu_u[:, 0], sigma2_p)))

        # KL for subsequent time steps
        kl_ut = 0.0
        for t in range(1, u.size(1)):
            prior_mean = alpha * mu_u[:, t-1]
            prior_logvar = torch.log(torch.full_like(prior_mean, sigma2_e))
            kl_t = kl_div_gaussian(mu_u[:, t], logvar_u[:, t], prior_mean, prior_logvar)
            kl_ut += kl_t.mean()

        # Reduce both KL components to scalars
        kl_u = kl_u1.mean() + kl_ut

        # L2 regularization
        l2_reg = self.compute_recurrent_l2_reg()

        # Total loss
        total_loss = rec_loss + kl_g0.mean() + kl_u + 0.01 * l2_reg

        loss_dict = {
            "total_loss": total_loss,
            "rec_loss": rec_loss,
            "kl_g0": kl_g0.mean(),
            "kl_u": kl_u,
            "l2_loss": l2_reg
        }

        return total_loss, loss_dict
